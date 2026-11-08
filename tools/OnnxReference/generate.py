# Regenerates tests/PdfPecker.Ocr.Tests/Fixtures/onnx/*.ref.gz: ONNX Runtime's outputs for the three bundled models on
# fixed inputs, which tests/PdfPecker.Ocr.Tests/Inference/ReferenceTests.cs compares PdfPecker.Ocr's engine with.
# Development tool only; nothing here ships.
#
# Usage (from the repository root), with Python 3.10 or later:
#   python -m pip install onnxruntime==1.31.1 numpy opencv-python-headless
#   python -I tools/OnnxReference/generate.py [--wheel path/to/rapidocr-3.10.0-py3-none-any.whl]
# Without --wheel it reads the wheel that tools/ModelBuilder downloads into its cache folder in the temporary directory.
# The wheel and the models are checked against the same SHA-256 hashes as the model builder.
#
# Inputs are rendered text (OpenCV's Hershey fonts) on a light background, stored in the fixture as 8-bit pixels; the
# model input is (pixel / 255 - 0.5) / 0.5 in float32, channel c of the tensor taken from byte c of each pixel. ONNX
# Runtime runs on the CPU with one thread and its default graph optimisations, as rapidocr uses it.
#
# Fixture layout, little-endian, gzip-compressed (mtime 0, so reruns are byte-identical):
#   "OREF", u32 version (1), u8 name length, name ("det", "rec" or "cls")
#   u8 input rank, i32 dims (N, 3, H, W), then N*H*W*3 pixel bytes in N, H, W, C order
#   u8 output rank, i32 dims
#   f64 mean, f64 standard deviation, f32 minimum, f32 maximum of the whole output
#   u32 count, then (u32 flat index, f32 value) pairs: all values when the output is small, otherwise a sample (see
#       sample_indices: random positions, plus the uncertain pixels of a detection map, plus the five largest
#       probabilities of every time step of the recogniser)
#   u32 rows, then per row of the last axis (time steps for rec, items for cls; 0 rows for det): u32 argmax, f32 max
import argparse
import gzip
import hashlib
import io
import os
import struct
import sys
import tempfile
import zipfile

import cv2
import numpy as np
import onnxruntime as ort

WHEEL_SHA256 = "2fc34e26cd0f48514804a94f8832283039c7334076bff60768b1fed9ad068050"
MODELS = {
    "det": ("rapidocr/models/PP-OCRv6_det_small.onnx", "090f04abcd9d9a7498bc4ebf677e4cb9bdce1fe4197ddb7e529f1ef44e1ff94f"),
    "rec": ("rapidocr/models/PP-OCRv6_rec_small.onnx", "6f327246b50388f3c176ae304bd95767ea6dc0c9ae92153ef8cbe210b3c14884"),
    "cls": ("rapidocr/models/ch_ppocr_mobile_v2.0_cls_mobile.onnx", "e47acedf663230f8863ff1ab0e64dd2d82b838fceb5957146dab185a89d6215c"),
}
FULL_LIMIT = 16384
SAMPLES = 4096

root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
out_dir = os.path.join(root, "tests", "PdfPecker.Ocr.Tests", "Fixtures", "onnx")


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def text_image(height, width, lines, scale, thickness, seed, background=(245, 242, 238), ink=(20, 24, 30)):
    """A light page with dark text lines and a little deterministic noise, as a scan would have."""
    img = np.empty((height, width, 3), np.uint8)
    img[:] = background
    y = int(round(30 * scale)) + 4
    for i, line in enumerate(lines):
        font = cv2.FONT_HERSHEY_SIMPLEX if i % 2 == 0 else cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(img, line, (6, y), font, scale, ink, thickness, cv2.LINE_AA)
        y += int(round(38 * scale))
    noise = np.random.default_rng(seed).integers(-6, 7, img.shape)
    return np.clip(img.astype(np.int32) + noise, 0, 255).astype(np.uint8)


def line_image(height, width, text, scale, seed):
    img = text_image(height, width, [], scale, 1, seed)
    cv2.putText(img, text, (4, int(height * 0.72)), cv2.FONT_HERSHEY_SIMPLEX, scale, (20, 24, 30), 2, cv2.LINE_AA)
    return img


def inputs():
    """(model, name, uint8 images [N, H, W, 3]) for every fixture."""
    page = ["Invoice 2026-0042", "Example Customer Ltd.", "Total due: EUR 1,234.56", "VAT 25% included"]
    yield "det", "det_96x128", text_image(96, 128, ["Total 42", "EUR 9.50"], 0.8, 2, 1)[None]
    yield "det", "det_320x320", text_image(320, 320, page + ["The quick brown fox", "jumps over 13 dogs"], 0.85, 2, 2)[None]
    yield "rec", "rec_1x48x320", line_image(48, 320, "Invoice 2026-0042", 1.0, 3)[None]
    yield "rec", "rec_2x48x160", np.stack([line_image(48, 160, "Total 98.50", 0.9, 4), line_image(48, 160, "Hello World", 0.9, 5)])
    upright = line_image(48, 192, "Example Ltd.", 0.9, 6)
    yield "cls", "cls_2x48x192", np.stack([upright, upright[::-1, ::-1]])


def to_tensor(images):
    x = images.astype(np.float32) / np.float32(255.0)
    x = (x - np.float32(0.5)) / np.float32(0.5)
    return np.ascontiguousarray(x.transpose(0, 3, 1, 2))


def sample_indices(model, output, rng):
    flat = output.ravel()
    if flat.size <= FULL_LIMIT:
        return np.arange(flat.size)
    picked = set(rng.choice(flat.size, SAMPLES, replace=False).tolist())
    if model == "det":
        uncertain = np.flatnonzero((flat > 0.05) & (flat < 0.95))
        if uncertain.size > SAMPLES:
            uncertain = rng.choice(uncertain, SAMPLES, replace=False)
        picked.update(uncertain.tolist())
    if model == "rec":
        classes = output.shape[-1]
        rows = output.reshape(-1, classes)
        top = np.argsort(-rows, axis=1, kind="stable")[:, :5]
        for r in range(rows.shape[0]):
            picked.update((r * classes + top[r]).tolist())
    return np.array(sorted(picked), dtype=np.int64)


def write_fixture(path, model, images, output):
    rng = np.random.default_rng(12345)
    buf = io.BytesIO()
    buf.write(b"OREF")
    buf.write(struct.pack("<I", 1))
    buf.write(struct.pack("<B", len(model)) + model.encode("ascii"))
    n, h, w, _ = images.shape
    buf.write(struct.pack("<B", 4) + struct.pack("<4i", n, 3, h, w))
    buf.write(images.tobytes())
    buf.write(struct.pack("<B", output.ndim) + struct.pack(f"<{output.ndim}i", *output.shape))
    flat = output.ravel().astype(np.float32)
    buf.write(struct.pack("<ddff", float(flat.astype(np.float64).mean()), float(flat.astype(np.float64).std()), float(flat.min()), float(flat.max())))
    idx = sample_indices(model, output, rng)
    buf.write(struct.pack("<I", idx.size))
    pairs = np.empty(idx.size, dtype=[("i", "<u4"), ("v", "<f4")])
    pairs["i"] = idx
    pairs["v"] = flat[idx]
    buf.write(pairs.tobytes())
    if model == "det":
        buf.write(struct.pack("<I", 0))
    else:
        rows = output.reshape(-1, output.shape[-1])
        buf.write(struct.pack("<I", rows.shape[0]))
        best = np.empty(rows.shape[0], dtype=[("i", "<u4"), ("v", "<f4")])
        best["i"] = rows.argmax(axis=1)
        best["v"] = rows.max(axis=1)
        buf.write(best.tobytes())
    with open(path, "wb") as f:
        with gzip.GzipFile(fileobj=f, mode="wb", compresslevel=9, mtime=0, filename="") as gz:
            gz.write(buf.getvalue())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wheel", default=os.path.join(tempfile.gettempdir(), "pdfpecker-ocr-modelbuilder", "rapidocr-3.10.0-py3-none-any.whl"))
    args = parser.parse_args()
    wheel = open(args.wheel, "rb").read()
    if sha256(wheel) != WHEEL_SHA256:
        sys.exit(f"{args.wheel} is not the pinned rapidocr 3.10.0 wheel")
    sessions = {}
    with zipfile.ZipFile(io.BytesIO(wheel)) as z:
        for name, (entry, digest) in MODELS.items():
            data = z.read(entry)
            if sha256(data) != digest:
                sys.exit(f"{entry} does not match its pinned hash")
            options = ort.SessionOptions()
            options.intra_op_num_threads = 1
            options.inter_op_num_threads = 1
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            sessions[name] = ort.InferenceSession(data, options, providers=["CPUExecutionProvider"])
    os.makedirs(out_dir, exist_ok=True)
    for model, name, images in inputs():
        session = sessions[model]
        output = session.run(None, {session.get_inputs()[0].name: to_tensor(images)})[0]
        path = os.path.join(out_dir, name + ".ref.gz")
        write_fixture(path, model, images, output)
        extra = ""
        if model != "det":
            extra = " argmax " + str(output.reshape(-1, output.shape[-1]).argmax(axis=1).tolist()[:48])
        print(f"{name}: output {list(output.shape)} -> {os.path.relpath(path, root)} ({os.path.getsize(path)} bytes){extra}")
    print(f"onnxruntime {ort.__version__}, numpy {np.__version__}, opencv {cv2.__version__}")


if __name__ == "__main__":
    main()
