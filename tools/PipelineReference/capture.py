"""Records what rapidocr 3.10.0 does on a few rendered test images, as fixtures for the pipeline tests.

This is a development tool, not part of the package. It renders the test images with Pillow and system fonts, runs
rapidocr (Apache-2.0, https://pypi.org/project/rapidocr/3.10.0/) with ONNX Runtime and hooks into its stages, so the C#
tests can check each stage of PdfPecker.Ocr's pipeline against the reference on the same inputs:

- the image rapidocr's detector saw (after its own resizing and padding) and the det input size;
- the det probability map, quantised to 8 bits, and DBPostProcess's boxes and scores computed on the *dequantised*
  map, so the C# post-processing gets exactly the input the expected boxes came from;
- each line crop's rec input tensor (dequantised to 8-bit pixels), the cls outputs, and the rec outputs as the top
  three classes per time step, with rapidocr's decoded texts, scores and word boxes.

Usage (from any directory, with the venv that has rapidocr, onnxruntime, opencv and Pillow):

    python -I tools/PipelineReference/capture.py <output folder>

The output folder is tests/PdfPecker.Ocr.Tests/Fixtures/pipeline. Each case writes <case>.json.gz and gzip files of
raw bytes: <case>.image.gz, <case>.detimage.gz when rapidocr resized or padded the image, <case>.detinput.gz,
<case>.prob.gz and, for a few cases, <case>.rec.gz. The JSON says the shapes. The character list of the rec model
is written to characters.txt.gz (one entry per line, without the blank and the space that rapidocr adds).
"""

import gzip
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from rapidocr import RapidOCR

FONTS = Path("/usr/share/fonts/truetype")
SANS = FONTS / "dejavu" / "DejaVuSans.ttf"
SANS_BOLD = FONTS / "dejavu" / "DejaVuSans-Bold.ttf"
SERIF = FONTS / "liberation" / "LiberationSerif-Regular.ttf"
LSANS = FONTS / "liberation" / "LiberationSans-Regular.ttf"
MONO = FONTS / "liberation" / "LiberationMono-Regular.ttf"


def font(path, size):
    return ImageFont.truetype(str(path), size)


def blank(w, h):
    return Image.new("L", (w, h), 255)


def render_invoice():
    """An A4 page at 150 dpi: letterhead, address, a table with amounts and a total line with a wide gap."""
    img = blank(1240, 1754)
    d = ImageDraw.Draw(img)
    d.text((100, 90), "Example Supplies Ltd.", font=font(SANS_BOLD, 40), fill=0)
    d.text((100, 145), "12 Harbour Road, 8000 Aarhus, Denmark", font=font(SANS, 20), fill=0)
    d.text((100, 260), "Invoice 2026-0412", font=font(SANS_BOLD, 30), fill=0)
    d.text((820, 260), "Date: 8 November 2026", font=font(SANS, 20), fill=0)
    d.text((100, 330), "Bill to: Example Customer Ltd.", font=font(SANS, 22), fill=0)
    d.text((100, 362), "Attn. Accounts Payable, Main Street 5", font=font(SANS, 22), fill=0)
    y = 470
    d.line((100, y - 12, 1140, y - 12), fill=0, width=2)
    for x, t in ((100, "Item"), (640, "Qty"), (780, "Price"), (980, "Amount")):
        d.text((x, y), t, font=font(SANS_BOLD, 22), fill=0)
    d.line((100, y + 40, 1140, y + 40), fill=0, width=1)
    rows = [
        ("Printer paper A4, 80 g", "10", "4.95", "49.50"),
        ("Toner cartridge, black", "2", "89.00", "178.00"),
        ("Desk lamp with LED bulb", "1", "34.90", "34.90"),
        ("Archive boxes (pack of 5)", "4", "12.25", "49.00"),
        ("Delivery", "1", "15.00", "15.00"),
    ]
    y += 60
    for item, qty, price, amount in rows:
        d.text((100, y), item, font=font(SANS, 22), fill=0)
        d.text((640, y), qty, font=font(SANS, 22), fill=0)
        d.text((780, y), price, font=font(SANS, 22), fill=0)
        d.text((980, y), amount, font=font(SANS, 22), fill=0)
        y += 44
    d.line((100, y + 4, 1140, y + 4), fill=0, width=1)
    y += 30
    d.text((640, y), "Subtotal", font=font(SANS, 22), fill=0)
    d.text((980, y), "326.40", font=font(SANS, 22), fill=0)
    y += 60
    # The total and the VAT rate sit on one line with a gap the detector closes, so rapidocr reads one box.
    d.text((560, y), "Total EUR 1,234.56", font=font(SANS_BOLD, 22), fill=0)
    d.text((842, y), "VAT 25%", font=font(SANS_BOLD, 22), fill=0)
    y += 120
    body = [
        "Payment within 30 days of the invoice date. Please quote the invoice number",
        "with your payment. Bank transfer to IBAN DK50 0040 0440 1162 43, BIC EXAMDKKK.",
        "Thank you for your order. Questions about this invoice: accounts@example.com",
    ]
    for line in body:
        d.text((100, y), line, font=font(SERIF, 21), fill=0)
        y += 34
    d.text((100, 1650), "Page 1 of 1", font=font(SANS, 16), fill=0)
    return img


def render_small_text():
    """Small text: 11 and 13 pixel fonts, the size of 8 pt text on a page rendered at about 100 dpi."""
    img = blank(900, 260)
    d = ImageDraw.Draw(img)
    lines = [
        (11, "Terms and conditions apply. Prices include VAT where stated."),
        (11, "Goods remain our property until paid in full (section 4.2)."),
        (13, "The quick brown fox jumps over the lazy dog 0123456789."),
        (13, "Customer number 55-1023, order reference PO/2026/118."),
    ]
    y = 30
    for size, t in lines:
        d.text((40, y), t, font=font(LSANS, size), fill=0)
        y += size * 3
    return img


def paste_rotated(img, text, fnt, center, angle):
    """Draws text on its own layer, rotates it counter-clockwise by angle degrees and pastes it centred at center."""
    bbox = fnt.getbbox(text)
    w, h = bbox[2] + 20, bbox[3] + 20
    layer = Image.new("L", (w, h), 0)
    ImageDraw.Draw(layer).text((10, 10), text, font=fnt, fill=255)
    layer = layer.rotate(angle, expand=True, resample=Image.BICUBIC)
    x, y = int(center[0] - layer.width / 2), int(center[1] - layer.height / 2)
    img.paste(0, (x, y), layer)


def render_rotated():
    """Lines rotated by 8 and -15 degrees, and a vertical line running bottom to top (rotated 90 degrees)."""
    img = blank(1000, 700)
    d = ImageDraw.Draw(img)
    d.text((60, 40), "Horizontal reference line", font=font(SANS, 30), fill=0)
    paste_rotated(img, "Slightly tilted scan line", font(SANS, 30), (420, 200), 8)
    paste_rotated(img, "Rotated the other way round", font(SERIF, 32), (430, 420), -15)
    paste_rotated(img, "Vertical spine label", font(SANS_BOLD, 30), (900, 380), 90)
    return img


def render_columns():
    """Two columns of a newsletter, set close enough that the columns must be told apart by layout."""
    img = blank(1100, 640)
    d = ImageDraw.Draw(img)
    d.text((60, 40), "Quarterly Newsletter", font=font(SANS_BOLD, 36), fill=0)
    left = [
        "Our new office opens in",
        "March. The move will take",
        "two weeks, and phones stay",
        "on throughout the move.",
        "Parking is free for guests.",
    ]
    right = [
        "The annual meeting is held",
        "on 14 May at 10:00. Please",
        "register by 30 April so we",
        "can plan the catering for",
        "everyone who attends.",
    ]
    y = 140
    for a, b in zip(left, right):
        d.text((60, y), a, font=font(SERIF, 28), fill=0)
        d.text((580, y), b, font=font(SERIF, 28), fill=0)
        y += 44
    d.text((60, 520), "Contact: office@example.org, phone +45 12 34 56 78", font=font(SANS, 24), fill=0)
    return img


def render_table():
    """A ruled table with numbers, where grid lines touch the cell text boxes."""
    img = blank(900, 420)
    d = ImageDraw.Draw(img)
    cols = [40, 300, 520, 700, 860]
    rows = [40, 100, 160, 220, 280, 340]
    for x in cols:
        d.line((x, rows[0], x, rows[-1]), fill=0, width=2)
    for y in rows:
        d.line((cols[0], y, cols[-1], y), fill=0, width=2)
    data = [
        ("Region", "Units", "Revenue", "Share"),
        ("North", "1,204", "48,160", "31%"),
        ("South", "987", "39,480", "25%"),
        ("East", "1,530", "61,200", "39%"),
        ("West", "201", "8,040", "5%"),
    ]
    for r, row in enumerate(data):
        for c, t in enumerate(row):
            f = font(SANS_BOLD if r == 0 else SANS, 24)
            d.text((cols[c] + 14, rows[r] + 16), t, font=f, fill=0)
    return img


def render_upside_down():
    """A short letter turned 180 degrees, as a page scanned upside down."""
    img = blank(900, 360)
    d = ImageDraw.Draw(img)
    lines = ["Dear Sir or Madam,", "please find enclosed the signed contract.", "Kind regards, Anna Jensen"]
    y = 50
    for t in lines:
        d.text((50, y), t, font=font(SERIF, 34), fill=0)
        y += 80
    return img.rotate(180)


def render_nordic():
    """Danish and German text with æ, ø, å, ä, ö, ü and ß, in colour on a tinted background."""
    img = Image.new("RGB", (1000, 420), (250, 246, 232))
    d = ImageDraw.Draw(img)
    lines = [
        ("Rødgrød med fløde og æblekage på Ærø", (20, 20, 120)),
        ("Blåbærsyltetøj, smørrebrød og øl i Århus", (0, 0, 0)),
        ("Größere Straßen führen über die Brücke", (120, 10, 10)),
        ("Schöne Grüße aus München, Köln und Düsseldorf", (0, 0, 0)),
    ]
    y = 40
    for t, color in lines:
        d.text((40, y), t, font=font(SANS, 32), fill=color)
        y += 90
    return img


def render_gap():
    """Amounts and a VAT rate at growing distances in DejaVu Sans 16, where rapidocr 3.10 reads one line as
    "EUR 1,234.56VAT 25%": the detector joins the two parts and the recogniser emits no space between them."""
    f = font(SANS, 16)
    gaps = [int(16 * k) for k in (0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0)]
    img = blank(1000, 48 * len(gaps) + 40)
    d = ImageDraw.Draw(img)
    for i, gap in enumerate(gaps):
        y = 20 + 48 * i
        d.text((40, y), "EUR 1,234.56", font=f, fill=0)
        d.text((40 + f.getlength("EUR 1,234.56") + gap, y), "VAT 25%", font=f, fill=0)
    return img


CASES = {
    "invoice": render_invoice,
    "small_text": render_small_text,
    "rotated": render_rotated,
    "columns": render_columns,
    "table": render_table,
    "upside_down": render_upside_down,
    "nordic": render_nordic,
    "gap": render_gap,
}


# The cases whose rec input tensors are stored, for the crop tests.
CROP_CASES = {"rotated", "upside_down", "table", "nordic", "gap"}


class Recorder:
    """Wraps an ONNX Runtime session of rapidocr and keeps every input and output it sees."""

    def __init__(self, session):
        self.session = session
        self.calls = []

    def __call__(self, x):
        y = self.session(x)
        self.calls.append((x, y))
        return y

    def __getattr__(self, name):
        return getattr(self.session, name)


def gz(path, data):
    # A fixed mtime keeps the files byte-identical between runs.
    with open(path, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", compresslevel=9, mtime=0) as f:
        f.write(data)


def to_pixels(t):
    """Undoes rapidocr's (px / 255 - 0.5) / 0.5 normalisation, which maps back to whole pixel values exactly."""
    return np.clip(np.rint((t * 0.5 + 0.5) * 255.0), 0, 255).astype(np.uint8)


def capture_case(ocr, name, img, out):
    if img.mode == "RGB":
        rgb = np.asarray(img)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        image_bytes, channels = rgb.tobytes(), 3
    else:
        gray = np.asarray(img)
        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        image_bytes, channels = gray.tobytes(), 1
    gz(out / f"{name}.image.gz", image_bytes)

    det, cls, rec = ocr.text_det, ocr.text_cls, ocr.text_rec
    det.session.calls.clear()
    cls.session.calls.clear()
    rec.session.calls.clear()

    # Record what the stages see: the image the detector gets (after preprocessing and padding) and the crops.
    seen = {}
    orig_det_call = det.__class__.__call__

    def det_call(self, image):
        seen["det_image"] = image
        return orig_det_call(self, image)

    det.__class__.__call__ = det_call
    orig_rec_call = rec.__class__.__call__

    def rec_call(self, args):
        seen["rec_images"] = [i.copy() for i in args.img]
        return orig_rec_call(self, args)

    rec.__class__.__call__ = rec_call
    orig_cls_call = cls.__class__.__call__

    def cls_call(self, images):
        seen["cls_images"] = [i.copy() for i in images]
        return orig_cls_call(self, images)

    cls.__class__.__call__ = cls_call
    try:
        result = ocr(bgr, return_word_box=True)
    finally:
        det.__class__.__call__ = orig_det_call
        rec.__class__.__call__ = orig_rec_call
        cls.__class__.__call__ = orig_cls_call

    det_image = seen["det_image"]
    det_in, det_out = det.session.calls[0]
    prob = det_out[0, 0]
    q = np.clip(np.rint(prob * 255.0), 0, 255).astype(np.uint8)
    gz(out / f"{name}.prob.gz", q.tobytes())

    # The expected boxes come from running rapidocr's own post-processing on the dequantised map.
    dq = (q.astype(np.float32) / 255.0)[None, None]
    src_h, src_w = det_image.shape[:2]
    pp = det.postprocess_op
    raw_boxes, raw_scores = pp.boxes_from_bitmap(
        dq[:, 0][0],
        cv2.dilate((dq[0, 0] > pp.thresh).astype(np.uint8), np.array([[1, 1], [1, 1]])),
        src_w,
        src_h,
    )
    boxes, scores = pp(dq, (src_h, src_w))
    sorted_boxes = det.sorted_boxes(boxes) if len(boxes) else boxes
    # rapidocr sorts after filtering; scores are not reordered by sorted_boxes, so record them per filtered box.

    det_image_gray = det_image[:, :, 1] if channels == 1 else cv2.cvtColor(det_image, cv2.COLOR_BGR2RGB)
    pre = {
        "ratio_h": float(ocr_preprocess_ratio(ocr, img)[0]),
        "ratio_w": float(ocr_preprocess_ratio(ocr, img)[1]),
        "det_image_width": int(src_w),
        "det_image_height": int(src_h),
        "det_input_shape": list(det_in.shape),
    }
    # The image the detector saw is stored only when rapidocr's preprocessing changed it.
    det_image_is_image = det_image_gray.tobytes() == image_bytes
    if not det_image_is_image:
        gz(out / f"{name}.detimage.gz", np.ascontiguousarray(det_image_gray).tobytes())
    # The network input as pixels: plane 0 only for grey images, else the B, G and R planes rapidocr feeds.
    det_planes = to_pixels(det_in[0, :1] if channels == 1 else det_in[0])
    gz(out / f"{name}.detinput.gz", np.ascontiguousarray(det_planes).tobytes())
    det_input_ok = bool(np.array_equal(to_pixels(det_in[0, 0]), cv2.resize(det_image, (det_in.shape[3], det_in.shape[2]))[:, :, 0]))

    # Classifier: inputs in batch order with their outputs; rapidocr sorts crops by aspect ratio first.
    cls_batch = np.concatenate([c[1] for c in cls.session.calls]) if cls.session.calls else np.zeros((0, 2))
    cls_shapes = [list(c[0].shape) for c in cls.session.calls]
    crops = seen["cls_images"]
    cls_order = np.argsort(np.array([i.shape[1] / float(i.shape[0]) for i in crops]))
    cls_out = np.zeros_like(cls_batch)
    for k, idx in enumerate(cls_order):
        cls_out[int(idx)] = cls_batch[k]
    crop_sizes = [[int(i.shape[1]), int(i.shape[0])] for i in crops]

    # Recognition: each batch's input tensor and the top three classes per time step of each line.
    rec_images = seen["rec_images"]
    order = np.argsort(np.array([i.shape[1] / float(i.shape[0]) for i in rec_images]))
    rec_lines = []
    rec_pixels = bytearray()
    k = 0
    for x, y in rec.session.calls:
        for b in range(x.shape[0]):
            idx = int(order[k])
            k += 1
            top = np.argsort(-y[b], axis=1)[:, :3]
            probs = np.take_along_axis(y[b], top, axis=1)
            # Crop pixels are kept for a few cases only, to keep the fixtures small; grey images keep one plane.
            keep = name in CROP_CASES
            px = to_pixels(x[b][:1] if channels == 1 else x[b]) if keep else None
            rec_lines.append(
                {
                    "line": idx,
                    "crop_width": int(rec_images[idx].shape[1]),
                    "crop_height": int(rec_images[idx].shape[0]),
                    "input_shape": list(x.shape),
                    "pixel_offset": len(rec_pixels) if keep else -1,
                    "top_classes": top.astype(int).tolist(),
                    "top_probs": [[float(f"{v:.6g}") for v in row] for row in probs],
                }
            )
            if keep:
                rec_pixels += px.tobytes()
    if name in CROP_CASES:
        gz(out / f"{name}.rec.gz", bytes(rec_pixels))

    final = []
    if result is not None and getattr(result, "txts", None):
        for i, (box, txt, score) in enumerate(zip(result.boxes, result.txts, result.scores)):
            words = []
            if result.word_results and i < len(result.word_results):
                for w in result.word_results[i]:
                    words.append({"text": w[0], "score": float(w[1]), "box": np.asarray(w[2]).astype(int).tolist()})
            final.append(
                {"box": np.asarray(box).astype(float).round(2).tolist(), "text": txt, "score": float(score), "words": words}
            )

    rec_texts = rec_texts_by_line(ocr, rec, rec_images)
    doc = {
        "case": name,
        "width": img.width,
        "height": img.height,
        "channels": channels,
        "preprocess": pre,
        "det_input_matches_resize": det_input_ok,
        "det_image_is_image": det_image_is_image,
        "det_raw_boxes": np.asarray(raw_boxes).astype(int).tolist(),
        "det_raw_scores": [float(s) for s in raw_scores],
        "det_boxes": np.asarray(boxes).astype(float).tolist(),
        "det_scores": [float(s) for s in scores],
        "det_sorted_boxes": np.asarray(sorted_boxes).astype(float).tolist(),
        "orig_map_box_count": None,
        "crop_sizes": crop_sizes,
        "cls_shapes": cls_shapes,
        "cls_outputs": cls_out.astype(float).round(6).tolist(),
        "rec_lines": sorted(rec_lines, key=lambda r: r["line"]),
        "rec_texts": rec_texts,
        "final": final,
    }
    # The box count on the unquantised map, to show the quantisation does not change the detection.
    pp_boxes, _ = pp(det_out, (src_h, src_w))
    doc["orig_map_box_count"] = int(len(pp_boxes))
    gz(out / f"{name}.json.gz", json.dumps(doc, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    print(name, img.size, "det", det_in.shape, "boxes", len(boxes), "lines", len(final))
    for f in final:
        print("   ", f["text"], round(f["score"], 3), [w["text"] for w in f["words"]])


def rec_texts_by_line(ocr, rec, rec_images):
    """rapidocr's decoded text and score for each crop, from the recorded outputs, before any score filtering."""
    order = np.argsort(np.array([i.shape[1] / float(i.shape[0]) for i in rec_images]))
    texts = [None] * len(rec_images)
    k = 0
    for x, y in rec.session.calls:
        res, _ = rec.postprocess_op(y, False)
        for t, s in res:
            texts[int(order[k])] = {"text": t, "score": float(s)}
            k += 1
    return texts


def ocr_preprocess_ratio(ocr, img):
    arr = np.zeros((img.height, img.width, 3), np.uint8)
    _, rec = ocr.preprocess_img(arr)
    return rec["preprocess"]["ratio_h"], rec["preprocess"]["ratio_w"]


def main():
    if len(sys.argv) != 2:
        print("usage: capture.py <output folder>", file=sys.stderr)
        sys.exit(2)
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    ocr = RapidOCR(params={"Global.log_level": "warning"})
    ocr._load_det_model()
    ocr._load_cls_model()
    ocr._load_rec_model()
    ocr.text_det.session = Recorder(ocr.text_det.session)
    ocr.text_cls.session = Recorder(ocr.text_cls.session)
    ocr.text_rec.session = Recorder(ocr.text_rec.session)

    chars = ocr.text_rec.session.get_character_list()
    gz(out / "characters.txt.gz", "\n".join(chars).encode("utf-8"))

    for name, render in CASES.items():
        capture_case(ocr, name, render(), out)


if __name__ == "__main__":
    main()
