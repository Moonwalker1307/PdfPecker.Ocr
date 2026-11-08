"""A reference OCR server for bench/TesseractComparison: rapidocr (Apache-2.0) on ONNX Runtime, the engine whose models
PdfPecker.Ocr runs, so that the comparison has a reference for accuracy and time.

It speaks a small binary protocol on stdin and stdout, one request at a time:
  in:  "INIT <threads>\\n"                       -> out: "READY\\n"
  in:  "REQ <width> <height> <channels> <reset>\\n" + width*height*channels raw 8-bit pixels (RGB order for 3 channels)
       -> out: "L\\t<score>\\t<x0,y0,x1,y1,x2,y2,x3,y3>\\t<text>\\n" per line, "W\\t..." per word (same columns),
               "T\\t<engine_ms>\\t<cpu_ms>\\t<peak_kb>\\n", "END\\n"
  in:  "QUIT\\n"
Pixels are handed over as raw bytes so that no image decoding is timed; rapidocr itself resizes the page as its defaults do
(longest side limited to 2000 pixels). With reset=1 the process's peak resident set is reset first (Linux), so peak_kb is
the peak of this request alone. Run it with python -I from the venv, and keep it outside any downloaded folder.
"""
import resource
import sys
import time

import cv2
import numpy as np


def read_line():
    return sys.stdin.buffer.readline().decode("utf-8").strip()


def read_exact(n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sys.stdin.buffer.read(n - len(buf))
        if not chunk:
            raise EOFError
        buf.extend(chunk)
    return bytes(buf)


def reset_peak():
    try:
        with open("/proc/self/clear_refs", "w") as f:
            f.write("5")
        return True
    except OSError:
        return False


def peak_kb():
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1])
    except OSError:
        pass
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def fmt_box(box):
    return ",".join(str(int(round(v))) for p in box for v in p)


def main():
    from rapidocr import RapidOCR

    out = sys.stdout.buffer
    engine = None
    while True:
        line = read_line()
        if not line or line == "QUIT":
            return
        parts = line.split()
        if parts[0] == "INIT":
            threads = int(parts[1])
            params = {"Global.return_word_box": True}
            if threads > 0:
                params["EngineConfig.onnxruntime.intra_op_num_threads"] = threads
                params["EngineConfig.onnxruntime.inter_op_num_threads"] = 1
            engine = RapidOCR(params=params)
            out.write(b"READY\n")
            out.flush()
        elif parts[0] == "REQ":
            w, h, c, reset = (int(v) for v in parts[1:5])
            data = read_exact(w * h * c)
            img = np.frombuffer(data, dtype=np.uint8).reshape(h, w, c) if c > 1 else np.frombuffer(data, dtype=np.uint8).reshape(h, w)
            if c == 1:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            if reset:
                reset_peak()
            cpu0 = time.process_time()
            t0 = time.perf_counter()
            result = engine(img)
            engine_ms = (time.perf_counter() - t0) * 1000
            cpu_ms = (time.process_time() - cpu0) * 1000
            peak = peak_kb()
            lines = []
            if result.txts is not None and result.boxes is not None:
                for box, txt, score in zip(result.boxes, result.txts, result.scores):
                    lines.append(f"L\t{score:.4f}\t{fmt_box(box)}\t{txt}\n")
                words = result.word_results or ()
                for word_line in words:
                    for item in word_line or ():
                        if item is None or len(item) < 3 or item[2] is None:
                            continue
                        txt, score, box = item[0], item[1], item[2]
                        lines.append(f"W\t{score:.4f}\t{fmt_box(box)}\t{txt}\n")
            for l in lines:
                out.write(l.replace("\r", " ").encode("utf-8"))
            out.write(f"T\t{engine_ms:.3f}\t{cpu_ms:.3f}\t{peak}\n".encode())
            out.write(b"END\n")
            out.flush()


if __name__ == "__main__":
    main()
