# Benchmarks

- `PdfPecker.Ocr.Benchmarks` – timing, allocation and peak memory of PdfPecker.Ocr itself: a page at 300 and 150 dpi, one thread and
  all cores, and a searchable PDF, through the public API (`-p:OcrApi=false` lists them as pending, for comparisons with a
  version from before the API). Behind `-p:OcrInternals=true`: detection and recognition on their own, the `stages`
  command, which reads an A4 invoice page at 300 dpi cold and warm and splits its time by stage, and the `layers` command,
  which times each network on that page layer by layer (`--threads n --runs n --top n`). Run `dotnet run -c Release
  --project bench/PdfPecker.Ocr.Benchmarks`. Optional `--corpus bench/TesseractComparison/corpus` adds corpus pages, `--threads n`
  and `--filter text` select cases.
  `profile [--corpus dir] [--only id,=id] [--threads n] [--runs n]` (also behind `-p:OcrInternals=true`) reads corpus pages with
  every layer of the three networks timed, and prints the time by operator, the slowest layers and the recognizer's batches with
  the share of their time steps that belongs to the lines rather than to padding.
- `compare-versions.sh [git-ref] [rounds]` – the working tree against another version on this machine, in turns, each in a fresh
  process (the ref is built with this version's benchmark program, so only the library differs). Pass the feature switches both
  versions understand in `OCR_BENCH_BUILD_ARGS`, for example `OCR_BENCH_BUILD_ARGS=-p:OcrInternals=true bench/compare-versions.sh v1.24.0`
  (the API cases are on by default; `-p:OcrApi=false` for a version from before the API).
  As in PdfPecker, only same-machine comparisons are meaningful, and the GC gen0 budget is fixed (`DOTNET_GCgen0size`) so that
  peak memory compares across machines.
- `TesseractComparison` – accuracy, speed and memory against Tesseract and rapidocr on the synthetic corpus; see its README. On
  2026-11-09 (4-vCPU VM, otherwise idle, 65 pages) PdfPecker.Ocr had the lowest error rates (CER 1.7% in reading order, 0.4%
  ignoring it, against 10.4% and 3.3% for Tesseract 5.5.4 with `tessdata_fast`) and took 2.12 s per page at one thread and
  0.58 s at four, against 0.61 s and 0.57 s for Tesseract and 5.2 s and 1.9 s for rapidocr (measured on 2026-11-08); a
  300 dpi page peaked at 226 MB, against 67 MB for Tesseract.
