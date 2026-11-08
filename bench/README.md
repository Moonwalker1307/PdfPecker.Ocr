# Benchmarks

- `PdfPecker.Ocr.Benchmarks` – timing, allocation and peak memory of PdfPecker.Ocr itself: a page at 300 and 150 dpi, one thread and
  all cores, and a searchable PDF, through the public API (`-p:OcrApi=false` lists them as pending, for comparisons with a
  version from before the API). Behind `-p:OcrInternals=true`: detection and recognition on their own, and the `stages`
  command, which reads an A4 invoice page at 300 dpi cold and warm and splits its time by stage. Run `dotnet run -c Release
  --project bench/PdfPecker.Ocr.Benchmarks`. Optional `--corpus bench/TesseractComparison/corpus` adds corpus pages, `--threads n`
  and `--filter text` select cases.
- `compare-versions.sh [git-ref] [rounds]` – the working tree against another version on this machine, in turns, each in a fresh
  process (the ref is built with this version's benchmark program, so only the library differs). Pass the feature switches both
  versions understand in `OCR_BENCH_BUILD_ARGS`, for example `OCR_BENCH_BUILD_ARGS=-p:OcrInternals=true bench/compare-versions.sh v0.1.0`
  (the API cases are on by default; `-p:OcrApi=false` for a version from before the API).
  As in PdfPecker, only same-machine comparisons are meaningful, and the GC gen0 budget is fixed (`DOTNET_GCgen0size`) so that
  peak memory compares across machines.
- `TesseractComparison` – accuracy, speed and memory against Tesseract and rapidocr on the synthetic corpus; see its README.
