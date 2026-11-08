# Benchmarks

- `PdfPecker.Ocr.Benchmarks` – timing, allocation and peak memory of PdfPecker.Ocr itself: a page at 300 and 150 dpi, one thread and
  all cores, a searchable PDF, and (behind `-p:OcrInternals=true`) detection, recognition and the kernels. Cases that need the API
  are switched on with `-p:OcrApi=true`; until then they are listed as pending. Run `dotnet run -c Release --project
  bench/PdfPecker.Ocr.Benchmarks`. Optional `--corpus bench/TesseractComparison/corpus` adds corpus pages, `--threads n` and
  `--filter text` select cases.
- `compare-versions.sh [git-ref] [rounds]` – the working tree against another version on this machine, in turns, each in a fresh
  process (the ref is built with this version's benchmark program, so only the library differs). Pass the feature switches both
  versions understand in `OCR_BENCH_BUILD_ARGS`, for example `OCR_BENCH_BUILD_ARGS=-p:OcrApi=true bench/compare-versions.sh v0.1.0`.
  As in PdfPecker, only same-machine comparisons are meaningful, and the GC gen0 budget is fixed (`DOTNET_GCgen0size`) so that
  peak memory compares across machines.
- `TesseractComparison` – accuracy, speed and memory against Tesseract and rapidocr on the synthetic corpus; see its README.
