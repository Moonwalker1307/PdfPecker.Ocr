# TesseractComparison

Measures OCR engines on the synthetic corpus of `tools/Corpus`: character and word error rate, word box recall, time, CPU time and
peak memory per page, at one thread and at all cores. It is development tooling for PdfPecker.Ocr (MIT); it compares with
Tesseract (Apache-2.0, built by `tools/Tesseract/build.sh`) and with rapidocr (Apache-2.0, Python, the reference for the models
PdfPecker.Ocr runs), and neither is ever shipped.

## Engines

| Engine | How it runs |
| --- | --- |
| `tesseract` | In process, through Tesseract's C API (`TessBaseAPIInit2` with `OEM_LSTM_ONLY`, `TessBaseAPISetImage` with raw 8-bit gray or RGB pixels, `TessBaseAPIRecognize`, `TessBaseAPIGetUTF8Text`, the result iterator for word boxes and confidences). Imported with source-generated `LibraryImport`; the library path comes from `--lib` and `NativeLibrary.SetDllImportResolver`. Models: `tessdata_fast`, the page's own language (`--tess-lang auto`; `eng+dan+deu+fra` with `multi`). Page segmentation 3, and 1 (with orientation detection) for the upside-down pages (`--tess-psm auto`). |
| `rapidocr` | A Python child process (`rapidocr_server.py`, venv with rapidocr 3.10.0 and ONNX Runtime). Pixels are sent as raw bytes over a pipe; the defaults of rapidocr's `config.yaml` apply, including its limit of 2000 px on the longest side, with word boxes on. ONNX Runtime's intra-op thread count is set from `--threads`. |
| `pdfpecker-ocr` | In process, through `IOcrEngineAdapter`. A **stub** until the app is built with `-p:WithPdfPeckerOcr=true`, which references `src/PdfPecker.Ocr` and compiles the real adapter in `PdfPeckerOcrAdapter.cs` (written against the design brief's API, `ImageOcr.RecognizeGray` and `RecognizeRgba`; check it against the code when the API exists). |

All three sit behind `IOcrEngineAdapter` (`Adapters.cs`): `Initialize(threads)`, `Prepare(page)` (language and page mode, not
timed), `Recognize(pixels, width, height, format)` returning text, word boxes with confidences, and for out-of-process engines their
own CPU time and peak memory.

## Build and run

```
tools/Tesseract/build.sh
dotnet run -c Release --project tools/Corpus
dotnet publish bench/TesseractComparison -r linux-x64 -c Release -o /tmp/tc          # Native AOT
PDFPECKER_PYTHON=/path/to/venv/bin/python /tmp/tc/TesseractComparison run --engines tesseract,rapidocr --threads 1,max --runs 5
/tmp/tc/TesseractComparison report --in bench/TesseractComparison/results/<run>     # rewrite results.md from the CSVs
```

`run` options: `--engines`, `--threads 1,max` (`max` is the processor count), `--runs` (timed runs per page, default 5), `--warmup`
(default 1), `--corpus`, `--out`, `--lib`, `--tessdata`, `--tess-lang`, `--tess-psm`, `--python`, `--only id,id`, `--title`.
Results go to `bench/TesseractComparison/results/<timestamp>/` (git-ignored): `results.csv` (one row per engine, thread count and
page), `results.md`, the recognised text of every page under `hyp/`, and each child's log. `dotnet run -c Release` works as well
as the AOT build; `dotnet publish -r linux-x64` needs the ILCompiler package from nuget.org and `clang`.

## Method

- **One process per engine and thread count.** `run` starts a fresh child for each, with `OMP_THREAD_LIMIT` and `OMP_NUM_THREADS`
  set to the thread count (Tesseract's OpenMP reads them once, at load). A child loads the engine, then goes page by page.
  Language models are loaded when the language changes (pages are sorted by language), outside the timed region.
- **Time.** Wall time of `Recognize` is the median of N runs after the warm-up runs, per page. The input is already decoded
  pixels, so no image decoding is timed. For `rapidocr` the pipe transfer of the page (a few milliseconds) is included.
  CPU time is the process's total (`Environment.CpuUsage`, all threads); for `rapidocr` it is the Python process's own.
- **Peak memory.** The child's resident peak (`VmHWM`) is reset after the warm-up (by writing 5 to `/proc/self/clear_refs`) and read
  after the timed runs of the page, so it is the peak while recognising that page, including the model and input already
  resident. The report gives the median and the maximum over pages, and the resident size after the first model load.
  If the kernel refuses the reset, `peak_reset` is False in the `.meta` file and the peak is the process's running maximum.
- **Managed allocations** (`alloc_mb`, only for managed engines): `GC.GetTotalAllocatedBytes` per run, all threads.
- **Normalisation** (ground truth and output alike, `Metrics.Normalize`): Latin letters with combining accents composed (form C for the common marks); typographic single and double quotes,
  dashes and minus folded to ASCII; no-break and narrow spaces count as spaces; every whitespace run, line breaks included,
  becomes one space; the ends are trimmed. Case, accents and punctuation are kept. Reading order is therefore the order of the
  text, line by line.
- **CER** is the Levenshtein distance between the normalised page texts divided by the length of the ground truth text (it can
  exceed 100%). **WER** is the same over words. The summary gives **micro** averages (total edits over total reference length,
  so long pages count more) and the **macro** mean of pages for CER. **Word bag F1** is the F1 of the word multisets: it
  ignores order, so it separates a recognition error from a reading order difference (columns, or an upside-down page).
- **Word boxes**: each ground truth word is matched greedily to the unused engine word of highest IoU, with at least 0.5.
  Recall is matched ground truth words over all of them, precision matched engine words over all engine words, and mean IoU is
  over matches. Ground truth boxes hug the ink (see `tools/Corpus/README.md`); rapidocr's boxes are padded by its unclip ratio, so
  its IoU is lower than its recall would suggest.
- Machines that run other jobs give noisy times. The numbers in `baseline/` were measured on a 4-core VM shared with other
  builds; compare engines within one run, and use `bench/compare-versions.sh` (same machine, in turns) for a change to PdfPecker.Ocr.

## Files

`Program.cs` (commands), `Child.cs` (the measuring child and the CSV row), `Metrics.cs`, `Report.cs`, `CorpusData.cs` (manifest and
PNG decoder, shared with `bench/PdfPecker.Ocr.Benchmarks`), `Adapters.cs`, `TesseractNative.cs` and `TesseractAdapter.cs`,
`RapidOcrAdapter.cs` with `rapidocr_server.py`, `PdfPeckerOcrAdapter.cs`, `ProcMem.cs`.
