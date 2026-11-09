# TesseractComparison

Measures OCR engines on the synthetic corpus of `tools/Corpus`: character and word error rate, word box recall, time, CPU time and
peak memory per page, at one thread and at all cores. It is development tooling for PdfPecker.Ocr (MIT); it compares with
Tesseract (Apache-2.0, built by `tools/Tesseract/build.sh`) and with rapidocr (Apache-2.0, Python, the reference for the models
PdfPecker.Ocr runs), and neither is ever shipped.

## Engines

| Engine | How it runs |
| --- | --- |
| `tesseract` | In process, through Tesseract's C API (`TessBaseAPIInit2` with `OEM_LSTM_ONLY`, `TessBaseAPISetImage` with raw 8-bit gray or RGB pixels, `TessBaseAPIRecognize`, `TessBaseAPIGetUTF8Text`, the result iterator for word boxes and confidences). Imported with source-generated `LibraryImport`; the library path comes from `--lib` and `NativeLibrary.SetDllImportResolver`. Models: `tessdata_fast`, the page's own language (`--tess-lang auto`; `eng+dan+deu+fra` with `multi`). Page segmentation 3, and 1 (with orientation detection) for the upside-down pages (`--tess-psm auto`). |
| `tesseract-best` | The same, with `tessdata_best`'s `eng` model (from `--tessdata-best`, which also needs `osd.traineddata` for the upside-down pages; `build.sh` copies it there), on the English pages only, since only `eng` is fetched. |
| `rapidocr` | A Python child process (`rapidocr_server.py`, venv with rapidocr 3.10.0 and ONNX Runtime). Pixels are sent as raw bytes over a pipe; the defaults of rapidocr's `config.yaml` apply, including its limit of 2000 px on the longest side, with word boxes on. ONNX Runtime's intra-op thread count is set from `--threads`. |
| `pdfpecker-ocr` | In process, through `IOcrEngineAdapter` (`PdfPeckerOcrAdapter.cs`): `ImageOcr.RecognizeGray` for gray pages, read in place, and `RecognizeRgba` for colour ones, with the package's defaults and `MaxDegreeOfParallelism` set from `--threads`; `Initialize` reads a blank page, which loads the models and builds the engine's worker outside the timed runs. Built in by default from `src/PdfPecker.Ocr`; `-p:WithPdfPeckerOcr=false` leaves it out (a stub). |

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
(default 1), `--corpus`, `--out`, `--lib`, `--tessdata`, `--tessdata-best`, `--tess-lang`, `--tess-psm`, `--python`, `--only id,id`
(ids containing the text, or `=id` for exactly that page), `--title`.
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
  resident. A reset only brings the peak down to what is resident at that moment, so the figure also holds whatever the engine
  kept from earlier pages; for a page's own peak, run the child on that page alone (`--only =id --runs 1 --warmup 0`). The
  `.meta` file's `process_peak_mb` is the whole child's peak (the readings before each reset are kept), and the report gives the
  resident size after the first model load. If the kernel refuses the reset, `peak_reset` is False in the `.meta` file and the
  peak is the process's running maximum.
- **Managed allocations** (`alloc_mb`, only for managed engines): `GC.GetTotalAllocatedBytes` per run, all threads.
- **Normalisation** (ground truth and output alike, `Metrics.Normalize`): Latin letters with combining accents composed (form C for the common marks); typographic single and double quotes,
  dashes and minus folded to ASCII; no-break and narrow spaces count as spaces; every whitespace run, line breaks included,
  becomes one space; the ends are trimmed. Case, accents and punctuation are kept. Reading order is therefore the order of the
  text, line by line.
- **CER** is the Levenshtein distance between the normalised page texts divided by the length of the ground truth text (it can
  exceed 100%). **WER** is the same over words. The summary gives **micro** averages (total edits over total reference length,
  so long pages count more) and the **macro** mean of pages for CER. **Word bag F1** is the F1 of the word multisets: it
  ignores order, so it separates a recognition error from a reading order difference (columns, or an upside-down page). The
  **order-free CER** is the character distance between the sorted words of reference and output: reading order costs nothing,
  but a misread first letter can move a word in the sort, and a joined or split word sorts elsewhere, so it errs on the high
  side.
- **Word boxes**: each ground truth word is matched greedily to the unused engine word of highest IoU, with at least 0.5.
  Recall is matched ground truth words over all of them, precision matched engine words over all engine words, and mean IoU is
  over matches. Ground truth boxes hug the ink (see `tools/Corpus/README.md`); rapidocr's boxes are padded by its unclip ratio, so
  its IoU is lower than its recall would suggest.
- **Instruction set.** The app is compiled with `IlcInstructionSet=native`: Native AOT otherwise targets a baseline x64
  processor, PdfPecker.Ocr's AVX2, FMA and AVX-512 kernels are left out, and a page takes two and a half to four times as
  long as under the JIT. Tesseract picks its SIMD code at run time, so both use the machine's processor.
- **In parts.** `--only` with exact ids lets a run be split into parts, one child each, so that no single measurement holds a
  shared machine for long; `report` merges the parts of one configuration (the first part's start-up figures, the highest
  process peak).
- Machines that run other jobs give noisy times. The numbers in `baseline/` were measured on a 4-core VM that other builds share (idle
  during the run); compare engines within one run, and use `bench/compare-versions.sh` (same machine, in turns) for a change to PdfPecker.Ocr.
- **Tesseract's OpenMP threads and a busy machine.** Tesseract's LSTM opens an OpenMP parallel region for every step of a line,
  and its threads wait for each other at the end of each one. When other processes take cores, a page at 4 threads can stall
  for seconds or minutes (in a first attempt at the run of 2026-11-08, 420 s for one 150 dpi page against 0.46 s at best); at 1 thread it
  cannot. Measure
  Tesseract's multi-threaded rows when the machine is otherwise idle, and look at `wall_ms_min` next to the median.

## Results of 2026-11-09

`baseline/2026-11-09/` holds the report (`results.md`, with the method, the change against the first run of 2026-11-08,
the results per page group and the causes found), the per-page rows (`results.csv`) and one page per fresh process
(`fresh.csv`). Shared 4-vCPU cloud VM (Xeon at 2.1 GHz with AVX-512), otherwise idle (load average 0.2 to 3.2, median 1.6,
the higher readings from this run's own four-thread children); 65 pages, 3 timed runs after 1 warm-up, Tesseract best on
the 35 English pages. rapidocr was not run again: its rows are those of 2026-11-08, measured on the busier machine, and
only its fresh-process peaks were measured again.

| | PdfPecker.Ocr | Tesseract 5.5.4 fast | Tesseract 5.5.4 best (en) | rapidocr |
| --- | ---: | ---: | ---: | ---: |
| CER in reading order / order-free | 1.71% / 0.38% | 10.37% / 3.30% | 15.21% / 2.19% | 23.32% / 6.40% |
| Word bag F1 | 0.9960 | 0.9713 | 0.9729 | 0.9601 |
| Median ms per page, 1 thread / 4 threads | 2122 / 581 | 605 / 572 | 1382 / 828 | 5204 / 1905 |
| CPU ms per page, 1 thread | 2184 | 597 | 1569 | 5015 |
| Peak RSS, fresh process, A4 invoice at 300 dpi | 226 MB | 67 MB | 80 MB | 548 MB |

- Every engine wrote exactly the text of the run of 2026-11-08 on every page, so the accuracy figures are the same; the
  review fixes of PdfPecker.Ocr change no page of this corpus, and Tesseract 5.5.4 reads as 5.5.1 did.
- Against 2026-11-08, PdfPecker.Ocr is 10% faster at one thread (CPU time per page 2420 to 2184 ms) and 21% at four
  (733 to 581 ms, part of it the quieter machine), and a fresh process peaks at 226 MB instead of 431 MB for an invoice
  and 236 MB instead of 906 MB for a page of long lines: buffers no longer grow batch by batch.
- PdfPecker.Ocr reads every word of the skewed and upside-down pages, but on pages skewed by 2 or 3 degrees its reading order
  takes table and letter columns one after the other (10.7% CER in reading order, 0.00% order-free): `ReadingOrder` corrects
  quarter turns only.
- Over a whole run the process still reaches 668 MB, because the adapter converts the four colour pages from RGB to a new
  RGBA array in every run (35 MB at 300 dpi) and the engine allocates too little for the garbage collector to run; a
  gray page read repeatedly stays at about 225 MB.
- Tesseract reads the invoice tables column by column (about 38% CER in reading order), drops most cells of the bordered tables at
  300 dpi (F1 0.43), and reads single-digit columns badly; rapidocr interleaves the article's columns and does not turn
  upside-down pages as a whole.
- Tesseract best needs `osd.traineddata` next to `eng` in the `--tessdata-best` folder for the upside-down pages;
  `tools/Tesseract/build.sh` copies the fast `osd` there (builds before 2026-11-09 did not).

## Files

`Program.cs` (commands), `Child.cs` (the measuring child and the CSV row), `Metrics.cs`, `Report.cs`, `CorpusData.cs` (manifest and
PNG decoder, shared with `bench/PdfPecker.Ocr.Benchmarks`), `Adapters.cs`, `TesseractNative.cs` and `TesseractAdapter.cs`,
`RapidOcrAdapter.cs` with `rapidocr_server.py`, `PdfPeckerOcrAdapter.cs`, `ProcMem.cs`.
