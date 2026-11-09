> Checked-in copy of the report. The raw logs, every child's output and the recognised text of every page (`raw/` below)
> are not in the repository; they were kept with the run.

# PdfPecker.Ocr against Tesseract 5.5.4 and rapidocr, 2026-11-09

A second run of the comparison of 2026-11-08, on the final engine build: the same 65-page corpus (byte-identical files),
the same method, drivers and scoring. What changed since then: PdfPecker.Ocr got twelve output-identical speed-ups and
memory cuts and the review fixes (detection of elongated pages, lone narrow characters), and `tools/Tesseract/build.sh`
builds Tesseract 5.5.4 with Leptonica 1.87.0 instead of 5.5.1 with 1.85.0. **rapidocr was not run again**: its engine,
models and settings are unchanged and its corpus run takes about 35 minutes, so its rows in `results.csv` and in the
tables below are carried over from 2026-11-08 (measured then on the busier machine). Only its fresh-process rows
(`fresh.csv`, two minutes) were measured again.

## In short

- **Accuracy is unchanged, to the byte.** Every engine wrote exactly the text of 2026-11-08 on every page (the
  recognised text of all 65 pages, compared file by file, for PdfPecker.Ocr at both thread counts and for Tesseract
  5.5.4 fast and best against 5.5.1). So the scores are those of the first run: CER in reading order 1.7% for
  PdfPecker.Ocr, 10.4% for Tesseract fast, 15.2% for Tesseract best (English pages) and 23.3% for rapidocr; ignoring
  reading order 0.38%, 3.3%, 2.2% and 6.4%; word bag F1 0.996, 0.971, 0.973 and 0.960. The review fixes change no page of
  this corpus.
- **PdfPecker.Ocr is 10% faster at one thread and 21% faster at four.** The median page takes 2.12 s at one thread
  (2.35 s before) and 0.58 s at four (0.73 s), and its CPU time per page fell from 2.42 to 2.18 s. At four threads it
  now matches Tesseract fast (0.57 s) and beats Tesseract best (0.83 s); per core, Tesseract fast still needs a little
  over a quarter of its CPU time (0.60 s).
- **Its memory is about halved.** A fresh process reading one A4 invoice at 300 dpi peaks at 226 MB (431 MB before),
  and the page of long monospaced lines at 236 MB (906 MB): the step-by-step buffer growth found in the first run is
  gone, so a page's peak no longer depends on its longest line. Tesseract needs 67 and 76 MB, rapidocr 548 and 571 MB.
  Over a whole run of pages the process still reaches 668 MB, but that comes from the benchmark's own colour
  conversion, not from the engine (see "Odd things in this run").
- **Tesseract 5.5.4 performs as 5.5.1 did**: the same text, and times within 1–4% (fast) and 2–10% (best) of the first
  run, which is about the noise between two runs on this machine. Its peaks are unchanged (67 and 80 MB).
- **PdfPecker.Ocr's remaining loss on accuracy is still reading order on skewed pages** (10.7% CER in reading order on
  the nine skewed pages, every word right); `ReadingOrder` still corrects quarter turns only.

The machine is the same shared cloud VM with 4 vCPUs (Intel Xeon at 2.1 GHz, AVX-512) and 15 GB of RAM. This time no
other jobs ran: the 1-minute load average was 0.2 to 3.2, median 1.6, during the runs (`raw/loadavg.log`), and the
higher readings are this benchmark's own four-thread children. The first run had a median of 4.1 and a maximum of 12.2,
so its four-thread times were noisier; part of PdfPecker.Ocr's gain at four threads (and its much lower p90) is the
quieter machine, while the one-thread CPU time (−10%) is the fairer measure of the engine's own speed-up. No four-thread
run stalled: on every page the slowest of the three timed runs was within 1.5 times the fastest, so no part had to be
repeated.

## Headline table

All 65 pages, except Tesseract best (only the 35 English pages, for which `tessdata_best` has a model). Times are the
median over pages of each page's median of 3 timed runs after 1 warm-up. Peak memory comes from a fresh process that
reads one page once (`fresh.csv`). rapidocr's accuracy, time and whole-run rows are from 2026-11-08.

| | PdfPecker.Ocr | Tesseract 5.5.4 fast | Tesseract 5.5.4 best (en) | rapidocr 3.10 + ONNX Runtime |
| --- | ---: | ---: | ---: | ---: |
| CER, reading order | 1.71% | 10.37% | 15.21% | 23.32% |
| CER, order-free (sorted words) | 0.38% | 3.30% | 2.19% | 6.40% |
| WER, reading order | 2.35% | 10.75% | 14.58% | 29.96% |
| Word bag F1 | 0.9960 | 0.9713 | 0.9729 | 0.9601 |
| Same, English pages only: CER / order-free CER / F1 | 1.72% / 0.60% / 0.9934 | 15.26% / 3.45% / 0.9660 | 15.21% / 2.19% / 0.9729 | 32.87% / 8.73% / 0.9505 |
| Median ms per page, 1 thread | 2122 | 605 | 1382 | 5204 |
| p90 ms per page, 1 thread | 2350 | 796 | 2036 | 5527 |
| Median ms per page, 4 threads | 581 | 572 | 828 | 1905 |
| p90 ms per page, 4 threads | 673 | 780 | 1165 | 4727 |
| CPU ms per page, mean, 1 / 4 threads | 2184 / 2261 | 597 / 1644 | 1569 / 2758 | 5015 / 7631 |
| Peak RSS, fresh process, A4 invoice at 300 dpi | 226 MB | 67 MB | 80 MB | 548 MB (Python process) |
| Peak RSS, fresh process, A4 log of long monospaced lines | 236 MB | 76 MB | 90 MB | 571 MB |
| Whole run, process peak (65 pages, 1 thread) | 668 MB (386 MB before the first colour page at 300 dpi; see below) | 251 MB | 211 MB | 798 MB |
| Fresh process, one 300 dpi invoice, 1 thread: total / of which start-up and model load | 2.52–2.57 s / about 0.25 s (0.07–0.08 s start-up, the model load inside the first call) | 0.65–0.68 s / 0.13–0.15 s | 1.46–1.51 s / 0.18–0.23 s | 7.5–8.2 s / 1.5–1.7 s |
| Size on disk | 35 MB self-contained Native AOT executable (models embedded); or 18.0 MB `PdfPecker.Ocr.dll` + 6.1 MB `PdfPecker.dll` on a .NET runtime | 7.0 MB `libtesseract.so` (Leptonica linked in, no image codecs) + 1.1–4.1 MB per language; 10.6 MB more for orientation detection | 7.0 MB + 15.4 MB for `eng` | 31 MB of ONNX models + about 360 MB of Python packages (ONNX Runtime, NumPy, OpenCV, Shapely, ...) + Python |

Accuracy does not change with the thread count for any engine (PdfPecker.Ocr's output is deterministic; the rows for 1
and 4 threads are identical).

## Change against 2026-11-08

| | 2026-11-08 | 2026-11-09 | Change |
| --- | ---: | ---: | ---: |
| Accuracy, all engines (CER, order-free CER, WER, F1) | | | identical text on every page |
| PdfPecker.Ocr, median ms per page, 1 / 4 threads | 2354 / 733 | 2122 / 581 | −10% / −21% |
| PdfPecker.Ocr, p90 ms per page, 1 / 4 threads | 2647 / 1442 | 2350 / 673 | −11% / −53% |
| PdfPecker.Ocr, CPU ms per page, 1 / 4 threads | 2420 / 2587 | 2184 / 2261 | −10% / −13% |
| PdfPecker.Ocr, speed-up from 1 to 4 threads | 3.2 times | 3.65 times | |
| PdfPecker.Ocr, fresh-process peak, invoice / long lines | 431 / 906 MB | 226 / 236 MB | −48% / −74% |
| PdfPecker.Ocr, fresh-process peak, invoice at 150 dpi | 413 MB | 221 MB | −46% |
| PdfPecker.Ocr, whole-run process peak | 1267 MB | 668 MB | −47% |
| PdfPecker.Ocr, per-page peak in the long runs, median | 575 MB | 323 MB | −44% |
| PdfPecker.Ocr, long monospaced page in a fresh process, 1 thread | 7.29 s | 6.36 s | −13% |
| PdfPecker.Ocr, fresh process for one invoice, 1 thread | 3.1–3.3 s | 2.52–2.57 s | about −20% |
| Tesseract fast, median ms per page, 1 / 4 threads | 583 / 569 | 605 / 572 | +4% / +1% |
| Tesseract fast, CPU ms per page, 1 / 4 threads | 587 / 1615 | 597 / 1644 | +2% / +2% |
| Tesseract best, median ms per page, 1 / 4 threads | 1351 / 755 | 1382 / 828 | +2% / +10% |
| Tesseract best, CPU ms per page, 1 / 4 threads | 1565 / 2650 | 1569 / 2758 | 0% / +4% |
| Tesseract fast / best, fresh-process peak, invoice | 67 / 80 MB | 67 / 80 MB | none |
| rapidocr, fresh-process peak, invoice / long lines | 548 / 571 MB | 548 / 571 MB | none (re-measured) |

Per page, PdfPecker.Ocr's one-thread median fell by 8% (median of the 65 per-page ratios; between −35% and +4%), and its
four-thread median by 22%. Tesseract fast took 2.5% longer per page at one thread on the median (between −15% and +19%),
with byte-identical output, which puts the noise between the two runs at a few per cent.

## Accuracy by page group

Unchanged from 2026-11-08, since every engine wrote the same text. CER in reading order / order-free CER / word bag F1,
all pages of the group (Tesseract best: English pages of the group).

| Group | Pages | PdfPecker.Ocr | Tesseract fast | Tesseract best | rapidocr |
| --- | ---: | --- | --- | --- | --- |
| Clean corpus pages, 300 and 150 dpi | 36 | 0.09% / 0.39% / 0.994 | 11.45% / 3.65% / 0.967 | 18.78% / 3.16% / 0.959 | 16.94% / 9.43% / 0.940 |
| Degraded (noise, blur, skew, JPEG, low contrast, rot180, scan) | 27 | 3.99% / 0.29% / 0.999 | 9.13% / 2.87% / 0.978 | 11.64% / 1.19% / 0.985 | 32.84% / 2.29% / 0.985 |
| Skewed 1, 2 and 3 degrees | 9 | 10.71% / 0.00% / 1.000 | 11.78% / 6.37% / 0.964 | 11.04% / 1.03% / 0.988 | 29.75% / 0.00% / 1.000 |
| Turned 180 degrees | 3 | 0.00% / 0.00% / 1.000 | 9.35% / 0.89% / 0.986 | 14.31% / 1.41% / 0.983 | 72.29% / 15.74% / 0.870 |
| Tesseract's own samples (phototest, eurotext) | 2 | 0.29% / 3.16% / 0.984 | 0.86% / 1.29% / 0.952 | 0.00% / 0.00% / 1.000 (phototest only) | 0.14% / 3.02% / 0.992 |
| Danish, all pages | 8 | 0.02% / 0.08% / 0.999 | 11.68% / 6.77% / 0.959 | n/a | 4.08% / 5.43% / 0.951 |
| German, all pages | 15 | 3.04% / 0.00% / 1.000 | 2.65% / 2.83% / 0.982 | n/a | 16.80% / 1.78% / 0.983 |
| English, all pages | 35 | 1.72% / 0.60% / 0.993 | 15.26% / 3.45% / 0.966 | 15.21% / 2.19% / 0.973 | 32.87% / 8.73% / 0.951 |
| French, all pages | 6 | 0.00% / 0.00% / 1.000 | 0.12% / 0.47% / 0.993 | n/a | 2.65% / 5.11% / 0.968 |

The order-free CER is the character distance between the sorted words of the page and of the output, so it ignores
reading order completely; a misread first letter can move a word in the sort, so it errs on the high side. It can be
higher than the CER in reading order when an engine joins or splits words (PdfPecker.Ocr's "|https://..." on the scan
page, or the eurotext quotes), because the joined word sorts elsewhere.

The causes found on 2026-11-08 still hold, because the text is the same: Tesseract reads the invoice table and the
monospaced log column by column (about 38% CER in reading order on every invoice, 3–4% order-free) and drops most cells
of the bordered tables at 300 dpi (F1 0.43–0.44); rapidocr interleaves the two article columns and does not turn
upside-down pages as a whole; PdfPecker.Ocr reads skewed tables and letters column by column, because
`Layout/ReadingOrder.cs` turns the page by quarter turns only before its XY cuts (invoice-en.300.skew2 31% CER,
letter-de.300.skew3 46%, every word right). Its smaller recognition errors are the same too ("INF0" and "wARN" on the
Liberation Mono page, "|https://..." on the scan and JPEG pages, one "∅" for "Ø" at 150 dpi).

## Speed and memory of PdfPecker.Ocr

- **Resolution does not matter**: clean 300 dpi pages take 2.21 s at one thread and 150 dpi pages 2.12 s (median),
  because detection always runs at 1,536 px on the longer side and line crops are always 48 px high; Tesseract fast gets a
  third faster at 150 dpi (0.66 against 0.44 s).
- **Threads scale better than before**: 3.65 times from one to four threads (3.2 times on 2026-11-08), for 4% more CPU
  time per page (2184 against 2261 ms). Tesseract fast gets 5% faster at four threads for 2.75 times the CPU time,
  Tesseract best 40% faster for 1.76 times.
- **The recognition work itself is unchanged**: batches are still padded to at least 320 px (`TextRecognizer`, as
  rapidocr does), so short table cells still cost as much as a 320 px line; the per-layer profile of 2026-11-08 (a quarter
  of the time in detection, three quarters in recognition, mostly the backbone's 1x1 convolutions) was not repeated.
- **Memory no longer grows with the line length**: a fresh process peaks at 221–236 MB on all four test pages, the long
  monospaced page included (906 MB before). The resident set after the models are loaded is no longer reported (see
  below).
- **Against rapidocr with ONNX Runtime on the same models** PdfPecker.Ocr is now 2.5 times faster at one thread (2122
  against 5204 ms), with rapidocr's figures from the first run; at one thread both engines used a whole core then, so
  this ratio is fair. At four threads the ratio (3.3 times) mixes a quiet and a loaded machine.

## Tesseract notes

- **5.5.4 against 5.5.1**: byte-identical text on all 65 pages with `tessdata_fast` and on the 35 English pages with
  `tessdata_best`, at both thread counts; times and memory as before within the run-to-run noise.
- **OpenMP**: with the machine otherwise idle no four-thread page stalled (the slowest of three runs at most 1.5 times the
  fastest on every page, against 420 s for one page under load in the first run). Four threads still buy Tesseract fast
  only 5% for 2.75 times the CPU time.
- **The build has no PNG support** in Leptonica, as before, so Tesseract stores each line image as BMP instead of PNG
  (Leptonica prints "png library missing; output bmp format"); that is slightly faster than a stock build.

## Odd things in this run

- **`tools/Tesseract/out/linux-x64/libtesseract.so` was still Tesseract 5.5.1.** The 5.5.4 build had reused the work
  folder of the 5.5.1 build, which still holds `libtesseract.so.5.5.1` next to the new `libtesseract.so.5.5.4`, and
  `build.sh` copies the first `libtesseract*.so*` that `find` lists, which was the old file (same SHA-256 as the library
  of 2026-11-08). The `tesseract` program next to it reports 5.5.4, because it loads the library from the work folder.
  This run passed the 5.5.4 library explicitly (`--lib`, a copy of `work-linux-x64/tesseract/libtesseract.so.5.5.4`,
  SHA-256 `7236f2eb…74e96a3`), and the report's engine lines confirm "Tesseract 5.5.4" from `TessVersion()`. A clean
  build, or `cp -L "$work/tesseract/libtesseract.$ext"` in `build.sh`, avoids it.
- **Tesseract best needs `osd.traineddata` beside `eng`.** `tools/Tesseract/out/tessdata-best` holds only `eng`; the
  first attempt of this run used it as it is, and on the two upside-down English pages Tesseract printed "Failed loading
  language 'osd'" and read them upside down (F1 0.03 and 0.00). The run of 2026-11-08 had used a folder with both models
  linked, so Tesseract best was measured again with such a folder (`raw/best-without-osd/` keeps the failed attempt).
- **PdfPecker.Ocr's start-up and model load no longer show in the engine lines** ("Start-up 0 ms", "6 MB resident after
  loading"). The adapter's `Initialize` reads a blank page to load the models, and the engine now skips pages of one
  colour before detection, so the models load on the first real page instead: in the long runs inside the untimed
  warm-up, in the fresh processes inside the one timed call (2.44–2.49 s for the invoice against 2.28 s warm, so the
  model load is about 0.15–0.2 s, against 0.5–0.6 s on 2026-11-08). The timed medians are unaffected.
- **The whole-run peak (668 MB) is the benchmark's colour conversion, not the engine.** The four colour pages
  (invoice-de and invoice-fr) go through `RecognizeRgba`, and the adapter converts their RGB pixels to a new RGBA array
  inside every run (35 MB at 300 dpi, the 34 MB of managed allocation per run in `alloc_mb`). The engine allocates almost
  nothing else, so the garbage collector seldom runs and those arrays stay resident: reading invoice-de.300 once in a
  fresh process peaks at 290 MB, four times at 392 MB, ten times at 593 MB, while invoice-en.300 (gray) stays at 225–228
  MB however often it is read (`raw/probe-memory/`). In the long runs the resident set jumps by 180–220 MB at
  invoice-de.300 and invoice-fr.300 and stays there; before the first colour page at 300 dpi no part went above 386 MB.
- rapidocr's four-thread rows still show the load of the first run (eight pages whose slowest run took 1.5–2.3 times the
  fastest), since they were carried over.

## Method

- Corpus: `tools/Corpus` with `--tesseract-samples`, seed 20261108, the files of 2026-11-08 (copied, all 196 files
  compared by SHA-256): 63 synthetic A4 pages drawn by PdfPecker 1.23.0 (invoices, letters, two-column articles, tables,
  small print, headings, a monospaced log, identifiers; English, Danish, German, French) at 300 and 150 dpi, 27 degraded
  300 dpi variants, and Tesseract's `phototest.tif` and `eurotext.tif`. The pages are clean renderings or simulated
  degradations, not scans of paper, which flatters every engine.
- App: `bench/TesseractComparison` at commit 2da8615 of `claude/ocr-engine`, with PdfPecker.Ocr built in from the same
  commit, published with Native AOT, `IlcInstructionSet=native`. Every engine sits behind the same adapter interface and
  is timed by the same code on decoded pixels; no image decoding is timed.
  - PdfPecker.Ocr: in process, `ImageOcr.RecognizeGray` (gray pages) or `RecognizeRgba` (colour pages, converted from
    RGB inside the timed region), default options, `MaxDegreeOfParallelism` = threads.
  - Tesseract 5.5.4 with Leptonica 1.87.0 (`tools/Tesseract/build.sh`; the library passed with `--lib`, see above), in
    process through the C API, LSTM engine, `tessdata_fast` in the page's language (eng, dan, deu, fra; eng+deu+fra for
    eurotext), page segmentation 3, and 1 with orientation detection for the rot180 pages. `tesseract-best`:
    `tessdata_best` eng with the `osd` model beside it, English pages only. `OMP_THREAD_LIMIT` and `OMP_NUM_THREADS` =
    threads.
  - rapidocr 3.10.0 with ONNX Runtime 1.31.1 in a Python 3.13 child process (rows of 2026-11-08; fresh-process rows
    measured again with the same venv), defaults of its `config.yaml`, word boxes on, `intra_op_num_threads` = threads.
- Runs: one fresh process per engine, thread count and quarter of the corpus (pages dealt round robin), one after another
  under the shared benchmark lock (`flock`), never two at once; 3 timed runs after 1 warm-up per page. Tesseract at four
  threads ran under a ten-minute cut-off, which no part reached. Peak memory per page is the resident peak after a reset
  before the timed runs; it includes memory retained from earlier pages, so the table uses the fresh-process figures
  (`fresh.csv`: one page, one run, no warm-up, two repetitions, the same four pages as before).
- Scores: text normalised the same way for ground truth and output (Unicode composition of Latin accents, typographic
  quotes and dashes to ASCII, whitespace runs to one space). CER and WER are Levenshtein distances over the page text;
  group figures are micro averages (edits summed over pages). Word bag F1 counts matching words regardless of order.
- Raw data: `results.csv` (one row per engine, thread count and page; rapidocr's rows from 2026-11-08), `fresh.csv`, and
  under `raw/children/` every child's CSV, `.meta` and log and the recognised text of every page (`raw/children/hyp/`),
  the drivers (`run-compare.sh`, `run-fresh.sh`, `loadavg.sh`) and their logs (`full-driver.log` for the main run,
  `best-driver.log` for Tesseract best again with the `osd` model, `fresh-driver.log`), the load log, the failed
  Tesseract best attempt (`raw/best-without-osd/`) and the memory probes (`raw/probe-memory/`). The generated
  per-condition and per-page tables follow.

---

## Generated report: PdfPecker.Ocr, Tesseract 5.5.4 and rapidocr on the synthetic corpus

Generated by `bench/TesseractComparison`. All engines read the same synthetic corpus (`tools/Corpus`), one engine and thread count per fresh process.

rapidocr's rows (`rapidocr/1`, `rapidocr/4`) are the child outputs of 2026-11-08, carried over unchanged; every other row was measured on 2026-11-09.

### Engines

- `pdfpecker-ocr/1`: PdfPecker.Ocr 0.1.0.0, MaxDegreeOfParallelism 1. Start-up 0 ms plus first model load 0 ms, 6 MB resident after loading, 3 timed runs after 1 warm-up per page.
- `pdfpecker-ocr/4`: PdfPecker.Ocr 0.1.0.0, MaxDegreeOfParallelism 4. Start-up 0 ms plus first model load 0 ms, 6 MB resident after loading, 3 timed runs after 1 warm-up per page.
- `rapidocr/1`: rapidocr 3.10.0 (PP-OCRv6 small models) on ONNX Runtime, defaults of its config.yaml, word boxes on, in a Python child process. Start-up 1321 ms plus first model load 0 ms, 174 MB resident after loading, 3 timed runs after 1 warm-up per page.
- `rapidocr/4`: rapidocr 3.10.0 (PP-OCRv6 small models) on ONNX Runtime, defaults of its config.yaml, word boxes on, in a Python child process. Start-up 870 ms plus first model load 0 ms, 175 MB resident after loading, 3 timed runs after 1 warm-up per page.
- `tesseract-best/1`: Tesseract 5.5.4, LSTM engine, models in tools/Tesseract/out/tessdata-best (with osd.traineddata linked beside eng), language eng, page segmentation auto, OMP_THREAD_LIMIT=1. Start-up 0 ms plus first model load 110 ms, 37 MB resident after loading, 3 timed runs after 1 warm-up per page.
- `tesseract-best/4`: Tesseract 5.5.4, LSTM engine, models in tools/Tesseract/out/tessdata-best (with osd.traineddata linked beside eng), language eng, page segmentation auto, OMP_THREAD_LIMIT=4. Start-up 0 ms plus first model load 103 ms, 37 MB resident after loading, 3 timed runs after 1 warm-up per page.
- `tesseract/1`: Tesseract 5.5.4, LSTM engine, models in tools/Tesseract/out/tessdata, language auto, page segmentation auto, OMP_THREAD_LIMIT=1. Start-up 0 ms plus first model load 78 ms, 23 MB resident after loading, 3 timed runs after 1 warm-up per page.
- `tesseract/4`: Tesseract 5.5.4, LSTM engine, models in tools/Tesseract/out/tessdata, language auto, page segmentation auto, OMP_THREAD_LIMIT=4. Start-up 0 ms plus first model load 103 ms, 23 MB resident after loading, 3 timed runs after 1 warm-up per page.

### Summary

Accuracy: CER and WER over all text of the pages (micro), word bag F1 over all words (micro, order-free), and the CER of the sorted words (order-free). Time: the per-page medians of the timed runs, then their median, 90th percentile and mean over pages. CPU is all threads of the engine's process. Peak RSS per page is the resident peak while that page was read, models included; the process peak is the whole child's.

| Config | Pages | CER | CER (mean of pages) | WER | Word bag F1 | CER, order-free | Box recall @IoU 0.5 | Median ms/page | p90 ms/page | Mean ms/page | CPU ms/page, mean | Peak RSS per page, median / max (MB) | Process peak (MB) | Start-up + model load (ms) | Failed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pdfpecker-ocr/1 | 65 | 1.71% | 1.96% | 2.35% | 0.9960 | 0.38% | 69.34% | 2122 | 2350 | 2204 | 2184 | 323 / 667 | 668 | 0 | 0 |
| pdfpecker-ocr/4 | 65 | 1.71% | 1.96% | 2.35% | 0.9960 | 0.38% | 69.34% | 581 | 673 | 601 | 2261 | 324 / 669 | 669 | 0 | 0 |
| rapidocr/1 | 65 | 23.32% | 18.32% | 29.96% | 0.9601 | 6.40% | 70.43% | 5204 | 5527 | 5056 | 5015 | 746 / 791 | 798 | 1321 | 0 |
| rapidocr/4 | 65 | 23.32% | 18.32% | 29.96% | 0.9601 | 6.40% | 70.43% | 1905 | 4727 | 2908 | 7631 | 747 / 782 | 782 | 870 | 0 |
| tesseract-best/1 | 35 | 15.21% | 13.99% | 14.58% | 0.9729 | 2.19% | 96.70% | 1382 | 2036 | 1582 | 1569 | 154 / 210 | 211 | 110 | 0 |
| tesseract-best/4 | 35 | 15.21% | 13.99% | 14.58% | 0.9729 | 2.19% | 96.70% | 828 | 1165 | 871 | 2758 | 154 / 206 | 207 | 103 | 0 |
| tesseract/1 | 65 | 10.37% | 9.89% | 10.75% | 0.9713 | 3.30% | 96.26% | 605 | 796 | 602 | 597 | 167 / 251 | 251 | 78 | 0 |
| tesseract/4 | 65 | 10.37% | 9.89% | 10.75% | 0.9713 | 3.30% | 96.26% | 572 | 780 | 589 | 1644 | 166 / 251 | 251 | 103 | 0 |

Managed allocation per page (mean): pdfpecker-ocr/1 1.9 MB, pdfpecker-ocr/4 1.9 MB.

### Character error rate by condition

| Condition | Pages | pdfpecker-ocr/1 | pdfpecker-ocr/4 | rapidocr/1 | rapidocr/4 | tesseract-best/1 | tesseract-best/4 | tesseract/1 | tesseract/4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| all pages | 65 | 1.71% | 1.71% | 23.32% | 23.32% | 15.21% | 15.21% | 10.37% | 10.37% |
| clean (undegraded corpus pages, both resolutions) | 36 | 0.09% | 0.09% | 16.94% | 16.94% | 18.78% | 18.78% | 11.45% | 11.45% |
| degraded (all variants) | 27 | 3.99% | 3.99% | 32.84% | 32.84% | 11.64% | 11.64% | 9.13% | 9.13% |
| degraded, without rot180 | 24 | 4.49% | 4.49% | 27.91% | 27.91% | 11.31% | 11.31% | 9.10% | 9.10% |
| skewed (1, 2 and 3 degrees) | 9 | 10.71% | 10.71% | 29.75% | 29.75% | 11.04% | 11.04% | 11.78% | 11.78% |
| rotated 180 degrees | 3 | 0.00% | 0.00% | 72.29% | 72.29% | 14.31% | 14.31% | 9.35% | 9.35% |
| clean, 300 dpi | 18 | 0.08% | 0.08% | 17.65% | 17.65% | 22.19% | 22.19% | 14.97% | 14.97% |
| clean, 150 dpi | 18 | 0.10% | 0.10% | 16.24% | 16.24% | 15.36% | 15.36% | 7.93% | 7.93% |
| Tesseract's sample images | 2 | 0.29% | 0.29% | 0.14% | 0.14% | 0.00% | 0.00% | 0.86% | 0.86% |
| degraded: blur | 3 | 0.00% | 0.00% | 26.78% | 26.78% | 14.31% | 14.31% | 9.35% | 9.35% |
| degraded: jpeg30 | 3 | 0.04% | 0.04% | 26.86% | 26.86% | 14.31% | 14.31% | 9.35% | 9.35% |
| degraded: lowcontrast | 3 | 0.00% | 0.00% | 26.78% | 26.78% | 14.31% | 14.31% | 9.24% | 9.24% |
| degraded: noise | 3 | 0.00% | 0.00% | 26.82% | 26.82% | 14.37% | 14.37% | 9.27% | 9.27% |
| degraded: rot180 | 3 | 0.00% | 0.00% | 72.29% | 72.29% | 14.31% | 14.31% | 9.35% | 9.35% |
| degraded: scan | 3 | 3.73% | 3.73% | 26.78% | 26.78% | 0.06% | 0.06% | 0.26% | 0.26% |
| degraded: skew1 | 3 | 0.00% | 0.00% | 26.78% | 26.78% | 14.25% | 14.25% | 9.42% | 9.42% |
| degraded: skew2 | 3 | 7.68% | 7.68% | 33.62% | 33.62% | 0.00% | 0.00% | 0.00% | 0.00% |
| degraded: skew3 | 3 | 24.46% | 24.46% | 28.85% | 28.85% | 18.87% | 18.87% | 25.93% | 25.93% |
| clean 300 dpi, article | 2 | 0.00% | 0.00% | 67.10% | 67.10% | 0.00% | 0.00% | 0.00% | 0.00% |
| clean 300 dpi, headings | 2 | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| clean 300 dpi, identifiers | 1 | 0.00% | 0.00% | 7.88% | 7.88% | 0.00% | 0.00% | 0.30% | 0.30% |
| clean 300 dpi, invoice | 4 | 0.00% | 0.00% | 5.88% | 5.88% | 38.37% | 38.37% | 18.87% | 18.87% |
| clean 300 dpi, letter | 4 | 0.00% | 0.00% | 0.03% | 0.03% | 0.00% | 0.00% | 0.00% | 0.00% |
| clean 300 dpi, mono | 1 | 0.53% | 0.53% | 40.53% | 40.53% | 54.40% | 54.40% | 54.78% | 54.78% |
| clean 300 dpi, smallprint | 2 | 0.00% | 0.00% | 4.51% | 4.51% | 0.00% | 0.00% | 0.00% | 0.00% |
| clean 300 dpi, table | 2 | 0.00% | 0.00% | 0.12% | 0.12% | 65.34% | 65.34% | 68.26% | 68.26% |
| clean 300 dpi, language da | 4 | 0.00% | 0.00% | 2.19% | 2.19% | n/a | n/a | 23.20% | 23.20% |
| clean 300 dpi, language de | 3 | 0.00% | 0.00% | 31.11% | 31.11% | n/a | n/a | 0.07% | 0.07% |
| clean 300 dpi, language en | 8 | 0.16% | 0.16% | 22.01% | 22.01% | 22.19% | 22.19% | 22.36% | 22.36% |
| clean 300 dpi, language fr | 3 | 0.00% | 0.00% | 5.31% | 5.31% | n/a | n/a | 0.03% | 0.03% |
| all pages, language da | 8 | 0.02% | 0.02% | 4.08% | 4.08% | n/a | n/a | 11.68% | 11.68% |
| all pages, language de | 15 | 3.04% | 3.04% | 16.80% | 16.80% | n/a | n/a | 2.65% | 2.65% |
| all pages, language en | 35 | 1.72% | 1.72% | 32.87% | 32.87% | 15.21% | 15.21% | 15.26% | 15.26% |
| all pages, language fr | 6 | 0.00% | 0.00% | 2.65% | 2.65% | n/a | n/a | 0.12% | 0.12% |
| all pages, language multi | 1 | 0.24% | 0.24% | 0.24% | 0.24% | n/a | n/a | 1.46% | 1.46% |

### Word error rate by condition

| Condition | Pages | pdfpecker-ocr/1 | pdfpecker-ocr/4 | rapidocr/1 | rapidocr/4 | tesseract-best/1 | tesseract-best/4 | tesseract/1 | tesseract/4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| all pages | 65 | 2.35% | 2.35% | 29.96% | 29.96% | 14.58% | 14.58% | 10.75% | 10.75% |
| clean (undegraded corpus pages, both resolutions) | 36 | 0.62% | 0.62% | 19.74% | 19.74% | 17.80% | 17.80% | 11.55% | 11.55% |
| degraded (all variants) | 27 | 4.58% | 4.58% | 43.98% | 43.98% | 11.94% | 11.94% | 9.91% | 9.91% |
| degraded, without rot180 | 24 | 5.16% | 5.16% | 37.96% | 37.96% | 11.61% | 11.61% | 9.88% | 9.88% |
| skewed (1, 2 and 3 degrees) | 9 | 12.29% | 12.29% | 40.17% | 40.17% | 11.22% | 11.22% | 12.52% | 12.52% |
| rotated 180 degrees | 3 | 0.00% | 0.00% | 92.17% | 92.17% | 14.63% | 14.63% | 10.14% | 10.14% |
| clean, 300 dpi | 18 | 0.56% | 0.56% | 20.62% | 20.62% | 24.53% | 24.53% | 16.99% | 16.99% |
| clean, 150 dpi | 18 | 0.68% | 0.68% | 18.86% | 18.86% | 11.07% | 11.07% | 6.12% | 6.12% |
| Tesseract's sample images | 2 | 1.59% | 1.59% | 0.79% | 0.79% | 0.00% | 0.00% | 4.76% | 4.76% |
| degraded: blur | 3 | 0.00% | 0.00% | 36.41% | 36.41% | 14.63% | 14.63% | 9.91% | 9.91% |
| degraded: jpeg30 | 3 | 0.46% | 0.46% | 37.10% | 37.10% | 14.63% | 14.63% | 10.14% | 10.14% |
| degraded: lowcontrast | 3 | 0.00% | 0.00% | 36.41% | 36.41% | 14.63% | 14.63% | 10.83% | 10.83% |
| degraded: noise | 3 | 0.00% | 0.00% | 36.87% | 36.87% | 14.97% | 14.97% | 9.68% | 9.68% |
| degraded: rot180 | 3 | 0.00% | 0.00% | 92.17% | 92.17% | 14.63% | 14.63% | 10.14% | 10.14% |
| degraded: scan | 3 | 3.92% | 3.92% | 36.41% | 36.41% | 0.34% | 0.34% | 0.92% | 0.92% |
| degraded: skew1 | 3 | 0.00% | 0.00% | 36.41% | 36.41% | 14.63% | 14.63% | 10.14% | 10.14% |
| degraded: skew2 | 3 | 8.29% | 8.29% | 45.62% | 45.62% | 0.00% | 0.00% | 0.00% | 0.00% |
| degraded: skew3 | 3 | 28.57% | 28.57% | 38.48% | 38.48% | 19.05% | 19.05% | 27.42% | 27.42% |
| clean 300 dpi, article | 2 | 0.00% | 0.00% | 81.20% | 81.20% | 0.00% | 0.00% | 0.00% | 0.00% |
| clean 300 dpi, headings | 2 | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| clean 300 dpi, identifiers | 1 | 0.00% | 0.00% | 6.38% | 6.38% | 0.00% | 0.00% | 4.96% | 4.96% |
| clean 300 dpi, invoice | 4 | 0.00% | 0.00% | 9.87% | 9.87% | 44.79% | 44.79% | 23.54% | 23.54% |
| clean 300 dpi, letter | 4 | 0.00% | 0.00% | 0.17% | 0.17% | 0.00% | 0.00% | 0.00% | 0.00% |
| clean 300 dpi, mono | 1 | 4.95% | 4.95% | 48.41% | 48.41% | 73.50% | 73.50% | 75.62% | 75.62% |
| clean 300 dpi, smallprint | 2 | 0.00% | 0.00% | 5.20% | 5.20% | 0.00% | 0.00% | 0.00% | 0.00% |
| clean 300 dpi, table | 2 | 0.00% | 0.00% | 0.68% | 0.68% | 71.64% | 71.64% | 76.03% | 76.03% |
| clean 300 dpi, language da | 4 | 0.00% | 0.00% | 2.96% | 2.96% | n/a | n/a | 26.91% | 26.91% |
| clean 300 dpi, language de | 3 | 0.00% | 0.00% | 39.76% | 39.76% | n/a | n/a | 0.24% | 0.24% |
| clean 300 dpi, language en | 8 | 1.15% | 1.15% | 25.02% | 25.02% | 24.53% | 24.53% | 25.68% | 25.68% |
| clean 300 dpi, language fr | 3 | 0.00% | 0.00% | 7.34% | 7.34% | n/a | n/a | 0.43% | 0.43% |
| all pages, language da | 8 | 0.12% | 0.12% | 5.80% | 5.80% | n/a | n/a | 13.83% | 13.83% |
| all pages, language de | 15 | 3.92% | 3.92% | 21.15% | 21.15% | n/a | n/a | 3.16% | 3.16% |
| all pages, language en | 35 | 2.49% | 2.49% | 42.44% | 42.44% | 14.58% | 14.58% | 15.14% | 15.14% |
| all pages, language fr | 6 | 0.00% | 0.00% | 3.67% | 3.67% | n/a | n/a | 0.86% | 0.86% |
| all pages, language multi | 1 | 1.52% | 1.52% | 1.52% | 1.52% | n/a | n/a | 9.09% | 9.09% |

### Word bag F1 by condition (ignores reading order)

| Condition | Pages | pdfpecker-ocr/1 | pdfpecker-ocr/4 | rapidocr/1 | rapidocr/4 | tesseract-best/1 | tesseract-best/4 | tesseract/1 | tesseract/4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| all pages | 65 | 0.9960 | 0.9960 | 0.9601 | 0.9601 | 0.9729 | 0.9729 | 0.9713 | 0.9713 |
| clean (undegraded corpus pages, both resolutions) | 36 | 0.9938 | 0.9938 | 0.9397 | 0.9397 | 0.9592 | 0.9592 | 0.9666 | 0.9666 |
| degraded (all variants) | 27 | 0.9992 | 0.9992 | 0.9849 | 0.9849 | 0.9851 | 0.9851 | 0.9780 | 0.9780 |
| degraded, without rot180 | 24 | 0.9991 | 0.9991 | 0.9990 | 0.9990 | 0.9853 | 0.9853 | 0.9769 | 0.9769 |
| skewed (1, 2 and 3 degrees) | 9 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9881 | 0.9881 | 0.9639 | 0.9639 |
| rotated 180 degrees | 3 | 1.0000 | 1.0000 | 0.8699 | 0.8699 | 0.9830 | 0.9830 | 0.9862 | 0.9862 |
| clean, 300 dpi | 18 | 0.9944 | 0.9944 | 0.9323 | 0.9323 | 0.9211 | 0.9211 | 0.9421 | 0.9421 |
| clean, 150 dpi | 18 | 0.9932 | 0.9932 | 0.9472 | 0.9472 | 0.9975 | 0.9975 | 0.9910 | 0.9910 |
| Tesseract's sample images | 2 | 0.9841 | 0.9841 | 0.9920 | 0.9920 | 1.0000 | 1.0000 | 0.9524 | 0.9524 |
| degraded: blur | 3 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9830 | 0.9830 | 0.9873 | 0.9873 |
| degraded: jpeg30 | 3 | 0.9965 | 0.9965 | 0.9954 | 0.9954 | 0.9830 | 0.9830 | 0.9862 | 0.9862 |
| degraded: lowcontrast | 3 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9762 | 0.9762 | 0.9690 | 0.9690 |
| degraded: noise | 3 | 1.0000 | 1.0000 | 0.9965 | 0.9965 | 0.9796 | 0.9796 | 0.9896 | 0.9896 |
| degraded: rot180 | 3 | 1.0000 | 1.0000 | 0.8699 | 0.8699 | 0.9830 | 0.9830 | 0.9862 | 0.9862 |
| degraded: scan | 3 | 0.9965 | 0.9965 | 1.0000 | 1.0000 | 0.9966 | 0.9966 | 0.9908 | 0.9908 |
| degraded: skew1 | 3 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9830 | 0.9830 | 0.9862 | 0.9862 |
| degraded: skew2 | 3 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| degraded: skew3 | 3 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9813 | 0.9813 | 0.9015 | 0.9015 |
| clean 300 dpi, article | 2 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| clean 300 dpi, headings | 2 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| clean 300 dpi, identifiers | 1 | 1.0000 | 1.0000 | 0.9463 | 0.9463 | 1.0000 | 1.0000 | 0.9614 | 0.9614 |
| clean 300 dpi, invoice | 4 | 1.0000 | 1.0000 | 0.9131 | 0.9131 | 0.9479 | 0.9479 | 0.9759 | 0.9759 |
| clean 300 dpi, letter | 4 | 1.0000 | 1.0000 | 0.9983 | 0.9983 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| clean 300 dpi, mono | 1 | 0.9505 | 0.9505 | 0.5737 | 0.5737 | 0.7855 | 0.7855 | 0.7618 | 0.7618 |
| clean 300 dpi, smallprint | 2 | 1.0000 | 1.0000 | 0.9563 | 0.9563 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| clean 300 dpi, table | 2 | 1.0000 | 1.0000 | 0.9932 | 0.9932 | 0.4419 | 0.4419 | 0.4362 | 0.4362 |
| clean 300 dpi, language da | 4 | 1.0000 | 1.0000 | 0.9740 | 0.9740 | n/a | n/a | 0.9204 | 0.9204 |
| clean 300 dpi, language de | 3 | 1.0000 | 1.0000 | 0.9759 | 0.9759 | n/a | n/a | 0.9976 | 0.9976 |
| clean 300 dpi, language en | 8 | 0.9885 | 0.9885 | 0.9016 | 0.9016 | 0.9211 | 0.9211 | 0.9095 | 0.9095 |
| clean 300 dpi, language fr | 3 | 1.0000 | 1.0000 | 0.9357 | 0.9357 | n/a | n/a | 0.9968 | 0.9968 |
| all pages, language da | 8 | 0.9988 | 0.9988 | 0.9508 | 0.9508 | n/a | n/a | 0.9585 | 0.9585 |
| all pages, language de | 15 | 1.0000 | 1.0000 | 0.9827 | 0.9827 | n/a | n/a | 0.9818 | 0.9818 |
| all pages, language en | 35 | 0.9934 | 0.9934 | 0.9505 | 0.9505 | 0.9729 | 0.9729 | 0.9660 | 0.9660 |
| all pages, language fr | 6 | 1.0000 | 1.0000 | 0.9680 | 0.9680 | n/a | n/a | 0.9930 | 0.9930 |
| all pages, language multi | 1 | 0.9848 | 0.9848 | 0.9848 | 0.9848 | n/a | n/a | 0.9091 | 0.9091 |

### Character error rate of the sorted words by condition (ignores reading order)

| Condition | Pages | pdfpecker-ocr/1 | pdfpecker-ocr/4 | rapidocr/1 | rapidocr/4 | tesseract-best/1 | tesseract-best/4 | tesseract/1 | tesseract/4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| all pages | 65 | 0.38% | 0.38% | 6.40% | 6.40% | 2.19% | 2.19% | 3.30% | 3.30% |
| clean (undegraded corpus pages, both resolutions) | 36 | 0.39% | 0.39% | 9.43% | 9.43% | 3.16% | 3.16% | 3.65% | 3.65% |
| degraded (all variants) | 27 | 0.29% | 0.29% | 2.29% | 2.29% | 1.19% | 1.19% | 2.87% | 2.87% |
| degraded, without rot180 | 24 | 0.32% | 0.32% | 0.61% | 0.61% | 1.16% | 1.16% | 3.12% | 3.12% |
| skewed (1, 2 and 3 degrees) | 9 | 0.00% | 0.00% | 0.00% | 0.00% | 1.03% | 1.03% | 6.37% | 6.37% |
| rotated 180 degrees | 3 | 0.00% | 0.00% | 15.74% | 15.74% | 1.41% | 1.41% | 0.89% | 0.89% |
| clean, 300 dpi | 18 | 0.30% | 0.30% | 10.43% | 10.43% | 6.28% | 6.28% | 6.25% | 6.25% |
| clean, 150 dpi | 18 | 0.49% | 0.49% | 8.43% | 8.43% | 0.03% | 0.03% | 1.05% | 1.05% |
| Tesseract's sample images | 2 | 3.16% | 3.16% | 3.02% | 3.02% | 0.00% | 0.00% | 1.29% | 1.29% |
| degraded: blur | 3 | 0.00% | 0.00% | 0.00% | 0.00% | 1.41% | 1.41% | 0.85% | 0.85% |
| degraded: jpeg30 | 3 | 0.11% | 0.11% | 2.44% | 2.44% | 1.41% | 1.41% | 0.89% | 0.89% |
| degraded: lowcontrast | 3 | 0.00% | 0.00% | 0.00% | 0.00% | 1.86% | 1.86% | 2.18% | 2.18% |
| degraded: noise | 3 | 0.00% | 0.00% | 2.48% | 2.48% | 1.46% | 1.46% | 0.78% | 0.78% |
| degraded: rot180 | 3 | 0.00% | 0.00% | 15.74% | 15.74% | 1.41% | 1.41% | 0.89% | 0.89% |
| degraded: scan | 3 | 2.48% | 2.48% | 0.00% | 0.00% | 0.06% | 0.06% | 1.15% | 1.15% |
| degraded: skew1 | 3 | 0.00% | 0.00% | 0.00% | 0.00% | 1.41% | 1.41% | 0.96% | 0.96% |
| degraded: skew2 | 3 | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| degraded: skew3 | 3 | 0.00% | 0.00% | 0.00% | 0.00% | 1.69% | 1.69% | 18.14% | 18.14% |
| clean 300 dpi, article | 2 | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| clean 300 dpi, headings | 2 | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| clean 300 dpi, identifiers | 1 | 0.00% | 0.00% | 10.40% | 10.40% | 0.00% | 0.00% | 8.32% | 8.32% |
| clean 300 dpi, invoice | 4 | 0.00% | 0.00% | 10.20% | 10.20% | 3.78% | 3.78% | 1.16% | 1.16% |
| clean 300 dpi, letter | 4 | 0.00% | 0.00% | 0.11% | 0.11% | 0.00% | 0.00% | 0.00% | 0.00% |
| clean 300 dpi, mono | 1 | 1.90% | 1.90% | 45.26% | 45.26% | 10.90% | 10.90% | 14.13% | 14.13% |
| clean 300 dpi, smallprint | 2 | 0.00% | 0.00% | 6.62% | 6.62% | 0.00% | 0.00% | 0.00% | 0.00% |
| clean 300 dpi, table | 2 | 0.00% | 0.00% | 0.95% | 0.95% | 65.34% | 65.34% | 64.80% | 64.80% |
| clean 300 dpi, language da | 4 | 0.00% | 0.00% | 2.80% | 2.80% | n/a | n/a | 12.08% | 12.08% |
| clean 300 dpi, language de | 3 | 0.00% | 0.00% | 2.22% | 2.22% | n/a | n/a | 0.07% | 0.07% |
| clean 300 dpi, language en | 8 | 0.58% | 0.58% | 15.51% | 15.51% | 6.28% | 6.28% | 8.57% | 8.57% |
| clean 300 dpi, language fr | 3 | 0.00% | 0.00% | 10.23% | 10.23% | n/a | n/a | 0.10% | 0.10% |
| all pages, language da | 8 | 0.08% | 0.08% | 5.43% | 5.43% | n/a | n/a | 6.77% | 6.77% |
| all pages, language de | 15 | 0.00% | 0.00% | 1.78% | 1.78% | n/a | n/a | 2.83% | 2.83% |
| all pages, language en | 35 | 0.60% | 0.60% | 8.73% | 8.73% | 2.19% | 2.19% | 3.45% | 3.45% |
| all pages, language fr | 6 | 0.00% | 0.00% | 5.11% | 5.11% | n/a | n/a | 0.47% | 0.47% |
| all pages, language multi | 1 | 5.10% | 5.10% | 5.10% | 5.10% | n/a | n/a | 2.18% | 2.18% |

### Median time per page by condition (ms)

| Condition | Pages | pdfpecker-ocr/1 | pdfpecker-ocr/4 | rapidocr/1 | rapidocr/4 | tesseract-best/1 | tesseract-best/4 | tesseract/1 | tesseract/4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| all pages | 65 | 2122 | 581 | 5204 | 1905 | 1382 | 828 | 605 | 572 |
| clean (undegraded corpus pages, both resolutions) | 36 | 2180 | 594 | 4924 | 1886 | 1372 | 775 | 531 | 540 |
| degraded (all variants) | 27 | 2122 | 574 | 5238 | 1959 | 1547 | 870 | 635 | 618 |
| degraded, without rot180 | 24 | 2121 | 578 | 5237 | 1948 | 1463 | 811 | 624 | 611 |
| skewed (1, 2 and 3 degrees) | 9 | 2058 | 583 | 5237 | 2899 | 1376 | 811 | 616 | 610 |
| rotated 180 degrees | 3 | 2122 | 558 | 5257 | 3850 | 1784 | 1088 | 927 | 878 |
| clean, 300 dpi | 18 | 2213 | 608 | 5355 | 1886 | 1578 | 884 | 659 | 654 |
| clean, 150 dpi | 18 | 2115 | 582 | 4465 | 2120 | 1114 | 619 | 441 | 430 |
| Tesseract's sample images | 2 | 832 | 230 | 1517 | 1198 | 423 | 209 | 216 | 185 |
| degraded: blur | 3 | 2132 | 557 | 5402 | 3319 | 1533 | 787 | 625 | 618 |
| degraded: jpeg30 | 3 | 2178 | 590 | 5293 | 1959 | 1505 | 804 | 691 | 685 |
| degraded: lowcontrast | 3 | 2218 | 667 | 5398 | 3657 | 1484 | 806 | 674 | 612 |
| degraded: noise | 3 | 2119 | 525 | 5275 | 1793 | 1501 | 905 | 661 | 637 |
| degraded: rot180 | 3 | 2122 | 558 | 5257 | 3850 | 1784 | 1088 | 927 | 878 |
| degraded: scan | 3 | 2160 | 515 | 5084 | 1789 | 1346 | 776 | 602 | 570 |
| degraded: skew1 | 3 | 2058 | 602 | 5232 | 3879 | 1388 | 812 | 661 | 610 |
| degraded: skew2 | 3 | 2015 | 625 | 5237 | 3418 | 1328 | 759 | 616 | 628 |
| degraded: skew3 | 3 | 2117 | 567 | 5238 | 1867 | 1343 | 797 | 475 | 517 |
| clean 300 dpi, article | 2 | 2192 | 622 | 5618 | 1838 | 1777 | 1069 | 723 | 705 |
| clean 300 dpi, headings | 2 | 1546 | 451 | 4651 | 2391 | 1164 | 682 | 592 | 551 |
| clean 300 dpi, identifiers | 1 | 2626 | 833 | 6035 | 4789 | 2572 | 1380 | 900 | 855 |
| clean 300 dpi, invoice | 4 | 2267 | 648 | 5312 | 3174 | 1362 | 828 | 716 | 693 |
| clean 300 dpi, letter | 4 | 2103 | 574 | 5164 | 3042 | 1492 | 853 | 635 | 660 |
| clean 300 dpi, mono | 1 | 5838 | 1391 | 9231 | 3095 | 5518 | 2922 | 1944 | 1887 |
| clean 300 dpi, smallprint | 2 | 2143 | 590 | 5515 | 1866 | 1664 | 915 | 624 | 653 |
| clean 300 dpi, table | 2 | 2090 | 610 | 5368 | 2738 | 473 | 347 | 414 | 430 |
| clean 300 dpi, language da | 4 | 2100 | 561 | 5200 | 2531 | n/a | n/a | 695 | 640 |
| clean 300 dpi, language de | 3 | 2287 | 670 | 5447 | 4255 | n/a | n/a | 793 | 760 |
| clean 300 dpi, language en | 8 | 2037 | 585 | 5355 | 1821 | 1578 | 884 | 607 | 630 |
| clean 300 dpi, language fr | 3 | 2242 | 618 | 5236 | 1866 | n/a | n/a | 617 | 667 |
| all pages, language da | 8 | 2079 | 559 | 4751 | 2531 | n/a | n/a | 531 | 555 |
| all pages, language de | 15 | 2117 | 625 | 5225 | 1936 | n/a | n/a | 661 | 628 |
| all pages, language en | 35 | 2124 | 581 | 5237 | 1959 | 1382 | 828 | 505 | 520 |
| all pages, language fr | 6 | 2220 | 599 | 4915 | 1834 | n/a | n/a | 505 | 532 |
| all pages, language multi | 1 | 1008 | 274 | 1700 | 1898 | n/a | n/a | 298 | 245 |

### 90th percentile time per page by condition (ms)

| Condition | Pages | pdfpecker-ocr/1 | pdfpecker-ocr/4 | rapidocr/1 | rapidocr/4 | tesseract-best/1 | tesseract-best/4 | tesseract/1 | tesseract/4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| all pages | 65 | 2350 | 673 | 5527 | 4727 | 2036 | 1165 | 796 | 780 |
| clean (undegraded corpus pages, both resolutions) | 36 | 2570 | 706 | 5723 | 4847 | 3457 | 1736 | 813 | 795 |
| degraded (all variants) | 27 | 2235 | 664 | 5410 | 4470 | 1736 | 1005 | 767 | 734 |
| degraded, without rot180 | 24 | 2248 | 666 | 5417 | 4390 | 1716 | 906 | 695 | 671 |
| skewed (1, 2 and 3 degrees) | 9 | 2175 | 671 | 5307 | 4466 | 1567 | 893 | 668 | 647 |
| rotated 180 degrees | 3 | 2171 | 572 | 5260 | 4432 | 1969 | 1186 | 965 | 959 |
| clean, 300 dpi | 18 | 2439 | 767 | 5900 | 4762 | 3456 | 1842 | 850 | 826 |
| clean, 150 dpi | 18 | 2555 | 656 | 4906 | 4954 | 2738 | 1294 | 620 | 605 |
| Tesseract's sample images | 2 | 973 | 266 | 1664 | 1758 | 423 | 209 | 282 | 233 |
| degraded: blur | 3 | 2153 | 570 | 5614 | 4182 | 1654 | 874 | 668 | 627 |
| degraded: jpeg30 | 3 | 2292 | 605 | 5358 | 3889 | 1695 | 872 | 695 | 699 |
| degraded: lowcontrast | 3 | 2253 | 709 | 5447 | 4486 | 1656 | 870 | 708 | 621 |
| degraded: noise | 3 | 2123 | 573 | 5393 | 3829 | 1687 | 1058 | 740 | 659 |
| degraded: rot180 | 3 | 2171 | 572 | 5260 | 4432 | 1969 | 1186 | 965 | 959 |
| degraded: scan | 3 | 2305 | 599 | 5180 | 1851 | 1509 | 873 | 618 | 600 |
| degraded: skew1 | 3 | 2105 | 688 | 5293 | 4692 | 1533 | 859 | 686 | 661 |
| degraded: skew2 | 3 | 2148 | 654 | 5293 | 4171 | 1502 | 847 | 633 | 638 |
| degraded: skew3 | 3 | 2162 | 580 | 5249 | 1874 | 1520 | 892 | 603 | 604 |
| clean 300 dpi, article | 2 | 2326 | 660 | 5797 | 1891 | 1777 | 1069 | 779 | 749 |
| clean 300 dpi, headings | 2 | 1552 | 482 | 4742 | 3058 | 1164 | 682 | 677 | 608 |
| clean 300 dpi, identifiers | 1 | 2626 | 833 | 6035 | 4789 | 2572 | 1380 | 900 | 855 |
| clean 300 dpi, invoice | 4 | 2287 | 719 | 5398 | 6737 | 1362 | 828 | 802 | 788 |
| clean 300 dpi, letter | 4 | 2254 | 639 | 5384 | 4601 | 1492 | 853 | 758 | 753 |
| clean 300 dpi, mono | 1 | 5838 | 1391 | 9231 | 3095 | 5518 | 2922 | 1944 | 1887 |
| clean 300 dpi, smallprint | 2 | 2222 | 613 | 5560 | 1867 | 1664 | 915 | 631 | 664 |
| clean 300 dpi, table | 2 | 2220 | 660 | 5556 | 3544 | 473 | 347 | 494 | 520 |
| clean 300 dpi, language da | 4 | 2232 | 657 | 5514 | 3590 | n/a | n/a | 769 | 742 |
| clean 300 dpi, language de | 3 | 2345 | 724 | 5763 | 4460 | n/a | n/a | 821 | 803 |
| clean 300 dpi, language en | 8 | 3590 | 1000 | 6994 | 5660 | 3456 | 1842 | 1213 | 1164 |
| clean 300 dpi, language fr | 3 | 2251 | 663 | 5415 | 4173 | n/a | n/a | 717 | 716 |
| all pages, language da | 8 | 2283 | 643 | 5396 | 4687 | n/a | n/a | 728 | 694 |
| all pages, language de | 15 | 2330 | 694 | 5441 | 4621 | n/a | n/a | 814 | 792 |
| all pages, language en | 35 | 2501 | 696 | 5629 | 4705 | 2036 | 1165 | 852 | 825 |
| all pages, language fr | 6 | 2271 | 655 | 5348 | 4631 | n/a | n/a | 679 | 697 |
| all pages, language multi | 1 | 1008 | 274 | 1700 | 1898 | n/a | n/a | 298 | 245 |

### Mean CPU time per page by condition (ms)

| Condition | Pages | pdfpecker-ocr/1 | pdfpecker-ocr/4 | rapidocr/1 | rapidocr/4 | tesseract-best/1 | tesseract-best/4 | tesseract/1 | tesseract/4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| all pages | 65 | 2184 | 2261 | 5015 | 7631 | 1569 | 2758 | 597 | 1644 |
| clean (undegraded corpus pages, both resolutions) | 36 | 2316 | 2385 | 5052 | 7734 | 1766 | 3112 | 597 | 1711 |
| degraded (all variants) | 27 | 2108 | 2201 | 5225 | 7838 | 1457 | 2555 | 625 | 1630 |
| degraded, without rot180 | 24 | 2109 | 2212 | 5230 | 7774 | 1417 | 2518 | 592 | 1591 |
| skewed (1, 2 and 3 degrees) | 9 | 2048 | 2260 | 5184 | 7943 | 1340 | 2422 | 569 | 1514 |
| rotated 180 degrees | 3 | 2097 | 2115 | 5183 | 8354 | 1776 | 2849 | 889 | 1946 |
| clean, 300 dpi | 18 | 2312 | 2429 | 5495 | 7973 | 1980 | 3509 | 715 | 1889 |
| clean, 150 dpi | 18 | 2321 | 2342 | 4609 | 7496 | 1551 | 2716 | 480 | 1533 |
| Tesseract's sample images | 2 | 819 | 825 | 1509 | 2988 | 423 | 745 | 215 | 619 |
| degraded: blur | 3 | 2106 | 2122 | 5377 | 8194 | 1526 | 2434 | 597 | 1619 |
| degraded: jpeg30 | 3 | 2180 | 2155 | 5260 | 7892 | 1493 | 2541 | 623 | 1705 |
| degraded: lowcontrast | 3 | 2215 | 2494 | 5334 | 8401 | 1470 | 2560 | 620 | 1654 |
| degraded: noise | 3 | 2075 | 2077 | 5262 | 7285 | 1495 | 2941 | 630 | 1675 |
| degraded: rot180 | 3 | 2097 | 2115 | 5183 | 8354 | 1776 | 2849 | 889 | 1946 |
| degraded: scan | 3 | 2154 | 2071 | 5055 | 6590 | 1336 | 2404 | 561 | 1530 |
| degraded: skew1 | 3 | 2039 | 2355 | 5181 | 8348 | 1371 | 2544 | 610 | 1618 |
| degraded: skew2 | 3 | 2043 | 2269 | 5189 | 8853 | 1320 | 2282 | 573 | 1556 |
| degraded: skew3 | 3 | 2063 | 2154 | 5182 | 6628 | 1329 | 2438 | 525 | 1369 |
| clean 300 dpi, article | 2 | 2185 | 2341 | 5582 | 6812 | 1763 | 3324 | 720 | 2044 |
| clean 300 dpi, headings | 2 | 1525 | 1663 | 4649 | 6822 | 1150 | 1957 | 583 | 1446 |
| clean 300 dpi, identifiers | 1 | 2611 | 3122 | 5972 | 10230 | 2521 | 4557 | 891 | 2519 |
| clean 300 dpi, invoice | 4 | 2242 | 2378 | 5279 | 8293 | 1355 | 2569 | 687 | 1721 |
| clean 300 dpi, letter | 4 | 2106 | 2167 | 5175 | 7986 | 1485 | 2586 | 659 | 1810 |
| clean 300 dpi, mono | 1 | 5781 | 5364 | 9118 | 11243 | 5473 | 9652 | 1917 | 5514 |
| clean 300 dpi, smallprint | 2 | 2129 | 2223 | 5462 | 6860 | 1622 | 2838 | 622 | 1751 |
| clean 300 dpi, table | 2 | 2074 | 2298 | 5312 | 7963 | 471 | 586 | 413 | 680 |
| clean 300 dpi, language da | 4 | 1984 | 2136 | 5151 | 7675 | n/a | n/a | 671 | 1675 |
| clean 300 dpi, language de | 3 | 2296 | 2514 | 5529 | 8394 | n/a | n/a | 756 | 2015 |
| clean 300 dpi, language en | 8 | 2522 | 2604 | 5743 | 8010 | 1980 | 3509 | 746 | 2037 |
| clean 300 dpi, language fr | 3 | 2204 | 2267 | 5260 | 7847 | n/a | n/a | 649 | 1652 |
| all pages, language da | 8 | 1991 | 2081 | 4687 | 7449 | n/a | n/a | 568 | 1628 |
| all pages, language de | 15 | 2140 | 2279 | 5140 | 7633 | n/a | n/a | 654 | 1721 |
| all pages, language en | 35 | 2281 | 2338 | 5157 | 7807 | 1569 | 2758 | 602 | 1668 |
| all pages, language fr | 6 | 2183 | 2221 | 4863 | 7448 | n/a | n/a | 514 | 1471 |
| all pages, language multi | 1 | 989 | 954 | 1700 | 4024 | n/a | n/a | 296 | 817 |

### Peak resident memory by condition, median (MB)

| Condition | Pages | pdfpecker-ocr/1 | pdfpecker-ocr/4 | rapidocr/1 | rapidocr/4 | tesseract-best/1 | tesseract-best/4 | tesseract/1 | tesseract/4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| all pages | 65 | 323 | 324 | 746 | 747 | 154 | 154 | 167 | 166 |
| clean (undegraded corpus pages, both resolutions) | 36 | 314 | 315 | 678 | 681 | 165 | 166 | 164 | 164 |
| degraded (all variants) | 27 | 325 | 326 | 747 | 748 | 140 | 138 | 175 | 186 |
| degraded, without rot180 | 24 | 326 | 327 | 747 | 748 | 140 | 138 | 166 | 166 |
| skewed (1, 2 and 3 degrees) | 9 | 369 | 371 | 747 | 748 | 156 | 154 | 193 | 204 |
| rotated 180 degrees | 3 | 313 | 314 | 746 | 747 | 155 | 155 | 175 | 194 |
| clean, 300 dpi | 18 | 378 | 380 | 748 | 753 | 154 | 153 | 164 | 164 |
| clean, 150 dpi | 18 | 302 | 304 | 598 | 600 | 165 | 166 | 133 | 133 |
| Tesseract's sample images | 2 | 362 | 380 | 341 | 342 | 169 | 169 | 171 | 171 |
| degraded: blur | 3 | 302 | 303 | 746 | 747 | 136 | 136 | 117 | 117 |
| degraded: jpeg30 | 3 | 302 | 303 | 746 | 747 | 149 | 149 | 137 | 137 |
| degraded: lowcontrast | 3 | 370 | 371 | 747 | 748 | 119 | 117 | 189 | 189 |
| degraded: noise | 3 | 326 | 327 | 748 | 749 | 114 | 114 | 109 | 110 |
| degraded: rot180 | 3 | 313 | 314 | 746 | 747 | 155 | 155 | 175 | 194 |
| degraded: scan | 3 | 354 | 356 | 747 | 748 | 137 | 137 | 130 | 130 |
| degraded: skew1 | 3 | 377 | 378 | 747 | 748 | 136 | 134 | 204 | 204 |
| degraded: skew2 | 3 | 324 | 325 | 746 | 748 | 172 | 172 | 193 | 211 |
| degraded: skew3 | 3 | 369 | 371 | 747 | 748 | 146 | 140 | 147 | 147 |
| clean 300 dpi, article | 2 | 290 | 291 | 747 | 748 | 83 | 83 | 118 | 131 |
| clean 300 dpi, headings | 2 | 300 | 301 | 747 | 748 | 123 | 119 | 118 | 118 |
| clean 300 dpi, identifiers | 1 | 576 | 577 | 748 | 757 | 115 | 115 | 213 | 213 |
| clean 300 dpi, invoice | 4 | 395 | 396 | 764 | 762 | 132 | 130 | 175 | 176 |
| clean 300 dpi, letter | 4 | 413 | 431 | 748 | 754 | 191 | 191 | 175 | 175 |
| clean 300 dpi, mono | 1 | 386 | 387 | 747 | 756 | 190 | 186 | 174 | 174 |
| clean 300 dpi, smallprint | 2 | 527 | 528 | 748 | 757 | 210 | 206 | 204 | 204 |
| clean 300 dpi, table | 2 | 317 | 319 | 748 | 752 | 175 | 175 | 132 | 131 |
| clean 300 dpi, language da | 4 | 239 | 241 | 747 | 748 | n/a | n/a | 81 | 82 |
| clean 300 dpi, language de | 3 | 488 | 489 | 748 | 766 | n/a | n/a | 165 | 165 |
| clean 300 dpi, language en | 8 | 386 | 387 | 747 | 756 | 154 | 153 | 168 | 168 |
| clean 300 dpi, language fr | 3 | 563 | 564 | 756 | 757 | n/a | n/a | 243 | 243 |
| all pages, language da | 8 | 229 | 230 | 672 | 673 | n/a | n/a | 60 | 60 |
| all pages, language de | 15 | 308 | 310 | 747 | 748 | n/a | n/a | 110 | 116 |
| all pages, language en | 35 | 328 | 329 | 746 | 747 | 154 | 154 | 175 | 186 |
| all pages, language fr | 6 | 467 | 468 | 679 | 682 | n/a | n/a | 210 | 235 |
| all pages, language multi | 1 | 338 | 373 | 355 | 358 | n/a | n/a | 185 | 185 |

### Per page

Character error rate and median time per page, one column pair per configuration. `-` is a failed page.

| Page | pdfpecker-ocr/1 CER | pdfpecker-ocr/1 ms | pdfpecker-ocr/4 CER | pdfpecker-ocr/4 ms | rapidocr/1 CER | rapidocr/1 ms | rapidocr/4 CER | rapidocr/4 ms | tesseract-best/1 CER | tesseract-best/1 ms | tesseract-best/4 CER | tesseract-best/4 ms | tesseract/1 CER | tesseract/1 ms | tesseract/4 CER | tesseract/4 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| article-de.150 | 0.00% | 2532 | 0.00% | 663 | 68.91% | 4760 | 68.91% | 1648 | - | - | - | - | 0.08% | 631 | 0.08% | 538 |
| article-de.300 | 0.00% | 2359 | 0.00% | 670 | 68.91% | 5842 | 68.91% | 1905 | - | - | - | - | 0.00% | 793 | 0.00% | 760 |
| article-en.150 | 0.00% | 2224 | 0.00% | 567 | 65.14% | 4369 | 65.14% | 3888 | 0.00% | 1382 | 0.00% | 723 | 0.00% | 475 | 0.00% | 425 |
| article-en.300 | 0.00% | 2025 | 0.00% | 575 | 65.14% | 5393 | 65.14% | 1771 | 0.00% | 1777 | 0.00% | 1069 | 0.00% | 654 | 0.00% | 650 |
| article-en.300.blur | 0.00% | 2158 | 0.00% | 557 | 65.14% | 5187 | 65.14% | 4398 | 0.00% | 1684 | 0.00% | 896 | 0.00% | 679 | 0.00% | 618 |
| article-en.300.jpeg30 | 0.00% | 2178 | 0.00% | 609 | 65.14% | 5161 | 65.14% | 4371 | 0.00% | 1743 | 0.00% | 889 | 0.00% | 691 | 0.00% | 703 |
| article-en.300.lowcontrast | 0.00% | 2261 | 0.00% | 719 | 65.14% | 5459 | 65.14% | 3657 | 0.00% | 1698 | 0.00% | 886 | 0.00% | 717 | 0.00% | 612 |
| article-en.300.noise | 0.00% | 2119 | 0.00% | 524 | 65.14% | 5275 | 65.14% | 1793 | 0.00% | 1733 | 0.00% | 1096 | 0.00% | 760 | 0.00% | 664 |
| article-en.300.rot180 | 0.00% | 2183 | 0.00% | 575 | 71.52% | 5257 | 71.52% | 1844 | 0.00% | 2015 | 0.00% | 1211 | 0.00% | 927 | 0.00% | 878 |
| article-en.300.scan | 0.00% | 2341 | 0.00% | 515 | 65.14% | 5055 | 65.14% | 1760 | 0.00% | 1550 | 0.00% | 897 | 0.00% | 602 | 0.00% | 608 |
| article-en.300.skew1 | 0.00% | 2029 | 0.00% | 602 | 65.14% | 5232 | 65.14% | 3879 | 0.00% | 1569 | 0.00% | 870 | 0.00% | 661 | 0.00% | 674 |
| article-en.300.skew2 | 0.00% | 1995 | 0.00% | 572 | 70.89% | 5185 | 70.89% | 2899 | 0.00% | 1545 | 0.00% | 869 | 0.00% | 616 | 0.00% | 640 |
| article-en.300.skew3 | 3.77% | 1987 | 3.77% | 583 | 59.57% | 5179 | 59.57% | 1876 | 0.18% | 1565 | 0.18% | 916 | 0.54% | 635 | 0.54% | 625 |
| headings-da.150 | 0.00% | 1549 | 0.00% | 392 | 6.59% | 3856 | 6.59% | 1268 | - | - | - | - | 0.00% | 451 | 0.00% | 435 |
| headings-da.300 | 0.00% | 1554 | 0.00% | 490 | 0.00% | 4764 | 0.00% | 3225 | - | - | - | - | 0.00% | 698 | 0.00% | 622 |
| headings-en.150 | 0.00% | 1518 | 0.00% | 434 | 0.00% | 3744 | 0.00% | 5066 | 0.00% | 973 | 0.00% | 443 | 0.00% | 306 | 0.00% | 342 |
| headings-en.300 | 0.00% | 1539 | 0.00% | 412 | 0.00% | 4537 | 0.00% | 1556 | 0.00% | 1164 | 0.00% | 682 | 0.00% | 485 | 0.00% | 481 |
| identifiers-en.150 | 0.00% | 2608 | 0.00% | 647 | 8.77% | 5247 | 8.77% | 2438 | 0.00% | 2050 | 0.00% | 952 | 0.07% | 615 | 0.07% | 693 |
| identifiers-en.300 | 0.00% | 2626 | 0.00% | 833 | 7.88% | 6035 | 7.88% | 4789 | 0.00% | 2572 | 0.00% | 1380 | 0.30% | 900 | 0.30% | 855 |
| invoice-da.150 | 0.00% | 2087 | 0.00% | 630 | 8.57% | 4362 | 8.57% | 4593 | - | - | - | - | 0.30% | 477 | 0.30% | 499 |
| invoice-da.300 | 0.00% | 2184 | 0.00% | 622 | 8.42% | 5308 | 8.42% | 1836 | - | - | - | - | 39.55% | 691 | 39.55% | 658 |
| invoice-de.150 | 0.00% | 2117 | 0.00% | 653 | 0.00% | 4448 | 0.00% | 1629 | - | - | - | - | 0.14% | 480 | 0.14% | 436 |
| invoice-de.300 | 0.00% | 2280 | 0.00% | 738 | 7.43% | 5432 | 7.43% | 4511 | - | - | - | - | 0.29% | 828 | 0.29% | 813 |
| invoice-en.150 | 0.00% | 2281 | 0.00% | 584 | 0.00% | 4647 | 0.00% | 1656 | 0.00% | 954 | 0.00% | 529 | 0.30% | 347 | 0.30% | 311 |
| invoice-en.300 | 0.00% | 2290 | 0.00% | 597 | 0.00% | 5317 | 0.00% | 7691 | 38.37% | 1362 | 38.37% | 828 | 38.37% | 505 | 38.37% | 518 |
| invoice-en.300.blur | 0.00% | 2081 | 0.00% | 574 | 0.00% | 5668 | 0.00% | 1834 | 38.37% | 1382 | 38.37% | 678 | 38.22% | 494 | 38.22% | 520 |
| invoice-en.300.jpeg30 | 0.15% | 2320 | 0.15% | 590 | 0.30% | 5293 | 0.30% | 1959 | 38.37% | 1266 | 38.37% | 718 | 38.22% | 500 | 38.22% | 493 |
| invoice-en.300.lowcontrast | 0.00% | 2218 | 0.00% | 618 | 0.00% | 5398 | 0.00% | 1818 | 38.37% | 1269 | 38.37% | 726 | 37.76% | 485 | 37.76% | 510 |
| invoice-en.300.noise | 0.00% | 2124 | 0.00% | 586 | 0.15% | 5423 | 0.15% | 4338 | 38.52% | 1268 | 38.52% | 714 | 37.92% | 480 | 37.92% | 482 |
| invoice-en.300.rot180 | 0.00% | 2122 | 0.00% | 558 | 77.34% | 5261 | 77.34% | 4578 | 38.37% | 1554 | 38.37% | 966 | 38.22% | 779 | 38.22% | 781 |
| invoice-en.300.scan | 15.26% | 2160 | 15.26% | 620 | 0.00% | 5204 | 0.00% | 1866 | 0.15% | 1143 | 0.15% | 656 | 0.30% | 464 | 0.30% | 450 |
| invoice-en.300.skew1 | 0.00% | 2116 | 0.00% | 557 | 0.00% | 5308 | 0.00% | 1750 | 38.22% | 1207 | 38.22% | 754 | 38.37% | 488 | 38.37% | 506 |
| invoice-en.300.skew2 | 31.42% | 2181 | 31.42% | 661 | 18.28% | 5237 | 18.28% | 4359 | 0.00% | 1111 | 0.00% | 649 | 0.00% | 484 | 0.00% | 466 |
| invoice-en.300.skew3 | 29.00% | 2174 | 29.00% | 567 | 17.82% | 5238 | 17.82% | 1766 | 50.30% | 1122 | 50.30% | 678 | 50.30% | 471 | 50.30% | 517 |
| invoice-fr.150 | 0.00% | 2199 | 0.00% | 635 | 0.00% | 4348 | 0.00% | 4512 | - | - | - | - | 0.69% | 405 | 0.69% | 405 |
| invoice-fr.300 | 0.00% | 2254 | 0.00% | 674 | 7.41% | 5198 | 7.41% | 1802 | - | - | - | - | 0.14% | 741 | 0.14% | 728 |
| letter-da.150 | 0.12% | 2072 | 0.12% | 520 | 6.82% | 4280 | 6.82% | 4905 | - | - | - | - | 0.00% | 547 | 0.00% | 567 |
| letter-da.300 | 0.00% | 2016 | 0.00% | 501 | 0.12% | 5091 | 0.12% | 1830 | - | - | - | - | 0.00% | 799 | 0.00% | 777 |
| letter-de.150 | 0.00% | 2088 | 0.00% | 573 | 0.00% | 4649 | 0.00% | 1514 | - | - | - | - | 0.00% | 453 | 0.00% | 443 |
| letter-de.300 | 0.00% | 2287 | 0.00% | 658 | 0.00% | 5447 | 0.00% | 4255 | - | - | - | - | 0.00% | 665 | 0.00% | 697 |
| letter-de.300.blur | 0.00% | 2132 | 0.00% | 541 | 0.00% | 5402 | 0.00% | 3319 | - | - | - | - | 0.00% | 625 | 0.00% | 629 |
| letter-de.300.jpeg30 | 0.00% | 2079 | 0.00% | 533 | 0.00% | 5374 | 0.00% | 1936 | - | - | - | - | 0.00% | 696 | 0.00% | 685 |
| letter-de.300.lowcontrast | 0.00% | 2187 | 0.00% | 667 | 0.00% | 5223 | 0.00% | 4694 | - | - | - | - | 0.00% | 674 | 0.00% | 624 |
| letter-de.300.noise | 0.00% | 2015 | 0.00% | 525 | 0.00% | 5225 | 0.00% | 1756 | - | - | - | - | 0.00% | 661 | 0.00% | 637 |
| letter-de.300.rot180 | 0.00% | 2052 | 0.00% | 551 | 69.64% | 5154 | 69.64% | 3850 | - | - | - | - | 0.00% | 975 | 0.00% | 980 |
| letter-de.300.scan | 0.00% | 2035 | 0.00% | 498 | 0.00% | 5084 | 0.00% | 1789 | - | - | - | - | 0.54% | 622 | 0.54% | 570 |
| letter-de.300.skew1 | 0.00% | 2058 | 0.00% | 709 | 0.00% | 5098 | 0.00% | 4895 | - | - | - | - | 0.11% | 692 | 0.11% | 610 |
| letter-de.300.skew2 | 0.00% | 2015 | 0.00% | 625 | 0.00% | 5307 | 0.00% | 3418 | - | - | - | - | 0.00% | 638 | 0.00% | 628 |
| letter-de.300.skew3 | 45.92% | 2117 | 45.92% | 550 | 0.00% | 5252 | 0.00% | 1867 | - | - | - | - | 38.95% | 475 | 38.95% | 452 |
| letter-en.150 | 0.00% | 2019 | 0.00% | 514 | 0.00% | 4342 | 0.00% | 1607 | 0.00% | 1182 | 0.00% | 708 | 0.00% | 432 | 0.00% | 383 |
| letter-en.300 | 0.00% | 2030 | 0.00% | 596 | 0.00% | 5084 | 0.00% | 1774 | 0.00% | 1492 | 0.00% | 853 | 0.00% | 582 | 0.00% | 622 |
| letter-fr.150 | 0.00% | 2287 | 0.00% | 580 | 0.00% | 4519 | 0.00% | 1481 | - | - | - | - | 0.00% | 385 | 0.00% | 492 |
| letter-fr.300 | 0.00% | 2177 | 0.00% | 552 | 0.00% | 5236 | 0.00% | 4750 | - | - | - | - | 0.00% | 605 | 0.00% | 572 |
| mono-en.150 | 0.61% | 6074 | 0.61% | 1455 | 34.97% | 8125 | 34.97% | 6834 | 50.32% | 4341 | 50.32% | 2092 | 50.55% | 1402 | 50.55% | 1255 |
| mono-en.300 | 0.53% | 5838 | 0.53% | 1391 | 40.53% | 9231 | 40.53% | 3095 | 54.40% | 5518 | 54.40% | 2922 | 54.78% | 1944 | 54.78% | 1887 |
| smallprint-en.150 | 0.00% | 2112 | 0.00% | 581 | 0.00% | 4482 | 0.00% | 3957 | 0.10% | 1047 | 0.10% | 482 | 0.00% | 343 | 0.00% | 315 |
| smallprint-en.300 | 0.00% | 2043 | 0.00% | 561 | 0.00% | 5571 | 0.00% | 1867 | 0.00% | 1664 | 0.00% | 915 | 0.00% | 632 | 0.00% | 638 |
| smallprint-fr.150 | 0.00% | 2074 | 0.00% | 522 | 0.00% | 4633 | 0.00% | 1604 | - | - | - | - | 0.09% | 354 | 0.09% | 380 |
| smallprint-fr.300 | 0.00% | 2242 | 0.00% | 618 | 8.40% | 5460 | 8.40% | 1866 | - | - | - | - | 0.00% | 617 | 0.00% | 667 |
| table-da.150 | 0.00% | 2356 | 0.00% | 598 | 0.21% | 4739 | 0.21% | 1802 | - | - | - | - | 0.41% | 397 | 0.41% | 426 |
| table-da.300 | 0.00% | 2253 | 0.00% | 672 | 0.00% | 5603 | 0.00% | 3746 | - | - | - | - | 70.37% | 514 | 70.37% | 542 |
| table-en.150 | 0.00% | 2024 | 0.00% | 591 | 0.00% | 4255 | 0.00% | 3931 | 0.28% | 576 | 0.28% | 297 | 0.28% | 227 | 0.28% | 209 |
| table-en.300 | 0.00% | 1927 | 0.00% | 549 | 0.28% | 5133 | 0.28% | 1730 | 65.34% | 473 | 65.34% | 347 | 65.34% | 314 | 65.34% | 318 |
| tesseract-eurotext | 0.24% | 1008 | 0.24% | 274 | 0.24% | 1700 | 0.24% | 1898 | - | - | - | - | 1.46% | 298 | 1.46% | 245 |
| tesseract-phototest | 0.35% | 656 | 0.35% | 185 | 0.00% | 1334 | 0.00% | 498 | 0.00% | 423 | 0.00% | 209 | 0.00% | 133 | 0.00% | 125 |
