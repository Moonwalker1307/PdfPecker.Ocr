> Checked-in copy of the report. The raw logs, every child's output and the recognised text of every page (`raw/` below)
> are not in the repository; they were kept with the run.

# PdfPecker.Ocr against Tesseract 5.5.1 and rapidocr, 2026-11-08

## In short

- **Accuracy: PdfPecker.Ocr reads the synthetic corpus best.** Over all 65 pages its character error rate (CER) in
  reading order is 1.7%, against 10.4% for Tesseract with `tessdata_fast` and 23.3% for rapidocr. If reading order is
  ignored, the CER is 0.38% against 3.3% for Tesseract and 6.4% for rapidocr, and word bag F1 is 0.996 against 0.971
  and 0.960. Tesseract with `tessdata_best` on the 35 English pages has an order-free CER of 2.2% (PdfPecker.Ocr 0.60%
  on the same pages).
- **Speed: Tesseract is about four times faster per core.** At one thread the median page takes 2.35 s with
  PdfPecker.Ocr, 0.58 s with Tesseract fast, 1.35 s with Tesseract best (English pages) and 5.2 s with rapidocr. At four
  threads PdfPecker.Ocr's median is 0.73 s, close to Tesseract fast's 0.57 s and Tesseract best's 0.76 s: Tesseract
  fast hardly gets faster with more threads (2%) but uses 2.7 times the CPU time. Measured as CPU time per page, which is what a busy
  server pays, PdfPecker.Ocr needs 2.4 s, Tesseract fast 0.59 s, Tesseract best 1.6 s and rapidocr 5.0 s.
- **Memory: PdfPecker.Ocr needs five to six times what Tesseract needs.** A fresh process reading one A4 page at
  300 dpi peaks at 431 MB (906 MB for a page of long monospaced lines), Tesseract fast at 67 MB (76 MB) and rapidocr's
  Python process at 548 MB (571 MB).
- **PdfPecker.Ocr is about 2.2 times faster than rapidocr with ONNX Runtime at one thread and 2.6 times at four**, on the
  same models. The comparison is not exact: rapidocr limits the detection input to 2,000 px against PdfPecker.Ocr's
  1,536 px, and its own settings differ.
- **Where PdfPecker.Ocr loses on accuracy, it is reading order on skewed pages**: on the nine pages skewed by 1 to 3
  degrees its CER in reading order is 10.7%, while every word is read correctly (order-free CER 0.00%, F1 1.000).
- **Where Tesseract loses**: bordered tables at 300 dpi (on both it dropped most of the cells, F1 0.43–0.44), single-digit table
  columns ("4 8 3 6 7" read as "N DOD W OO Sf"), invoices read column by column, and a stall of OpenMP threads under
  load (one page took 420 s at 4 threads while other jobs ran).

The machine is a shared cloud VM with 4 vCPUs (Intel Xeon at 2.1 GHz, AVX-512) and 15 GB of RAM. Other agents ran builds
and tests on it at the same time: the 1-minute load average had a median of 4.1 and a maximum of 12.2 during the runs
(`raw/loadavg.log`). At one thread every engine got close to a whole core: CPU time per page is 99% of its wall time.
The four-thread times are noisier. Only compare these numbers with each other, never with a run on another machine.

## Headline table

All 65 pages, except Tesseract best (only the 35 English pages, for which `tessdata_best` has a model). Times are the
median over pages of each page's median of 3 timed runs after 1 warm-up. Peak memory comes from a fresh process that
reads one page once (`fresh.csv`).

| | PdfPecker.Ocr | Tesseract 5.5.1 fast | Tesseract 5.5.1 best (en) | rapidocr 3.10 + ONNX Runtime |
| --- | ---: | ---: | ---: | ---: |
| CER, reading order | 1.71% | 10.37% | 15.21% | 23.32% |
| CER, order-free (sorted words) | 0.38% | 3.30% | 2.19% | 6.40% |
| WER, reading order | 2.35% | 10.75% | 14.58% | 29.96% |
| Word bag F1 | 0.9960 | 0.9713 | 0.9729 | 0.9601 |
| Same, English pages only: CER / order-free CER / F1 | 1.72% / 0.60% / 0.9934 | 15.26% / 3.45% / 0.9660 | 15.21% / 2.19% / 0.9729 | 32.87% / 8.73% / 0.9505 |
| Median ms per page, 1 thread | 2354 | 583 | 1351 | 5204 |
| p90 ms per page, 1 thread | 2647 | 764 | 1990 | 5527 |
| Median ms per page, 4 threads | 733 | 569 | 755 | 1905 |
| p90 ms per page, 4 threads | 1442 | 775 | 1104 | 4727 |
| CPU ms per page, mean, 1 / 4 threads | 2420 / 2587 | 587 / 1615 | 1565 / 2650 | 5015 / 7631 |
| Peak RSS, fresh process, A4 invoice at 300 dpi | 431 MB | 67 MB | 80 MB | 548 MB (Python process) |
| Peak RSS, fresh process, A4 log of long monospaced lines | 906 MB | 76 MB | 90 MB | 571 MB |
| Whole run, process peak (65 pages, 1 thread) | 1267 MB | 251 MB | 211 MB | 798 MB |
| Fresh process, one 300 dpi invoice, 1 thread: total / of which start-up and model load | 3.1–3.3 s / 0.6–0.7 s | 0.69–0.76 s / 0.19 s | 1.4–1.7 s / 0.17–0.19 s | 7.7–8.5 s / 1.6 s |
| Size on disk | 34 MB self-contained Native AOT executable (models embedded); or 17.2 MB `PdfPecker.Ocr.dll` + 6.1 MB `PdfPecker.dll` on a .NET runtime | 7.0 MB `libtesseract.so` (Leptonica linked in, no image codecs) + 1.1–4.1 MB per language; 10.6 MB more for orientation detection | 7.0 MB + 15.4 MB for `eng` | 31 MB of ONNX models + about 360 MB of Python packages (ONNX Runtime, NumPy, OpenCV, Shapely, ...) + Python |

Accuracy does not change with the thread count for any engine (PdfPecker.Ocr's output is deterministic; the rows for 1
and 4 threads are identical).

## Accuracy by page group

CER in reading order / order-free CER / word bag F1, all pages of the group (Tesseract best: English pages of the group).

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

### Where the ground truth's reading order penalises an engine

The ground truth reads tables row by row and the two-column article left column first. That is one valid order, not the
only one, so the CER in reading order punishes layouts an engine reads differently:

- **Tesseract reads the invoice table column by column** (its page segmentation makes a block of each column), so every
  invoice page loses about 38% CER in reading order while its order-free CER is 3–4%. It also reads the monospaced log
  as columns (timestamps first, then levels, then messages): 50–55% CER, 3% order-free at 150 dpi.
- **rapidocr interleaves the two columns** of the article (it orders boxes by their top edge in 10 px bands): 65–71% CER on
  every article page with an order-free CER of 0.00%. It also does not turn upside-down pages as a whole, so the rot180
  pages come out bottom line first.
- **PdfPecker.Ocr reads skewed tables and letters out of order** (next section).
- **Tesseract on the rot180 pages** was run with page segmentation mode 1 (orientation detection), which a caller who
  expects upside-down scans would choose; it then reads them as well as the upright ones.

## Where PdfPecker.Ocr loses, and why

### Speed and CPU: the recognition network is heavy, and short lines are padded

PdfPecker.Ocr spends about a quarter of a page in detection and three quarters in recognition, independent of the
resolution (detection always runs at 1,536 px on the longer side, and line crops are always 48 px high), which is why
150 dpi pages are not faster (2.30 s against 2.43 s), while Tesseract gets a third faster at 150 dpi.

A per-layer profile (`raw/profile-t1.txt`, `raw/profile-t4.txt`, `raw/profile-t1b.txt`, new `profile` command of
`bench/PdfPecker.Ocr.Benchmarks`) of four 300 dpi pages at one thread:

| Page | Lines | Detection | Orientation | Recognition | Time steps computed / of the lines themselves |
| --- | ---: | ---: | ---: | ---: | --- |
| invoice-en.300 | 42 | 598 ms | 90 ms | 1656 ms | 2664 / 1545 (58%) |
| letter-en.300 | 22 | 561 ms | 72 ms | 1410 ms | 2346 / 1986 (85%) |
| table-en.300 | 56 | 609 ms | 66 ms | 1431 ms | 2384 / 749 (31%) |
| mono-en.300 | 36 | 619 ms | 124 ms | 5692 ms | 8052 / 7705 (96%) |

- **Recognition time is in the backbone's 1x1 convolutions, not in the output layer.** 1x1 convolutions take 57–66% of
  the recognition network, attention 3–14% (it grows with the square of the line length: 792 ms on the mono page), and
  the final 120 to 18,710 class projection only 5–7%, plus 4% for its softmax. The integration report's guess that the
  projection dominates does not hold; a fused max-probability output would save about a tenth of recognition.
- **The kernels are not the gap.** On the same models PdfPecker.Ocr is 2.2 times faster than ONNX Runtime at one thread
  (2354 against 5204 ms median, and on every one of the 65 pages), so its convolutions are not slow by the standard of a
  mature runtime. The gap to Tesseract is the amount of arithmetic: PP-OCRv6's recognizer is a convolutional backbone with
  attention over every line plus a detection network over the whole page, against Tesseract's small LSTM over lines found
  by classical layout analysis. I did not measure the kernels' share of the processor's peak.
- **Short lines are padded to 320 px.** The recognizer pads every batch to at least 320 px (rapidocr's `rec_img_shape`),
  so a table cell of one number costs as much as a 320 px line. On the table page only 31% of the computed time steps
  belong to the lines; on the invoice 58%. rapidocr with the minimum lowered from 320 to 48 px read 410 lines of ten
  pages with one line different, and that line closer to the ground truth (`raw/padexp.txt`: table-en.300 went from 8 to
  2 character edits). Padding each batch only to its widest line would cut the table page's recognition work by about
  two thirds and the invoice's by about 40%.
- **Detection costs 560–650 ms at one thread on every page**, a quarter of the total, because its input size is fixed.
  The two 3x3 convolutions at 384 x 272 with 24 output channels (#5 and #75) take 120 ms of it.
- **Threads scale well**: 2354 to 733 ms median (3.2 times) at 4 threads.

### Memory: buffers that grow step by step leave their old copies on the heap

The engine's buffers grow to exactly what the current input needs (`InferenceContext.Pointer` and `OutputBuffer`
allocate a new pinned array when the old one is too small and drop the old one). The recognizer sorts lines by width,
shortest first, so within one page every batch is a little wider than the last and every batch allocates a new arena,
new scratch and a new output buffer of batch × steps × 18,710 floats (about 110 MB for six 1,964 px lines). The engine
allocates almost nothing else (0.5–1 MB of managed memory per warm page), so the garbage collector does not run and the
old pinned arrays stay resident. Evidence:

- In a fresh process the long-line page peaks at 906 MB and the invoice at 431 MB; six output buffers of 6 × 198–245
  steps × 18,710 floats add up to about 600 MB, which matches the difference.
- In one long run the resident set only grows: 396 MB after the first page, 1,231 MB after the mono page, and it stays
  there for every later page (`raw/children/pdfpecker-ocr-t1-p0.csv`, column `peak_rss_mb`), although later pages need less.
- 161 MB are resident after the models are loaded, before any page.

Running the widest batch first (so the first allocation is the largest), growing buffers geometrically, or returning the
old buffer to a pool would remove most of this; capping the batch width or splitting very long lines would bound it.

### Reading order on skewed pages

`Layout/ReadingOrder.cs` turns the page by quarter turns only and then makes XY cuts on axis-aligned boxes. On a page
skewed by 2 or 3 degrees, the boxes of one table row drift by 30–100 px from left to right, so no horizontal gap crosses
the whole table; the cut falls between the columns instead, and the table is read column by column (invoice-en.300.skew2
CER 31%, skew3 29%, letter-de.300.skew3 46%: the address block and paragraphs out of order). Every word is right on all
nine skewed pages. Measuring the dominant small angle of the lines and rotating the boxes by it before the cuts would fix
this; Tesseract deskews before its layout analysis and has 0% CER in reading order on invoice-en.300.skew2.

### Smaller recognition errors

- Liberation Mono capitals: "INFO" is read as "INF0" and "WARN" as "wARN" on every line of the mono page (rapidocr reads
  the same, so it is the model).
- Separators: "| https://..." becomes "|https://..." on the scan and JPEG pages.
- Danish "Ø" read as "∅" once at 150 dpi; "over" as "ovér" on phototest.

## Tesseract notes

- **OpenMP under load**: with 4 OpenMP threads and other processes on the machine, Tesseract stalled: in the first
  attempt invoice-da.150 took 0.46 s at best and 420 s at worst, article-en.300.jpeg30 had a median of 51 s, and a fresh
  process needed 230 s for one invoice. The same pages took 0.4–0.8 s when the load was lower. Those runs are kept in
  `raw/loaded-p0/`; the 4-thread Tesseract rows in the tables come from a second run of all four parts at a load average
  of 2–3. Even when it works, 4 threads make Tesseract fast only 2% faster for 2.7 times the CPU time (Tesseract best,
  whose LSTM is larger, gets 44% faster for 1.7 times the CPU time); Tesseract's documentation recommends
  `OMP_THREAD_LIMIT=1` for servers that read several pages at once.
- **Bordered tables**: on table-da.300 and table-en.300 Tesseract fast and best dropped most of the cells (on table-en.300
  it returned the title, "North 370", four more numbers and the footnote; F1 0.43–0.44); at 150 dpi the same tables are
  read fine (F1 0.985–0.994).
- **The build has no PNG support** in Leptonica, so Tesseract's `ImageData::SetPixInternal` stores each line image as BMP
  instead of PNG (Leptonica prints "png library missing; output bmp format" to stderr). That is slightly faster than a
  stock build, which compresses every line as PNG.

## Method

- Corpus: `tools/Corpus` with `--tesseract-samples`, seed 20261108: 63 synthetic A4 pages drawn by PdfPecker 1.23.0
  (invoices, letters, two-column articles, tables, small print, headings, a monospaced log, identifiers; English, Danish,
  German, French) at 300 and 150 dpi, 27 degraded 300 dpi variants, and Tesseract's `phototest.tif` and `eurotext.tif`.
  The pages are clean renderings or simulated degradations, not scans of paper, which flatters every engine.
- App: `bench/TesseractComparison` at commit b65d738 (on PdfPecker.Ocr f481c43 of `wip/integrate`), published with
  Native AOT, `IlcInstructionSet=native`. Every engine sits behind the same adapter interface and is timed by the same
  code on decoded pixels; no image decoding is timed.
  - PdfPecker.Ocr: in process, `ImageOcr.RecognizeGray` (gray pages) or `RecognizeRgba` (colour pages, converted from
    RGB inside the timed region), default options, `MaxDegreeOfParallelism` = threads.
  - Tesseract 5.5.1 with Leptonica 1.85.0 (`tools/Tesseract/build.sh`), in process through the C API, LSTM engine,
    `tessdata_fast` in the page's language (eng, dan, deu, fra; eng+deu+fra for eurotext), page segmentation 3, and 1 with
    orientation detection for the rot180 pages. `tesseract-best`: `tessdata_best` eng, English pages only.
    `OMP_THREAD_LIMIT` and `OMP_NUM_THREADS` = threads.
  - rapidocr 3.10.0 with ONNX Runtime 1.31.1 in a Python 3.13 child process, defaults of its `config.yaml`, word boxes on,
    `intra_op_num_threads` = threads; raw pixels over a pipe (the transfer is timed, a few ms).
- Runs: one fresh process per engine, thread count and quarter of the corpus (pages dealt round robin), each under the
  shared benchmark lock (`flock`) so no other measurement ran at the same time; 3 timed runs after 1 warm-up per page.
  Peak memory per page is the resident peak after a reset before the timed runs; it includes memory retained from earlier
  pages, so the table uses the fresh-process figures (`fresh.csv`: one page, one run, no warm-up, two repetitions).
- Scores: text normalised the same way for ground truth and output (Unicode composition of Latin accents, typographic
  quotes and dashes to ASCII, whitespace runs to one space). CER and WER are Levenshtein distances over the page text;
  group figures are micro averages (edits summed over pages). Word bag F1 counts matching words regardless of order.
- Raw data: `results.csv` (one row per engine, thread count and page), `fresh.csv`, and under `raw/children/` every child's CSV,
  `.meta` and log, the recognised text of every page (`raw/hyp/`), the drivers (`run-compare.sh`, `run-fresh2.sh`,
  `run-deferred.sh`, `run-p0t4.sh`), the load log, the per-layer profiles and the padding experiment
  (`raw/padexp.txt`, `raw/padexp.py`). The generated per-condition and per-page tables follow.

---

## Generated report: PdfPecker.Ocr, Tesseract 5.5.1 and rapidocr on the synthetic corpus

Generated by `bench/TesseractComparison`. All engines read the same synthetic corpus (`tools/Corpus`), one engine and thread count per fresh process.

### Engines

- `pdfpecker-ocr/1`: PdfPecker.Ocr 0.1.0.0, MaxDegreeOfParallelism 1. Start-up 548 ms plus first model load 0 ms, 161 MB resident after loading, 3 timed runs after 1 warm-up per page.
- `pdfpecker-ocr/4`: PdfPecker.Ocr 0.1.0.0, MaxDegreeOfParallelism 4. Start-up 559 ms plus first model load 0 ms, 162 MB resident after loading, 3 timed runs after 1 warm-up per page.
- `rapidocr/1`: rapidocr 3.10.0 (PP-OCRv6 small models) on ONNX Runtime, defaults of its config.yaml, word boxes on, in a Python child process. Start-up 1321 ms plus first model load 0 ms, 174 MB resident after loading, 3 timed runs after 1 warm-up per page.
- `rapidocr/4`: rapidocr 3.10.0 (PP-OCRv6 small models) on ONNX Runtime, defaults of its config.yaml, word boxes on, in a Python child process. Start-up 870 ms plus first model load 0 ms, 175 MB resident after loading, 3 timed runs after 1 warm-up per page.
- `tesseract-best/1`: Tesseract 5.5.1, LSTM engine, models in tools/Tesseract/out/tessdata-best, language eng, page segmentation auto, OMP_THREAD_LIMIT=1. Start-up 0 ms plus first model load 103 ms, 38 MB resident after loading, 3 timed runs after 1 warm-up per page.
- `tesseract-best/4`: Tesseract 5.5.1, LSTM engine, models in tools/Tesseract/out/tessdata-best, language eng, page segmentation auto, OMP_THREAD_LIMIT=4. Start-up 0 ms plus first model load 106 ms, 37 MB resident after loading, 3 timed runs after 1 warm-up per page.
- `tesseract/1`: Tesseract 5.5.1, LSTM engine, models in tools/Tesseract/out/tessdata, language auto, page segmentation auto, OMP_THREAD_LIMIT=1. Start-up 0 ms plus first model load 82 ms, 23 MB resident after loading, 3 timed runs after 1 warm-up per page.
- `tesseract/4`: Tesseract 5.5.1, LSTM engine, models in tools/Tesseract/out/tessdata, language auto, page segmentation auto, OMP_THREAD_LIMIT=4. Start-up 0 ms plus first model load 73 ms, 23 MB resident after loading, 3 timed runs after 1 warm-up per page.

### Summary

Accuracy: CER and WER over all text of the pages (micro), word bag F1 over all words (micro, order-free), and the CER of the sorted words (order-free). Time: the per-page medians of the timed runs, then their median, 90th percentile and mean over pages. CPU is all threads of the engine's process. Peak RSS per page is the resident peak while that page was read, models included; the process peak is the whole child's.

| Config | Pages | CER | CER (mean of pages) | WER | Word bag F1 | CER, order-free | Box recall @IoU 0.5 | Median ms/page | p90 ms/page | Mean ms/page | CPU ms/page, mean | Peak RSS per page, median / max (MB) | Process peak (MB) | Start-up + model load (ms) | Failed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pdfpecker-ocr/1 | 65 | 1.71% | 1.96% | 2.35% | 0.9960 | 0.38% | 69.34% | 2354 | 2647 | 2439 | 2420 | 575 / 1267 | 1267 | 548 | 0 |
| pdfpecker-ocr/4 | 65 | 1.71% | 1.96% | 2.35% | 0.9960 | 0.38% | 69.34% | 733 | 1442 | 901 | 2587 | 577 / 1275 | 1275 | 559 | 0 |
| rapidocr/1 | 65 | 23.32% | 18.32% | 29.96% | 0.9601 | 6.40% | 70.43% | 5204 | 5527 | 5056 | 5015 | 746 / 791 | 798 | 1321 | 0 |
| rapidocr/4 | 65 | 23.32% | 18.32% | 29.96% | 0.9601 | 6.40% | 70.43% | 1905 | 4727 | 2908 | 7631 | 747 / 782 | 782 | 870 | 0 |
| tesseract-best/1 | 35 | 15.21% | 13.99% | 14.58% | 0.9729 | 2.19% | 96.70% | 1351 | 1990 | 1576 | 1565 | 147 / 210 | 211 | 103 | 0 |
| tesseract-best/4 | 35 | 15.21% | 13.99% | 14.58% | 0.9729 | 2.19% | 96.70% | 755 | 1104 | 838 | 2650 | 154 / 211 | 212 | 106 | 0 |
| tesseract/1 | 65 | 10.37% | 9.89% | 10.75% | 0.9713 | 3.30% | 96.26% | 583 | 764 | 591 | 587 | 166 / 251 | 251 | 82 | 0 |
| tesseract/4 | 65 | 10.37% | 9.89% | 10.75% | 0.9713 | 3.30% | 96.26% | 569 | 775 | 582 | 1615 | 166 / 251 | 251 | 73 | 0 |

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
| all pages | 65 | 2354 | 733 | 5204 | 1905 | 1351 | 755 | 583 | 569 |
| clean (undegraded corpus pages, both resolutions) | 36 | 2386 | 843 | 4924 | 1886 | 1307 | 720 | 533 | 542 |
| degraded (all variants) | 27 | 2291 | 707 | 5238 | 1959 | 1509 | 823 | 623 | 625 |
| degraded, without rot180 | 24 | 2340 | 705 | 5237 | 1948 | 1407 | 778 | 616 | 585 |
| skewed (1, 2 and 3 degrees) | 9 | 2272 | 728 | 5237 | 2899 | 1340 | 778 | 607 | 558 |
| rotated 180 degrees | 3 | 2261 | 710 | 5257 | 3850 | 1760 | 1091 | 922 | 934 |
| clean, 300 dpi | 18 | 2432 | 853 | 5355 | 1886 | 1539 | 860 | 632 | 626 |
| clean, 150 dpi | 18 | 2302 | 825 | 4465 | 2120 | 1114 | 530 | 416 | 421 |
| Tesseract's sample images | 2 | 861 | 271 | 1517 | 1198 | 415 | 214 | 199 | 196 |
| degraded: blur | 3 | 2291 | 707 | 5402 | 3319 | 1494 | 778 | 618 | 569 |
| degraded: jpeg30 | 3 | 2354 | 702 | 5293 | 1959 | 1506 | 812 | 636 | 626 |
| degraded: lowcontrast | 3 | 2623 | 1289 | 5398 | 3657 | 1583 | 793 | 625 | 660 |
| degraded: noise | 3 | 2386 | 696 | 5275 | 1793 | 1484 | 776 | 638 | 631 |
| degraded: rot180 | 3 | 2261 | 710 | 5257 | 3850 | 1760 | 1091 | 922 | 934 |
| degraded: scan | 3 | 2248 | 670 | 5084 | 1789 | 1382 | 747 | 583 | 590 |
| degraded: skew1 | 3 | 2434 | 679 | 5232 | 3879 | 1528 | 820 | 607 | 637 |
| degraded: skew2 | 3 | 2272 | 657 | 5237 | 3418 | 1302 | 729 | 614 | 558 |
| degraded: skew3 | 3 | 2216 | 754 | 5238 | 1867 | 1318 | 771 | 483 | 488 |
| clean 300 dpi, article | 2 | 2462 | 1104 | 5618 | 1838 | 1830 | 887 | 697 | 660 |
| clean 300 dpi, headings | 2 | 2131 | 705 | 4651 | 2391 | 1168 | 624 | 584 | 521 |
| clean 300 dpi, identifiers | 1 | 3059 | 842 | 6035 | 4789 | 2462 | 1264 | 893 | 834 |
| clean 300 dpi, invoice | 4 | 2490 | 841 | 5312 | 3174 | 1264 | 755 | 699 | 671 |
| clean 300 dpi, letter | 4 | 2303 | 1061 | 5164 | 3042 | 1504 | 863 | 663 | 626 |
| clean 300 dpi, mono | 1 | 6784 | 1898 | 9231 | 3095 | 5468 | 2747 | 1862 | 1897 |
| clean 300 dpi, smallprint | 2 | 2454 | 1055 | 5515 | 1866 | 1573 | 857 | 609 | 626 |
| clean 300 dpi, table | 2 | 2426 | 1089 | 5368 | 2738 | 440 | 341 | 458 | 429 |
| clean 300 dpi, language da | 4 | 2378 | 877 | 5200 | 2531 | n/a | n/a | 647 | 602 |
| clean 300 dpi, language de | 3 | 2628 | 1484 | 5447 | 4255 | n/a | n/a | 767 | 691 |
| clean 300 dpi, language en | 8 | 2382 | 717 | 5355 | 1821 | 1539 | 860 | 588 | 603 |
| clean 300 dpi, language fr | 3 | 2540 | 864 | 5236 | 1866 | n/a | n/a | 612 | 661 |
| all pages, language da | 8 | 2317 | 891 | 4751 | 2531 | n/a | n/a | 571 | 543 |
| all pages, language de | 15 | 2354 | 805 | 5225 | 1936 | n/a | n/a | 628 | 626 |
| all pages, language en | 35 | 2326 | 710 | 5237 | 1959 | 1351 | 755 | 534 | 539 |
| all pages, language fr | 6 | 2469 | 885 | 4915 | 1834 | n/a | n/a | 490 | 485 |
| all pages, language multi | 1 | 1018 | 313 | 1700 | 1898 | n/a | n/a | 269 | 248 |

### 90th percentile time per page by condition (ms)

| Condition | Pages | pdfpecker-ocr/1 | pdfpecker-ocr/4 | rapidocr/1 | rapidocr/4 | tesseract-best/1 | tesseract-best/4 | tesseract/1 | tesseract/4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| all pages | 65 | 2647 | 1442 | 5527 | 4727 | 1990 | 1104 | 764 | 775 |
| clean (undegraded corpus pages, both resolutions) | 36 | 2754 | 1506 | 5723 | 4847 | 3408 | 1734 | 793 | 810 |
| degraded (all variants) | 27 | 2600 | 1384 | 5410 | 4470 | 1899 | 971 | 732 | 724 |
| degraded, without rot180 | 24 | 2612 | 1356 | 5417 | 4390 | 1830 | 907 | 687 | 665 |
| skewed (1, 2 and 3 degrees) | 9 | 2529 | 1381 | 5307 | 4466 | 1707 | 879 | 641 | 643 |
| rotated 180 degrees | 3 | 2262 | 1287 | 5260 | 4432 | 1955 | 1145 | 993 | 934 |
| clean, 300 dpi | 18 | 2775 | 1536 | 5900 | 4762 | 3364 | 1709 | 841 | 822 |
| clean, 150 dpi | 18 | 2635 | 1398 | 4906 | 4954 | 2686 | 1328 | 559 | 562 |
| Tesseract's sample images | 2 | 987 | 305 | 1664 | 1758 | 415 | 214 | 255 | 238 |
| degraded: blur | 3 | 2426 | 710 | 5614 | 4182 | 1725 | 873 | 626 | 647 |
| degraded: jpeg30 | 3 | 2454 | 734 | 5358 | 3889 | 1662 | 921 | 685 | 711 |
| degraded: lowcontrast | 3 | 2662 | 1384 | 5447 | 4486 | 1879 | 890 | 723 | 661 |
| degraded: noise | 3 | 2545 | 725 | 5393 | 3829 | 1666 | 842 | 705 | 659 |
| degraded: rot180 | 3 | 2262 | 1287 | 5260 | 4432 | 1955 | 1145 | 993 | 934 |
| degraded: scan | 3 | 2310 | 744 | 5180 | 1851 | 1499 | 814 | 646 | 618 |
| degraded: skew1 | 3 | 2451 | 1238 | 5293 | 4692 | 1807 | 883 | 620 | 659 |
| degraded: skew2 | 3 | 2443 | 1246 | 5293 | 4171 | 1462 | 797 | 631 | 577 |
| degraded: skew3 | 3 | 2605 | 1195 | 5249 | 1874 | 1493 | 841 | 628 | 601 |
| clean 300 dpi, article | 2 | 2604 | 1408 | 5797 | 1891 | 1830 | 887 | 753 | 685 |
| clean 300 dpi, headings | 2 | 2342 | 890 | 4742 | 3058 | 1168 | 624 | 642 | 569 |
| clean 300 dpi, identifiers | 1 | 3059 | 842 | 6035 | 4789 | 2462 | 1264 | 893 | 834 |
| clean 300 dpi, invoice | 4 | 2607 | 1330 | 5398 | 6737 | 1264 | 755 | 801 | 788 |
| clean 300 dpi, letter | 4 | 2580 | 1433 | 5384 | 4601 | 1504 | 863 | 758 | 754 |
| clean 300 dpi, mono | 1 | 6784 | 1898 | 9231 | 3095 | 5468 | 2747 | 1862 | 1897 |
| clean 300 dpi, smallprint | 2 | 2523 | 1350 | 5560 | 1867 | 1573 | 857 | 612 | 654 |
| clean 300 dpi, table | 2 | 2607 | 1460 | 5556 | 3544 | 440 | 341 | 579 | 525 |
| clean 300 dpi, language da | 4 | 2575 | 1368 | 5514 | 3590 | n/a | n/a | 729 | 749 |
| clean 300 dpi, language de | 3 | 2637 | 1520 | 5763 | 4460 | n/a | n/a | 808 | 792 |
| clean 300 dpi, language en | 8 | 4176 | 1583 | 6994 | 5660 | 3364 | 1709 | 1184 | 1153 |
| clean 300 dpi, language fr | 3 | 2576 | 1312 | 5415 | 4173 | n/a | n/a | 731 | 707 |
| all pages, language da | 8 | 2487 | 1422 | 5396 | 4687 | n/a | n/a | 687 | 677 |
| all pages, language de | 15 | 2624 | 1463 | 5441 | 4621 | n/a | n/a | 798 | 767 |
| all pages, language en | 35 | 2794 | 1432 | 5629 | 4705 | 1990 | 1104 | 835 | 794 |
| all pages, language fr | 6 | 2562 | 1183 | 5348 | 4631 | n/a | n/a | 686 | 690 |
| all pages, language multi | 1 | 1018 | 313 | 1700 | 1898 | n/a | n/a | 269 | 248 |

### Mean CPU time per page by condition (ms)

| Condition | Pages | pdfpecker-ocr/1 | pdfpecker-ocr/4 | rapidocr/1 | rapidocr/4 | tesseract-best/1 | tesseract-best/4 | tesseract/1 | tesseract/4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| all pages | 65 | 2420 | 2587 | 5015 | 7631 | 1565 | 2650 | 587 | 1615 |
| clean (undegraded corpus pages, both resolutions) | 36 | 2571 | 2764 | 5052 | 7734 | 1740 | 2959 | 584 | 1684 |
| degraded (all variants) | 27 | 2335 | 2473 | 5225 | 7838 | 1474 | 2482 | 619 | 1595 |
| degraded, without rot180 | 24 | 2346 | 2471 | 5230 | 7774 | 1440 | 2431 | 587 | 1560 |
| skewed (1, 2 and 3 degrees) | 9 | 2328 | 2478 | 5184 | 7943 | 1373 | 2371 | 560 | 1466 |
| rotated 180 degrees | 3 | 2244 | 2492 | 5183 | 8354 | 1746 | 2887 | 878 | 1868 |
| clean, 300 dpi | 18 | 2651 | 2804 | 5495 | 7973 | 1945 | 3240 | 706 | 1843 |
| clean, 150 dpi | 18 | 2490 | 2724 | 4609 | 7496 | 1534 | 2678 | 462 | 1524 |
| Tesseract's sample images | 2 | 861 | 937 | 1509 | 2988 | 409 | 747 | 199 | 655 |
| degraded: blur | 3 | 2283 | 2413 | 5377 | 8194 | 1486 | 2455 | 574 | 1578 |
| degraded: jpeg30 | 3 | 2328 | 2433 | 5260 | 7892 | 1496 | 2545 | 609 | 1684 |
| degraded: lowcontrast | 3 | 2560 | 2670 | 5334 | 8401 | 1576 | 2570 | 633 | 1679 |
| degraded: noise | 3 | 2379 | 2422 | 5262 | 7285 | 1466 | 2467 | 624 | 1616 |
| degraded: rot180 | 3 | 2244 | 2492 | 5183 | 8354 | 1746 | 2887 | 878 | 1868 |
| degraded: scan | 3 | 2232 | 2394 | 5055 | 6590 | 1374 | 2301 | 572 | 1528 |
| degraded: skew1 | 3 | 2346 | 2450 | 5181 | 8348 | 1517 | 2577 | 581 | 1559 |
| degraded: skew2 | 3 | 2277 | 2418 | 5189 | 8853 | 1299 | 2222 | 566 | 1456 |
| degraded: skew3 | 3 | 2361 | 2566 | 5182 | 6628 | 1302 | 2313 | 533 | 1382 |
| clean 300 dpi, article | 2 | 2442 | 2662 | 5582 | 6812 | 1789 | 2849 | 692 | 1892 |
| clean 300 dpi, headings | 2 | 2114 | 1777 | 4649 | 6822 | 1161 | 1844 | 566 | 1412 |
| clean 300 dpi, identifiers | 1 | 3045 | 2919 | 5972 | 10230 | 2438 | 4115 | 892 | 2457 |
| clean 300 dpi, invoice | 4 | 2472 | 2728 | 5279 | 8293 | 1261 | 2376 | 673 | 1676 |
| clean 300 dpi, letter | 4 | 2325 | 2619 | 5175 | 7986 | 1490 | 2633 | 657 | 1761 |
| clean 300 dpi, mono | 1 | 6714 | 6723 | 9118 | 11243 | 5426 | 8881 | 1852 | 5630 |
| clean 300 dpi, smallprint | 2 | 2443 | 2603 | 5462 | 6860 | 1559 | 2631 | 606 | 1694 |
| clean 300 dpi, table | 2 | 2391 | 2675 | 5312 | 7963 | 434 | 590 | 457 | 675 |
| clean 300 dpi, language da | 4 | 2360 | 2594 | 5151 | 7675 | n/a | n/a | 662 | 1667 |
| clean 300 dpi, language de | 3 | 2608 | 2816 | 5529 | 8394 | n/a | n/a | 771 | 1861 |
| clean 300 dpi, language en | 8 | 2904 | 2973 | 5743 | 8010 | 1945 | 3240 | 727 | 2002 |
| clean 300 dpi, language fr | 3 | 2408 | 2620 | 5260 | 7847 | n/a | n/a | 645 | 1637 |
| all pages, language da | 8 | 2218 | 2564 | 4687 | 7449 | n/a | n/a | 559 | 1610 |
| all pages, language de | 15 | 2363 | 2607 | 5140 | 7633 | n/a | n/a | 645 | 1680 |
| all pages, language en | 35 | 2535 | 2633 | 5157 | 7807 | 1565 | 2650 | 590 | 1650 |
| all pages, language fr | 6 | 2395 | 2549 | 4863 | 7448 | n/a | n/a | 511 | 1388 |
| all pages, language multi | 1 | 1017 | 1082 | 1700 | 4024 | n/a | n/a | 269 | 824 |

### Peak resident memory by condition, median (MB)

| Condition | Pages | pdfpecker-ocr/1 | pdfpecker-ocr/4 | rapidocr/1 | rapidocr/4 | tesseract-best/1 | tesseract-best/4 | tesseract/1 | tesseract/4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| all pages | 65 | 575 | 577 | 746 | 747 | 147 | 154 | 166 | 166 |
| clean (undegraded corpus pages, both resolutions) | 36 | 640 | 651 | 678 | 681 | 162 | 166 | 164 | 164 |
| degraded (all variants) | 27 | 556 | 558 | 747 | 748 | 131 | 140 | 175 | 175 |
| degraded, without rot180 | 24 | 554 | 556 | 747 | 748 | 126 | 140 | 166 | 166 |
| skewed (1, 2 and 3 degrees) | 9 | 556 | 560 | 747 | 748 | 148 | 156 | 193 | 192 |
| rotated 180 degrees | 3 | 650 | 652 | 746 | 747 | 154 | 154 | 175 | 175 |
| clean, 300 dpi | 18 | 734 | 741 | 748 | 753 | 148 | 153 | 164 | 164 |
| clean, 150 dpi | 18 | 565 | 568 | 598 | 600 | 164 | 166 | 133 | 133 |
| Tesseract's sample images | 2 | 1199 | 1201 | 341 | 342 | 169 | 170 | 171 | 171 |
| degraded: blur | 3 | 543 | 545 | 746 | 747 | 136 | 136 | 117 | 117 |
| degraded: jpeg30 | 3 | 563 | 565 | 746 | 747 | 149 | 145 | 137 | 137 |
| degraded: lowcontrast | 3 | 549 | 551 | 747 | 748 | 119 | 119 | 190 | 190 |
| degraded: noise | 3 | 520 | 522 | 748 | 749 | 102 | 114 | 110 | 110 |
| degraded: rot180 | 3 | 650 | 652 | 746 | 747 | 154 | 154 | 175 | 175 |
| degraded: scan | 3 | 536 | 538 | 747 | 748 | 125 | 137 | 130 | 130 |
| degraded: skew1 | 3 | 553 | 555 | 747 | 748 | 136 | 136 | 204 | 204 |
| degraded: skew2 | 3 | 658 | 660 | 746 | 748 | 172 | 172 | 193 | 192 |
| degraded: skew3 | 3 | 556 | 558 | 747 | 748 | 134 | 146 | 147 | 147 |
| clean 300 dpi, article | 2 | 472 | 474 | 747 | 748 | 83 | 83 | 118 | 119 |
| clean 300 dpi, headings | 2 | 479 | 494 | 747 | 748 | 124 | 124 | 118 | 118 |
| clean 300 dpi, identifiers | 1 | 795 | 828 | 748 | 757 | 123 | 115 | 213 | 213 |
| clean 300 dpi, invoice | 4 | 669 | 671 | 764 | 762 | 129 | 130 | 176 | 176 |
| clean 300 dpi, letter | 4 | 828 | 846 | 748 | 754 | 166 | 191 | 175 | 175 |
| clean 300 dpi, mono | 1 | 1137 | 1139 | 747 | 756 | 191 | 190 | 174 | 174 |
| clean 300 dpi, smallprint | 2 | 1017 | 1064 | 748 | 757 | 210 | 211 | 204 | 204 |
| clean 300 dpi, table | 2 | 773 | 789 | 748 | 752 | 175 | 176 | 131 | 131 |
| clean 300 dpi, language da | 4 | 409 | 424 | 747 | 748 | n/a | n/a | 81 | 81 |
| clean 300 dpi, language de | 3 | 673 | 674 | 748 | 766 | n/a | n/a | 164 | 165 |
| clean 300 dpi, language en | 8 | 824 | 855 | 747 | 756 | 148 | 153 | 168 | 168 |
| clean 300 dpi, language fr | 3 | 1254 | 1256 | 756 | 757 | n/a | n/a | 243 | 243 |
| all pages, language da | 8 | 398 | 406 | 672 | 673 | n/a | n/a | 60 | 60 |
| all pages, language de | 15 | 516 | 518 | 747 | 748 | n/a | n/a | 110 | 109 |
| all pages, language en | 35 | 658 | 660 | 746 | 747 | 147 | 154 | 175 | 175 |
| all pages, language fr | 6 | 1071 | 1119 | 679 | 682 | n/a | n/a | 210 | 210 |
| all pages, language multi | 1 | 1254 | 1256 | 355 | 358 | n/a | n/a | 185 | 185 |

### Per page

Character error rate and median time per page, one column pair per configuration. `-` is a failed page.

| Page | pdfpecker-ocr/1 CER | pdfpecker-ocr/1 ms | pdfpecker-ocr/4 CER | pdfpecker-ocr/4 ms | rapidocr/1 CER | rapidocr/1 ms | rapidocr/4 CER | rapidocr/4 ms | tesseract-best/1 CER | tesseract-best/1 ms | tesseract-best/4 CER | tesseract-best/4 ms | tesseract/1 CER | tesseract/1 ms | tesseract/4 CER | tesseract/4 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| article-de.150 | 0.00% | 2514 | 0.00% | 862 | 68.91% | 4760 | 68.91% | 1648 | - | - | - | - | 0.08% | 532 | 0.08% | 546 |
| article-de.300 | 0.00% | 2639 | 0.00% | 1484 | 68.91% | 5842 | 68.91% | 1905 | - | - | - | - | 0.00% | 767 | 0.00% | 691 |
| article-en.150 | 0.00% | 2259 | 0.00% | 670 | 65.14% | 4369 | 65.14% | 3888 | 0.00% | 1351 | 0.00% | 686 | 0.00% | 428 | 0.00% | 415 |
| article-en.300 | 0.00% | 2285 | 0.00% | 724 | 65.14% | 5393 | 65.14% | 1771 | 0.00% | 1830 | 0.00% | 887 | 0.00% | 627 | 0.00% | 630 |
| article-en.300.blur | 0.00% | 2205 | 0.00% | 710 | 65.14% | 5187 | 65.14% | 4398 | 0.00% | 1783 | 0.00% | 897 | 0.00% | 618 | 0.00% | 667 |
| article-en.300.jpeg30 | 0.00% | 2202 | 0.00% | 649 | 65.14% | 5161 | 65.14% | 4371 | 0.00% | 1701 | 0.00% | 949 | 0.00% | 636 | 0.00% | 733 |
| article-en.300.lowcontrast | 0.00% | 2672 | 0.00% | 1408 | 65.14% | 5459 | 65.14% | 3657 | 0.00% | 1953 | 0.00% | 915 | 0.00% | 748 | 0.00% | 660 |
| article-en.300.noise | 0.00% | 2386 | 0.00% | 689 | 65.14% | 5275 | 65.14% | 1793 | 0.00% | 1712 | 0.00% | 859 | 0.00% | 638 | 0.00% | 666 |
| article-en.300.rot180 | 0.00% | 2262 | 0.00% | 710 | 71.52% | 5257 | 71.52% | 1844 | 0.00% | 2003 | 0.00% | 1158 | 0.00% | 922 | 0.00% | 935 |
| article-en.300.scan | 0.00% | 2248 | 0.00% | 632 | 65.14% | 5055 | 65.14% | 1760 | 0.00% | 1528 | 0.00% | 830 | 0.00% | 662 | 0.00% | 590 |
| article-en.300.skew1 | 0.00% | 2455 | 0.00% | 1378 | 65.14% | 5232 | 65.14% | 3879 | 0.00% | 1877 | 0.00% | 899 | 0.00% | 607 | 0.00% | 637 |
| article-en.300.skew2 | 0.00% | 2123 | 0.00% | 657 | 70.89% | 5185 | 70.89% | 2899 | 0.00% | 1502 | 0.00% | 815 | 0.00% | 614 | 0.00% | 581 |
| article-en.300.skew3 | 3.77% | 2216 | 3.77% | 728 | 59.57% | 5179 | 59.57% | 1876 | 0.18% | 1537 | 0.18% | 858 | 0.54% | 665 | 0.54% | 629 |
| headings-da.150 | 0.00% | 1620 | 0.00% | 485 | 6.59% | 3856 | 6.59% | 1268 | - | - | - | - | 0.00% | 424 | 0.00% | 427 |
| headings-da.300 | 0.00% | 2394 | 0.00% | 937 | 0.00% | 4764 | 0.00% | 3225 | - | - | - | - | 0.00% | 656 | 0.00% | 580 |
| headings-en.150 | 0.00% | 1622 | 0.00% | 486 | 0.00% | 3744 | 0.00% | 5066 | 0.00% | 894 | 0.00% | 480 | 0.00% | 295 | 0.00% | 360 |
| headings-en.300 | 0.00% | 1868 | 0.00% | 473 | 0.00% | 4537 | 0.00% | 1556 | 0.00% | 1168 | 0.00% | 624 | 0.00% | 511 | 0.00% | 462 |
| identifiers-en.150 | 0.00% | 2855 | 0.00% | 865 | 8.77% | 5247 | 8.77% | 2438 | 0.00% | 1971 | 0.00% | 953 | 0.07% | 618 | 0.07% | 600 |
| identifiers-en.300 | 0.00% | 3059 | 0.00% | 842 | 7.88% | 6035 | 7.88% | 4789 | 0.00% | 2462 | 0.00% | 1264 | 0.30% | 893 | 0.30% | 834 |
| invoice-da.150 | 0.00% | 2272 | 0.00% | 1366 | 8.57% | 4362 | 8.57% | 4593 | - | - | - | - | 0.30% | 473 | 0.30% | 475 |
| invoice-da.300 | 0.00% | 2362 | 0.00% | 817 | 8.42% | 5308 | 8.42% | 1836 | - | - | - | - | 39.55% | 637 | 39.55% | 623 |
| invoice-de.150 | 0.00% | 2258 | 0.00% | 805 | 0.00% | 4448 | 0.00% | 1629 | - | - | - | - | 0.14% | 460 | 0.14% | 439 |
| invoice-de.300 | 0.00% | 2617 | 0.00% | 1529 | 7.43% | 5432 | 7.43% | 4511 | - | - | - | - | 0.29% | 819 | 0.29% | 817 |
| invoice-en.150 | 0.00% | 2541 | 0.00% | 771 | 0.00% | 4647 | 0.00% | 1656 | 0.00% | 954 | 0.00% | 482 | 0.30% | 338 | 0.30% | 389 |
| invoice-en.300 | 0.00% | 2396 | 0.00% | 710 | 0.00% | 5317 | 0.00% | 7691 | 38.37% | 1264 | 38.37% | 755 | 38.37% | 484 | 38.37% | 539 |
| invoice-en.300.blur | 0.00% | 2460 | 0.00% | 707 | 0.00% | 5668 | 0.00% | 1834 | 38.37% | 1205 | 38.37% | 659 | 38.22% | 482 | 38.22% | 476 |
| invoice-en.300.jpeg30 | 0.15% | 2479 | 0.15% | 742 | 0.30% | 5293 | 0.30% | 1959 | 38.37% | 1311 | 38.37% | 676 | 38.22% | 509 | 38.22% | 510 |
| invoice-en.300.lowcontrast | 0.00% | 2623 | 0.00% | 694 | 0.00% | 5398 | 0.00% | 1818 | 38.37% | 1213 | 38.37% | 671 | 37.76% | 533 | 37.76% | 458 |
| invoice-en.300.noise | 0.00% | 2584 | 0.00% | 696 | 0.15% | 5423 | 0.15% | 4338 | 38.52% | 1256 | 38.52% | 693 | 37.92% | 534 | 37.92% | 472 |
| invoice-en.300.rot180 | 0.00% | 2261 | 0.00% | 644 | 77.34% | 5261 | 77.34% | 4578 | 38.37% | 1516 | 38.37% | 1024 | 38.22% | 718 | 38.22% | 718 |
| invoice-en.300.scan | 15.26% | 2326 | 15.26% | 762 | 0.00% | 5204 | 0.00% | 1866 | 0.15% | 1235 | 0.15% | 663 | 0.30% | 479 | 0.30% | 443 |
| invoice-en.300.skew1 | 0.00% | 2434 | 0.00% | 679 | 0.00% | 5308 | 0.00% | 1750 | 38.22% | 1178 | 38.22% | 742 | 38.37% | 528 | 38.37% | 464 |
| invoice-en.300.skew2 | 31.42% | 2272 | 31.42% | 657 | 18.28% | 5237 | 18.28% | 4359 | 0.00% | 1101 | 0.00% | 643 | 0.00% | 456 | 0.00% | 482 |
| invoice-en.300.skew3 | 29.00% | 2703 | 29.00% | 1305 | 17.82% | 5238 | 17.82% | 1766 | 50.30% | 1098 | 50.30% | 684 | 50.30% | 483 | 50.30% | 488 |
| invoice-fr.150 | 0.00% | 2288 | 0.00% | 941 | 0.00% | 4348 | 0.00% | 4512 | - | - | - | - | 0.69% | 398 | 0.69% | 380 |
| invoice-fr.300 | 0.00% | 2585 | 0.00% | 864 | 7.41% | 5198 | 7.41% | 1802 | - | - | - | - | 0.14% | 760 | 0.14% | 719 |
| letter-da.150 | 0.12% | 2032 | 0.12% | 1350 | 6.82% | 4280 | 6.82% | 4905 | - | - | - | - | 0.00% | 534 | 0.00% | 537 |
| letter-da.300 | 0.00% | 2121 | 0.00% | 726 | 0.12% | 5091 | 0.12% | 1830 | - | - | - | - | 0.00% | 760 | 0.00% | 803 |
| letter-de.150 | 0.00% | 2377 | 0.00% | 679 | 0.00% | 4649 | 0.00% | 1514 | - | - | - | - | 0.00% | 454 | 0.00% | 449 |
| letter-de.300 | 0.00% | 2628 | 0.00% | 1396 | 0.00% | 5447 | 0.00% | 4255 | - | - | - | - | 0.00% | 753 | 0.00% | 638 |
| letter-de.300.blur | 0.00% | 2291 | 0.00% | 677 | 0.00% | 5402 | 0.00% | 3319 | - | - | - | - | 0.00% | 628 | 0.00% | 569 |
| letter-de.300.jpeg30 | 0.00% | 2354 | 0.00% | 702 | 0.00% | 5374 | 0.00% | 1936 | - | - | - | - | 0.00% | 697 | 0.00% | 626 |
| letter-de.300.lowcontrast | 0.00% | 2463 | 0.00% | 1289 | 0.00% | 5223 | 0.00% | 4694 | - | - | - | - | 0.00% | 625 | 0.00% | 662 |
| letter-de.300.noise | 0.00% | 2212 | 0.00% | 733 | 0.00% | 5225 | 0.00% | 1756 | - | - | - | - | 0.00% | 722 | 0.00% | 631 |
| letter-de.300.rot180 | 0.00% | 2235 | 0.00% | 1432 | 69.64% | 5154 | 69.64% | 3850 | - | - | - | - | 0.00% | 1011 | 0.00% | 934 |
| letter-de.300.scan | 0.00% | 2169 | 0.00% | 670 | 0.00% | 5084 | 0.00% | 1789 | - | - | - | - | 0.54% | 583 | 0.54% | 625 |
| letter-de.300.skew1 | 0.00% | 2244 | 0.00% | 645 | 0.00% | 5098 | 0.00% | 4895 | - | - | - | - | 0.11% | 623 | 0.11% | 665 |
| letter-de.300.skew2 | 0.00% | 2486 | 0.00% | 1394 | 0.00% | 5307 | 0.00% | 3418 | - | - | - | - | 0.00% | 636 | 0.00% | 558 |
| letter-de.300.skew3 | 45.92% | 2212 | 45.92% | 754 | 0.00% | 5252 | 0.00% | 1867 | - | - | - | - | 38.95% | 467 | 38.95% | 477 |
| letter-en.150 | 0.00% | 2191 | 0.00% | 647 | 0.00% | 4342 | 0.00% | 1607 | 0.00% | 1193 | 0.00% | 579 | 0.00% | 407 | 0.00% | 369 |
| letter-en.300 | 0.00% | 2469 | 0.00% | 1448 | 0.00% | 5084 | 0.00% | 1774 | 0.00% | 1504 | 0.00% | 863 | 0.00% | 571 | 0.00% | 615 |
| letter-fr.150 | 0.00% | 2496 | 0.00% | 905 | 0.00% | 4519 | 0.00% | 1481 | - | - | - | - | 0.00% | 406 | 0.00% | 388 |
| letter-fr.300 | 0.00% | 2136 | 0.00% | 687 | 0.00% | 5236 | 0.00% | 4750 | - | - | - | - | 0.00% | 574 | 0.00% | 582 |
| mono-en.150 | 0.61% | 6563 | 0.61% | 1824 | 34.97% | 8125 | 34.97% | 6834 | 50.32% | 4355 | 50.32% | 2203 | 50.55% | 1313 | 50.55% | 1387 |
| mono-en.300 | 0.53% | 6784 | 0.53% | 1898 | 40.53% | 9231 | 40.53% | 3095 | 54.40% | 5468 | 54.40% | 2747 | 54.78% | 1862 | 54.78% | 1897 |
| smallprint-en.150 | 0.00% | 2315 | 0.00% | 753 | 0.00% | 4482 | 0.00% | 3957 | 0.10% | 1036 | 0.10% | 472 | 0.00% | 333 | 0.00% | 315 |
| smallprint-en.300 | 0.00% | 2369 | 0.00% | 686 | 0.00% | 5571 | 0.00% | 1867 | 0.00% | 1573 | 0.00% | 857 | 0.00% | 605 | 0.00% | 592 |
| smallprint-fr.150 | 0.00% | 2441 | 0.00% | 707 | 0.00% | 4633 | 0.00% | 1604 | - | - | - | - | 0.09% | 339 | 0.09% | 350 |
| smallprint-fr.300 | 0.00% | 2540 | 0.00% | 1424 | 8.40% | 5460 | 8.40% | 1866 | - | - | - | - | 0.00% | 612 | 0.00% | 661 |
| table-da.150 | 0.00% | 2416 | 0.00% | 845 | 0.21% | 4739 | 0.21% | 1802 | - | - | - | - | 0.41% | 405 | 0.41% | 452 |
| table-da.300 | 0.00% | 2653 | 0.00% | 1553 | 0.00% | 5603 | 0.00% | 3746 | - | - | - | - | 70.37% | 609 | 70.37% | 549 |
| table-en.150 | 0.00% | 2079 | 0.00% | 1471 | 0.00% | 4255 | 0.00% | 3931 | 0.28% | 577 | 0.28% | 291 | 0.28% | 216 | 0.28% | 244 |
| table-en.300 | 0.00% | 2200 | 0.00% | 626 | 0.28% | 5133 | 0.28% | 1730 | 65.34% | 440 | 65.34% | 341 | 65.34% | 306 | 65.34% | 308 |
| tesseract-eurotext | 0.24% | 1018 | 0.24% | 313 | 0.24% | 1700 | 0.24% | 1898 | - | - | - | - | 1.46% | 269 | 1.46% | 248 |
| tesseract-phototest | 0.35% | 704 | 0.35% | 228 | 0.00% | 1334 | 0.00% | 498 | 0.00% | 415 | 0.00% | 214 | 0.00% | 129 | 0.00% | 145 |
