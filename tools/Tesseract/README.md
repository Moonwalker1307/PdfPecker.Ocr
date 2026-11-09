# Tesseract for comparison

PdfPecker.Ocr is MIT licensed. This folder builds [Tesseract](https://github.com/tesseract-ocr/tesseract) (Apache-2.0) and
[Leptonica](https://github.com/DanBloomberg/leptonica) (BSD-2-Clause) from source, and fetches Tesseract's `tessdata_fast`
language models (Apache-2.0), for one purpose: `bench/TesseractComparison` calls the library to compare recognition
accuracy, speed and memory with PdfPecker.Ocr. **Nothing built or downloaded here is part of the PdfPecker.Ocr package or ships
with it**, and no project in the repository references it at build time. Output goes to `tools/Tesseract/out/`, which is
git-ignored.

```
tools/Tesseract/build.sh            # about 10 minutes on 2 cores
tools/Tesseract/build.sh --jobs 4   # or TESSERACT_BUILD_JOBS=4
```

Needs `git`, `cmake`, `ninja`, a C and C++ compiler with OpenMP (`libgomp`), and `curl`. It builds on Linux (x64, arm64) and
macOS; the comparison app is only run on Linux so far.

> **Status:** Tesseract 5.5.4 with Leptonica 1.87.0 built on 2026-11-09 on Linux x64 (the build takes about ten minutes on
> four cores), and used for the comparison in `bench/TesseractComparison` (results in its README and in
> `bench/TesseractComparison/baseline/2026-11-09/`). Earlier builds did not copy the `tesseract` program, which CMake links
> as `bin/tesseract`; the library was unaffected. A build into the work folder of an older version can copy the old
> library to `out/<rid>/`, because that folder keeps `libtesseract.so.5.5.1` next to `libtesseract.so.5.5.4`; check
> `TessVersion()` (the comparison's engine lines print it) or delete `out/work-<rid>` before building another version.

## What the script does

1. Clones Leptonica `1.87.0` and Tesseract `5.5.4` with `git clone --depth 1 --branch <tag>`, then **checks the commit** each tag
   resolves to against the pins at the top of the script and refuses to build on a mismatch.
2. Builds Leptonica as a static, position independent library, Release `-O3`, **without any image codec** (no libpng, libjpeg,
   libtiff, zlib, giflib, webp or openjpeg): the comparison passes raw pixels to `TessBaseAPISetImage`, so Leptonica never reads
   a file. This keeps the result self-contained.
3. Builds Tesseract as a shared library, Release `-O3`, with the training tools, curl and libarchive off, and links the static
   Leptonica **into** `libtesseract` (the script fails if `ldd` shows a separate Leptonica). The result is
   `out/<rid>/libtesseract.so`, plus the `tesseract` command line program next to it.
4. Downloads `eng`, `dan`, `deu`, `fra` and `osd` from `tessdata_fast` at a pinned commit, and `eng` from `tessdata_best`, into
   `out/tessdata` and `out/tessdata-best`, and **verifies the SHA-256 of every file**.

| Pin | Value |
| --- | --- |
| Leptonica | tag `1.87.0`, commit `13275a278eb55b5746e33f95fbf5a2c8f604b3ab` |
| Tesseract | tag `5.5.4`, commit `7e548e6eff172f38f97664afe7f7ce40db36c8d6` |
| tessdata_fast | commit `87416418657359cb625c412a48b6e1d6d41c29bd` |
| tessdata_best | commit `e12c65a915945e4c28e237a9b52bc4a8f39a0cec` (`eng` only) |

The SHA-256 of each model is in the script (`FAST_FILES`, `BEST_ENG_SHA256`).

## OpenMP and threads

Tesseract's own CMake build has OpenMP on by default, and so does this one (`-DOPENMP_BUILD=ON`), because that is what a
user of the stock build gets: the LSTM engine then uses up to four OpenMP threads per recognition. The thread count is not an
API setting. **Set `OMP_THREAD_LIMIT` (and `OMP_NUM_THREADS`) in the environment of the process that loads the library**, before
it starts: the OpenMP runtime reads it once. `bench/TesseractComparison` does this by starting a fresh child process per engine
and thread count, which is also what makes each engine's peak memory its own. Tesseract's own advice for servers that run many
recognitions at once is `OMP_THREAD_LIMIT=1`.

The library is built without `-march=native`: Tesseract picks its AVX2/FMA/SSE dot-product code at run time.

What the threads bought in the comparison of 2026-11-09 (4-vCPU VM, otherwise idle, 65 pages): with `tessdata_fast`, 4 threads
made the median page 5% faster (605 to 572 ms) for 2.75 times the CPU time; with `tessdata_best`, whose LSTM is larger, 40%
faster (1382 to 828 ms) for 1.76 times the CPU time. In the first comparison, on 2026-11-08, other processes were using the
cores, and 4-thread runs stalled for seconds to minutes per page (420 s once), because the OpenMP threads wait for each other
after every step of a line; at 1 thread nothing of the kind happened, and on the idle machine no 4-thread page stalled.

## Leptonica without image codecs

Leptonica is built without libpng, which costs nothing when reading, since the app passes raw pixels. But Tesseract's LSTM
recognizer stores each line image in an `ImageData` (`src/ccstruct/imagedata.cpp`, `SetPixInternal`), PNG-compressed when
Leptonica can write PNG. Here Leptonica falls back to BMP and prints `Warning in changeFormatForMissingLib: png library
missing; output bmp format` to stderr for each line. BMP is cheaper than PNG to write and read, so this build is, if anything,
a little faster per line than a stock build with libpng.

## Verifying a build by hand

```
tools/Tesseract/out/linux-x64/tesseract --version
tools/Tesseract/out/linux-x64/tesseract page.png - --tessdata-dir tools/Tesseract/out/tessdata -l eng
```

(Run the program with `LD_LIBRARY_PATH=tools/Tesseract/out/linux-x64` if it does not find `libtesseract.so`; the build sets an
`$ORIGIN` run path, which normally makes that unnecessary.)

## Licences

Tesseract and the tessdata models are Apache-2.0 (Copyright the Tesseract authors, and Google for the models); Leptonica is
BSD-2-Clause (Copyright Daniel Bloomberg and contributors). They are developer tooling here, not dependencies, and not
redistributed, so `THIRD-PARTY-NOTICES.md` does not list them.
