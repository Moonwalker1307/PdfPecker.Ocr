# PdfPecker.Ocr fuzzer

A dependency-free mutation fuzzer for the OCR package's handling of untrusted input: encoded images, raw pixel buffers
and PDFs. It is a console project in the solution and references `src/PdfPecker.Ocr` as a project and, through it, the
released PdfPecker package; it has no other package references. Its infrastructure is ported from PdfPecker's fuzzer.

There is no coverage feedback (the .NET runtime offers none without a third-party tool), so it relies on seeds and on
broad mutations. It is good at finding hangs, huge allocations, deep recursion and unexpected exception types; it will
not find wrong text or other logic bugs. Limits on damaged input belong in the code's own work budget, counted before
allocating, not in the fuzzer.

## Run it

```
dotnet build -c Release fuzz/PdfPecker.Ocr.Fuzz
dotnet fuzz/PdfPecker.Ocr.Fuzz/bin/Release/net10.0/PdfPecker.Ocr.Fuzz.dll --target ocr-smoke --duration 2m --out fuzz-findings
```

| Option | Meaning |
| --- | --- |
| `--target <name\|all>` | What to fuzz; `--list` prints the names. `all` runs them one after another, `--duration` each. |
| `--duration 90s\|10m\|1h` | How long to run per target (default 1m). |
| `--seed <n>` | Seed of the random generator (default 1). The same seed gives the same inputs; timeouts still depend on the machine. |
| `--jobs <n>` | Workers to run side by side per target (default 1); use the number of cores. Each worker gets its own seed and a share of the seed inputs, and they share the findings folder. `fuzz.yml` passes `$(nproc)`. |
| `--out <dir>` | Findings folder (default `fuzz-findings`); each target gets a sub-folder. |
| `--timeout <s>` | Wall-clock limit per input (default 10). A target can raise it with `MinTimeout` when its legitimate worst case is slower. |
| `--max-memory-mb <n>` | Heap ceiling per input (default 1536). |
| `--no-minimize` | Save inputs as found. |
| `--repo <dir>` | The repository root, when the fuzzer runs from outside it. By default it looks upwards for `PdfPecker.Ocr.slnx`. |
| `--repro <file> --target <name>` | Run one saved input once; prints the verdict and exits with 10 when it fails. |

The process exits with 1 when the findings folder has any finding, which is what fails the CI job, and with 2 for a
wrong command line.

Each target runs in a worker process, which the supervisor restarts when an input hangs beyond cancellation or crashes
the process. A restarted worker skips the seeds that were already run (`--skip-seeds`, passed by the supervisor), so a
seed that hangs is reported once instead of on every restart. Seeds never come from `bin`, `obj` or a `fuzz-findings`
folder: a saved finding read back as a seed would hang every worker on the same input.

## Targets

| Target | Status | What it does |
| --- | --- | --- |
| `ocr-smoke` | In place | A placeholder that reads `OcrInfo.Version`, so the supervisor, workers, mutations and `fuzz.yml` can be checked end to end before the real targets exist. It never finds anything; remove it once the targets below are in. |
| `ocr-image` | Planned | Mutated PNG, JPEG, WebP, TIFF, BMP and GIF files through `ImageOcr.Recognize`. Seeds: the test fixtures and the pages `SeedContext.RenderedImages` renders with PdfPecker. |
| `ocr-pixels` | Planned | Random raw pixel buffers through `ImageOcr.RecognizeRgba` and `RecognizeGray`. The first bytes of the input choose the width, height and channel count (including zero, negative, huge and mismatched sizes), the rest are the pixels, so the buffer length and the stated size disagree in every way. `ArgumentException` is the documented failure for a buffer that does not match its size. |
| `ocr-pdf` | Planned | Mutated PDFs through `PdfOcr.MakeSearchable` with `OcrPageMode.Force`, so every page is rendered and recognised. Seeds: `SeedContext.GeneratedPdfs` and the scanned fixtures. |
| `searchable-layer` | Planned | Random `OcrPage` data (text with control characters, lone surrogates and right-to-left runs, quads that are degenerate, inverted, rotated or far off the page, NaN and infinite coordinates, confidence outside 0 to 1) through the searchable PDF layer writer and `PdfStamper.Stamp`. The output must open again with PdfPecker. |

The model targets are slow per input compared with a parser, so they set `MaxSize` and `MinTimeout` to match: a seed
page at 72 dpi is a few hundred milliseconds of work, and a mutated size field can ask for far more pixels than that,
which `OcrOptions.MaxPixels` must refuse before any buffer is allocated.

## Adding a target

1. Add a file under `Targets/`, a static class with a `Target` property, like `Targets/OcrSmokeTarget.cs`. Give it a
   name, the extension of its saved inputs, a `Run` that passes the cancellation token on, its seeds and, when its API
   documents more failure modes than the common ones, `ExtraExpected`.
2. Add one line for it to `Targets.All` in `Targets.cs`.
3. Add its name to the `ALL` list in `.github/workflows/fuzz.yml` and to the table above.
4. Run it locally for a few minutes, and fix or report what it finds before the nightly run does.

Seeds come from `SeedContext`: `Files` reads files of given extensions in place from the repository (the tests'
`Fixtures` and `Golden` folders), and `GeneratedPdfs` and `RenderedImages` write and render short pages of printed text
with PdfPecker at start-up, so the image and PDF targets have seeds that hold real text.

## What counts as a finding

* **exception**: anything other than `FormatException`, `InvalidDataException`, `PdfPeckerException` (which covers
  PdfPecker's `PdfLimitException` and `PdfPasswordException`, and the OCR package's own exceptions, which derive from
  it), `OperationCanceledException` and `NotSupportedException`, plus whatever a target's `ExtraExpected` adds for its
  own API.
* **timeout**: an input that runs longer than `--timeout`. The token is cancelled first; a target that does not react
  is waited for three more seconds, then its thread is abandoned and the process restarted.
* **memory**: the live heap grows by more than `--max-memory-mb` over what stays live between inputs, or the runtime's
  hard heap limit (2.5 GB, set in the project file) raises `OutOfMemoryException`. The heap belongs to the whole
  process, so the baseline is measured again before a memory finding is confirmed; heap that keeps growing between
  inputs is a leak in the package and is printed as a `LEAK` line rather than blamed on the input that happened to cross
  the ceiling. Once the leak alone passes the ceiling, the worker is restarted.
  A timeout or memory finding is run a second time and only counts if it fails again, because a busy machine (or the
  state left by earlier inputs) can make an innocent input slow; the progress line counts those as "unconfirmed".
  Check a saved one with `--repro` before trusting it.
* **crash**: the process died, for example from a stack overflow. Each target runs in a child process, so a crash costs
  one restart. The input is written to `.current` before every run, which is how the supervisor finds it; a crash that
  does not happen again with that input alone is not saved.

Every finding is minimised (whole tokens first for text inputs, then runs of bytes) and saved as
`<target>/<kind>-<hash>.<ext>` with a `<kind>-<hash>.txt` that has the signature, the sizes, the seed and the stack
trace. Findings with the same signature (exception type plus the top frame in PdfPecker.Ocr or PdfPecker) are reported
once per run, and signatures already in the folder are skipped.

## Mutations

One to four per input: bit flips, byte changes, inserted, deleted and duplicated chunks, interesting integers (0, -1,
0x7fffffff, 0x80000000, 0xffff and so on in 1, 2 and 4 bytes, both byte orders), splicing between seeds, truncation,
and for text inputs token-level duplication, deep nesting, deletion, swaps, dictionary words and interesting numbers.
Inputs that run cleanly are sometimes kept and mutated further.

`--target selftest-crash` and `--target selftest-hang` (not listed, not part of `all`) fail on purpose, to check the
supervisor and the abandoned-thread handling.

## From finding to regression test

1. Reproduce: `dotnet ... PdfPecker.Ocr.Fuzz.dll --repro fuzz-findings/ocr-image/exception-1234abcd.png --target ocr-image`.
2. Copy the minimised input into the test. It is usually small enough to inline as a `byte[]` literal or as
   `Convert.FromBase64String("...")` (`base64 -w0 file`).
3. Add a test to `tests/PdfPecker.Ocr.Tests/HostileInputTests.cs`, bounded like the others so that a regression fails
   instead of hanging, that asserts either the documented exception or a valid result. It must fail, or time out,
   before the fix.
4. Fix the package, confirm the test passes and run the full suite. Do not commit the findings folder.
