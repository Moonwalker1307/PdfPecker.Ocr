# AGENTS.md

Guidance for coding agents (and people) working in this repository. `CLAUDE.md` imports this file, so Claude Code and
other agents read the same rules.

PdfPecker.Ocr is a dependency-free OCR package for .NET in the PdfPecker family: it reads printed text from images and
scanned PDFs and makes scanned PDFs searchable with an invisible text layer. It runs PaddleOCR's PP-OCRv6 small models
(detection, recognition and a 0/180 degree line orientation classifier) on a neural network runtime of its own, in C#.
The NuGet package depends only on the released [PdfPecker](https://www.nuget.org/packages/PdfPecker) package, which it
uses for image decoding, PDF rendering and PDF writing through PdfPecker's public API; the library itself lives in
[Moonwalker1307/PdfPecker](https://github.com/Moonwalker1307/PdfPecker).

## Read first

Before non-trivial work, read:

- `README.md` – what the package does and promises; it is also the nuget.org page.
- `BUILD.md` – SDKs, target frameworks, build commands, warnings as errors, Native AOT.
- `TESTING.md` – the test suites, ONNX Runtime references, golden baselines, hostile input, fuzzing, benchmarks and
  the Tesseract comparison.
- `CONFIG.md` – the repository's build settings, environment variables and CI settings.
- `RELEASING.md` – versioning, the PdfPecker dependency and how a release reaches nuget.org.
- `THIRD-PARTY-NOTICES.md` – where the models come from and the Apache-2.0 terms they carry.

`CONTRIBUTING.md` is the short version for outside contributors, and `fuzz/README.md` covers the fuzzer. PdfPecker's
own `AGENTS.md` is the source of the shared rules below; when they disagree on something PdfPecker-specific, this file
wins here.

## Core rules

- **No dependencies besides PdfPecker.** `src/PdfPecker.Ocr` references the released PdfPecker package at
  `$(PdfPeckerVersion)` from `Directory.Build.props`, and nothing else. Never add another `PackageReference` to it,
  never reference an unreleased PdfPecker build, and never copy PdfPecker's code into this repository. When the package
  needs something PdfPecker cannot do (a decoder, a public hook, a writer feature), fix PdfPecker in its own repository
  and pull request first, and raise `PdfPeckerVersion` here after its release. Tests and tools may use test packages;
  dev-only tools may use Python (`tools/OnnxReference`) or native code (`tools/Tesseract`), but nothing they use ships.
- **No native code and no network.** The package ships no native binaries, does not P/Invoke into anything outside the
  .NET runtime, and does not use ONNX Runtime, Tesseract or a GPU. It never downloads anything: the models are
  embedded resources.
- **Models are generated data.** `src/PdfPecker.Ocr/Models/*.bin` are written by `tools/ModelBuilder` from the ONNX
  files in the rapidocr wheel on PyPI, pinned by SHA-256. Change the builder, never its output, and keep the model
  notices in `THIRD-PARTY-NOTICES.md` accurate: the weights are Apache-2.0, copyright Baidu and/or the PaddleOCR rights
  holders, and the converted files must say that and how they were changed. A new model, a new source or a change to
  what the builder does to the weights updates the notices in the same pull request.
- **Licences.** The code is MIT (`LICENSE`, unchanged); the package's licence expression is `MIT AND Apache-2.0`, and
  it carries `THIRD-PARTY-NOTICES.md` with the full Apache-2.0 text. Pre- and post-processing follows RapidOCR and
  PaddleOCR (both Apache-2.0); it is written as our own C# code from their Python code, not translated line by line
  from anything under another licence. Never put a licence that is not OSI- or FSF-approved in
  `PackageLicenseExpression`.
- **No breaking API changes before 2.0** once 0.1.0 is released. `dotnet pack` checks the public API against the last
  release from then on (`RELEASING.md`); a change it rejects needs a new overload or type instead, not a suppression.
- **Trimmable and AOT-compatible.** No reflection, `dynamic` or runtime code generation. CI publishes the sample with
  Native AOT and runs it.
- **Sealed, static and span-based.** Classes are `sealed` (or `static`) unless something derives from them, and
  methods that use no instance data are `static`. `src/PdfPecker.Ocr/.globalconfig` turns the .NET performance
  analyzers (CA18xx) and the culture rules into build errors.
- **Culture-independent output.** Parse and format numbers with `CultureInfo.InvariantCulture` and compare strings
  ordinally. hOCR, the PDF text layer and every number the package writes must not depend on the machine's culture.
- **Malformed input fails only with documented exceptions**, within bounded time and memory. Images, PDFs and raw
  pixel buffers are untrusted: check sizes before allocating, respect `OcrOptions.MaxPixels`, pass the
  `CancellationToken` into every long loop, and never let an `IndexOutOfRangeException`, `NullReferenceException`,
  `OverflowException` or `OutOfMemoryException` escape.
- **Deterministic output.** The same input and options give the same text, positions and confidences whatever
  `MaxDegreeOfParallelism` is and whichever SIMD path the processor takes: reductions use a fixed order, and the
  scalar, 128-bit, 256-bit and 512-bit kernels must agree within the tolerance the reference tests set.
- **Every PR checks `README.md`** (features, languages, limitations, examples, the performance table) and updates it
  in the same PR when the change is visible to users.
- **Never name a user's private application or employer** in code, tests, commits, the README or PR descriptions. Use
  neutral names such as "MyApp", "a production app" or "Example Customer Ltd.", and use only synthetic documents in
  fixtures: never a scan of a real person's paperwork.

## Repo map

- `src/PdfPecker.Ocr/` – the package. The public API sits at the top level (`ImageOcr`, `PdfOcr`, `OcrOptions`,
  `OcrDocument`, `OcrPage`, `OcrLine`, `OcrWord`, `OcrQuad`, `OcrBox`, `OcrInfo`). Below it:
  - `Inference/` – tensors, the graph executor and the kernels (GEMM, convolutions, depthwise convolutions,
    activations, normalisation, attention), with buffers planned when a model loads.
  - `Models/` – the generated model data (embedded resources), its loader and `ModelStore`, which loads the detection,
    recognition and orientation models lazily and holds the character list.
  - `Detection/` – the differentiable binarisation post-processing that turns the detection map into text quads.
  - `Recognition/` – line crops, the orientation step, CTC decoding and words with their positions.
  - `Layout/` – lines into reading order, and the `OcrPage` built from them.
  - `Pdf/` – the searchable text layer (a glyphless font, invisible text scaled to each word), stamped onto the
    original pages with PdfPecker.
  - `Input/` – images and PDF pages to pixels through PdfPecker.
- `tools/ModelBuilder/` – downloads the rapidocr wheel from PyPI (pinned by SHA-256), reads the ONNX files with its own
  protobuf reader, simplifies the graph and writes `src/PdfPecker.Ocr/Models/*.bin`.
- `tools/OnnxReference/` – Python, dev only: runs ONNX Runtime on fixed inputs and writes the reference outputs the
  tests compare with.
- `tests/PdfPecker.Ocr.Tests/` – xUnit: kernels, model outputs against the ONNX Runtime references, the pipeline,
  golden tests, searchable PDF goldens and `HostileInputTests`. `Fixtures/` holds inputs, `Golden/` the baselines.
- `fuzz/PdfPecker.Ocr.Fuzz/` – the dependency-free mutation fuzzer; targets register in `Targets.cs`.
- `bench/PdfPecker.Ocr.Benchmarks/` – the timing and allocation harness (no BenchmarkDotNet), and
  `bench/compare-versions.sh`, which runs two versions in turns on one machine.
- `bench/TesseractComparison/` and `tools/Tesseract/build.sh` – a native build of Tesseract and Leptonica from pinned
  git tags, called through its C API from a Native AOT comparison app, with a synthetic corpus and its ground truth.
  Tesseract never ships in the package.
- `samples/PdfPecker.Ocr.Sample` – a console app that prints the text of an image; CI publishes it with Native AOT.
- `.github/workflows/` – `nuget.yml`, `codeql.yml`, `fuzz.yml`, `perf.yml`, `dependabot-auto-merge.yml`; see
  `CONFIG.md`.

## Build and verify

```
dotnet build -c Release
dotnet test -c Release --no-build
dotnet test -f net10.0 --filter "FullyQualifiedName~GoldenTests"    # one class, one runtime
dotnet publish samples/PdfPecker.Ocr.Sample -c Release -p:PublishAot=true -o aot
```

Before pushing, the build must be warning-free (warnings are errors), the tests must pass, and the golden baselines
must be unchanged unless the output change is intended. If only one runtime is installed, pass `-f net10.0`; CI runs
`net8.0`, `net9.0` and `net10.0` on Linux and `net10.0` on Windows. A change to the public API also needs
`dotnet pack src/PdfPecker.Ocr -c Release -o ./nupkg`. Details are in `BUILD.md` and `TESTING.md`.

## Change-specific guidance

- **Output changes** (text, positions, confidences, hOCR, the PDF text layer): regenerate the goldens with
  `PDFPECKER_OCR_UPDATE_GOLDEN=1` (command in `TESTING.md`) and list in the commit and PR which baselines changed and
  why. A refactor or a pure speed-up must leave every golden unchanged.
- **Model regeneration**: run `tools/ModelBuilder`, commit its output with the builder change, regenerate the ONNX
  Runtime references if the inputs changed, and update `THIRD-PARTY-NOTICES.md` when the source, the hashes or what is
  done to the weights changes. Never hand-edit `Models/*.bin`.
- **Kernel and inference changes**: the reference tests (model outputs within 1e-3 of ONNX Runtime) must pass on every
  SIMD path the kernel has, the scalar one included; add a reference test for a new kernel or fusion. Compare speed and
  allocations with `bench/compare-versions.sh <base-ref>` on one machine and put the numbers in the PR.
- **Accuracy or speed claims against Tesseract**: run `bench/TesseractComparison` on the same machine and corpus, and
  ask the owner before changing figures in the README; they come from measured runs.
- **Public API**: code is `internal` unless it is meant for callers (tests can see internals). Every public member
  needs an XML doc comment (CS1591 is an error). Add overloads rather than changing signatures, and add a README line
  for a new feature. Prefer the shapes of PdfPecker's API (static entry points, options classes, byte array, stream and
  file overloads, a `CancellationToken` last).
- **Something PdfPecker cannot do**: don't work around it here with fragile code; fix PdfPecker in its own PR first,
  and use it here after its release.
- **Newer .NET APIs**: the package also targets .NET 8, which lacks some APIs (`Vector512` is there; some span and
  dictionary APIs are not). Guard them with `#if NET9_0_OR_GREATER` and keep a .NET 8 path.
- **Untrusted input** (images, PDFs, pixel buffers, sizes): add resource limits, a fuzz target in
  `fuzz/PdfPecker.Ocr.Fuzz` (see `fuzz/README.md`), and for each bug fix a regression test in `HostileInputTests`.
- **Generated data**: change the generator, never its output.
  - `tools/ModelBuilder` → `src/PdfPecker.Ocr/Models/*.bin`
  - `tools/OnnxReference` → the reference outputs in `tests/PdfPecker.Ocr.Tests/Fixtures`
- **CI**: actions are pinned to the same commit SHAs as PdfPecker's workflows; update both together.
- **Releases**: follow `RELEASING.md`; only the repository owner publishes a release.

## Avoid

- Adding a package to `src/PdfPecker.Ocr`, a native binary, ONNX Runtime, Tesseract or any download at run time.
- Copying PdfPecker's code instead of calling its public API.
- Regenerating goldens or references just to make a failing test pass.
- Skipping, disabling or loosening a test, a tolerance or a time bound to get CI green.
- Reflection, `dynamic`, runtime code generation, or anything else that breaks trimming or Native AOT.
- `ToString()`/`Parse` without an invariant culture, and culture-sensitive string comparison.
- Per-operation allocations in the inference loop; buffers come from the plan made when the model loads.
- Results that depend on thread count or timing, such as parallel reductions in an unfixed order.
- Hand-editing generated model data, or dropping the model notices.
- Committing real scans, fuzz findings, benchmark outputs or the downloaded wheel.
- Publishing, tagging or creating a GitHub release; that is the owner's step.

## Working mindset

- Read the code around a change first and match it: naming, comment density, file layout, error handling.
- Prefer the smallest change that fixes the cause, and keep a pull request to one change.
- Correctness first: match ONNX Runtime's outputs and RapidOCR's text before making anything faster.
- Measure before claiming a speed-up, on one machine, against the base version.
- When RapidOCR and this package disagree, RapidOCR with ONNX Runtime is the reference for what the models produce.
- Write down what you checked: the tests run, the baselines changed, the benchmark compared.

## Commit guidance

- Branch from the latest `origin/main`, one branch per piece of work.
- Name the branch after the work it carries, in lowercase words joined by hyphens, with a `claude/` prefix when an
  agent creates it (e.g. `claude/depthwise-conv-avx2`). Never use a generic or generated name such as
  `claude/project-thread-xxxx`, even when a tool suggests one. Don't rename a branch that already has an open pull
  request.
- Never reuse a branch name. Once a pull request is merged or closed, its branch is finished: follow-up work goes on a
  new branch with a new name, even when it continues the same feature.
- Commit subjects are short imperative or descriptive sentences without prefixes, e.g. "Fuse hard swish into the
  depthwise convolution".
- The body explains why, and names any golden baselines or model files that changed and the reason.
- Fill in `.github/pull_request_template.md`; CI must be green before review.

## Wisdom

- Code comments and docs are plain, full sentences explaining why, not what.
- The recognition model's character list (18,708 entries, read from the ONNX metadata key `character`) has Latin with
  accented letters, Greek, Chinese and Japanese, but no Cyrillic, Korean, Arabic or Indic letters; check it before
  promising a language.
- The fuzzer has no coverage feedback. It finds hangs, huge allocations, deep recursion and wrong exception types, not
  wrong text; damaged-input limits belong in the code's own work budget, counted before allocating.
- Cloud VMs and GitHub runners differ too much in CPU to compare a single benchmark run with an earlier one; only
  version-against-version runs on one machine are meaningful.
