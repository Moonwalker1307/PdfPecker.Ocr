# Testing PdfPecker.Ocr

```
dotnet test -c Release --no-build                                    # everything, every runtime
dotnet test -f net10.0 --filter "FullyQualifiedName~GoldenTests"     # one class, one runtime
```

The tests are xUnit in `tests/PdfPecker.Ocr.Tests`, one file per area. Inputs live in `Fixtures/`, baselines in
`Golden/`. Tests can see the package's internals. Fixtures are synthetic: pages rendered by PdfPecker from HTML, or
images made for the purpose, never scans of real documents.

## Unit tests

Each module has tests of its own: tensor shapes and storage, every kernel against a plain scalar implementation on
random inputs of awkward sizes (lengths that are not multiples of the vector width, one-pixel images, single
channels), the graph executor and its buffer plan, the detection post-processing on hand-made probability maps, CTC
decoding, word splitting, reading order, hOCR and the searchable layer. A kernel with several SIMD paths is tested on
each path the machine offers and on its scalar path, and the paths must agree.

## ONNX Runtime reference tests

The model tests run each model on fixed inputs and compare every output with ONNX Runtime's, which must agree within
1e-3. The reference outputs are generated, like PdfPecker's HarfBuzz shaping reference, by `tools/OnnxReference`, a
Python script that runs ONNX Runtime on the original ONNX files from the rapidocr wheel and writes the inputs and
outputs to `Fixtures/`. The tests do not need Python; only regenerating the references does:

```
python3 -m venv .venv && .venv/bin/pip install onnxruntime onnx numpy
.venv/bin/python -I tools/OnnxReference/generate.py
```

Regenerate them only when the inputs or the models change, and say so in the pull request. For pre- and
post-processing the reference is RapidOCR itself (`pip install rapidocr`): the goal for recognised text is to be equal
to what RapidOCR reads from the same page, so compare with it when changing how pages are prepared or outputs read.

## Golden baselines

`GoldenTests` render each `Golden/*.html` page with PdfPecker, run OCR on it and compare the recognised text with the
checked-in `*.expected.txt`. The searchable PDF goldens compare the PDFs `PdfOcr.MakeSearchable` writes with
`*.expected.pdf`, after inflating their streams, because .NET 9's zlib compresses the same bytes differently from .NET
8 and 10. Goldens should also run under the ar-SA and fa-IR cultures, whose minus and plus signs differ from ASCII,
and with `MaxDegreeOfParallelism` of 1 and of the processor count, to catch culture- and thread-dependent output.

When a change is meant to alter output, regenerate them and say in the commit and PR which baselines changed and why:

```
PDFPECKER_OCR_UPDATE_GOLDEN=1 dotnet test -f net10.0 --filter "FullyQualifiedName~GoldenTests"
```

Never regenerate just to make a failing test pass. A refactor or a speed-up must leave every golden unchanged.

## Hostile input

`HostileInputTests` run truncated and corrupted images and PDFs, pixel buffers that do not match their stated size,
zero, negative and huge dimensions, pages above `OcrOptions.MaxPixels`, PDFs with thousands of pages and a cancelled
token, each within a time bound. Malformed input may only surface as the exceptions the public API documents. A fix for
a crash, hang or huge allocation on bad input gets a regression test there, and new handling of untrusted input also
gets resource limits and a fuzz target.

## Fuzzing

`fuzz/PdfPecker.Ocr.Fuzz` mutates seeds against the package's handling of untrusted input and runs nightly in
`fuzz.yml`. How to run a target, reproduce a finding, add a target and turn a finding into a test is in
`fuzz/README.md`.

## Benchmarks

`bench/PdfPecker.Ocr.Benchmarks` is the package's own timing and allocation harness (no BenchmarkDotNet): it times
each model and the whole pipeline on fixed pages after a warm-up, and reports the time per page, allocations per page
and peak memory. To check a change of your own, run both versions in turns on one machine:

```
bench/compare-versions.sh main        # the working tree against a branch, tag or commit
```

Cloud VMs differ too much to compare a run with the README table, or with a run on another machine. `perf.yml` runs the
benchmark on demand on main and keeps the results.

## Tesseract comparison

`bench/TesseractComparison` compares accuracy, speed and peak memory with Tesseract 5.5.1 on a synthetic corpus with
known ground truth. `tools/Tesseract/build.sh` builds Tesseract and Leptonica from pinned git tags, and the comparison
app, compiled with Native AOT, calls Tesseract through its C API in the same process, so neither side pays for starting
a process. Tesseract is dev tooling only and never ships in the package. The README's figures come from this
comparison; change them only from a measured run, and name the machine.

## Coverage

CI collects line coverage on `net10.0` in a job of its own, alongside the build job, without the time-bound hostile
input tests (instrumentation slows them down), and never fails on it.
