# Building PdfPecker.Ocr

## Requirements

- The .NET 10 SDK. `global.json` pins `10.0.100` with `rollForward: latestFeature`, so any 10.0 feature band works.
- The .NET 8 and 9 runtimes to run the tests on every target framework, as CI does. With only .NET 10 installed, pass
  `-f net10.0`.
- Access to nuget.org for the first restore, which fetches the PdfPecker package and the test packages.
- Nothing else for the package, tests, fuzzer and sample: no native libraries and no Python. The dev tools have their
  own requirements: `tools/OnnxReference` needs Python 3 with `onnxruntime`, `onnx` and `numpy`, and
  `tools/Tesseract/build.sh` a C++ compiler and CMake; neither is needed to build or test.

## Commands

```
dotnet build -c Release                                          # package, tests, fuzzer, sample, benchmarks, tools
dotnet test -c Release --no-build                                # all tests on net8.0, net9.0 and net10.0
dotnet test -f net10.0 --filter "FullyQualifiedName~GoldenTests" # one class, one runtime
dotnet pack src/PdfPecker.Ocr -c Release -o ./nupkg              # the package
dotnet run --project samples/PdfPecker.Ocr.Sample -c Release -- scan.png   # prints the text of an image
dotnet publish samples/PdfPecker.Ocr.Sample -c Release -p:PublishAot=true -o aot   # Native AOT, as CI checks it
dotnet run --project tools/ModelBuilder -c Release               # regenerates src/PdfPecker.Ocr/Models (see TESTING.md)
```

## What breaks the build

- **Warnings are errors.** `Directory.Build.props` turns on `TreatWarningsAsErrors` and nullable reference types for
  every project.
- **Missing XML doc comments.** The package generates documentation, so a public member without a comment is CS1591,
  an error.
- **Analyzers.** `src/PdfPecker.Ocr/.globalconfig` raises the .NET performance rules (CA18xx: sealed classes, static
  methods, span overloads and so on) and the culture rules (CA1305, CA1310) to warnings, and so to errors.
- **Trimming and AOT analyzers.** `IsTrimmable` and `IsAotCompatible` are on, so reflection-based code shows up as a
  warning, and therefore an error.
- **File-scoped namespaces** are required by `.editorconfig` (`csharp_style_namespace_declarations = file_scoped:warning`).
- **Breaking API changes**, once 1.24.0 is released: `dotnet pack` then runs package validation against
  `PackageValidationBaselineVersion`; see `RELEASING.md`.

## Target frameworks

The package and tests target `net8.0`, `net9.0` and `net10.0`. APIs that .NET 8 lacks go behind
`#if NET9_0_OR_GREATER` with a .NET 8 path next to them. The SIMD kernels use `System.Runtime.Intrinsics`
(`Vector128`, `Vector256`, `Vector512` and the x86 and Arm intrinsics), which all three have; each kernel checks
`IsSupported` or `IsHardwareAccelerated` at run time and keeps a scalar path.

## Unsafe code

`AllowUnsafeBlocks` is on for the package, because the inner loops of the kernels use pointers and
`Unsafe`/`MemoryMarshal` to avoid bounds checks. Keep unsafe code to `Inference/` and the pixel conversion in
`Input/`, behind checks of every size it relies on.

## Line endings

`.gitattributes` checks text out with LF on every platform and marks images, PDFs and the model data as binary, so
tests that compare text or bytes behave the same on Windows.
