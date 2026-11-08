## What changes

<!-- What a user of PdfPecker.Ocr sees before and after this change. Link the issue it fixes, if any. -->

## How

<!-- A short note on the approach, if it isn't obvious from the diff. -->

## Checklist

- [ ] `dotnet build -c Release` and `dotnet test -c Release --no-build` pass
- [ ] No new `PackageReference` in `src/PdfPecker.Ocr` (PdfPecker is the only one), no native code and no reflection
- [ ] Golden baselines unchanged, or regenerated on purpose and listed here with the reason
- [ ] Model data unchanged, or regenerated with `tools/ModelBuilder` and the reason given here
- [ ] Kernel or inference changes: the ONNX Runtime reference tests pass, and `bench/compare-versions.sh` results are below
- [ ] New public members have XML doc comments
- [ ] Bug fixes for malformed or hostile input have a regression test in `HostileInputTests`
- [ ] `README.md` updated if a feature, a limitation or a measured figure changed
