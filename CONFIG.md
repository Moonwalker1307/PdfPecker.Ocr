# Configuration

This file covers how the repository is configured: build settings, environment variables and CI. How callers
configure the package itself (`OcrOptions`) is in the XML doc comments and the README.

## Build settings

| File | What it sets |
| --- | --- |
| `global.json` | .NET SDK 10.0.100, rolling forward to the latest feature band. |
| `Directory.Build.props` | Nullable and `TreatWarningsAsErrors` for every project, the local `-dev` version, and `PdfPeckerVersion`, the PdfPecker release the package builds against and depends on at least. |
| `src/PdfPecker.Ocr/PdfPecker.Ocr.csproj` | Target frameworks, trimming and AOT analyzers, unsafe code, XML docs, package metadata, the licence expression `MIT AND Apache-2.0 AND BSL-1.0`, the README, notices and licence packed with it, symbols, and `InternalsVisibleTo` for the tests, fuzzer, benchmarks, ModelBuilder and HeadTuner. On GitHub Actions it also sets `ContinuousIntegrationBuild` for reproducible PDBs. Package validation is off until the first release (see `RELEASING.md`). |
| `src/PdfPecker.Ocr/.globalconfig` | The performance (CA18xx) and culture (CA1305, CA1310) analyzer rules as warnings, and so as errors. |
| `.editorconfig` | UTF-8, four-space indents (two for YAML and JSON), file-scoped namespaces, `System` usings first. |
| `.gitattributes` | LF line endings everywhere and binary handling for images, PDFs, fonts and the model data. |

## Environment variables

| Variable | Used by | Effect |
| --- | --- | --- |
| `PDFPECKER_OCR_UPDATE_GOLDEN=1` | `GoldenTests` | Rewrites the golden baselines instead of comparing. |
| `SOURCE_DATE_EPOCH` | PdfPecker, through the searchable PDF writer | When no creation date is set, fixes the dates and file identifier, so the same input gives a byte-identical PDF. |

## CI

| Workflow | When | What |
| --- | --- | --- |
| `nuget.yml` | Every PR, pushes to main, published releases | Builds and tests net8.0, net9.0 and net10.0 on Linux and net10.0 on Windows, publishes the sample with Native AOT and runs it, reports coverage (main only) and packs PdfPecker.Ocr, checking that it depends on PdfPecker alone and carries the notices. On a release it attaches the package to the release and publishes it to nuget.org. |
| `codeql.yml` | PRs, pushes to main touching `src`, and a weekly schedule | CodeQL analysis of the package. |
| `fuzz.yml` | Nightly, and on demand with `minutes` and `target` inputs | Fuzzes every target, one job per target, one worker per core; findings fail the job and are uploaded. |
| `perf.yml` | On demand, with `runner` and `compare` inputs | Runs `bench/PdfPecker.Ocr.Benchmarks` on the branch it is started on and keeps the results for 90 days; with `compare`, it also times that branch, tag or commit in turns with this one, layer by layer and by stage. A nightly schedule and a comparison with earlier runs, as in PdfPecker, come once the benchmark has settled. |
| `vulnerability-scan.yml` | Daily, and PRs that change it or the Tesseract SBOM | OSV-Scanner checks every NuGet package (transitive ones too, from lock files the restore writes) against osv.dev and reports to Security → Code scanning, failing the run when something is affected. A second job asks osv.dev about the Tesseract and Leptonica commits listed in `tools/Tesseract/sbom.cdx.json` and fails when one is affected; `tools/Tesseract/build.sh` must build the same commits. |
| `dependabot-auto-merge.yml` | Dependabot PRs | Turns on auto-merge (squash) for patch and minor updates of actions and test packages, so GitHub merges them once the required checks pass. Major updates wait for a person. |

Each push to main gets its own run, while a newer push to a PR cancels that PR's older run. Package and coverage
artifacts are kept for 7 days (release packages for 90), as artifact storage is shared by the account's repositories.

Actions are pinned to commit SHAs, the same ones PdfPecker's workflows use, and jobs get read-only permissions unless
they need more: the release job may attach files, and the publish job may request an OIDC token.

Every job that restores packages caches `~/.nuget/packages` through `setup-dotnet`, keyed on the project files and
`Directory.Build.props`. The build of a published release is the exception: it restores from nuget.org without the
cache, so the package it publishes never depends on a cache an earlier run saved.

Dependabot (`.github/dependabot.yml`) proposes updates of the pinned actions and the test packages weekly. It does not
propose PdfPecker: `PdfPeckerVersion` is the minimum the package depends on, raised by hand when the package needs a
newer release.

Repository settings Dependabot auto-merge relies on: "Allow auto-merge" under Settings → General, and a branch
ruleset (or branch protection) on `main` that requires the status checks `build`, `windows`, `native-aot` and
`analyze` and no approving review. Without required checks, enabling auto-merge fails and the PR waits for a person.

Repository settings the release relies on: the `nuget` environment, and the `NUGET_USER` variable naming the
nuget.org account that owns the Trusted Publishing policy and the package ID. No NuGet API key is stored. See
`RELEASING.md`.
