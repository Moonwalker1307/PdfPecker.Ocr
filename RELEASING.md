# Releasing PdfPecker.Ocr

Only the repository owner publishes a release. Agents and contributors prepare the pull requests around it, but never
create tags or GitHub releases.

A release publishes one package, PdfPecker.Ocr, with its symbols package.

## Versions

- PdfPecker.Ocr has PdfPecker's version, as PdfPecker.Fonts does, starting at 1.24.0: each release follows a PdfPecker
  release and carries its number, so callers can see which PdfPecker it belongs with. `nuget.yml` refuses a release
  tag that is not `PdfPeckerVersion` in `Directory.Build.props`.
- The package depends on PdfPecker at least at `PdfPeckerVersion`, the release it is built and tested against. NuGet
  treats that as a minimum, so callers may use any later PdfPecker, which keeps its public API until 2.0. After each
  PdfPecker release, a pull request raises `PdfPeckerVersion` once nuget.org serves that release. A fix that cannot
  wait for the next PdfPecker release goes out with the next PdfPecker patch release, which can be made for it.
- No breaking changes to the public API before 2.0, as in PdfPecker.
- Local builds pack as `<PdfPeckerVersion>-dev` (the `Version` in `Directory.Build.props`). CI builds of main and pull
  requests are `<next patch>-preview.<run number>`, computed from the latest `v*` tag (`<PdfPeckerVersion>-preview.<run
  number>` before the first release), so they sort between two releases.
- Package validation is off until the first release, because there is no published version to compare the API with.
  After 1.24.0 it is on, and `dotnet pack` compares the public API with `PackageValidationBaselineVersion` and fails on
  a break; a deliberate break, in a major version only, needs a `CompatibilitySuppressions.xml` entry.

## Before a release

1. Make sure main is green in `nuget.yml`, and that the last nightly `fuzz.yml` run has no findings.
2. Check that the README's performance table and language list match the release: figures from a measured run on a
   named machine, and languages from the model's character list.
3. Check that `THIRD-PARTY-NOTICES.md` matches the models in the package (sources, hashes and what `tools/ModelBuilder`
   does to them).
4. Write the release notes: what is new, the PdfPecker version it needs, and a separate list of intended output
   changes (text, positions, confidences, the PDF text layer), since callers who compare results need to know.

## Publishing

1. The owner creates a GitHub release with a tag `vX.Y.Z` (or `vX.Y.Z-suffix` for a pre-release) and the notes.
2. `nuget.yml` builds and tests with that version, checks that the package depends on PdfPecker alone and carries the
   notices, attaches the `.nupkg` to the release, and pushes it to nuget.org with Trusted Publishing: `NuGet/login`
   swaps the workflow's OIDC token for a short-lived key, in the `nuget` environment with the `NUGET_USER` repository
   variable. There is no stored API key. The first push creates the package ID on nuget.org under that account; the
   Trusted Publishing policy on nuget.org must name this repository and `nuget.yml` before that first release.

## After a release

Open one pull request that:

- sets `EnablePackageValidation` to `true` and `PackageValidationBaselineVersion` to the new version in
  `src/PdfPecker.Ocr/PdfPecker.Ocr.csproj` (after the first release, both lines are added; afterwards only the
  baseline moves), and `Version` in `Directory.Build.props` to the next patch's `-dev` (CI warns while the baseline lags
  behind the latest tag); the pull request that raises `PdfPeckerVersion` for the next release sets `Version` to its
  `-dev`;
- updates the version placeholder in `.github/ISSUE_TEMPLATE/bug_report.yml`;
- updates the README if it names the release, such as in the `bench/compare-versions.sh` example.
