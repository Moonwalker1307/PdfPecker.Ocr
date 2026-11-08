# Releasing PdfPecker.Ocr

Only the repository owner publishes a release. Agents and contributors prepare the pull requests around it, but never
create tags or GitHub releases.

A release publishes one package, PdfPecker.Ocr, with its symbols package.

## Versions

- PdfPecker.Ocr has its own version line, starting at 0.1.0; it does not follow PdfPecker's version numbers.
- Semantic versioning. New features are minor versions, fixes are patches, and a breaking change to the public API
  belongs only in a major version. No breaking changes are planned before 2.0. Releases before 1.0 keep that promise
  too, once 0.1.0 is out, because callers build on them.
- The package depends on PdfPecker at least at `PdfPeckerVersion` in `Directory.Build.props`, the release it is built
  and tested against. NuGet treats that as a minimum, so callers may use any later PdfPecker, which keeps its public
  API until 2.0. Raise `PdfPeckerVersion` only when the package needs something a newer PdfPecker release has, and only
  once nuget.org serves that release; the release notes say when the minimum moved.
- Local builds pack as `0.1.0-dev` (the `Version` in `Directory.Build.props`). CI builds of main and pull requests are
  `<next patch>-preview.<run number>`, computed from the latest `v*` tag (`0.1.0-preview.<run number>` before the first
  release), so they sort between two releases.
- Package validation is off until the first release, because there is no published version to compare the API with.
  After 0.1.0 it is on, and `dotnet pack` compares the public API with `PackageValidationBaselineVersion` and fails on
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
  behind the latest tag);
- updates the version placeholder in `.github/ISSUE_TEMPLATE/bug_report.yml`;
- updates the README if it names the release, such as in the `bench/compare-versions.sh` example.
