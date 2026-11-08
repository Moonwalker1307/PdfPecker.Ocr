#!/usr/bin/env bash
# Benchmarks PdfPecker.Ocr from the working tree against another version (a tag, branch or commit; the latest release tag, or
# origin/main when there is none) on this machine, running the two in turns. Usage: bench/compare-versions.sh [git-ref] [rounds]
#   OCR_BENCH_BUILD_ARGS="-p:OcrApi=true"   MSBuild arguments for both builds: the switches that turn the cases on (see the csproj).
#                                           Both builds get the same ones, so a case the other version cannot compile shows as n/a
#                                           only if its code is behind a switch that version lacks; pass only what both support.
#   DOTNET_GCgen0size                        fixes the GC allocation budget (default 0x1000000, 16 MB) so memory compares across machines.
set -euo pipefail
repo=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
ref=${1:-$(git -C "$repo" describe --tags --abbrev=0 2>/dev/null || echo origin/main)}
rounds=${2:-3}
build_args=${OCR_BENCH_BUILD_ARGS:-}
other=$(mktemp -d)
trap 'git -C "$repo" worktree remove --force "$other" >/dev/null 2>&1 || rm -rf "$other"' EXIT

git -C "$repo" worktree add --detach "$other" "$ref" >/dev/null
# The other version is measured with this version's benchmark program, so both run exactly the same code around the
# recognition; only the library differs. The program shares the corpus reader with the Tesseract comparison, so that file goes too.
rm -rf "$other/bench/PdfPecker.Ocr.Benchmarks"
mkdir -p "$other/bench/TesseractComparison"
cp -r "$repo/bench/PdfPecker.Ocr.Benchmarks" "$other/bench/PdfPecker.Ocr.Benchmarks"
cp "$repo/bench/TesseractComparison/CorpusData.cs" "$other/bench/TesseractComparison/CorpusData.cs"
rm -rf "$other/bench/PdfPecker.Ocr.Benchmarks/bin" "$other/bench/PdfPecker.Ocr.Benchmarks/obj"

# shellcheck disable=SC2086
dotnet build -c Release "$other/bench/PdfPecker.Ocr.Benchmarks" -v q -nologo $build_args >&2
# shellcheck disable=SC2086
dotnet build -c Release "$repo/bench/PdfPecker.Ocr.Benchmarks" -v q -nologo $build_args >&2
# The ref is escaped before it goes into the sed replacements, because a branch such as origin/main or claude/foo
# carries a slash that would otherwise end the s command.
label=$(printf '%s' "$ref" | sed 's/[\\/|&]/\\&/g')
dotnet "$repo/bench/PdfPecker.Ocr.Benchmarks/bin/Release/net10.0/PdfPecker.Ocr.Benchmarks.dll" versions \
  "$other/bench/PdfPecker.Ocr.Benchmarks/bin/Release/net10.0/PdfPecker.Ocr.Benchmarks.dll" "$rounds" |
  sed "s|other = .*|other = $label|; s/Other build/$label/; s/This build/working tree/"
