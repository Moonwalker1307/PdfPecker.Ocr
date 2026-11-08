#!/usr/bin/env bash
# Builds Leptonica and Tesseract from pinned source into one self-contained libtesseract shared library, with the
# traineddata files the comparison needs. This is dev tooling for bench/TesseractComparison: Tesseract (Apache-2.0) is
# only called to compare accuracy, speed and memory with PdfPecker.Ocr, and nothing built here is ever shipped.
#
# Usage: tools/Tesseract/build.sh [--jobs N] [--clean]
#   TESSERACT_BUILD_JOBS  parallel compile jobs (default 2: the machine is usually shared)
#   TESSERACT_SRC_CACHE   where the git clones live (default tools/Tesseract/src, git-ignored)
#
# Output: tools/Tesseract/out/<rid>/libtesseract.so (or .dylib) and tools/Tesseract/out/tessdata/*.traineddata.
# The Leptonica static library is linked into libtesseract, and Leptonica is built without image file formats (no
# libpng, libjpeg, libtiff, zlib, giflib, webp or openjpeg), because the comparison hands Tesseract raw pixels through
# TessBaseAPISetImage and never asks it to read a file.
#
# OpenMP is left on, as Tesseract's own CMake build does by default, so that a user running the stock build sees the same
# behaviour: the LSTM engine then uses up to four OpenMP threads per recognition. Set OMP_THREAD_LIMIT=1 (or
# OMP_NUM_THREADS=1) in the environment of the process that loads the library to get single-threaded numbers; it must be
# set before the process starts, because the OpenMP runtime reads it once. The comparison app does this by starting a
# child process per engine and thread count.
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)

# Pins. Tags move rarely, but a commit does not: the script checks the commit each tag resolves to.
LEPTONICA_TAG=1.85.0
LEPTONICA_COMMIT=63aef18d98432b8582a1565e241f7bd2ee9cc8d9
TESSERACT_TAG=5.5.1
TESSERACT_COMMIT=0995615022dc368f65c6b88f31784d8bc384f855

# tessdata_fast and tessdata_best commits, and the SHA-256 of every file taken from them.
TESSDATA_FAST_COMMIT=87416418657359cb625c412a48b6e1d6d41c29bd
TESSDATA_BEST_COMMIT=e12c65a915945e4c28e237a9b52bc4a8f39a0cec
FAST_FILES=(
  "eng 7d4322bd2a7749724879683fc3912cb542f19906c83bcc1a52132556427170b2"
  "dan acb1fd074487a31d1294fcdfd7d7c673467ffd8aeacb2ccd61ebcbf04eb4e2fa"
  "deu 19d219bbb6672c869d20a9636c6816a81eb9a71796cb93ebe0cb1530e2cdb22d"
  "fra ced037562e8c80c13122dece28dd477d399af80911a28791a66a63ac1e3445ca"
  "osd 9cf5d576fcc47564f11265841e5ca839001e7e6f38ff7f7aacf46d15a96b00ff"
)
BEST_ENG_SHA256=8280aed0782fe27257a68ea10fe7ef324ca0f8d85bd2fd145d1c2b560bcb66ba

jobs=${TESSERACT_BUILD_JOBS:-2}
clean=0
while [ $# -gt 0 ]; do
  case $1 in
    --jobs) jobs=$2; shift 2 ;;
    --clean) clean=1; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

case "$(uname -s)-$(uname -m)" in
  Linux-x86_64) rid=linux-x64; ext=so ;;
  Linux-aarch64) rid=linux-arm64; ext=so ;;
  Darwin-arm64) rid=osx-arm64; ext=dylib ;;
  Darwin-x86_64) rid=osx-x64; ext=dylib ;;
  *) echo "unsupported platform $(uname -s)-$(uname -m)" >&2; exit 2 ;;
esac

src=${TESSERACT_SRC_CACHE:-$here/src}
out=$here/out
prefix=$out/leptonica-$rid
work=$out/work-$rid
if [ "$clean" = 1 ]; then rm -rf "$out" "$src"; fi
mkdir -p "$src" "$out/$rid" "$out/tessdata" "$work"

clone() { # name url tag commit
  local dir=$src/$1
  if [ ! -d "$dir/.git" ]; then
    git clone --quiet --depth 1 --branch "$3" "$2" "$dir"
  fi
  local actual
  actual=$(git -C "$dir" rev-parse HEAD)
  if [ "$actual" != "$4" ]; then
    echo "$1 $3 is at $actual, expected $4: refusing to build" >&2
    exit 1
  fi
}

fetch() { # url sha256 destination
  if [ -f "$3" ] && [ "$(sha256sum "$3" | cut -d' ' -f1)" = "$2" ]; then return; fi
  curl --silent --show-error --fail --location --output "$3.part" "$1"
  local actual
  actual=$(sha256sum "$3.part" | cut -d' ' -f1)
  if [ "$actual" != "$2" ]; then
    rm -f "$3.part"
    echo "$1 has SHA-256 $actual, expected $2" >&2
    exit 1
  fi
  mv "$3.part" "$3"
}

clone leptonica https://github.com/DanBloomberg/leptonica.git "$LEPTONICA_TAG" "$LEPTONICA_COMMIT"
clone tesseract https://github.com/tesseract-ocr/tesseract.git "$TESSERACT_TAG" "$TESSERACT_COMMIT"

# Leptonica: static, position independent so that it can go into a shared library, and no image codecs.
cmake -S "$src/leptonica" -B "$work/leptonica" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_FLAGS_RELEASE="-O3 -DNDEBUG" \
  -DCMAKE_INSTALL_PREFIX="$prefix" -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  -DBUILD_SHARED_LIBS=OFF -DSW_BUILD=OFF -DBUILD_PROG=OFF \
  -DCMAKE_DISABLE_FIND_PACKAGE_PNG=ON -DCMAKE_DISABLE_FIND_PACKAGE_JPEG=ON \
  -DCMAKE_DISABLE_FIND_PACKAGE_TIFF=ON -DCMAKE_DISABLE_FIND_PACKAGE_ZLIB=ON \
  -DCMAKE_DISABLE_FIND_PACKAGE_GIF=ON -DCMAKE_DISABLE_FIND_PACKAGE_WebP=ON \
  -DCMAKE_DISABLE_FIND_PACKAGE_OpenJPEG=ON
cmake --build "$work/leptonica" --parallel "$jobs"
cmake --install "$work/leptonica" >/dev/null

# Tesseract: the library only, no training tools, no network (curl) and no archive (libarchive) support. OpenMP stays on.
cmake -S "$src/tesseract" -B "$work/tesseract" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS_RELEASE="-O3 -DNDEBUG" \
  -DCMAKE_PREFIX_PATH="$prefix" -DBUILD_SHARED_LIBS=ON -DBUILD_TRAINING_TOOLS=OFF -DBUILD_TESSERACT_BINARY=ON \
  -DOPENMP_BUILD=ON -DDISABLE_CURL=ON -DDISABLE_ARCHIVE=ON -DGRAPHICS_DISABLED=ON -DDISABLE_TIFF=ON \
  -DSW_BUILD=OFF -DCMAKE_DISABLE_FIND_PACKAGE_TIFF=ON \
  -DCMAKE_INSTALL_PREFIX="$work/tesseract-install" -DCMAKE_INSTALL_RPATH='$ORIGIN'
cmake --build "$work/tesseract" --parallel "$jobs"

lib=$(find "$work/tesseract" -maxdepth 1 \( -name "libtesseract*.$ext" -o -name "libtesseract*.$ext.*" \) | head -n 1)
[ -n "$lib" ] || { echo "libtesseract was not built" >&2; exit 1; }
cp -L "$lib" "$out/$rid/libtesseract.$ext"
if [ -f "$work/tesseract/tesseract" ]; then cp "$work/tesseract/tesseract" "$out/$rid/tesseract"; fi

# Leptonica must be inside the library, not a separate dependency.
if command -v ldd >/dev/null && ldd "$out/$rid/libtesseract.$ext" | grep -qi lept; then
  echo "libtesseract still depends on a shared Leptonica" >&2
  exit 1
fi

for entry in "${FAST_FILES[@]}"; do
  set -- $entry
  fetch "https://raw.githubusercontent.com/tesseract-ocr/tessdata_fast/$TESSDATA_FAST_COMMIT/$1.traineddata" "$2" "$out/tessdata/$1.traineddata"
done
mkdir -p "$out/tessdata-best"
fetch "https://raw.githubusercontent.com/tesseract-ocr/tessdata_best/$TESSDATA_BEST_COMMIT/eng.traineddata" "$BEST_ENG_SHA256" "$out/tessdata-best/eng.traineddata"

echo "built $out/$rid/libtesseract.$ext (Tesseract $TESSERACT_TAG, Leptonica $LEPTONICA_TAG)"
echo "fast models in $out/tessdata, best eng in $out/tessdata-best"
