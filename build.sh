#!/bin/sh
# build.sh — compile the native libraries in-place for a source checkout.
#
# Thin wrapper over the cross-platform build module. Building a wheel
# (`uv build`) or installing the package (`uv add` / `uv pip install`) runs the
# same logic automatically via the hatchling build hook — you only need this for
# working directly in the source tree.
#
#   ./build.sh            # build if the libraries are missing
#   ./build.sh --force    # always rebuild
#   ./build.sh --test     # also install test/dev deps and build AOM inspect
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

TEST=0
for arg do
  shift
  case "$arg" in
    --test) TEST=1 ;;
    *) set -- "$@" "$arg" ;;
  esac
done

if [ "$TEST" -eq 1 ]; then
  if command -v uv >/dev/null 2>&1; then
    uv pip install -e ".[dev]"
  else
    "${PYTHON:-python3}" -m pip install -e ".[dev]"
  fi

  mkdir -p test/aom_build
  if [ ! -d test/aom/.git ]; then
    git clone https://aomedia.googlesource.com/aom test/aom
  fi
  # Disable _FORTIFY_SOURCE: aom examples/inspect.c advances a write pointer
  # then calls snprintf(buf, MAX_BUFFER, ...), which modern glibc fortify
  # correctly rejects as a buffer overflow.
  cmake -S test/aom -B test/aom_build \
    -DCONFIG_TUNE_VMAF=1 \
    -DENABLE_CCACHE=1 \
    -DCONFIG_INSPECTION=1 \
    -DCMAKE_C_FLAGS="-U_FORTIFY_SOURCE -D_FORTIFY_SOURCE=0" \
    -DCMAKE_CXX_FLAGS="-U_FORTIFY_SOURCE -D_FORTIFY_SOURCE=0"
  make -C test/aom_build -j"$(nproc)"
  git clone git@github.com:yohhoy/av1parser.git test/av1parser
fi

exec "${PYTHON:-python3}" -m dav1d_inspect._build "$@"
