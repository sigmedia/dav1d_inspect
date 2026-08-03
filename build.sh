#!/bin/bash
# build.sh — compile the native libraries in-place for a source checkout.
#
# Thin wrapper over the cross-platform build module. Building a wheel
# (`uv build`) or installing the package (`uv add` / `uv pip install`) runs the
# same logic automatically via the hatchling build hook — you only need this for
# working directly in the source tree.
#
#   ./build.sh            # build if the libraries are missing
#   ./build.sh --force    # always rebuild
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
exec "${PYTHON:-python3}" -m dav1d_inspect._build "$@"
