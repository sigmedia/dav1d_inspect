# dav1d-inspect

In-memory AV1 block-metadata extraction from motion-vector inspection.

Extracts per-frame motion vectors, reference maps, and block partition sizes from
AV1 bitstreams **without writing to disk** — fully in-memory via a patched `libdav1d`
with the `inspect_cb` callback.

## Features

- Zero-disk loader: no subprocess calls, no intermediate files during runtime
- Fast-path C shim: builds arrays in C, GIL-released, multi-threaded
- Fallback ctypes: works even without the C shim compiled
- Bit-exact validation against AOM inspect reference output
- 4x4 block granularity with dual-reference MVs (like AOM inspect's MI grid)

## Repository Changes

This package is set up so it can be installed directly into another project with
`uv add`.

- Wheel builds compile a fresh patched `libdav1d` for the target platform.
- Built native libraries are generated into `dav1d_inspect/_libs/` during
  install/build and bundled into the wheel.
- Stale local native artifacts are cleaned during forced/package builds.
- `test/compare.py` validates the package output against AOM inspect for the
  two provided IVF samples: `test/0017.ivf` and `test/0018.ivf`.
- `test/compare.py --pixels` also validates generated RGB frame data against
  PyAV/FFmpeg-decoded RGB frames.
- The validator has no extra Python parser dependency; it uses the standard
  library plus `numpy`. RGB validation additionally needs PyAV.

## Source Layout

The repository is intentionally kept to source files and reproducible fixtures:

- `dav1d_inspect/` — Python package, CLI, ctypes loader, native build helper.
- `native/` — patched dav1d source and the C inspection shim used at build time.
- `patches/` — inspection patch documentation/source patch.
- `test/` — IVF samples and the AOM/PyAV comparison script.
- `hatch_build.py` and `pyproject.toml` — package build configuration.

Generated artifacts are not kept in the source tree: `.venv/`, `dist/`,
`build/`, `native/dav1d/build/`, `dav1d_inspect/_libs/`, `__pycache__/`, and
native shared libraries are ignored. Recreate them with `uv add`,
`uv pip install -e .`, or `./build.sh`.

## Installation

The patched `libdav1d` and the C shim are compiled **automatically at install
time** and bundled into the package — there is no manual build step. The pinned,
already-patched dav1d source is vendored under `native/`, so the build runs
offline.

### As a dependency in another project

```bash
uv add git+https://github.com/sigmedia/dav1d_inspect      # from git
uv add /path/to/dav1d-inspect                            # from a local checkout
```

The install triggers a source build that produces a platform-specific wheel
(e.g. `py3-none-linux_x86_64`) with the native libraries inside it. `iter_frames`
then works out of the box.

After installation, use it from the consuming project:

```python
from dav1d_inspect import iter_frames

for frame in iter_frames("video.ivf", n_threads=4):
    motion_vectors = frame["motion_vectors"]
    reference_map = frame["reference_map"]
    block_map = frame["block_map"]
```

#### Build requirements (on the machine doing the install)

- A **C compiler** — `cc`/`clang`/`gcc` (Linux/macOS) or MSVC `cl` (Windows).
- **nasm** *(optional but recommended)* — for dav1d's hand-written assembly. If
  absent, the build falls back to `-Denable_asm=false` (correct, but slower).
- **meson** and **ninja** are pulled into the isolated build environment
  automatically; you do not need them installed system-wide.

If the C shim fails to compile, the install still succeeds and the library falls
back to a pure-`ctypes` decode path (same results, lower throughput).

### Working in a source checkout

```bash
uv pip install -e .
./build.sh                 # build native libraries in-place if needed
./build.sh --force         # rebuild native libraries from scratch
```

For a local install smoke test that matches a downstream project:

```bash
tmp=$(mktemp -d)
cd "$tmp"
uv init --bare
uv add /path/to/dav1d-inspect
uv run python -c "import dav1d_inspect; print(dav1d_inspect.__version__)"
```

## Quick start

```python
from dav1d_inspect import iter_frames

for frame in iter_frames("path/to/video.ivf", n_threads=4):
    mv = frame["motion_vectors"]   # (H/4, W/4, 4) int16  [mv0_x, mv0_y, mv1_x, mv1_y]
    ref = frame["reference_map"]   # (H/4, W/4, 2) int16  [ref0, ref1]
    bs = frame["block_map"]        # (H/4, W/4)   uint8  [AOM BLOCK_* enum]
```

### Pixel data

Pass `want_pixels=True` to also recover the reconstructed YUV planes for each
frame (in addition to the motion-vector metadata):

```python
for frame in iter_frames("path/to/video.ivf", n_threads=4, want_pixels=True):
    px = frame["pixels"]           # None if the frame produced no output picture
    y = px["y"]                    # (H, W)  uint8/uint16
    u, v = px["u"], px["v"]        # chroma planes (None for monochrome)
    print(px["layout"], px["bpc"]) # e.g. "I420", 8
```

Chroma-plane dimensions follow the pixel layout (I420 → H/2×W/2, I422 → H×W/2,
I444 → H×W); 10-bit samples are returned as uint16, LSB-aligned. Pixel
extraction always uses the ctypes decode path (the fast C shim decodes without
retaining the reconstructed pictures), so it is slower than metadata-only runs.

### RGB conversion

For RGB plus motion metadata, the recommended API is `iter_rgb_frames`.  It
performs one native dav1d decode with the GIL released, converts directly to
RGB in C, and can materialize only sampled/downscaled frames:

```python
from dav1d_inspect import iter_rgb_frames

for frame in iter_rgb_frames(
    "path/to/video.ivf",
    n_threads=4,
    step=2,
    max_frames=16,
    target_size=(224, 224),       # (width, height)
    process_motion=True,
    linear_interpolation=True,
    normalize_flow=True,
    nan_to_num=True,
):
    rgb = frame["rgb"]                 # (224, 224, 3) uint8
    flow = frame["motion_field"]       # (224, 224, 2) float32 [x, y]
```

The dense field lifts the primary/backward 4x4-block MV with nearest-neighbour
sampling. MVs are converted from 1/8-pel units to pixels; optional temporal
interpolation divides by the unwrapped reference distance, and normalization
divides x/y by the coded width/height and clips to `[-1, 1]`. Native buffers are
released frame-by-frame after NumPy has copied them. If the additive RGB shim
ABI is unavailable, the function falls back to the ctypes decoder.

The older conversion interface remains available for compatibility and for
callers that specifically need reconstructed YUV planes.

Pass `want_rgb=True` (implies `want_pixels`) to also get an `(H, W, 3)` uint8
RGB image per frame, or call `yuv_to_rgb` on a `pixels` dict directly:

```python
from dav1d_inspect import iter_frames, yuv_to_rgb

for frame in iter_frames("path/to/video.ivf", want_rgb=True):
    rgb = frame["rgb"]                       # (H, W, 3) uint8, or None

# or convert on demand:
for frame in iter_frames("path/to/video.ivf", want_pixels=True):
    rgb = yuv_to_rgb(frame["pixels"])        # uses the stream's matrix + range
    rgb = yuv_to_rgb(frame["pixels"], matrix="bt601", full_range=True)  # override
```

The converter reads the stream's colour matrix (`bt601`/`bt709`/`bt2020`/
`identity`) and range from the sequence header; when the stream leaves the
matrix unspecified it falls back to BT.601 (< 720p) or BT.709 (≥ 720p). Chroma
is nearest-neighbour upsampled; output is always 8-bit RGB.

### CLI

```bash
uv run dav1d-inspect test/0018.ivf --threads 4
uv run dav1d-inspect test/0018.ivf --threads 4 --pixels   # also report YUV planes

# Save decoded frames as PNGs (frame_<decode_seq>.png) to eyeball the pixels:
uv run dav1d-inspect test/0018.ivf --save-rgb frames_out
uv run dav1d-inspect test/0018.ivf --save-rgb frames_out --limit 5   # first 5 only
```

PNGs are written with a small built-in encoder (stdlib `zlib` only, no extra
dependency).

## Dependencies

Runtime (installed automatically):

- **numpy** — for array handling
- **libdav1d** — vendored patched source (`-Denable_inspection=true`, pinned
  commit); compiled and bundled into the wheel at install time

Build-time (see [Build requirements](#build-requirements-on-the-machine-doing-the-install)):

- a **C compiler**; **nasm** (optional); **meson** + **ninja** (auto-provided)

## Validation

The repository includes two IVF samples under `test/`. Validate the generated
metadata against AOM inspect after installing/building the checkout:

```bash
uv pip install -e .
python3 test/compare.py                  # compare test/0017.ivf and test/0018.ivf
python3 test/compare.py --threads 1,4,0  # compare single, 4-thread, and auto-thread output
python3 test/compare.py test/0018.ivf    # compare one file
python3 test/compare.py --pixels         # also compare RGB frames against PyAV
```

`compare.py` generates temporary AOM JSON references with
`/opt/aom_build/examples/inspect` by default. Pass `--aom-inspect` or set
`AOM_INSPECT` if the binary lives elsewhere. It checks:

- Motion vectors are bit-exact on all inter frames
- Block sizes match exactly once remapped to AOM's `BLOCK_*` ordering
- Reference indices share AOM's encoding (0 = intra, 1..7 = ref slot, -1 = none)
- Intra frames use zero MVs and ref = [0, -1] convention
- With `--pixels`, RGB frames match PyAV-decoded RGB within the configured
  per-channel tolerance. The default is `--rgb-tolerance 17`, which covers the
  small differences introduced by independent YUV-to-RGB conversion paths.

The dav1d inspection patch exposes block-size grids for inter/switch frames;
intra/key frame block-size grids are not exported by dav1d and are reported as
skipped by the validator.

Expected successful output looks like:

```text
0018.ivf threads=1: 111 frame(s)
  motion_vectors  100.00% (...)
  reference_map   100.00% (...)
  block_map       100.00% (...)
  block_map skipped 1 intra/key frame(s)
```

Optional validation commands:

```bash
python3 -m py_compile hatch_build.py dav1d_inspect/_build.py test/compare.py
uv run --with av python test/compare.py --pixels
uv run --with av python test/compare.py --pixels --rgb-tolerance 17
python3 test/compare.py --benchmark
python3 test/compare.py --write-references /tmp/dav1d-inspect-reference
python3 test/compare.py --reference-dir /tmp/dav1d-inspect-reference
```

Install all development/test dependencies in a checkout with:

```bash
uv pip install -e ".[dev]"
```

## Licensing

MIT License. dav1d is licensed under BSD 2-Clause (see its upstream repo).
