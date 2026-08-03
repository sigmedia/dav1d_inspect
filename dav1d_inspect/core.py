"""In-memory AV1 block-metadata extraction from motion-vector inspection."""

from __future__ import annotations

import ctypes as C
import errno
import os
import struct
import sys
from pathlib import Path
from typing import Generator

import numpy as np

# ---------------------------------------------------------------------------
# Data-layout constants
# ---------------------------------------------------------------------------

_BD_DAV1D_TO_AOM = np.array(
    [15, 14, 13, 12, 11, 21, 10, 9, 8, 19, 20, 7, 6, 5, 17, 18, 4, 3, 2, 16, 1, 0],
    dtype=np.uint8,
)

_BLOCK_DTYPE = np.dtype(
    [
        ("mv0_y", "<i2"),
        ("mv0_x", "<i2"),
        ("mv1_y", "<i2"),
        ("mv1_x", "<i2"),
        ("ref0", "i1"),
        ("ref1", "i1"),
        ("bs", "u1"),
        ("mf", "u1"),
    ]
)
assert _BLOCK_DTYPE.itemsize == 12

_EAGAIN = -errno.EAGAIN  # DAV1D_ERR(EAGAIN)

# ---------------------------------------------------------------------------
# ctypes mirrors of the dav1d ABI
# ---------------------------------------------------------------------------

class Dav1dPicAllocator(C.Structure):
    _fields_ = [
        ("cookie", C.c_void_p),
        ("alloc_picture_callback", C.c_void_p),
        ("release_picture_callback", C.c_void_p),
    ]


class Dav1dLogger(C.Structure):
    _fields_ = [
        ("cookie", C.c_void_p),
        ("callback", C.c_void_p),
    ]


class Dav1dInspectData(C.Structure):
    _fields_ = [
        ("decode_seq", C.c_uint),
        ("frame_offset", C.c_uint),
        ("frame_type", C.c_int),
        ("width", C.c_int),
        ("height", C.c_int),
        ("blk_w", C.c_int),
        ("blk_h", C.c_int),
        ("blk_stride", C.c_ssize_t),
        ("blocks", C.c_void_p),
        ("refidx", C.c_int8 * 7),
        ("refpoc", C.c_uint * 7),
    ]


_INSPECT_CB = C.CFUNCTYPE(None, C.c_void_p, C.POINTER(Dav1dInspectData))


class Dav1dSettings(C.Structure):
    _fields_ = [
        ("n_threads", C.c_int),
        ("max_frame_delay", C.c_int),
        ("apply_grain", C.c_int),
        ("operating_point", C.c_int),
        ("all_layers", C.c_int),
        ("frame_size_limit", C.c_uint),
        ("allocator", Dav1dPicAllocator),
        ("logger", Dav1dLogger),
        ("strict_std_compliance", C.c_int),
        ("output_invisible_frames", C.c_int),
        ("inloop_filters", C.c_int),
        ("decode_frame_type", C.c_int),
        ("inspect_cookie", C.c_void_p),
        ("inspect_cb", _INSPECT_CB),
        ("reserved", C.c_uint8 * 16),
    ]


class Dav1dData(C.Structure):
    """Full ABI layout so `sz` (bytes remaining) reads correctly."""
    _fields_ = [
        ("data", C.c_void_p),
        ("sz", C.c_size_t),
        ("ref", C.c_void_p),
        ("m_timestamp", C.c_int64),
        ("m_duration", C.c_int64),
        ("m_offset", C.c_int64),
        ("m_size", C.c_size_t),
        ("m_user_data", C.c_void_p),
        ("m_user_ref", C.c_void_p),
    ]


# Dav1dPicture is allocated as an opaque blob comfortably larger than the real
# struct (so dav1d_get_picture can safely fill it), then reinterpreted through
# _Dav1dPictureHead below when we need to read the planes.
class Dav1dPicture(C.Structure):
    _fields_ = [("_opaque", C.c_uint8 * 1024)]


class _Dav1dDataProps(C.Structure):
    """dav1d/common.h Dav1dDataProps — carried from input Dav1dData to the
    output picture, so we tag each input packet's timestamp and read it back."""
    _fields_ = [
        ("timestamp", C.c_int64),
        ("duration", C.c_int64),
        ("offset", C.c_int64),
        ("size", C.c_size_t),
        ("user_data_data", C.c_void_p),  # struct Dav1dUserData { const uint8_t*;
        ("user_data_ref", C.c_void_p),   #                        struct Dav1dRef*; }
    ]


class _Dav1dPictureHead(C.Structure):
    """The leading, ABI-stable fields of Dav1dPicture (dav1d/picture.h), up to
    and including `m`. Everything we read (planes, strides, params, timestamp)
    lives here; the rest of the real struct is left as opaque padding."""
    _fields_ = [
        ("seq_hdr", C.c_void_p),
        ("frame_hdr", C.c_void_p),
        ("data", C.c_void_p * 3),        # Y, U, V plane pointers
        ("stride", C.c_ssize_t * 2),     # bytes/row for luma[0], chroma[1]
        ("p_w", C.c_int),                # Dav1dPictureParameters.w
        ("p_h", C.c_int),                # Dav1dPictureParameters.h
        ("p_layout", C.c_int),           # enum Dav1dPixelLayout
        ("p_bpc", C.c_int),              # bits per component (8 or 10)
        ("m", _Dav1dDataProps),
    ]


class _Dav1dSeqHdrHead(C.Structure):
    """Leading, ABI-stable fields of Dav1dSequenceHeader (dav1d/headers.h), up
    to `color_range` — enough to pick the right YUV->RGB matrix and range. The
    large `operating_points[]` array follows and is left off."""
    _fields_ = [
        ("profile", C.c_uint8),
        ("max_width", C.c_int),
        ("max_height", C.c_int),
        ("layout", C.c_int),         # enum Dav1dPixelLayout
        ("pri", C.c_int),            # enum Dav1dColorPrimaries
        ("trc", C.c_int),            # enum Dav1dTransferCharacteristics
        ("mtrx", C.c_int),           # enum Dav1dMatrixCoefficients
        ("chr", C.c_int),            # enum Dav1dChromaSamplePosition
        ("hbd", C.c_uint8),
        ("color_range", C.c_uint8),  # 1 = full/JPEG range, 0 = limited/MPEG
    ]


# enum Dav1dPixelLayout (dav1d/headers.h): monochrome, 4:2:0, 4:2:2, 4:4:4.
_LAYOUT_NAMES = ("I400", "I420", "I422", "I444")

# enum Dav1dMatrixCoefficients (AV1 / ISO-23091) -> our conversion family.
# Only the values that map to a well-defined YCbCr matrix are listed; anything
# else (incl. 2 = unspecified) falls back to a resolution heuristic.
_MTRX_TO_FAMILY = {
    0: "identity",   # IDENTITY: planes are GBR, not YCbCr
    1: "bt709",      # BT.709
    5: "bt601",      # BT.470BG (BT.601 625)
    6: "bt601",      # BT.601 (SMPTE 170M, 525)
    9: "bt2020",     # BT.2020 non-constant-luminance
    10: "bt2020",    # BT.2020 constant-luminance (approx. with NCL matrix)
}

# Luma coefficients (Kr, Kb) per matrix family; Kg = 1 - Kr - Kb.
_MATRIX_KR_KB = {
    "bt601": (0.299, 0.114),
    "bt709": (0.2126, 0.0722),
    "bt2020": (0.2627, 0.0593),
}


def _matrix_family(mtrx: int, height: int) -> str:
    """Map AV1 matrix coefficients to a conversion family, guessing by
    resolution when the stream leaves it unspecified (swscale convention)."""
    fam = _MTRX_TO_FAMILY.get(mtrx)
    if fam is not None:
        return fam
    return "bt709" if height >= 720 else "bt601"


def _pixels_from_head(head: _Dav1dPictureHead) -> dict:
    """Copy a decoded picture's YUV planes out of dav1d's strided buffers.

    Returns {"y", "u", "v", "layout", "bpc"}. Planes are contiguous numpy
    arrays cropped to the visible dimensions; `u`/`v` are None for monochrome
    (I400). dtype is uint8 for 8-bit, uint16 for 10-bit (LSB-aligned samples).
    """
    w, h, bpc, layout = head.p_w, head.p_h, head.p_bpc, head.p_layout
    dtype = np.uint8 if bpc <= 8 else np.uint16
    itemsize = np.dtype(dtype).itemsize

    def _plane(ptr, stride_bytes, pw, ph):
        if not ptr or pw == 0 or ph == 0:
            return None
        buf = (C.c_char * (stride_bytes * ph)).from_address(ptr)
        rows = np.frombuffer(buf, dtype=dtype).reshape(ph, stride_bytes // itemsize)
        # Must copy: `rows` views dav1d's picture buffer, which is freed on
        # unref. A plain slice/ascontiguousarray can alias it when the crop is
        # already contiguous (stride == width), so copy unconditionally.
        return np.array(rows[:, :pw], dtype=dtype)

    y = _plane(head.data[0], head.stride[0], w, h)
    if layout == 1:      # I420
        cw, ch = (w + 1) // 2, (h + 1) // 2
    elif layout == 2:    # I422
        cw, ch = (w + 1) // 2, h
    elif layout == 3:    # I444
        cw, ch = w, h
    else:                # I400 (monochrome): no chroma
        cw, ch = 0, 0
    u = _plane(head.data[1], head.stride[1], cw, ch)
    v = _plane(head.data[2], head.stride[1], cw, ch)

    # Colour matrix / range from the sequence header (drives YUV->RGB). Defaults
    # cover streams with no seq_hdr pointer or unspecified coefficients.
    mtrx, full_range = 2, False  # 2 = MC_UNSPECIFIED
    if head.seq_hdr:
        seq = _Dav1dSeqHdrHead.from_address(head.seq_hdr)
        mtrx = int(seq.mtrx)
        full_range = bool(seq.color_range)

    return {
        "y": y,
        "u": u,
        "v": v,
        "layout": _LAYOUT_NAMES[layout] if 0 <= layout < 4 else str(layout),
        "bpc": int(bpc),
        "matrix": mtrx,
        "matrix_name": _matrix_family(mtrx, h),
        "full_range": full_range,
    }


def _upsample_chroma(c: np.ndarray, h: int, w: int) -> np.ndarray:
    """Nearest-neighbour upsample a chroma plane to luma resolution (h, w)."""
    ch, cw = c.shape
    if (ch, cw) == (h, w):
        return c
    c = np.repeat(c, -(-h // ch), axis=0)  # ceil-div replication, then crop
    c = np.repeat(c, -(-w // cw), axis=1)
    return c[:h, :w]


def yuv_to_rgb(
    pixels: dict,
    matrix: str | None = None,
    full_range: bool | None = None,
) -> np.ndarray | None:
    """Convert a `pixels` dict (from ``iter_frames(..., want_pixels=True)``) to
    an (H, W, 3) uint8 RGB image.

    Chroma is nearest-neighbour upsampled to luma resolution and the standard
    non-constant-luminance YCbCr matrix is applied. By default the stream's own
    matrix (``pixels["matrix_name"]``) and range (``pixels["full_range"]``) are
    used; pass ``matrix`` ("bt601"/"bt709"/"bt2020"/"identity") or ``full_range``
    to override. Output is always 8-bit RGB regardless of input bit depth.

    Returns None if the picture has no luma plane.
    """
    y = pixels["y"]
    if y is None:
        return None

    bpc = pixels["bpc"]
    maxv = float((1 << bpc) - 1)
    h, w = y.shape
    if matrix is None:
        matrix = pixels.get("matrix_name") or "bt709"
    if full_range is None:
        full_range = bool(pixels.get("full_range", False))

    yf = y.astype(np.float32)

    # Luma normalisation to [0, 1] (identical for all matrices).
    if full_range:
        def _norm_luma(p):
            return p / maxv
    else:
        black = float(16 << (bpc - 8))
        luma_span = float(219 << (bpc - 8))
        def _norm_luma(p):
            return (p - black) / luma_span

    if matrix == "identity":
        # IDENTITY: planes carry G, B, R directly (requires 4:4:4).
        u, v = pixels["u"], pixels["v"]
        if u is None or v is None:
            g = np.clip(_norm_luma(yf) * 255.0 + 0.5, 0, 255).astype(np.uint8)
            return np.stack([g, g, g], axis=-1)
        g = _norm_luma(yf)
        b = _norm_luma(u.astype(np.float32))
        r = _norm_luma(v.astype(np.float32))
        rgb = np.stack([r, g, b], axis=-1) * 255.0
        return np.clip(rgb + 0.5, 0, 255).astype(np.uint8)

    # Monochrome (no chroma): grey RGB.
    u, v = pixels["u"], pixels["v"]
    if u is None or v is None:
        g = np.clip(_norm_luma(yf) * 255.0 + 0.5, 0, 255).astype(np.uint8)
        return np.stack([g, g, g], axis=-1)

    uf = _upsample_chroma(u, h, w).astype(np.float32)
    vf = _upsample_chroma(v, h, w).astype(np.float32)

    mid = float(1 << (bpc - 1))
    if full_range:
        chroma_span = maxv
    else:
        chroma_span = float(224 << (bpc - 8))

    yn = _norm_luma(yf)
    un = (uf - mid) / chroma_span  # Cb, centred
    vn = (vf - mid) / chroma_span  # Cr, centred

    try:
        kr, kb = _MATRIX_KR_KB[matrix]
    except KeyError:
        raise ValueError(
            f"unknown matrix {matrix!r}; expected one of "
            f"{sorted(_MATRIX_KR_KB) + ['identity']}"
        )
    kg = 1.0 - kr - kb

    r = yn + vn * (2.0 - 2.0 * kr)
    b = yn + un * (2.0 - 2.0 * kb)
    g = yn - un * (2.0 * kb * (1.0 - kb) / kg) - vn * (2.0 * kr * (1.0 - kr) / kg)

    rgb = np.stack([r, g, b], axis=-1) * 255.0
    return np.clip(rgb + 0.5, 0, 255).astype(np.uint8)


class _Av1ofFrame(C.Structure):
    _fields_ = [
        ("decode_seq", C.c_uint),
        ("frame_offset", C.c_uint),
        ("frame_type", C.c_int),
        ("width", C.c_int),
        ("height", C.c_int),
        ("blk_w", C.c_int),
        ("blk_h", C.c_int),
        ("refidx", C.c_int8 * 7),
        ("refpoc", C.c_uint * 7),
        ("motion_vectors", C.POINTER(C.c_int16)),
        ("reference_map", C.POINTER(C.c_int16)),
        ("block_map", C.POINTER(C.c_uint8)),
    ]


# ---------------------------------------------------------------------------
# Library location helpers
# ---------------------------------------------------------------------------

_THIS_DIR = Path(__file__).resolve().parent

# Glob patterns for the built libraries, in the order we prefer to load them.
# Covers unversioned and versioned soname forms across Linux/macOS/Windows.
_DAV1D_PATTERNS = ("libdav1d.so", "libdav1d.so.*", "libdav1d.*.dylib",
                   "libdav1d.dylib", "dav1d.dll", "libdav1d.dll")
_SHIM_PATTERNS = ("libav1of_inspect.so", "libav1of_inspect.dylib",
                  "av1of_inspect.dll", "libav1of_inspect.dll")


def _find_lib(directory: Path, patterns: tuple[str, ...]) -> Path | None:
    """Return the first library matching any pattern in ``directory``."""
    if not directory.is_dir():
        return None
    for pat in patterns:
        matches = sorted(directory.glob(pat))
        if matches:
            return matches[0]
    return None


def _locate_libs() -> tuple[Path, Path | None]:
    """Find libdav1d and the (optional) C shim.

    Returns (libdav1d_path, shim_path_or_None). The libraries are expected in
    ``dav1d_inspect/_libs/`` — populated at wheel-build time by the hatchling
    build hook, or manually via ``python -m dav1d_inspect._build`` for a source
    checkout. Raises FileNotFoundError if libdav1d is not present.

    On Windows the ``_libs`` directory is registered with the DLL loader so the
    shim can resolve its libdav1d dependency alongside it (there is no rpath).
    """
    pkg_libs = _THIS_DIR / "_libs"
    libdav1d = _find_lib(pkg_libs, _DAV1D_PATTERNS)
    if libdav1d is None:
        raise FileNotFoundError(
            f"libdav1d not found in {pkg_libs}. Build the native libraries with "
            "`python -m dav1d_inspect._build` (from a source checkout), or "
            "reinstall the package so its build hook can compile them."
        )
    if sys.platform.startswith("win") and hasattr(os, "add_dll_directory"):
        try:
            os.add_dll_directory(str(pkg_libs))
        except OSError:
            pass
    shim = _find_lib(pkg_libs, _SHIM_PATTERNS)  # may legitimately be None
    return libdav1d, shim


# ---------------------------------------------------------------------------
# libdav1d loader
# ---------------------------------------------------------------------------


def _load_lib(lib_path: str | Path | None) -> C.CDLL:
    path = Path(lib_path) if lib_path else _locate_libs()[0]
    lib = C.CDLL(str(path))
    lib.dav1d_default_settings.argtypes = [C.POINTER(Dav1dSettings)]
    lib.dav1d_open.argtypes = [C.POINTER(C.c_void_p), C.POINTER(Dav1dSettings)]
    lib.dav1d_open.restype = C.c_int
    lib.dav1d_data_create.argtypes = [C.POINTER(Dav1dData), C.c_size_t]
    lib.dav1d_data_create.restype = C.POINTER(C.c_uint8)
    lib.dav1d_send_data.argtypes = [C.c_void_p, C.POINTER(Dav1dData)]
    lib.dav1d_send_data.restype = C.c_int
    lib.dav1d_get_picture.argtypes = [C.c_void_p, C.POINTER(Dav1dPicture)]
    lib.dav1d_get_picture.restype = C.c_int
    lib.dav1d_picture_unref.argtypes = [C.POINTER(Dav1dPicture)]
    lib.dav1d_close.argtypes = [C.POINTER(C.c_void_p)]
    return lib


# ---------------------------------------------------------------------------
# C shim loader
# ---------------------------------------------------------------------------


def _load_shim_for_path(shim_path: Path) -> C.CDLL | None:
    """Load the C shim for the fast-path."""
    # RTLD_DEEPBIND is not guaranteed in all Python builds (e.g. conda).
    _DEEPBIND = getattr(C, 'RTLD_DEEPBIND', None)
    mode = C.DEFAULT_MODE
    if _DEEPBIND is not None:
        mode = _DEEPBIND
    try:
        lib = C.CDLL(str(shim_path), mode=mode)
    except OSError:
        return None
    lib.av1of_decode.argtypes = [C.c_char_p, C.c_int, C.POINTER(C.c_void_p)]
    lib.av1of_decode.restype = C.c_int
    lib.av1of_num_frames.argtypes = [C.c_void_p]
    lib.av1of_num_frames.restype = C.c_int
    lib.av1of_get_frame.argtypes = [C.c_void_p, C.c_int]
    lib.av1of_get_frame.restype = C.POINTER(_Av1ofFrame)
    lib.av1of_free.argtypes = [C.c_void_p]
    return lib


# ---------------------------------------------------------------------------
# IVF demuxer
# ---------------------------------------------------------------------------


def _iter_ivf_packets(path: str | Path):
    with open(path, "rb") as f:
        header = f.read(32)
        if len(header) < 32 or header[:4] != b"DKIF":
            raise ValueError(f"Not a valid IVF file: {path}")
        while True:
            pkt_hdr = f.read(12)
            if len(pkt_hdr) < 12:
                break
            size = struct.unpack("<I", pkt_hdr[:4])[0]
            data = f.read(size)
            if len(data) < size:
                break
            yield data


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _decode_inspect_to_numpy(insp: Dav1dInspectData) -> dict:
    """Copy one callback payload into NumPy arrays."""
    frame = {
        "decode_seq": int(insp.decode_seq),
        "frame_offset": int(insp.frame_offset),
        "frame_type": int(insp.frame_type),
        "width": int(insp.width),
        "height": int(insp.height),
        "refidx": [int(x) for x in insp.refidx],
        "refpoc": [int(x) for x in insp.refpoc],
    }
    blk_w, blk_h, stride = insp.blk_w, insp.blk_h, insp.blk_stride
    if not insp.blocks or blk_w == 0 or blk_h == 0:
        frame["motion_vectors"] = np.zeros((blk_h, blk_w, 4), dtype=np.int16)
        ref = np.empty((blk_h, blk_w, 2), dtype=np.int16)
        ref[..., 0] = 0
        ref[..., 1] = -1
        frame["reference_map"] = ref
        frame["block_map"] = np.zeros((blk_h, blk_w), dtype=np.uint8)
        return frame

    n_bytes = blk_h * stride * _BLOCK_DTYPE.itemsize
    raw = (C.c_char * n_bytes).from_address(insp.blocks)
    grid = np.frombuffer(raw, dtype=_BLOCK_DTYPE).reshape(blk_h, stride)[:, :blk_w]

    mv = np.stack(
        [grid["mv0_x"], grid["mv0_y"], grid["mv1_x"], grid["mv1_y"]], axis=-1
    ).astype(np.int16)
    mv[mv == -32768] = 0
    ref = np.stack([grid["ref0"], grid["ref1"]], axis=-1).astype(np.int16)
    frame["motion_vectors"] = np.ascontiguousarray(mv)
    frame["reference_map"] = np.ascontiguousarray(ref)
    # Clamp out-of-range block-size indices to 0 (matches the C shim's
    # `bs < 22 ? bs : 0`); a stray value would otherwise index past the LUT.
    bs_idx = np.where(grid["bs"] < _BD_DAV1D_TO_AOM.size, grid["bs"], 0)
    frame["block_map"] = _BD_DAV1D_TO_AOM[bs_idx]
    return frame


def iter_frames(
    ivf_path: str | Path,
    n_threads: int = 0,
    want_pixels: bool = False,
    want_rgb: bool = False,
) -> Generator[dict, None, None]:
    """Decode an IVF/AV1 file and yield per-frame block metadata in memory.

    Yields one dict per decoded frame, in decode order, with keys:
        decode_seq, frame_offset, frame_type, width, height, refidx, refpoc,
        motion_vectors  (H/4, W/4, 4) int16  [mv0_x, mv0_y, mv1_x, mv1_y],
        reference_map   (H/4, W/4, 2) int16  [ref0, ref1],
        block_map       (H/4, W/4)   uint8  [AOM BLOCK_* enum].

    If ``want_pixels`` is True, each dict also carries:
        pixels  dict {"y", "u", "v", "layout", "bpc", "matrix", "matrix_name",
                "full_range"} — the decoded YUV planes as numpy arrays (u/v None
                for monochrome), or None if no output picture was produced for
                that frame.

    If ``want_rgb`` is True (implies ``want_pixels``), each dict also carries:
        rgb     (H, W, 3) uint8 — the frame converted to RGB via `yuv_to_rgb`
                using the stream's own colour matrix/range, or None.

    Motion-vector extraction uses the C shim (libav1of_inspect) as a fast-path
    when available, falling back to ctypes if the shim is not built. Pixel
    extraction (``want_pixels``/``want_rgb``) always uses the ctypes path, since
    the shim decodes without retaining the reconstructed pictures.
    """
    want_pixels = want_pixels or want_rgb
    libdav1d, shim_path = _locate_libs()
    if not want_pixels and shim_path is not None:
        shim = _load_shim_for_path(shim_path)
        if shim is not None:
            yield from _iter_frames_via_shim(shim, ivf_path, n_threads)
            return

    lib = _load_lib(libdav1d)
    for frame in _iter_frames_via_callback(ivf_path, n_threads, lib, want_pixels):
        if want_rgb:
            px = frame.get("pixels")
            frame["rgb"] = yuv_to_rgb(px) if px is not None else None
        yield frame


def _iter_frames_via_shim(shim: C.CDLL, ivf_path: str | Path, n_threads: int):
    handle = C.c_void_p()
    rc = shim.av1of_decode(str(ivf_path).encode(), int(n_threads), C.byref(handle))
    if rc != 0:
        raise RuntimeError(f"av1of_decode failed ({rc}) for {ivf_path}")
    try:
        n = shim.av1of_num_frames(handle)
        for i in range(n):
            f = shim.av1of_get_frame(handle, i).contents
            mv = np.ctypeslib.as_array(f.motion_vectors, shape=(f.blk_h, f.blk_w, 4)).copy()
            ref = np.ctypeslib.as_array(f.reference_map, shape=(f.blk_h, f.blk_w, 2)).copy()
            bs = np.ctypeslib.as_array(f.block_map, shape=(f.blk_h, f.blk_w)).copy()
            yield {
                "decode_seq": int(f.decode_seq),
                "frame_offset": int(f.frame_offset),
                "frame_type": int(f.frame_type),
                "width": int(f.width),
                "height": int(f.height),
                "refidx": [int(x) for x in f.refidx],
                "refpoc": [int(x) for x in f.refpoc],
                "motion_vectors": mv,
                "reference_map": ref,
                "block_map": bs,
            }
    finally:
        shim.av1of_free(handle)


def _iter_frames_via_callback(
    ivf_path: str | Path,
    n_threads: int,
    lib: C.CDLL,
    want_pixels: bool = False,
):
    collected: list[dict] = []

    @_INSPECT_CB
    def _cb(_cookie, data_ptr):
        collected.append(_decode_inspect_to_numpy(data_ptr.contents))

    settings = Dav1dSettings()
    lib.dav1d_default_settings(C.byref(settings))
    settings.n_threads = int(n_threads)
    settings.inspect_cb = _cb

    ctx = C.c_void_p()
    if lib.dav1d_open(C.byref(ctx), C.byref(settings)) < 0:
        raise RuntimeError("dav1d_open failed")

    pic = Dav1dPicture()

    # When capturing pixels, each input packet is tagged with its decode-order
    # index (via Dav1dData.m_timestamp); dav1d carries that onto the output
    # picture (pic.m.timestamp), letting us re-associate reordered pictures with
    # their decode_seq. Assumes one coded frame per IVF packet (true for AV1 in
    # IVF); frames with no output picture simply get pixels=None.
    pixels_by_seq: dict[int, dict] = {}

    def _drain():
        while True:
            res = lib.dav1d_get_picture(ctx, C.byref(pic))
            if res == _EAGAIN:
                break
            if res < 0:
                raise RuntimeError(f"dav1d_get_picture failed: {res}")
            if want_pixels:
                head = _Dav1dPictureHead.from_address(C.addressof(pic))
                pixels_by_seq[int(head.m.timestamp)] = _pixels_from_head(head)
            lib.dav1d_picture_unref(C.byref(pic))

    try:
        for seq, payload in enumerate(_iter_ivf_packets(ivf_path)):
            data = Dav1dData()
            buf = lib.dav1d_data_create(C.byref(data), len(payload))
            if not buf:
                raise RuntimeError("dav1d_data_create failed")
            C.memmove(buf, payload, len(payload))
            data.m_timestamp = seq
            while data.sz > 0:
                res = lib.dav1d_send_data(ctx, C.byref(data))
                if res < 0 and res != _EAGAIN:
                    raise RuntimeError(f"dav1d_send_data failed: {res}")
                _drain()
        _drain()
    finally:
        lib.dav1d_close(C.byref(ctx))

    collected.sort(key=lambda fr: fr["decode_seq"])
    if want_pixels:
        for fr in collected:
            fr["pixels"] = pixels_by_seq.get(fr["decode_seq"])
    yield from collected


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "test/0018.ivf"
    threads = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    n = 0
    for fr in iter_frames(src, n_threads=threads):
        if n < 8:
            mv = fr["motion_vectors"]
            ref = fr["reference_map"]
            bs = fr["block_map"]
            absmax = int(np.abs(mv).max()) if mv.size else 0
            rng = (int(ref.min()), int(ref.max())) if ref.size else (0, 0)
            bsrng = (int(bs.min()), int(bs.max())) if bs.size else (0, 0)
            print(
                f"seq={fr['decode_seq']:3d} frame_offset={fr['frame_offset']:3d} "
                f"type={fr['frame_type']} {fr['width']}x{fr['height']}  "
                f"mv={mv.shape} ref={ref.shape} bs={bs.shape} mv_absmax={absmax} "
                f"ref_range={rng} bs_range={bsrng}"
            )
        n += 1
    print(f"Total frames decoded: {n} (threads={threads})")
