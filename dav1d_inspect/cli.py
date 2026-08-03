"""CLI entry point for dav1d-inspect."""

from __future__ import annotations

import argparse
import struct
import sys
import zlib
from pathlib import Path

import numpy as np

_root = Path(__file__).resolve().parents[1]
if not Path(_root / "dav1d_inspect").is_dir():
    sys.path.insert(0, str(_root))

from dav1d_inspect.core import iter_frames


def _write_png(path: Path, rgb: np.ndarray) -> None:
    """Write an (H, W, 3) uint8 array to a PNG (stdlib-only, no dependencies)."""
    h, w, _ = rgb.shape
    # Each scanline is prefixed with a filter-type byte (0 = None).
    raw = np.zeros((h, 1 + w * 3), dtype=np.uint8)
    raw[:, 1:] = np.ascontiguousarray(rgb, dtype=np.uint8).reshape(h, w * 3)
    compressed = zlib.compress(raw.tobytes(), 6)

    def _chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)  # 8-bit, colour type 2 (RGB)
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(_chunk(b"IHDR", ihdr))
        f.write(_chunk(b"IDAT", compressed))
        f.write(_chunk(b"IEND", b""))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="dav1d-inspect",
        description="Extract block metadata from an IVF/AV1 file in memory.",
    )
    parser.add_argument(
        "ivf_path",
        help="Path to the .ivf input file",
    )
    parser.add_argument(
        "-j", "--threads",
        type=int,
        default=0,
        help="Number of threads (0 = auto)",
    )
    parser.add_argument(
        "-p", "--pixels",
        action="store_true",
        help="Also decode and report the reconstructed YUV pixel planes",
    )
    parser.add_argument(
        "--save-rgb",
        metavar="DIR",
        help="Convert each frame to RGB and save it as a PNG in DIR "
             "(frame_<decode_seq>.png), for visual verification",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="With --save-rgb, only save the first N frames (0 = all)",
    )
    args = parser.parse_args(argv)

    out_dir = None
    if args.save_rgb:
        out_dir = Path(args.save_rgb)
        out_dir.mkdir(parents=True, exist_ok=True)

    want_rgb = out_dir is not None
    saved = 0
    n = 0
    for fr in iter_frames(
        args.ivf_path,
        n_threads=args.threads,
        want_pixels=args.pixels,
        want_rgb=want_rgb,
    ):
        mv = fr["motion_vectors"]
        ref = fr["reference_map"]
        bs = fr["block_map"]
        absmax = int(np.abs(mv).max()) if mv.size else 0
        pix = ""
        if args.pixels:
            p = fr.get("pixels")
            pix = (f"  pixels={p['layout']}/{p['bpc']}bpc y={p['y'].shape}"
                   if p is not None else "  pixels=None")

        saved_note = ""
        if out_dir is not None and (args.limit <= 0 or saved < args.limit):
            rgb = fr.get("rgb")
            if rgb is not None:
                png = out_dir / f"frame_{fr['decode_seq']:04d}.png"
                _write_png(png, rgb)
                saved += 1
                saved_note = f"  saved={png.name}"

        print(
            f"  frame={fr['frame_offset']:>4d}  size={fr['width']}x{fr['height']}  "
            f"type={fr['frame_type']}  mv={mv.shape} mv_max={absmax}  "
            f"ref={ref.shape} ref_range=[{ref.min()},{ref.max()}]  "
            f"bs={bs.shape} bs_range=[{bs.min() if bs.size else 0}, ...]  "
            f"order_int=({[0] + list(fr['refpoc'])}){pix}{saved_note}"
        )
        n += 1

    print(f"\nTotal frames decoded: {n}", file=sys.stderr)
    if out_dir is not None:
        print(f"Saved {saved} RGB PNG(s) to {out_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
