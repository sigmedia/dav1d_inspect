"""Validate dav1d-inspect output against AOM inspect reference JSON.

By default this compares every ``test/*.ivf`` file using the AOM ``inspect``
binary as the reference producer. No reference JSON files are required in the
repository and no third-party Python parser is needed.

Examples:
    python3 test/compare.py
    python3 test/compare.py test/0018.ivf --threads 1,4,0
    python3 test/compare.py --pixels
    AOM_INSPECT=/path/to/inspect python3 test/compare.py --benchmark
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dav1d_inspect import iter_frames  # noqa: E402


DEFAULT_AOM_INSPECT = "/opt/aom_build/examples/inspect"


@dataclass
class FieldTotals:
    equal: int = 0
    total: int = 0

    @property
    def ok(self) -> bool:
        return self.equal == self.total

    def add(self, equal: int, total: int) -> None:
        self.equal += int(equal)
        self.total += int(total)

    def summary(self) -> str:
        pct = 100.0 * self.equal / self.total if self.total else 100.0
        return f"{pct:6.2f}% ({self.equal}/{self.total})"


def _resolve_aom_inspect(explicit: str | None) -> Path:
    candidates = [
        explicit,
        os.environ.get("AOM_INSPECT"),
        DEFAULT_AOM_INSPECT,
        shutil.which("inspect"),
    ]
    for candidate in candidates:
        if candidate:
            path = Path(candidate)
            if path.exists() and os.access(path, os.X_OK):
                return path
    raise FileNotFoundError(
        "AOM inspect binary not found. Pass --aom-inspect, set AOM_INSPECT, "
        f"or install it at {DEFAULT_AOM_INSPECT}."
    )


def _write_aom_json(aom_inspect: Path, ivf_path: Path, out_json: Path) -> float:
    t0 = time.perf_counter()
    with out_json.open("wb") as fh:
        proc = subprocess.run(
            [str(aom_inspect), str(ivf_path), "-mv", "-r", "-bs"],
            stdout=fh,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    if proc.returncode != 0:
        detail = proc.stderr.strip()
        raise RuntimeError(
            f"AOM inspect failed for {ivf_path} with exit code {proc.returncode}"
            + (f": {detail}" if detail else "")
        )
    return time.perf_counter() - t0


def _json_array_items(path: Path) -> Iterator[object]:
    """Yield top-level array items from a JSON file without loading it all."""
    decoder = json.JSONDecoder()
    buf = ""
    started = False

    with path.open("r", encoding="utf-8") as fh:
        while True:
            if not buf:
                chunk = fh.read(1 << 20)
                if not chunk:
                    return
                buf += chunk

            buf = buf.lstrip()
            if not started:
                if not buf:
                    continue
                if buf[0] != "[":
                    raise ValueError(f"{path} does not contain a JSON array")
                buf = buf[1:]
                started = True
                continue

            if not buf:
                continue
            if buf[0] == ",":
                buf = buf[1:]
                continue
            if buf[0] == "]":
                return

            while True:
                try:
                    item, idx = decoder.raw_decode(buf)
                    break
                except json.JSONDecodeError:
                    chunk = fh.read(1 << 20)
                    if not chunk:
                        raise
                    buf += chunk
            yield item
            buf = buf[idx:]


def _first_mismatch(mask: np.ndarray, lhs: np.ndarray, rhs: np.ndarray) -> str:
    idx = np.argwhere(~mask)
    if idx.size == 0:
        return ""
    pos = tuple(int(x) for x in idx[0])
    return f"at {pos}: reference={lhs[pos].tolist()} dav1d={rhs[pos].tolist()}"


def _compare_array(
    field: str,
    reference: np.ndarray,
    generated: np.ndarray,
    totals: FieldTotals,
) -> str | None:
    if reference.shape != generated.shape:
        totals.add(0, reference.size)
        return f"{field} shape mismatch: reference={reference.shape} dav1d={generated.shape}"

    eq = reference == generated
    if eq.ndim > 2:
        cells = eq.all(axis=-1)
    else:
        cells = eq
    totals.add(int(cells.sum()), int(cells.size))
    if cells.all():
        return None
    return f"{field} mismatch {_first_mismatch(cells, reference, generated)}"


def _load_pyav():
    try:
        import av  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "RGB pixel validation requires PyAV. Install test dependencies with "
            "`uv pip install -e '.[dev]'`, or run with "
            "`uv run --with av python test/compare.py --pixels`."
        ) from exc
    return av


def _iter_pyav_rgb(ivf_path: Path) -> Iterator[np.ndarray]:
    """Decode RGB frames with PyAV/FFmpeg in presentation order."""
    av = _load_pyav()
    with av.open(str(ivf_path)) as container:
        for frame in container.decode(video=0):
            yield frame.to_ndarray(format="rgb24")


def compare_pixels_file(ivf_path: Path, n_threads: int, tolerance: int) -> bool:
    """Compare dav1d-inspect RGB output against PyAV-decoded RGB frames."""
    totals = FieldTotals()
    first_error: str | None = None
    frames = 0

    dav_iter = iter(iter_frames(ivf_path, n_threads=n_threads, want_rgb=True))
    pyav_iter = iter(_iter_pyav_rgb(ivf_path))
    while True:
        try:
            dav = next(dav_iter)
        except StopIteration:
            try:
                next(pyav_iter)
            except StopIteration:
                break
            first_error = first_error or "dav1d produced fewer RGB frames than PyAV"
            break

        try:
            expected = next(pyav_iter)
        except StopIteration:
            first_error = first_error or "dav1d produced more RGB frames than PyAV"
            break

        actual = dav.get("rgb")
        if actual is None:
            first_error = first_error or f"dav1d frame {dav['frame_offset']} has no RGB output"
            continue

        frames += 1
        if expected.shape != actual.shape:
            totals.add(0, expected.shape[0] * expected.shape[1])
            first_error = first_error or (
                f"rgb shape mismatch at frame {dav['frame_offset']}: "
                f"pyav={expected.shape} dav1d={actual.shape}"
            )
            continue

        diff = np.abs(expected.astype(np.int16) - actual.astype(np.int16))
        cells = diff.max(axis=-1) <= tolerance
        totals.add(int(cells.sum()), int(cells.size))
        if not cells.all() and first_error is None:
            idx = np.argwhere(~cells)[0]
            y, x = int(idx[0]), int(idx[1])
            first_error = (
                f"rgb mismatch at frame {dav['frame_offset']} ({y}, {x}): "
                f"pyav={expected[y, x].tolist()} dav1d={actual[y, x].tolist()} "
                f"max_abs_diff={int(diff[y, x].max())}"
            )

    label = "auto" if n_threads == 0 else str(n_threads)
    print(f"{ivf_path.name} RGB threads={label}: {frames} frame(s)")
    print(f"  rgb_pixels      within +/-{tolerance}: {totals.summary()}")
    if first_error:
        print(f"  first failure: {first_error}")
        return False
    return totals.ok


def compare_file(ivf_path: Path, json_path: Path, n_threads: int) -> bool:
    totals = {
        "motion_vectors": FieldTotals(),
        "reference_map": FieldTotals(),
        "block_map": FieldTotals(),
    }
    first_error: str | None = None
    frames = 0
    block_frames_skipped = 0

    dav_iter = iter(iter_frames(ivf_path, n_threads=n_threads))
    for ref in _json_array_items(json_path):
        if not isinstance(ref, dict):  # AOM appends a trailing null.
            break
        try:
            dav = next(dav_iter)
        except StopIteration:
            first_error = first_error or f"dav1d stopped before reference frame {ref.get('frame')}"
            break

        frames += 1
        if int(dav["frame_offset"]) != int(ref["frame"]):
            first_error = first_error or (
                f"frame mismatch: reference={ref['frame']} dav1d={dav['frame_offset']}"
            )
            continue

        checks = [
            (
                "motion_vectors",
                np.asarray(ref["motionVectors"], dtype=np.int64),
                dav["motion_vectors"].astype(np.int64),
            ),
            (
                "reference_map",
                np.asarray(ref["referenceFrame"], dtype=np.int64),
                dav["reference_map"].astype(np.int64),
            ),
        ]
        if "blockSize" in ref and dav["frame_type"] in (1, 3):  # INTER / SWITCH
            checks.append(
                (
                    "block_map",
                    np.asarray(ref["blockSize"], dtype=np.int64),
                    dav["block_map"].astype(np.int64),
                )
            )
        elif "blockSize" in ref:
            block_frames_skipped += 1

        for field, expected, actual in checks:
            error = _compare_array(field, expected, actual, totals[field])
            if first_error is None and error is not None:
                first_error = error

    try:
        extra = next(dav_iter)
    except StopIteration:
        extra = None
    if extra is not None:
        first_error = first_error or f"dav1d produced extra frame {extra['frame_offset']}"

    label = "auto" if n_threads == 0 else str(n_threads)
    print(f"{ivf_path.name} threads={label}: {frames} frame(s)")
    for field, total in totals.items():
        print(f"  {field:<15} {total.summary()}")
    if block_frames_skipped:
        print(f"  block_map skipped {block_frames_skipped} intra/key frame(s)")
    if first_error:
        print(f"  first failure: {first_error}")
        return False
    return all(total.ok for total in totals.values())


def _time_dav1d(ivf_path: Path, n_threads: int) -> tuple[float, int]:
    t0 = time.perf_counter()
    frames = 0
    for frame in iter_frames(ivf_path, n_threads=n_threads):
        _ = frame["motion_vectors"].sum()
        _ = frame["reference_map"].sum()
        _ = frame["block_map"].sum()
        frames += 1
    return time.perf_counter() - t0, frames


def benchmark_file(ivf_path: Path, aom_seconds: float, thread_counts: Iterable[int]) -> None:
    print(f"{ivf_path.name} benchmark:")
    best = None
    for threads in thread_counts:
        seconds, frames = _time_dav1d(ivf_path, threads)
        best = seconds if best is None else min(best, seconds)
        label = "auto" if threads == 0 else str(threads)
        fps = frames / seconds if seconds else float("inf")
        print(f"  dav1d threads={label:>4}: {seconds:7.3f}s ({fps:6.1f} fps)")
    if best:
        print(f"  AOM inspect JSON generation: {aom_seconds:7.3f}s ({aom_seconds / best:4.2f}x slower)")


def _parse_threads(value: str) -> list[int]:
    try:
        threads = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not threads:
        raise argparse.ArgumentTypeError("at least one thread count is required")
    return threads


def _default_ivfs() -> list[Path]:
    return sorted((ROOT / "test").glob("*.ivf"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ivf", nargs="*", type=Path, default=_default_ivfs())
    parser.add_argument("--threads", type=_parse_threads, default=[1, 0])
    parser.add_argument("--aom-inspect", help="Path to the AOM inspect executable")
    parser.add_argument(
        "--reference-dir",
        type=Path,
        help="Use existing <stem>.json references from this directory",
    )
    parser.add_argument(
        "--write-references",
        type=Path,
        help="Write AOM JSON references to this directory and compare against them",
    )
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument(
        "--pixels",
        action="store_true",
        help="Also compare dav1d-inspect RGB output against PyAV RGB frames",
    )
    parser.add_argument(
        "--rgb-tolerance",
        type=int,
        default=17,
        help="Allowed per-channel RGB difference for --pixels (default: 17)",
    )
    args = parser.parse_args(argv)

    aom_inspect = None
    if args.reference_dir is None or args.write_references is not None:
        aom_inspect = _resolve_aom_inspect(args.aom_inspect)

    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        for ivf_path in args.ivf:
            ivf_path = ivf_path.resolve()
            if args.reference_dir is not None:
                json_path = args.reference_dir / f"{ivf_path.stem}.json"
                if not json_path.exists():
                    raise FileNotFoundError(f"missing reference JSON: {json_path}")
                aom_seconds = 0.0
            else:
                out_dir = args.write_references or tmp_dir
                out_dir.mkdir(parents=True, exist_ok=True)
                json_path = out_dir / f"{ivf_path.stem}.json"
                aom_seconds = _write_aom_json(aom_inspect, ivf_path, json_path)

            for threads in args.threads:
                ok = compare_file(ivf_path, json_path, threads) and ok
                if args.pixels:
                    ok = compare_pixels_file(ivf_path, threads, args.rgb_tolerance) and ok
            if args.benchmark:
                benchmark_file(ivf_path, aom_seconds, args.threads)

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
