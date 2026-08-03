"""Cross-platform native build for dav1d-inspect.

Builds the patched ``libdav1d`` (from the vendored source under ``native/dav1d``)
plus the ``av1of_inspect`` C shim, and drops the resulting shared libraries into
``dav1d_inspect/_libs/`` so the pure-ctypes loader in :mod:`dav1d_inspect.core`
can find them.

This module is the single source of truth for the native build. It is invoked
two ways:

  * at wheel-build time, by the hatchling build hook (``hatch_build.py``), so
    ``uv add`` / ``pip install`` produces a wheel with the libraries bundled; and
  * manually, for an editable/source checkout::

        python -m dav1d_inspect._build            # build if missing
        python -m dav1d_inspect._build --force    # always rebuild

``libdav1d`` is required; the C shim is an *optional* fast-path. If the shim
fails to compile (e.g. no usable C compiler on the target) the build still
succeeds and :func:`dav1d_inspect.core.iter_frames` transparently falls back to
the pure-ctypes decode path.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

_PKG_DIR = Path(__file__).resolve().parent          # .../dav1d_inspect
_PROJECT_ROOT = _PKG_DIR.parent                     # repo / sdist root
_NATIVE = _PROJECT_ROOT / "native"
_DAV1D_SRC = _NATIVE / "dav1d"
_SHIM_SRC = _NATIVE / "av1of_inspect_shim.c"
_LIBS_DIR = _PKG_DIR / "_libs"

_IS_WIN = sys.platform.startswith("win")
_IS_MAC = sys.platform == "darwin"


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _log(msg: str) -> None:
    print(f"[dav1d-inspect build] {msg}", flush=True)


def _run(cmd: list[str], cwd: Path | None = None, env: dict | None = None) -> None:
    _log("$ " + " ".join(str(c) for c in cmd))
    subprocess.run([str(c) for c in cmd], cwd=str(cwd) if cwd else None,
                   env=env, check=True)


def _meson_cmd() -> list[str]:
    """Prefer an importable meson (a build-time dependency) over PATH."""
    try:
        import mesonbuild  # noqa: F401
        return [sys.executable, "-m", "mesonbuild.mesonmain"]
    except Exception:
        exe = shutil.which("meson")
        if exe:
            return [exe]
    raise RuntimeError(
        "meson not found. Install it (`pip install meson`) or add it to PATH."
    )


def _ninja_exe() -> str:
    """Prefer the pip-installed ninja (a build-time dependency) over PATH."""
    try:
        import ninja
        cand = Path(ninja.BIN_DIR) / ("ninja.exe" if _IS_WIN else "ninja")
        if cand.exists():
            return str(cand)
    except Exception:
        pass
    exe = shutil.which("ninja")
    if exe:
        return exe
    raise RuntimeError(
        "ninja not found. Install it (`pip install ninja`) or add it to PATH."
    )


def _have_nasm() -> bool:
    return shutil.which("nasm") is not None or shutil.which("yasm") is not None


def _c_compiler() -> str | None:
    """Return a C compiler command for the shim, or None if none is usable."""
    if _IS_WIN:
        return shutil.which("cl")  # MSVC; requires a Developer prompt / vcvars
    return os.environ.get("CC") or shutil.which("cc") or shutil.which("clang") \
        or shutil.which("gcc")


# ---------------------------------------------------------------------------
# Build steps
# ---------------------------------------------------------------------------

def _build_dav1d(build_dir: Path) -> Path:
    """Configure + compile libdav1d. Returns build_dir/src (where the lib lands)."""
    if not _DAV1D_SRC.exists():
        raise RuntimeError(
            f"vendored dav1d source not found at {_DAV1D_SRC}. "
            "This build must run from the project tree / sdist."
        )

    setup = _meson_cmd() + [
        "setup", str(build_dir),
        "--buildtype", "release",
        "-Denable_inspection=true",
        "-Denable_tools=false",
        "-Denable_tests=false",
        "-Ddefault_library=shared",
    ]
    if not _have_nasm():
        _log("nasm/yasm not found — configuring without hand-written asm "
             "(slower decode). Install nasm for full performance.")
        setup.append("-Denable_asm=false")

    if build_dir.exists():
        # Reconfigure an existing build tree rather than failing.
        setup.append("--reconfigure")
    _run(setup, cwd=_DAV1D_SRC)
    _run([_ninja_exe(), "-C", str(build_dir)], cwd=_DAV1D_SRC)
    return build_dir / "src"


def _copy_libdav1d(lib_src_dir: Path) -> Path:
    """Copy the freshly built libdav1d into _libs/ under its runtime name.

    Returns the destination path. The name is chosen so the shim's recorded
    dependency (its soname / install-name / DLL name) resolves next to it.
    """
    _LIBS_DIR.mkdir(parents=True, exist_ok=True)

    if _IS_WIN:
        for name in ("dav1d.dll", "libdav1d.dll"):
            cand = lib_src_dir / name
            if cand.exists():
                dst = _LIBS_DIR / cand.name
                shutil.copy2(cand, dst)
                return dst
        raise RuntimeError(f"libdav1d DLL not found in {lib_src_dir}")

    if _IS_MAC:
        # Real versioned dylib, e.g. libdav1d.7.dylib (libdav1d.dylib is a symlink).
        real = Path(os.path.realpath(lib_src_dir / "libdav1d.dylib"))
        dst = _LIBS_DIR / real.name
        shutil.copy2(real, dst)
        return dst

    # Linux / other ELF: real file is libdav1d.so.<maj>.<min>.<rev>; the shim's
    # DT_NEEDED is the soname libdav1d.so.<maj>. Copy the real bytes under the
    # soname so no symlinks (fragile in wheels) are needed.
    real = Path(os.path.realpath(lib_src_dir / "libdav1d.so"))
    major = real.name.split(".so.")[1].split(".")[0]
    dst = _LIBS_DIR / f"libdav1d.so.{major}"
    shutil.copy2(real, dst)
    return dst


def _normalise_macho(shim: Path, dav1d_dst: Path) -> None:
    """Point the shim at @rpath/<dav1d> and give both @rpath ids (macOS)."""
    dav1d_id = f"@rpath/{dav1d_dst.name}"
    subprocess.run(["install_name_tool", "-id", dav1d_id, str(dav1d_dst)],
                   check=False)
    # Find the current dav1d dependency install-name recorded in the shim.
    out = subprocess.run(["otool", "-L", str(shim)], capture_output=True,
                         text=True, check=False).stdout
    old = None
    for line in out.splitlines():
        line = line.strip()
        if "libdav1d" in line:
            old = line.split(" ")[0]
            break
    if old and old != dav1d_id:
        subprocess.run(["install_name_tool", "-change", old, dav1d_id, str(shim)],
                       check=False)


def _build_shim(lib_src_dir: Path, dav1d_dst: Path) -> Path | None:
    """Compile the fast-path shim into _libs/. Returns its path, or None on failure.

    Failure is non-fatal: core.py falls back to the pure-ctypes decode path.
    """
    cc = _c_compiler()
    if cc is None:
        _log("no C compiler found — skipping the fast-path shim "
             "(ctypes fallback will be used).")
        return None

    inc = _DAV1D_SRC / "include"
    try:
        if _IS_WIN:
            out = _LIBS_DIR / "av1of_inspect.dll"
            implib = next((p for p in (lib_src_dir / "dav1d.lib",
                                       lib_src_dir / "libdav1d.lib") if p.exists()),
                          None)
            if implib is None:
                _log("dav1d import library (.lib) not found — skipping shim.")
                return None
            _run([cc, "/nologo", "/O2", "/LD", f"/I{inc}",
                  str(_SHIM_SRC), str(implib), f"/Fe:{out}"])
            return out

        if _IS_MAC:
            out = _LIBS_DIR / "libav1of_inspect.dylib"
            _run([cc, "-O3", "-shared", "-fPIC", "-I", str(inc),
                  "-o", str(out), str(_SHIM_SRC),
                  "-L", str(lib_src_dir), "-ldav1d",
                  "-Wl,-rpath,@loader_path"])
            _normalise_macho(out, dav1d_dst)
            return out

        # Linux / other ELF.
        out = _LIBS_DIR / "libav1of_inspect.so"
        _run([cc, "-O3", "-shared", "-fPIC", "-I", str(inc),
              "-o", str(out), str(_SHIM_SRC),
              "-L", str(lib_src_dir), "-ldav1d",
              "-Wl,-rpath,$ORIGIN"])
        return out
    except subprocess.CalledProcessError as e:
        _log(f"shim compile failed ({e}); continuing without the fast-path. "
             "Metadata extraction still works via ctypes.")
        return None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def _libs_present() -> bool:
    if not _LIBS_DIR.is_dir():
        return False
    globs = ("libdav1d.so*", "libdav1d*.dylib", "dav1d.dll", "libdav1d.dll")
    return any(next(iter(_LIBS_DIR.glob(g)), None) for g in globs)


def _clean_libs_dir() -> None:
    """Remove generated native artifacts before a forced rebuild."""
    if not _LIBS_DIR.is_dir():
        return
    globs = (
        "libdav1d.so*",
        "libdav1d*.dylib",
        "dav1d.dll",
        "libdav1d.dll",
        "libav1of_inspect.so",
        "libav1of_inspect.dylib",
        "av1of_inspect.dll",
        "libav1of_inspect.dll",
    )
    for glob in globs:
        for path in _LIBS_DIR.glob(glob):
            if path.is_file() or path.is_symlink():
                path.unlink()


def build(force: bool = False, build_dir: Path | None = None) -> Path:
    """Build the native libraries into ``dav1d_inspect/_libs/``.

    Returns the ``_libs`` directory. If ``force`` is False and a libdav1d is
    already present, the build is skipped (fast no-op for repeat installs).
    """
    if not force and _libs_present():
        _log(f"native libraries already present in {_LIBS_DIR} — skipping build.")
        return _LIBS_DIR

    if force:
        _clean_libs_dir()

    build_dir = build_dir or (_DAV1D_SRC / "build")
    _log(f"building libdav1d in {build_dir}")
    lib_src_dir = _build_dav1d(build_dir)
    dav1d_dst = _copy_libdav1d(lib_src_dir)
    _log(f"installed {dav1d_dst.name}")
    shim = _build_shim(lib_src_dir, dav1d_dst)
    if shim is not None:
        _log(f"installed {shim.name} (fast-path)")
    _log(f"done. platform tag: {sysconfig.get_platform()}")
    return _LIBS_DIR


def main(argv: list[str] | None = None) -> int:
    import argparse
    p = argparse.ArgumentParser(prog="python -m dav1d_inspect._build",
                                description="Build the native dav1d-inspect libraries.")
    p.add_argument("--force", action="store_true",
                   help="rebuild even if libraries already exist")
    args = p.parse_args(argv)
    build(force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
