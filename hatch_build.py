"""Hatchling build hook: compile the native libraries into the wheel.

Runs at wheel-build time (``uv add`` / ``pip install`` / ``uv build``). It drives
the cross-platform native build in ``dav1d_inspect/_build.py`` — producing the
patched ``libdav1d`` (+ optional fast-path shim) under ``dav1d_inspect/_libs/`` —
then force-includes those artifacts and tags the wheel as platform-specific.

The libraries are loaded via ctypes (no Python C-API), so a single wheel works
for every CPython 3.x on the build platform: the tag is ``py3-none-<platform>``
rather than a CPython-ABI tag.
"""

from __future__ import annotations

import importlib.util
import sysconfig
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


def _load_build_module(root: Path):
    """Import dav1d_inspect/_build.py standalone (without triggering the package
    __init__, which imports numpy — not available in the build env)."""
    path = root / "dav1d_inspect" / "_build.py"
    spec = importlib.util.spec_from_file_location("_dav1d_inspect_build", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _platform_tag() -> str:
    """wheel platform tag, e.g. linux_x86_64 / macosx_11_0_arm64 / win_amd64."""
    return sysconfig.get_platform().replace("-", "_").replace(".", "_")


class NativeBuildHook(BuildHookInterface):
    PLUGIN_NAME = "custom"

    def initialize(self, version: str, build_data: dict) -> None:
        root = Path(self.root)
        build_mod = _load_build_module(root)

        # Wheel builds must not reuse any checked-out or previously built native
        # libraries: a local checkout may contain artifacts for a different
        # platform. Build fresh libraries for the current target and bundle only
        # those artifacts.
        libs_dir = build_mod.build(
            force=True,
            build_dir=root / "build" / "dav1d-inspect-native",
        )

        # Ship every produced artifact at dav1d_inspect/_libs/<name> in the wheel.
        build_data.setdefault("force_include", {})
        build_data.setdefault("artifacts", [])
        for artifact in sorted(Path(libs_dir).iterdir()):
            if artifact.is_file():
                rel = f"dav1d_inspect/_libs/{artifact.name}"
                build_data["force_include"][str(artifact)] = rel
                build_data["artifacts"].append(rel)

        # Native, but ABI-agnostic (ctypes) -> platform-specific, any-Python tag.
        build_data["pure_python"] = False
        build_data["tag"] = f"py3-none-{_platform_tag()}"
