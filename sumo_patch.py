import os
import shutil
import sys
from pathlib import Path

_DLL_DIR_HANDLES = []


def _runtime_candidates(root: Path):
    yield root / "sumo_patch"


def _is_disabled() -> bool:
    return os.environ.get("DODE_USE_PATCHED_SUMO", "1").lower() in {"0", "false", "no", "off"}


def _project_root(start_file=None) -> Path | None:
    start = Path(start_file or __file__).resolve()
    search_root = start if start.is_dir() else start.parent
    for path in (search_root, *search_root.parents):
        if any((candidate / "bin" / "sumo.exe").is_file() for candidate in _runtime_candidates(path)):
            return path
    return None


def patched_sumo_root(start_file=None) -> Path | None:
    if _is_disabled():
        return None
    explicit = os.environ.get("DODE_PATCHED_SUMO_ROOT")
    if explicit:
        root = Path(explicit).resolve()
        return root if (root / "bin" / "sumo.exe").is_file() else None
    root = _project_root(start_file)
    if root is None:
        return None
    for patched in _runtime_candidates(root):
        if (patched / "bin" / "sumo.exe").is_file():
            return patched
    return None


def _prepend_path(path: Path) -> None:
    value = str(path)
    current = os.environ.get("PATH", "")
    parts = [p for p in current.split(os.pathsep) if p]
    if value not in parts:
        os.environ["PATH"] = value + os.pathsep + current


def _prepend_sys_path(path: Path) -> None:
    value = str(path)
    sys.path[:] = [p for p in sys.path if "Eclipse\\Sumo\\tools" not in os.path.normpath(p)]
    if value not in sys.path:
        sys.path.insert(0, value)


def configure_sumo_patch(start_file=None) -> Path | None:
    patched = patched_sumo_root(start_file)
    if patched is not None:
        bin_dir = patched / "bin"
        tools_dir = patched / "tools"
        os.environ["SUMO_HOME"] = str(patched)
        _prepend_path(bin_dir)
        _prepend_sys_path(tools_dir)
        if hasattr(os, "add_dll_directory"):
            _DLL_DIR_HANDLES.append(os.add_dll_directory(str(bin_dir)))
        return patched

    dg_bin = Path(sys.prefix) / "Lib" / "site-packages" / "sumo" / "bin"
    if dg_bin.is_dir():
        _prepend_path(dg_bin)
        if hasattr(os, "add_dll_directory"):
            _DLL_DIR_HANDLES.append(os.add_dll_directory(str(dg_bin)))
    sys.path[:] = [p for p in sys.path if "Eclipse\\Sumo\\tools" not in os.path.normpath(p)]
    return None


def get_sumo_binary(start_file=None) -> str:
    env_binary = os.environ.get("SUMO_BINARY")
    if env_binary:
        return env_binary
    patched = configure_sumo_patch(start_file)
    if patched is not None:
        return str(patched / "bin" / "sumo.exe")
    dg_binary = Path(sys.prefix) / "Lib" / "site-packages" / "sumo" / "bin" / "sumo.exe"
    if dg_binary.exists():
        return str(dg_binary)
    return shutil.which("sumo") or r"C:\Program Files (x86)\Eclipse\Sumo\bin\sumo.exe"
