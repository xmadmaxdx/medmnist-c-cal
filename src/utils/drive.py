"""Colab Drive autosave. No-op locally. Never fails closed runs."""
from __future__ import annotations

import logging
import shutil
from pathlib import Path

log = logging.getLogger("medmnist-c-cal")

_MOUNT = Path("/content/drive")
_OUT = _MOUNT / "MyDrive" / "medmnist-c-cal-outputs"


def drive_mounted() -> bool:
    return _OUT.parent.is_dir()


def ensure_drive_mounted() -> bool:
    """Mount Drive on Colab if possible. Returns True if usable."""
    try:
        if drive_mounted():
            return True
        import google.colab  # noqa: F401

        from google.colab import drive

        drive.mount(str(_MOUNT), force_remount=False)
        _OUT.mkdir(parents=True, exist_ok=True)
        log.info("drive mounted at %s", _OUT)
        return True
    except Exception as e:
        log.warning("drive not available, local-only mode: %s", e)
        return False


def mirror_file(local: str | Path) -> Path | None:
    """Copy one file to Drive mirror. Returns drive path or None."""
    try:
        src = Path(local)
        if not src.is_file():
            return None
        if not drive_mounted():
            return None
        # Preserve parent folder name: outputs/checkpoints-blood/best.pt -> drive/.../checkpoints-blood/best.pt
        dest = _OUT / src.parent.name / src.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        return dest
    except Exception as e:
        log.warning("drive mirror failed for %s: %s", local, e)
        return None
