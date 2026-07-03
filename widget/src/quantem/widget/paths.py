"""Import-light path helpers for notebooks and live workflows."""

from __future__ import annotations

from pathlib import Path


def first_existing(*candidates: str | Path) -> Path:
    """Return the first candidate path that exists.

    Raises with every attempted path so operators can see exactly what was
    checked when a notebook was pointed at the wrong mount.
    """
    if not candidates:
        raise ValueError("first_existing() needs at least one candidate path")
    paths = [Path(candidate).expanduser() for candidate in candidates]
    for path in paths:
        if path.exists():
            return path
    tried = ", ".join(str(path) for path in paths)
    raise FileNotFoundError(f"None of these paths exist: {tried}")


__all__ = ["first_existing"]
