"""Small folder-picking API shared by widget notebooks and live docs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class FolderPicker:
    """Record a selected folder path in notebook workflows."""

    path: Path | None = None

    @property
    def value(self) -> Path | None:
        """The selected folder path."""
        return self.path

    def select(self, path: str | Path) -> Path:
        """Set and return the selected folder path."""
        self.path = Path(path).expanduser()
        return self.path


def pick_folder(path: str | Path | None = None) -> Path | FolderPicker:
    """Return a folder path when provided, otherwise a picker object."""
    if path is None:
        return FolderPicker()
    return Path(path).expanduser()


__all__ = ["FolderPicker", "pick_folder"]
