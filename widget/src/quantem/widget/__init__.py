"""
quantem.widget: Interactive Jupyter widgets using anywidget + React.
"""

import importlib.metadata

try:
    __version__ = importlib.metadata.version("quantem-widget")
except importlib.metadata.PackageNotFoundError:
    __version__ = "unknown"

from quantem.widget.show3d import Show3D
from quantem.widget.show4dstem import Show4DSTEM

__all__ = ["Show3D", "Show4DSTEM"]
