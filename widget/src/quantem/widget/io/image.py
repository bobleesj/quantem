"""Read a single 2D image (Velox EMD HAADF, or .npy) into a Dataset2d.

``io.load`` is the iterative-4D-STEM loader (Arina/Dectris HDF5). A plain 2D
survey image - a HAADF saved by Velox as ``.emd``, or a ``.npy`` - needs a
different, tiny reader. This is it, returning a :class:`Dataset2d` that carries
the pixel size + raw metadata, so ``Show2D(io.read_image(path))`` draws a real
scale bar (in nm) with no extra arguments.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from quantem.widget.datastructures import Dataset2d


def read_image(path: str | Path) -> Dataset2d:
    """Return a :class:`Dataset2d` from a Velox ``.emd`` HAADF or a ``.npy`` file.

    Velox stores the image under ``Data/Image/<hash>/Data`` shaped (H, W, N) and
    a JSON metadata blob alongside; we take the first frame and pull the pixel
    size into ``sampling`` (nm) + keep the full metadata dict. ``.npy`` loads the
    array with no calibration (sampling defaults to pixels).
    """
    p = Path(path)
    if p.suffix == ".npy":
        return Dataset2d(np.load(p), name=p.stem)
    if p.suffix == ".emd":
        import h5py  # noqa: PLC0415  (lazy: keep importing quantem.widget.io cheap)
        with h5py.File(p, "r") as f:
            group = next(iter(f["Data/Image"]))  # one image signal per HAADF emd
            arr = f[f"Data/Image/{group}/Data"][...]
            meta = _read_velox_metadata(f, group)
        image = arr[:, :, 0] if arr.ndim == 3 else arr
        sampling, units = _velox_sampling(meta)
        return Dataset2d(image, sampling=sampling, units=units, name=p.stem, metadata=meta)
    raise ValueError(f"read_image: unsupported extension {p.suffix!r} (use .emd or .npy)")


def _read_velox_metadata(f, group) -> dict:
    """Decode the per-image Velox JSON metadata blob (null-padded uint8)."""
    raw = bytes(f[f"Data/Image/{group}/Metadata"][:, 0].tobytes()).split(b"\x00", 1)[0]
    if not raw:
        return {}
    try:
        return json.loads(raw.decode("utf-8", "ignore"))
    except json.JSONDecodeError:
        return {}


def _velox_sampling(meta: dict):
    """Pixel size (row, col) in nm + units from Velox ``BinaryResult.PixelSize`` (meters)."""
    px = meta.get("BinaryResult", {}).get("PixelSize")
    if not px:
        return None, None
    height_nm = float(px["height"]) * 1e9
    width_nm = float(px["width"]) * 1e9
    return (height_nm, width_nm), ["nm", "nm"]
