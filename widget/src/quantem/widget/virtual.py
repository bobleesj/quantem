"""Virtual images (DP / BF / ABF / ADF / HAADF / DF) with automatic probe fitting.

One call - ``virtual(data, "BF")`` - finds the probe (BF disk center + radius) from
the mean diffraction pattern, builds the detector mask for the requested mode, and
reduces over every scan position. MacBook (MPS) runs the raw-Metal masked-sum over
chunked uint16 buffers; CUDA / CPU runs torch. **No binning** on either path. The
result is a 2D array ready for ``Show2D``.

Modes (annular bands in units of the auto-detected BF radius ``r``):
  DP    - the mean diffraction pattern itself (detector space)
  BF    - bright field, disk ``<= r``
  ABF   - annular bright field, ``0.5r .. r``
  ADF   - annular dark field, ``r .. 2r``
  HAADF - high-angle ADF, ``2r .. 4r``
  DF    - dark field, ``> r``
  annular(inner=, outer=) - custom band in BF-radius units

Usage::

    from quantem.widget import load, virtual, Show2D
    Show2D(virtual(load("master.h5"), "ADF"))   # probe found automatically
"""
from __future__ import annotations

import numpy as np


def _resolve_backend(data):
    """Return a compute backend (MetalCompute on MPS chunks, TorchCompute on array)."""
    if hasattr(data, "_fields") and "data" in getattr(data, "_fields", ()):
        data = data.data
    # raw MPS chunks -> wrap so compute_backend sees a _is_gpu_frames source
    if hasattr(data, "chunks") and not getattr(data, "_is_gpu_frames", False):
        from quantem.widget.kernels.compute.mps import ChunkedFrames
        data = ChunkedFrames(data)
    from quantem.widget.kernels.compute.backends import compute_backend
    return compute_backend(data)


def auto_probe(mean_dp):
    """Detect the probe (BF disk) from the mean diffraction pattern.

    Threshold at ``mean + std``, take the centroid of the bright disk for the
    center, and ``radius = sqrt(area / pi)``. Matches Show4DSTEM.auto_detect_center.
    Returns ``((center_row, center_col), bf_radius)``.
    """
    dp = np.asarray(mean_dp, dtype=np.float32)
    thr = float(dp.mean()) + float(dp.std())
    mask = dp > thr
    total = int(mask.sum())
    if total == 0:
        h, w = dp.shape
        return (h / 2.0, w / 2.0), min(h, w) * 0.25
    rows = np.arange(dp.shape[0], dtype=np.float32)[:, None]
    cols = np.arange(dp.shape[1], dtype=np.float32)[None, :]
    cy = float((rows * mask).sum() / total)
    cx = float((cols * mask).sum() / total)
    radius = float(np.sqrt(total / np.pi))
    return (cy, cx), radius


def _detector_mask(mode, center, bf_radius, det_shape, inner, outer):
    """Boolean ``(det_row, det_col)`` mask for a virtual-detector mode."""
    cy, cx = center
    r = float(max(1.0, bf_radius))
    rows = np.arange(det_shape[0], dtype=np.float32)[:, None]
    cols = np.arange(det_shape[1], dtype=np.float32)[None, :]
    dist = np.sqrt((rows - cy) ** 2 + (cols - cx) ** 2)
    bands = {
        "BF": (0.0, r),
        "ABF": (0.5 * r, r),
        "ADF": (r, 2.0 * r),
        "HAADF": (2.0 * r, 4.0 * r),
        "DF": (r, np.inf),
    }
    if mode == "ANNULAR":
        lo, hi = (inner if inner is not None else 0.0) * r, (outer if outer is not None else np.inf) * r
    else:
        lo, hi = bands[mode]
    return (dist >= lo) & (dist <= hi)


def virtual(data, mode="BF", *, center=None, bf_radius=None, inner=None, outer=None):
    """Virtual image for ``mode`` with automatic probe fitting. See module docstring.

    ``mode`` is case-insensitive (DP/BF/ABF/ADF/HAADF/DF/annular). ``center`` and
    ``bf_radius`` override the auto-detected probe; ``inner``/``outer`` (BF-radius
    units) define a custom band when ``mode="annular"``. Returns a 2D float array
    (detector-space for DP, scan-space otherwise) for ``Show2D``.
    """
    backend = _resolve_backend(data)
    mean_dp = np.asarray(backend.mean_dp(), dtype=np.float32)
    mode = str(mode).strip().upper()
    if mode == "DP":
        return mean_dp
    if center is None or bf_radius is None:
        c_auto, r_auto = auto_probe(mean_dp)
        center = center if center is not None else c_auto
        bf_radius = bf_radius if bf_radius is not None else r_auto
    mask = _detector_mask(mode, center, bf_radius, mean_dp.shape, inner, outer)
    return np.asarray(backend.masked_sum(mask), dtype=np.float32)
