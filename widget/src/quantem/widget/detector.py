"""Virtual detectors (bright / annular-dark / dark field) for 4D-STEM.

Primary API - place a virtual detector on a dataset and get its image, with
collection angles in **mrad**::

    from quantem.widget import load, Dataset4dstemGPU, Show2D
    ds = Dataset4dstemGPU(load("master.h5"))
    Show2D(ds.bf())                       # bright field (the bright disk)
    Show2D(ds.adf())                       # annular dark field (auto band)
    Show2D(ds.adf(inner=50, outer=180))    # collection angles in mrad
    Show2D(ds.df())                        # outside the bright disk

``ds.bf()`` / ``.adf()`` / ``.df()`` are thin geometry over the shared compute
backend: they build a boolean detector mask and call the dataset's masked-sum -
the single fast reduction in ``kernels/compute`` that Show4DSTEM and any GUI also
use. The probe (disk center + size) auto-fits from the mean diffraction pattern;
``semiangle_mrad`` (from the load metadata, or ``ds.semiangle_mrad = ...``)
calibrates mrad because the bright disk spans exactly the convergence semi-angle.
MacBook (MPS) runs the raw-Metal masked-sum over chunked uint16 buffers; CUDA /
CPU runs torch. **No binning** on either path.

The lower-level :func:`virtual` function (below) is mode-based
(DP/BF/ABF/ADF/HAADF/DF, bands measured in the auto-detected disk radius) and is
mainly the reference path the parity tests pin; ``ds.bf()`` etc. are the API.
"""
from __future__ import annotations

import numpy as np


def _resolve_backend(data):
    """Return a compute backend (MetalCompute on MPS chunks, TorchCompute on array)."""
    if getattr(data, "_qw_dataset", False):  # Dataset4dstemGPU - backend already resolved
        return data.compute
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


# --- virtual detectors: thin geometry over the shared compute backend ---
# bf/adf/df build a boolean detector mask, then call the dataset's masked-sum
# (the single fast reduction in kernels/compute - the same one Show4DSTEM and any
# GUI use). The dataset holds the probe (center/bf_radius/semiangle_mrad) and the
# result cache; these functions are stateless and overridable.


def _mrad_to_px(ds, mrad: float) -> float:
    """Collection angle in mrad -> detector pixel radius. The bright disk radius
    spans ``semiangle_mrad``, so a mrad angle maps to
    ``mrad / semiangle_mrad * bf_radius_px``."""
    if not ds.semiangle_mrad:
        raise ValueError(
            "inner / outer are collection angles in mrad, but the convergence "
            "semi-angle is unknown for this dataset. Set it explicitly:\n"
            "    ds.semiangle_mrad = <convergence semi-angle in mrad>\n"
            "or pass detector pixels instead: ds.adf(inner=..., outer=..., unit='px')")
    return float(mrad) / float(ds.semiangle_mrad) * ds.bf_radius


def _to_px(ds, value: float, unit: str) -> float:
    """A collection-angle radius -> detector pixels. ``unit='mrad'`` (default)
    converts via the convergence semi-angle; ``unit='px'`` is already pixels
    (calibration-free, exact)."""
    unit = str(unit).lower()
    if unit in ("px", "pixel", "pixels"):
        return float(value)
    if unit == "mrad":
        return _mrad_to_px(ds, value)
    raise ValueError(f"unit must be 'mrad' or 'px', got {unit!r}")


def _detector_image(ds, name: str, lo_px: float, hi_px: float) -> np.ndarray:
    """Masked-sum image over the annulus ``lo_px .. hi_px`` detector pixels,
    cached on the dataset. The masked-sum runs on the shared compute backend."""
    cy, cx = ds.center
    key = (name, round(cy, 3), round(cx, 3), round(float(lo_px), 3), round(float(hi_px), 3))
    cached = ds._detector_cache.get(key)
    if cached is not None:
        return cached
    mean_dp = ds.mean_dp()
    rows = np.arange(mean_dp.shape[0], dtype=np.float32)[:, None]
    cols = np.arange(mean_dp.shape[1], dtype=np.float32)[None, :]
    dist = np.sqrt((rows - cy) ** 2 + (cols - cx) ** 2)
    mask = (dist >= lo_px) & (dist <= hi_px)
    img = np.asarray(ds.masked_sum(mask), dtype=np.float32)
    ds._detector_cache[key] = img
    return img


def bf(ds) -> np.ndarray:
    """Bright-field image of ``ds``: the bright disk (the unscattered probe)."""
    return _detector_image(ds, "bf", 0.0, ds.bf_radius)


def adf(ds, inner: float | None = None, outer: float | None = None,
        unit: str = "mrad") -> np.ndarray:
    """Annular-dark-field image of ``ds``, collected between ``inner`` and
    ``outer``. ``unit='mrad'`` (default, needs ``ds.semiangle_mrad``) or
    ``unit='px'`` (raw detector pixels). Omit either for the automatic band:
    ``inner`` = the bright-disk edge, ``outer`` = twice that."""
    radius = ds.bf_radius
    lo_px = radius if inner is None else _to_px(ds, inner, unit)
    hi_px = 2.0 * radius if outer is None else _to_px(ds, outer, unit)
    return _detector_image(ds, "adf", lo_px, hi_px)


def df(ds, inner: float | None = None, unit: str = "mrad") -> np.ndarray:
    """Dark-field image of ``ds``: everything collected beyond ``inner``.
    ``unit='mrad'`` (default, needs ``ds.semiangle_mrad``) or ``unit='px'``.
    Omit ``inner`` for everything outside the bright disk."""
    radius = ds.bf_radius
    lo_px = radius if inner is None else _to_px(ds, inner, unit)
    return _detector_image(ds, "df", lo_px, np.inf)


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
