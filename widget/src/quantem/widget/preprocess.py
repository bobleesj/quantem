"""Small reusable 4D-STEM preprocessing helpers."""

from __future__ import annotations

from typing import Any


def _array_namespace(array: Any):
    module = type(array).__module__.split(".", 1)[0]
    if module == "cupy":
        import cupy as cp  # type: ignore[import-not-found]

        return cp
    if module == "torch":
        import torch

        return torch
    import numpy as np

    return np


def dp_mean(data: Any):
    """Return the mean diffraction pattern from a 3D or 4D diffraction stack."""
    if getattr(data, "_qw_dataset", False):
        return data.mean_dp()
    if hasattr(data, "_fields") and "data" in getattr(data, "_fields", ()):
        data = data.data
    xp = _array_namespace(data)
    axes = (0,) if getattr(data, "ndim", 0) == 3 else (0, 1)
    if xp.__name__ == "torch":
        return data.to(dtype=xp.float32).mean(dim=axes)
    return data.mean(axis=axes)


def virtual_image(
    data: Any,
    center_row: float | Any | None = None,
    center_col: float | None = None,
    *,
    radius: float | None = None,
    inner_radius: float | None = None,
    outer_radius: float | None = None,
    mask: Any | None = None,
    chunk_size: int | None = None,
    center=None,
):
    """Return a virtual image by summing detector pixels selected by ``mask``.

    When ``mask`` is omitted, ``center`` and ``radius`` describe a circular
    detector in ``(row, col)`` coordinates. If neither is given, the bright-field
    disk is detected from the mean diffraction pattern.
    """
    if chunk_size is not None:
        raise NotImplementedError(
            "quantem.widget.virtual_image owns the shared compute path and does "
            "not accept live's old chunk_size override."
        )
    if mask is None and center_col is None and center is None and center_row is not None:
        candidate = center_row
        if hasattr(candidate, "shape") or isinstance(candidate, (list, tuple)):
            mask = candidate
            center_row = None
    if getattr(data, "_qw_dataset", False):
        if mask is not None:
            return data.masked_sum(mask)
        if center is None and center_row is not None and center_col is not None:
            center = (center_row, center_col)
        center, radius = data._probe(center, radius)
        from quantem.widget.detector import detector_mask

        lo = 0.0 if radius is not None else float(inner_radius)
        hi = float(radius) if radius is not None else float(outer_radius)
        mask = detector_mask(center, lo, hi, data.mean_dp().shape)
        return data.masked_sum(mask)
    if hasattr(data, "_fields") and "data" in getattr(data, "_fields", ()):
        data = data.data
    xp = _array_namespace(data)
    if mask is None:
        mean_dp = dp_mean(data)
        from quantem.widget.detector import auto_probe, detector_mask

        if center is None or radius is None:
            detected_center, detected_radius = auto_probe(mean_dp)
            center = detected_center if center is None else center
            radius = detected_radius if radius is None else radius
        if center_row is not None and center_col is not None:
            center = (float(center_row), float(center_col))
        lo = 0.0 if radius is not None else float(inner_radius)
        hi = float(radius) if radius is not None else float(outer_radius)
        mask = detector_mask(center, lo, hi, mean_dp.shape)
    if xp.__name__ == "torch":
        mask_tensor = xp.as_tensor(mask, device=data.device, dtype=xp.bool)
        image = data.reshape(-1, *data.shape[-2:])[:, mask_tensor].to(dtype=xp.float32).sum(dim=1)
        return image.reshape(data.shape[0], data.shape[1]) if data.ndim == 4 else image
    mask = xp.asarray(mask, dtype=bool)
    image = data.reshape(-1, *data.shape[-2:])[:, mask].sum(axis=1)
    return image.reshape(data.shape[0], data.shape[1]) if data.ndim == 4 else image


__all__ = ["dp_mean", "virtual_image"]
