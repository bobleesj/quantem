"""I/O helpers for DriftCorrection and known-drift workflows.

This module keeps microscope metadata parsing and synthetic known-drift H5
metadata in QuantEM rather than notebook-local helper functions. The known
drift metadata contract is intentionally small: a shared drift vector, scan
crop bounds, detector shape, and optional probe-position datasets.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

import quantem as em


DRIFT_METADATA_GROUP = "entry/quantem/drift"
DEFAULT_POSITION_UNITS = "scan pixels in shared physical sample coordinates"


@dataclass(frozen=True)
class KnownDriftMetadata:
    """Metadata stored with a known-drift 4D-STEM export.

    The drift vector is always ``(down_px, right_px)`` in the shared specimen
    frame. ``scan_crop_rows`` and ``scan_crop_cols`` describe which raw scan
    pixels from the generated acquisition were exported.
    """

    path: str
    label: str | None
    scan_crop_rows: tuple[int, int] | None
    scan_crop_cols: tuple[int, int] | None
    detector_shape_px: tuple[int, ...] | None
    source_master: str | None
    det_bin: int | None
    known_drift_total_px_down_right: tuple[float, float] | None
    position_units: str | None
    probe_positions_shape: tuple[int, ...] | None
    positions_offset_shape: tuple[int, ...] | None

    def as_manifest_row(self) -> dict[str, Any]:
        """Return a flat dictionary suitable for CSV/JSON manifests."""
        drift = self.known_drift_total_px_down_right
        rows = self.scan_crop_rows
        cols = self.scan_crop_cols
        return {
            "master_path": self.path,
            "label": self.label,
            "known_drift_total_px_down": None if drift is None else drift[0],
            "known_drift_total_px_right": None if drift is None else drift[1],
            "scan_crop_row_start": None if rows is None else rows[0],
            "scan_crop_row_stop": None if rows is None else rows[1],
            "scan_crop_col_start": None if cols is None else cols[0],
            "scan_crop_col_stop": None if cols is None else cols[1],
            "probe_positions_shape": self.probe_positions_shape,
            "positions_offset_shape": self.positions_offset_shape,
            "det_bin": self.det_bin,
            "detector_shape_px": self.detector_shape_px,
            "source_master": self.source_master,
            "position_units": self.position_units,
        }


@dataclass(frozen=True)
class Known4DSTEMExportStats:
    """Quantization statistics for one known-drift 4D-STEM export."""

    clipped_below: int
    clipped_above: int
    min_value: float
    max_value: float
    detector_shape_px: tuple[int, ...]
    scan_crop_rows: tuple[int, int]
    scan_crop_cols: tuple[int, int]


@dataclass(frozen=True)
class Known4DSTEMExportResult:
    """Result returned after writing one known-drift 4D-STEM export."""

    path: str
    label: str
    skipped: bool
    stats: Known4DSTEMExportStats | None


def _decode_attr(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    return value


def _optional_int_tuple(value: Any) -> tuple[int, ...] | None:
    if value is None:
        return None
    arr = np.asarray(value)
    if arr.size == 0:
        return None
    return tuple(int(x) for x in arr.reshape(-1))


def _optional_float_pair(value: Any, *, name: str) -> tuple[float, float] | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.size != 2:
        raise ValueError(
            f"{name} must contain exactly two values, "
            f"got shape {np.asarray(value).shape}"
        )
    return float(arr[0]), float(arr[1])


def _crop_to_bounds(
    scan_crop: tuple[slice, slice] | tuple[tuple[int, int], tuple[int, int]],
) -> tuple[tuple[int, int], tuple[int, int]]:
    rows, cols = scan_crop
    if isinstance(rows, slice) and isinstance(cols, slice):
        if rows.start is None or rows.stop is None or cols.start is None or cols.stop is None:
            raise ValueError("scan_crop slices must have explicit start and stop")
        return (int(rows.start), int(rows.stop)), (int(cols.start), int(cols.stop))
    row_bounds = tuple(int(x) for x in rows)  # type: ignore[arg-type]
    col_bounds = tuple(int(x) for x in cols)  # type: ignore[arg-type]
    if len(row_bounds) != 2 or len(col_bounds) != 2:
        raise ValueError(
            "scan_crop bounds must be "
            "((row_start, row_stop), (col_start, col_stop))"
        )
    return row_bounds, col_bounds


def drift_crop_slices(
    scan_crop_rows: tuple[int, int],
    scan_crop_cols: tuple[int, int],
) -> tuple[slice, slice]:
    """Convert stored crop bounds to row/column slices."""
    return slice(int(scan_crop_rows[0]), int(scan_crop_rows[1])), slice(
        int(scan_crop_cols[0]), int(scan_crop_cols[1])
    )


def _slice_4dstem_scan_crop(data: Any, scan_crop: tuple[slice, slice]) -> Any:
    return data[scan_crop[0], scan_crop[1]]


def _is_torch_tensor(value: Any) -> bool:
    return value.__class__.__module__.startswith("torch") and hasattr(value, "detach")


def _is_cupy_array(value: Any) -> bool:
    return value.__class__.__module__.startswith("cupy") and hasattr(value, "get")


def quantize_4dstem_scan_crop_uint16(
    data: Any,
    scan_crop: tuple[slice, slice],
) -> tuple[Any, Known4DSTEMExportStats]:
    """Crop scan axes and quantize a 4D-STEM block to ``uint16``.

    This helper is deliberately 4D-STEM-specific: the crop applies only to
    the leading scan axes, leaving detector axes unchanged. It accepts NumPy,
    CuPy, or torch tensors. CUDA torch tensors are converted to CuPy through
    DLPack so quantization can stay on the GPU before ``quantem.live`` saving.
    """
    row_bounds, col_bounds = _crop_to_bounds(scan_crop)
    crop = _slice_4dstem_scan_crop(data, scan_crop)

    if _is_torch_tensor(crop):
        import torch

        crop_t = crop.contiguous()
        if getattr(crop_t, "is_cuda", False):
            import cupy as cp

            crop_xp = cp.from_dlpack(torch.utils.dlpack.to_dlpack(crop_t))
            below = int(cp.count_nonzero(crop_xp < 0).get())
            above = int(cp.count_nonzero(crop_xp > np.iinfo(np.uint16).max).get())
            min_value = float(cp.min(crop_xp).get())
            max_value = float(cp.max(crop_xp).get())
            quantized = cp.clip(cp.rint(crop_xp), 0, np.iinfo(np.uint16).max).astype(cp.uint16)
            detector_shape = tuple(int(x) for x in quantized.shape[-2:])
            stats = Known4DSTEMExportStats(
                clipped_below=below,
                clipped_above=above,
                min_value=min_value,
                max_value=max_value,
                detector_shape_px=detector_shape,
                scan_crop_rows=row_bounds,
                scan_crop_cols=col_bounds,
            )
            return quantized, stats

        crop_np = crop_t.detach().cpu().numpy()
        below = int(np.count_nonzero(crop_np < 0))
        above = int(np.count_nonzero(crop_np > np.iinfo(np.uint16).max))
        min_value = float(np.min(crop_np))
        max_value = float(np.max(crop_np))
        quantized = np.clip(np.rint(crop_np), 0, np.iinfo(np.uint16).max).astype(np.uint16)
        detector_shape = tuple(int(x) for x in quantized.shape[-2:])
        stats = Known4DSTEMExportStats(
            clipped_below=below,
            clipped_above=above,
            min_value=min_value,
            max_value=max_value,
            detector_shape_px=detector_shape,
            scan_crop_rows=row_bounds,
            scan_crop_cols=col_bounds,
        )
        return quantized, stats

    if _is_cupy_array(crop):
        import cupy as cp

        crop_xp = cp.ascontiguousarray(crop)
        below = int(cp.count_nonzero(crop_xp < 0).get())
        above = int(cp.count_nonzero(crop_xp > np.iinfo(np.uint16).max).get())
        min_value = float(cp.min(crop_xp).get())
        max_value = float(cp.max(crop_xp).get())
        quantized = cp.clip(cp.rint(crop_xp), 0, np.iinfo(np.uint16).max).astype(cp.uint16)
        detector_shape = tuple(int(x) for x in quantized.shape[-2:])
        stats = Known4DSTEMExportStats(
            clipped_below=below,
            clipped_above=above,
            min_value=min_value,
            max_value=max_value,
            detector_shape_px=detector_shape,
            scan_crop_rows=row_bounds,
            scan_crop_cols=col_bounds,
        )
        return quantized, stats

    crop_np = np.ascontiguousarray(np.asarray(crop))
    below = int(np.count_nonzero(crop_np < 0))
    above = int(np.count_nonzero(crop_np > np.iinfo(np.uint16).max))
    min_value = float(np.min(crop_np))
    max_value = float(np.max(crop_np))
    quantized = np.clip(np.rint(crop_np), 0, np.iinfo(np.uint16).max).astype(np.uint16)
    detector_shape = tuple(int(x) for x in quantized.shape[-2:])
    stats = Known4DSTEMExportStats(
        clipped_below=below,
        clipped_above=above,
        min_value=min_value,
        max_value=max_value,
        detector_shape_px=detector_shape,
        scan_crop_rows=row_bounds,
        scan_crop_cols=col_bounds,
    )
    return quantized, stats


def write_known_4dstem_drift_metadata(
    master_path: str | Path,
    *,
    positions_px: np.ndarray | None = None,
    positions_offset_px: np.ndarray | None = None,
    scan_crop: tuple[slice, slice] | tuple[tuple[int, int], tuple[int, int]] | None = None,
    detector_shape_px: tuple[int, ...] | list[int] | np.ndarray | None = None,
    label: str | None = None,
    source_master: str | Path | None = None,
    det_bin: int | None = None,
    known_drift_total_px_down_right: tuple[float, float] | list[float] | np.ndarray | None = None,
    position_units: str = DEFAULT_POSITION_UNITS,
    extra_attrs: dict[str, Any] | None = None,
    compression: str | None = "gzip",
    compression_opts: int | None = 4,
) -> None:
    """Write known-drift metadata for a scan-axis-leading 4D-STEM export."""
    write_known_drift_metadata(
        master_path,
        positions_px=positions_px,
        positions_offset_px=positions_offset_px,
        scan_crop=scan_crop,
        detector_shape_px=detector_shape_px,
        label=label,
        source_master=source_master,
        det_bin=det_bin,
        known_drift_total_px_down_right=known_drift_total_px_down_right,
        position_units=position_units,
        extra_attrs=extra_attrs,
        compression=compression,
        compression_opts=compression_opts,
    )


def read_known_4dstem_drift_metadata(master_path: str | Path) -> KnownDriftMetadata:
    """Read known-drift metadata for a scan-axis-leading 4D-STEM export."""
    return read_known_drift_metadata(master_path)


def save_known_4dstem_drift_export(
    master_path: str | Path,
    data: Any,
    *,
    scan_crop: tuple[slice, slice],
    positions_px: np.ndarray,
    positions_offset_px: np.ndarray,
    label: str,
    source_master: str | Path,
    det_bin: int,
    known_drift_total_px_down_right: tuple[float, float] | list[float] | np.ndarray,
    save_func: Any,
    scan_shape: tuple[int, int],
    dtype: str = "u16",
    overwrite: bool = False,
    save_kwargs: dict[str, Any] | None = None,
) -> Known4DSTEMExportResult:
    """Quantize, save, and annotate one known-drift 4D-STEM export.

    ``save_func`` is usually ``quantem.live.io.save``. It is injected instead
    of imported here so QuantEM's drift metadata helpers remain a clean bridge
    to ``quantem.live`` without making import-time dependencies heavier.
    """
    path = Path(master_path)
    if path.exists() and not overwrite:
        return Known4DSTEMExportResult(
            path=str(path),
            label=label,
            skipped=True,
            stats=None,
        )

    quantized, stats = quantize_4dstem_scan_crop_uint16(data, scan_crop)
    metadata = {
        "quantem_export_kind": label,
        "source_master": str(source_master),
        "det_bin": int(det_bin),
        "known_drift_total_px_down_right": np.asarray(
            known_drift_total_px_down_right,
            dtype=np.float32,
        ),
    }
    save_func(
        str(path),
        quantized,
        scan_shape=scan_shape,
        dtype=dtype,
        metadata=metadata,
        **(save_kwargs or {}),
    )
    write_known_4dstem_drift_metadata(
        path,
        positions_px=positions_px[scan_crop[0], scan_crop[1]],
        positions_offset_px=positions_offset_px[scan_crop[0], scan_crop[1]],
        scan_crop=scan_crop,
        detector_shape_px=stats.detector_shape_px,
        label=label,
        source_master=source_master,
        det_bin=det_bin,
        known_drift_total_px_down_right=known_drift_total_px_down_right,
    )
    return Known4DSTEMExportResult(
        path=str(path),
        label=label,
        skipped=False,
        stats=stats,
    )


def write_known_drift_metadata(
    master_path: str | Path,
    *,
    positions_px: np.ndarray | None = None,
    positions_offset_px: np.ndarray | None = None,
    scan_crop: tuple[slice, slice] | tuple[tuple[int, int], tuple[int, int]] | None = None,
    detector_shape_px: tuple[int, ...] | list[int] | np.ndarray | None = None,
    label: str | None = None,
    source_master: str | Path | None = None,
    det_bin: int | None = None,
    known_drift_total_px_down_right: tuple[float, float] | list[float] | np.ndarray | None = None,
    position_units: str = DEFAULT_POSITION_UNITS,
    extra_attrs: dict[str, Any] | None = None,
    compression: str | None = "gzip",
    compression_opts: int | None = 4,
) -> None:
    """Write QuantEM known-drift metadata into a master H5 file.

    This is the canonical export contract for synthetic drift datasets. The
    input 4D-STEM data can be produced anywhere, but drift vectors and crop
    provenance should be written here so later notebooks, manifests, and live
    bridges read the same metadata.
    """
    path = Path(master_path)
    dataset_kwargs: dict[str, Any] = {}
    if compression is not None:
        dataset_kwargs["compression"] = compression
        if compression_opts is not None:
            dataset_kwargs["compression_opts"] = compression_opts

    with h5py.File(path, "r+") as f:
        group = f.require_group(DRIFT_METADATA_GROUP)
        if positions_px is not None:
            if "probe_positions_px" in group:
                del group["probe_positions_px"]
            group.create_dataset(
                "probe_positions_px",
                data=np.asarray(positions_px, dtype=np.float32),
                **dataset_kwargs,
            )
        if positions_offset_px is not None:
            if "positions_offset_px" in group:
                del group["positions_offset_px"]
            group.create_dataset(
                "positions_offset_px",
                data=np.asarray(positions_offset_px, dtype=np.float32),
                **dataset_kwargs,
            )
        if label is not None:
            group.attrs["label"] = str(label)
        if scan_crop is not None:
            row_bounds, col_bounds = _crop_to_bounds(scan_crop)
            group.attrs["scan_crop_rows"] = np.asarray(row_bounds, dtype=np.int32)
            group.attrs["scan_crop_cols"] = np.asarray(col_bounds, dtype=np.int32)
        if detector_shape_px is not None:
            group.attrs["detector_shape_px"] = np.asarray(detector_shape_px, dtype=np.int32)
        if source_master is not None:
            group.attrs["source_master"] = str(source_master)
        if det_bin is not None:
            group.attrs["det_bin"] = int(det_bin)
        if known_drift_total_px_down_right is not None:
            drift = _optional_float_pair(
                known_drift_total_px_down_right,
                name="known_drift_total_px_down_right",
            )
            group.attrs["known_drift_total_px_down_right"] = np.asarray(drift, dtype=np.float32)
        group.attrs["position_units"] = str(position_units)
        for key, value in (extra_attrs or {}).items():
            if value is not None:
                group.attrs[str(key)] = value


def read_known_drift_metadata(master_path: str | Path) -> KnownDriftMetadata:
    """Read QuantEM known-drift metadata from a master H5 file."""
    path = Path(master_path)
    with h5py.File(path, "r") as f:
        if DRIFT_METADATA_GROUP not in f:
            raise KeyError(f"{path} does not contain {DRIFT_METADATA_GROUP}")
        group = f[DRIFT_METADATA_GROUP]
        attrs = group.attrs
        return KnownDriftMetadata(
            path=str(path),
            label=_decode_attr(attrs.get("label")),
            scan_crop_rows=_optional_int_tuple(attrs.get("scan_crop_rows")),
            scan_crop_cols=_optional_int_tuple(attrs.get("scan_crop_cols")),
            detector_shape_px=_optional_int_tuple(attrs.get("detector_shape_px")),
            source_master=_decode_attr(attrs.get("source_master")),
            det_bin=None if attrs.get("det_bin") is None else int(attrs.get("det_bin")),
            known_drift_total_px_down_right=_optional_float_pair(
                attrs.get("known_drift_total_px_down_right"),
                name="known_drift_total_px_down_right",
            ),
            position_units=_decode_attr(attrs.get("position_units")),
            probe_positions_shape=(
                tuple(int(x) for x in group["probe_positions_px"].shape)
                if "probe_positions_px" in group
                else None
            ),
            positions_offset_shape=(
                tuple(int(x) for x in group["positions_offset_px"].shape)
                if "positions_offset_px" in group
                else None
            ),
        )


def read_emd_with_metadata(path: str | Path) -> dict:
    """Read a single-frame Velox EMD file and return data plus metadata.

    Returns
    -------
    dict
        {
            "data": quantem.Dataset2d,
            "scan_rotation_deg": float,         # ScanRotation in degrees
            "pixel_size_nm": float,             # detector pixel size in nm
            "shape": tuple[int, int],           # (rows, cols)
            "stack_shape": tuple[int, ...],     # raw stack shape from EMD
            "magnification": float | None,      # NominalMagnification
            "fov_nm": float | None,             # field of view, nm
            "path": str,
        }
    """
    path = str(path)
    data = em.io.read_2d(path)
    scan_rot_deg = _read_scan_rotation_deg(path)
    stack_shape = _read_stack_shape(path)
    px_nm = float(data.sampling[0])
    fov_nm = px_nm * data.shape[0]
    mag = _read_nominal_magnification(path)
    return {
        "data": data,
        "scan_rotation_deg": scan_rot_deg,
        "pixel_size_nm": px_nm,
        "shape": tuple(data.shape),
        "stack_shape": stack_shape,
        "magnification": mag,
        "fov_nm": fov_nm,
        "path": path,
    }


def read_emd_pair(path_0: str | Path, path_1: str | Path) -> dict:
    """Read a paired 0°/90° EMD acquisition and return a ready-to-use bundle.

    Returns
    -------
    dict
        {
            "data": [Dataset2d, Dataset2d],
            "scan_direction_degrees": tuple[float, float],
            "pixel_size_nm": float,
            "shape": tuple[int, int],
            "metadata": [m0, m1],   # full per-frame metadata dicts
        }
    """
    m0 = read_emd_with_metadata(path_0)
    m1 = read_emd_with_metadata(path_1)
    if m0["shape"] != m1["shape"]:
        raise ValueError(
            f"shape mismatch: {m0['shape']} != {m1['shape']}"
        )
    return {
        "data": [m0["data"], m1["data"]],
        "scan_direction_degrees": (m0["scan_rotation_deg"], m1["scan_rotation_deg"]),
        "pixel_size_nm": m0["pixel_size_nm"],
        "shape": m0["shape"],
        "metadata": [m0, m1],
    }


def _read_scan_rotation_deg(path: str) -> float:
    md = _read_velox_metadata(path)
    sr = md.get("Scan", {}).get("ScanRotation", None)
    if sr is None:
        return 0.0
    return float(sr) * 180.0 / np.pi


def _read_nominal_magnification(path: str) -> float | None:
    md = _read_velox_metadata(path)
    mag = md.get("Optics", {}).get("NominalMagnification", None)
    return float(mag) if mag is not None else None


def _read_stack_shape(path: str) -> tuple[int, ...]:
    with h5py.File(path, "r") as f:
        for img in f["Data/Image"]:
            return tuple(f[f"Data/Image/{img}/Data"].shape)
    return ()


def _read_velox_metadata(path: str) -> dict:
    """Parse the JSON metadata blob from a Velox EMD file.

    Multi-frame stacks interleave N copies of the bytes (one per frame); the
    parser strips nulls and tolerates the interleave by trying both raw and
    de-interleaved decodings before giving up.
    """
    with h5py.File(path, "r") as f:
        for img in f["Data/Image"]:
            raw = bytes(f[f"Data/Image/{img}/Metadata"][:]).replace(b"\x00", b"")
            for candidate in (raw, _deinterleave(raw)):
                try:
                    return json.loads(candidate.decode("utf-8", errors="ignore"))
                except json.JSONDecodeError:
                    continue
    return {}


def _deinterleave(raw: bytes) -> bytes:
    """Velox multi-frame metadata replicates each byte once per frame.

    Strip the interleave by sampling every Nth byte for N=1..16 and returning
    the first decode that parses as valid JSON-shaped text (starts with '{').
    """
    for n in range(2, 17):
        candidate = raw[::n]
        if candidate.startswith(b"{"):
            return candidate
    return raw
