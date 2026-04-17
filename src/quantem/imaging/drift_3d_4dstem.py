"""Dataset-path drift correction: ≥3-D inputs with scan axes leading.

The image-pair pipeline lives in :mod:`drift`; this module owns the
bigger hammer for *any* dataset shape with ``(scan_h, scan_w, …channels)``
layout:

* **3-D spectral cubes** (EDS / EELS) — ``(H, W, n_energy)``.
* **4-D STEM** scans — ``(H, W, det_h, det_w)``.

Both go through the same shape-agnostic ``apply_correction_to_dataset``
loop (chunked GPU ``grid_sample`` with pre-allocated memmap output).
The 4-D STEM-specific helpers — ``compute_vdf``, the paired merge
``generate_corrected_paired_datasets``, and the
:class:`PairedCorrectionResult` container — live alongside because they
share the dataset-shape concern.

Functions here take an already-built :class:`DriftCorrection` as their
first argument (``dc``) so :class:`drift.DriftCorrection` stays focused
on the algorithm story.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

if TYPE_CHECKING:
    from quantem.imaging.drift import DriftCorrection


@dataclass
class PairedCorrectionResult:
    """Container returned by paired 4D-STEM ``generate_corrected``.

    Holds the merged dataset (when ``merge=True``), per-side corrected
    datasets, the VDFs that drove the alignment, and a back-reference to
    the :class:`DriftCorrection` instance used.

    Attributes
    ----------
    corrected_a, corrected_b : np.ndarray
        Per-side drift-corrected datasets, scan-axis-leading layout.
    vdf_a, vdf_b : np.ndarray
        Pre-correction virtual-detector images, ``(scan_h, scan_w)``.
    merged : np.ndarray | None
        Half-sum of corrected_a + corrected_b in dataset A's frame.
        ``None`` when ``merge=False`` (heavy-data callers that defer
        the merge to disk).
    drift : DriftCorrection
        Back-reference for plotting / introspection (``result.drift.knots``,
        ``result.drift.print_drift_stats()``, etc.).
    """
    corrected_a: np.ndarray
    corrected_b: np.ndarray
    vdf_a: np.ndarray
    vdf_b: np.ndarray
    drift: object  # DriftCorrection — typed via TYPE_CHECKING above
    merged: np.ndarray | None = field(default=None)


def compute_vdf(
    ds_4d: np.ndarray,
    chunk_rows: int | None = None,
) -> np.ndarray:
    """Compute a virtual dark-field image from a 4D-STEM dataset.

    Averages over the detector dimensions to produce a 2D scan image.
    Supports memory-mapped inputs — when *chunk_rows* is set, only a
    few scan rows are loaded at a time, keeping host RAM usage low.

    Parameters
    ----------
    ds_4d : np.ndarray, shape ``(H, W, det_h, det_w)``
        4D-STEM dataset.  Can be a ``np.memmap``.
    chunk_rows : int or None
        Number of scan rows to process at a time.  ``None`` loads
        everything at once (fastest for in-memory arrays).

    Returns
    -------
    np.ndarray, shape ``(H, W)``, dtype float32
    """
    H, W = ds_4d.shape[:2]
    det_pixels = 1
    for d in range(2, ds_4d.ndim):
        det_pixels *= ds_4d.shape[d]
    # uint64 accumulator avoids the ~4x float64 transient from .mean().
    sum_dtype = np.uint64 if np.issubdtype(ds_4d.dtype, np.integer) else np.float64

    if chunk_rows is None:
        totals = ds_4d.reshape(H, W, det_pixels).sum(axis=2, dtype=sum_dtype)
        return (totals / det_pixels).astype(np.float32)

    vdf = np.empty((H, W), dtype=np.float32)
    for start in range(0, H, chunk_rows):
        end = min(start + chunk_rows, H)
        chunk = np.asarray(ds_4d[start:end])
        totals = chunk.reshape(end - start, W, det_pixels).sum(
            axis=2, dtype=sum_dtype)
        vdf[start:end] = (totals / det_pixels).astype(np.float32)
    return vdf


def apply_correction_to_dataset(
    dc: "DriftCorrection",
    ds_4d: torch.Tensor | np.ndarray | None = None,
    image_index: int = -1,
    mode: str = "bicubic",
    chunk_size: int | None = None,
    output_dtype: torch.dtype | np.dtype | str | None = None,
    output_device: str | torch.device | None = None,
    output: np.ndarray | None = None,
    verbose: bool = False,
) -> torch.Tensor | np.ndarray:
    """Apply drift correction to a ≥3-D dataset with scan axes leading.

    Internal worker for the 4D-STEM / spectral path of
    :meth:`DriftCorrection.apply_correction`.  Auto-selects single-shot GPU
    vs chunked processing based on free memory; supports pre-allocated
    ``output=`` for zero-copy memmap workflows.
    """
    # Resolve dataset from stored data when not provided explicitly
    if ds_4d is None:
        datasets = dc._datasets
        if datasets is None:
            raise ValueError(
                "No dataset provided and none stored. Pass ds_4d "
                "explicitly, or build this instance with "
                "DriftCorrection(ds_a, ds_b, ...)."
            )
        if image_index < 0:
            image_index = len(datasets) + image_index
        if image_index < 0 or image_index >= len(datasets):
            raise IndexError(
                f"image_index={image_index} out of range for "
                f"{len(datasets)} stored datasets"
            )
        ds_4d = datasets[image_index]

    is_numpy = isinstance(ds_4d, np.ndarray)
    original_shape = ds_4d.shape if is_numpy else tuple(ds_4d.shape)
    input_np_dtype = ds_4d.dtype if is_numpy else None
    use_external_output = output is not None

    if use_external_output:
        if not isinstance(output, np.ndarray):
            raise TypeError(
                "output must be a numpy ndarray (or np.memmap), "
                f"got {type(output).__name__}"
            )
        if tuple(output.shape) != tuple(original_shape):
            raise ValueError(
                f"output shape {output.shape} does not match "
                f"ds_4d shape {original_shape}"
            )

    return_numpy = (
        use_external_output
        or (is_numpy and output_device is None)
    )

    ndim = len(original_shape)
    if ndim < 3:
        raise ValueError(
            f"ds_4d must be at least 3D, got shape {original_shape}"
        )

    scan_h, scan_w = original_shape[0], original_shape[1]
    n_channels = 1
    for d in range(2, ndim):
        n_channels *= original_shape[d]

    device = torch.device(dc._device)

    # ── Drift from knots (canvas → raw frame) ──
    idx = image_index % len(dc.knots)
    # Validates preprocess+align ran.
    knot_h = dc._knot_delta_canvas(idx).shape[1]
    if knot_h != scan_h:
        raise ValueError(
            f"Drift grid has {knot_h} rows but ds_4d has "
            f"{scan_h} scan rows. Ensure reference image and ds_4d "
            f"have matching scan dimensions (check padding / resize).")

    drift = dc._drift(idx).to(device=device, dtype=torch.float32)
    # K=1 → drift is (2, H), broadcast across columns.
    # K>=2 → drift is (2, H, W), varies along fast axis.
    if drift.ndim == 2:
        drift_row = drift[0][:, None]
        drift_col = drift[1][:, None]
    else:
        drift_row = drift[0]
        drift_col = drift[1]
    row_coords = torch.arange(scan_h, device=device, dtype=torch.float32)
    col_coords = torch.arange(scan_w, device=device, dtype=torch.float32)
    sample_row = row_coords[:, None].expand(scan_h, scan_w) - drift_row
    sample_col = col_coords[None, :].expand(scan_h, scan_w) - drift_col
    # ── Pre-compute warp grid ONCE (tiny: 1×H×W×2 f32) ──
    warp_grid = torch.stack([
        2.0 * sample_col / (scan_w - 1) - 1.0,
        2.0 * sample_row / (scan_h - 1) - 1.0,
    ], dim=-1)[None]                                       # (1, H, W, 2)

    # ── Flatten input to (H, W, C) view ──
    flat = (
        torch.from_numpy(ds_4d.reshape(scan_h, scan_w, n_channels))
        if is_numpy
        else ds_4d.reshape(scan_h, scan_w, n_channels)
    )

    # ── Output dtype (for GPU intermediates) ──
    out_dt = torch.float32
    if output_dtype == "same":
        if is_numpy and input_np_dtype is not None:
            out_dt = torch.from_numpy(
                np.empty(0, dtype=input_np_dtype)
            ).dtype
        elif not is_numpy:
            out_dt = ds_4d.dtype
    elif isinstance(output_dtype, torch.dtype):
        out_dt = output_dtype

    # ── Target device ──
    if use_external_output:
        target = torch.device("cpu")
    elif output_device is not None:
        target = torch.device(output_device)
        if target.type == "cuda":
            target = device
    else:
        target = torch.device("cpu")

    # ── Chunk size (auto-fit within GPU memory) ──
    if chunk_size is None:
        bytes_per_ch = scan_h * scan_w * 4
        try:
            gpu_free, _ = torch.cuda.mem_get_info(device)
        except RuntimeError:
            gpu_free = 0
        if target.type == "cuda" and not use_external_output:
            out_elem = torch.tensor([], dtype=out_dt).element_size()
            gpu_free = max(
                0,
                gpu_free - n_channels * scan_h * scan_w * out_elem,
            )
        chunk_size = min(
            n_channels,
            max(1, int(gpu_free * 0.7 / (bytes_per_ch * 2))),
        )

    # ── Allocate output ──
    if use_external_output:
        out_flat = output.reshape(scan_h, scan_w, n_channels)
    else:
        internal_output = torch.empty(
            scan_h, scan_w, n_channels, dtype=out_dt, device=target,
        )

    # ── Numpy dtype for external output conversion ──
    if use_external_output:
        _out_np_dtype = output.dtype

    # ── Vectorized grid_sample with pre-computed grid ──
    chunks = range(0, n_channels, chunk_size)
    if verbose:
        chunks = tqdm(
            chunks,
            total=(n_channels + chunk_size - 1) // chunk_size,
            desc="apply_correction",
            unit="chunk",
        )
    for start in chunks:
        end = min(start + chunk_size, n_channels)
        warped = F.grid_sample(
            flat[:, :, start:end].permute(2, 0, 1).contiguous()
            .to(device=device, dtype=torch.float32)[None],
            warp_grid,
            mode=mode, align_corners=True, padding_mode="border",
        )[0].permute(1, 2, 0)

        if use_external_output:
            out_flat[:, :, start:end] = (
                warped.cpu().numpy().astype(_out_np_dtype)
            )
        else:
            internal_output[:, :, start:end] = warped.to(
                device=target, dtype=out_dt,
            )

    if use_external_output:
        return output

    result = internal_output.reshape(original_shape)
    if return_numpy:
        return result.cpu().numpy() if result.is_cuda else result.numpy()
    return result


def generate_corrected_paired_datasets(
    dc: "DriftCorrection",
    *,
    mode: str = "bicubic",
    chunk_size: int | None = None,
    merge: bool = True,
    verbose: bool = False,
    output_a: np.ndarray | None = None,
    output_b: np.ndarray | None = None,
) -> PairedCorrectionResult:
    """Internal worker for the 4D-STEM branch of ``generate_corrected``.

    Applies the correction to both stored datasets, rotates the second
    into the first scan's coordinate frame, and optionally merges them.
    """
    datasets = dc._datasets
    if len(datasets) < 2:
        raise ValueError(
            f"Need at least 2 datasets for paired correction, "
            f"got {len(datasets)}"
        )

    corrected_a = apply_correction_to_dataset(
        dc, None, image_index=0, mode=mode, chunk_size=chunk_size,
        output=output_a, verbose=verbose,
    )
    corrected_b = apply_correction_to_dataset(
        dc, None, image_index=1, mode=mode, chunk_size=chunk_size,
        output=output_b, verbose=verbose,
    )

    # Rotate dataset B into dataset A's coordinate frame
    sd = dc.scan_direction_degrees
    delta = float((sd[1] - sd[0]) % 360)
    rot_k = round(delta / 90) % 4
    if rot_k != 0:
        corrected_b = np.rot90(corrected_b, k=rot_k, axes=(0, 1)).copy()

    merged = None
    if merge:
        if corrected_a.shape != corrected_b.shape:
            raise ValueError(
                f"Cannot merge: corrected_a shape {corrected_a.shape} "
                f"!= rotated corrected_b shape {corrected_b.shape}. "
                f"Paired scans must have compatible scan dimensions "
                f"after rotation."
            )
        # In-place to avoid float64 promotion of multi-GB arrays.
        merged = np.empty_like(corrected_a, dtype=np.float32)
        np.add(corrected_a, corrected_b, out=merged, dtype=np.float32)
        merged *= 0.5

    # Extract VDFs from the stored alignment images
    vdf_a = np.asarray(dc.imgs[0].array)
    vdf_b = np.asarray(dc.imgs[1].array)

    return PairedCorrectionResult(
        merged=merged,
        corrected_a=corrected_a,
        corrected_b=corrected_b,
        drift=dc,
        vdf_a=vdf_a,
        vdf_b=vdf_b,
    )
