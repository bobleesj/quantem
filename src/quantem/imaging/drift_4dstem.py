"""4-D STEM drift correction (and 3-D spectral dataset path).

The scan-image pipeline lives in :mod:`drift`; this module owns the
bigger hammer for any dataset shape with ``(scan_h, scan_w, …channels)``
layout — primarily 4-D STEM ``(H, W, det_h, det_w)``, also 3-D spectral
datasets ``(H, W, n_energy)`` from EDS / EELS. Both go through the same
shape-agnostic ``apply_correction_to_dataset`` loop (chunked GPU
``grid_sample`` with pre-allocated memmap output).

The 4-D STEM-specific helpers — ``compute_vdf``, virtual-detector
integration, scan virtual-image correction, ``view_corrected_dp``,
``view_corrected_vdfs``, the 4D-STEM merge ``generate_corrected_4dstem_collection``,
and the 4D-STEM result containers — live here.

Functions take an already-built :class:`DriftCorrection` as their first
argument (``dc``) so :class:`drift.DriftCorrection` stays focused on the
algorithm story.
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
class CorrectionResult:
    """Container returned by 0/90 4D-STEM collection correction.

    This result represents the scan-derived corrected coordinate system:
    both input 4D-STEM datasets are treated as drifted scans and corrected toward a
    shared consensus frame before optional diffraction-pattern-level merge.

    Attributes
    ----------
    corrected_4dstem_0, corrected_4dstem_1 : np.ndarray | torch.Tensor
        Per-side drift-corrected 4D-STEM datasets, scan-axis-leading layout.
        Dataset 1 has already been oriented into dataset 0's display frame.
    corrected_4dstem : np.ndarray | torch.Tensor | None
        Diffraction-pattern-level average of ``corrected_4dstem_0`` and the
        oriented ``corrected_4dstem_1``. ``None`` when ``merge=False``.
    raw_vdf_0, raw_vdf_1 : np.ndarray
        Raw pre-correction virtual-detector images used to estimate drift.
    scalar_corrected_vdf : np.ndarray | None
        Scan-derived corrected VDF computed by correcting the raw alignment
        VDFs with the same operator used for 4D-STEM channels. This is not an
        external ground truth; it is the scalar virtual-image result implied
        by the learned scan drift fields.
    drift : DriftCorrection
        Back-reference for plotting / introspection.
    """
    corrected_4dstem_0: np.ndarray | torch.Tensor
    corrected_4dstem_1: np.ndarray | torch.Tensor
    raw_vdf_0: np.ndarray
    raw_vdf_1: np.ndarray
    drift: object  # DriftCorrection — typed via TYPE_CHECKING above
    corrected_4dstem: np.ndarray | torch.Tensor | None = field(default=None)
    scalar_corrected_vdf: np.ndarray | None = field(default=None)

    def virtual_image(
        self,
        detector_mask: np.ndarray | torch.Tensor | None = None,
        *,
        reduce: str = "mean",
        source: str = "corrected_4dstem",
        chunk_rows: int | None = None,
    ) -> np.ndarray:
        """Integrate a virtual image from a corrected 4D-STEM dataset.

        Parameters
        ----------
        detector_mask : ndarray or torch.Tensor, optional
            Boolean mask over detector / channel axes. ``None`` integrates
            all detector pixels.
        reduce : {"mean", "sum"}
            Whether selected detector pixels are averaged or summed.
        source : {"corrected_4dstem", "corrected_4dstem_0", "corrected_4dstem_1"}
            Which corrected 4D-STEM dataset to integrate.
        chunk_rows : int or None
            Optional scan-row chunking for memory-constrained CPU inputs.
        """
        source_key = source.lower()
        if source_key == "corrected_4dstem":
            if self.corrected_4dstem is None:
                raise RuntimeError(
                    "No merged corrected 4D-STEM dataset is available because "
                    "this result was generated with merge=False."
                )
            dataset = self.corrected_4dstem
        elif source_key == "corrected_4dstem_0":
            dataset = self.corrected_4dstem_0
        elif source_key == "corrected_4dstem_1":
            dataset = self.corrected_4dstem_1
        else:
            raise ValueError(f"unknown corrected 4D-STEM source {source!r}")
        return integrate_virtual_detector(
            dataset, detector_mask=detector_mask, reduce=reduce,
            chunk_rows=chunk_rows,
        )

    def probe_positions(self, image_index: int = 0, **kwargs) -> np.ndarray:
        """Return drift-updated probe positions from the fitted correction."""
        return self.drift.probe_positions(image_index=image_index, **kwargs)


def _rot90_to_image0_frame(dc: "DriftCorrection", image_index: int = 1) -> int:
    """Return the scan-axis rot90 needed to show ``image_index`` like image 0."""
    delta = float(dc.scan_direction_degrees[image_index] - dc.scan_direction_degrees[0])
    return (-int(round(delta / 90.0))) % 4


def integrate_virtual_detector(
    dataset,
    detector_mask: np.ndarray | torch.Tensor | None = None,
    *,
    reduce: str = "mean",
    chunk_rows: int | None = None,
) -> np.ndarray:
    """Integrate a virtual image from a scan-axis-leading dataset.

    Parameters
    ----------
    dataset : ndarray or torch.Tensor, shape ``(H, W, ...channels)``
        3-D/4-D dataset with scan axes first.
    detector_mask : ndarray or torch.Tensor, optional
        Boolean mask over the trailing detector / channel axes. ``None``
        selects every channel.
    reduce : {"mean", "sum"}
        Average or sum selected detector pixels.
    chunk_rows : int or None
        Optional row chunk size. Torch inputs auto-chunk when omitted to
        avoid large widened accumulators.

    Returns
    -------
    np.ndarray, shape ``(H, W)``, dtype float32
    """
    if reduce not in {"mean", "sum"}:
        raise ValueError(f"reduce must be 'mean' or 'sum', got {reduce!r}")

    if isinstance(dataset, torch.Tensor):
        H, W = dataset.shape[:2]
        n_channels = int(np.prod(tuple(dataset.shape[2:])))
        flat = dataset.reshape(H, W, n_channels)
        if detector_mask is None:
            channel_idx = None
            n_selected = n_channels
        else:
            mask_t = torch.as_tensor(
                detector_mask, dtype=torch.bool, device=dataset.device,
            ).flatten()
            if mask_t.numel() != n_channels:
                raise ValueError(
                    f"detector_mask has {mask_t.numel()} pixels but dataset has "
                    f"{n_channels} detector/channel pixels"
                )
            channel_idx = mask_t.nonzero().squeeze(-1)
            n_selected = int(channel_idx.numel())
        if n_selected == 0:
            raise ValueError("detector_mask selects zero detector pixels")

        if torch.is_floating_point(dataset):
            sum_dtype = torch.float64
        elif dataset.dtype in (torch.int8, torch.int16, torch.uint8):
            sum_dtype = torch.int32
        else:
            sum_dtype = torch.int64
        if chunk_rows is None:
            bytes_per_row = W * n_selected * sum_dtype.itemsize
            chunk_rows = max(1, int(1e9 / max(bytes_per_row, 1)))
        out = torch.empty(H, W, dtype=torch.float32, device=dataset.device)
        for r0 in range(0, H, chunk_rows):
            r1 = min(r0 + chunk_rows, H)
            chunk = flat[r0:r1]
            if channel_idx is not None:
                chunk = chunk[..., channel_idx]
            summed = chunk.sum(dim=2, dtype=sum_dtype).to(torch.float32)
            if reduce == "mean":
                summed = summed / n_selected
            out[r0:r1] = summed
        return out.cpu().numpy()

    dataset_np = np.asarray(dataset)
    H, W = dataset_np.shape[:2]
    n_channels = int(np.prod(dataset_np.shape[2:]))
    flat = dataset_np.reshape(H, W, n_channels)
    if detector_mask is None:
        mask = None
        n_selected = n_channels
    else:
        mask = np.asarray(detector_mask, dtype=bool).ravel()
        if mask.size != n_channels:
            raise ValueError(
                f"detector_mask has {mask.size} pixels but dataset has "
                f"{n_channels} detector/channel pixels"
            )
        n_selected = int(mask.sum())
    if n_selected == 0:
        raise ValueError("detector_mask selects zero detector pixels")
    sum_dtype = np.uint64 if np.issubdtype(dataset_np.dtype, np.integer) else np.float64

    def _reduce(chunk):
        if mask is not None:
            chunk = chunk[..., mask]
        summed = chunk.sum(axis=2, dtype=sum_dtype)
        if reduce == "mean":
            summed = summed / n_selected
        return summed.astype(np.float32)

    if chunk_rows is None:
        return _reduce(flat)
    out = np.empty((H, W), dtype=np.float32)
    for r0 in range(0, H, chunk_rows):
        r1 = min(r0 + chunk_rows, H)
        out[r0:r1] = _reduce(flat[r0:r1])
    return out


@torch.inference_mode()
def correct_virtual_images(
    dc: "DriftCorrection",
    image_0,
    image_1,
) -> dict[str, np.ndarray]:
    """Correct two scalar virtual images like matching 4D-STEM channels.

    Each scalar image is treated as a one-channel dataset with scan axes first,
    corrected with the same ``grid_sample`` operator used for every diffraction
    pixel, and image 1 is oriented into image 0's display frame before the
    average. The returned ``corrected_image`` should therefore match integrating
    the same virtual detector from ``generate_corrected_4dstem()`` output,
    up to output quantization.
    """
    if not hasattr(dc, "_initial_knots"):
        raise RuntimeError(
            "correct_virtual_images() requires preprocess() and "
            "align_affine() first."
        )
    if len(dc.imgs) != 2:
        raise ValueError(
            "correct_virtual_images() expects exactly two scan images"
        )
    images = [
        np.asarray(image_0, dtype=np.float32),
        np.asarray(image_1, dtype=np.float32),
    ]
    if images[0].shape != dc.imgs[0].shape or images[1].shape != dc.imgs[1].shape:
        raise ValueError(
            "virtual image shapes must match the raw scan images used for drift "
            f"alignment: got {images[0].shape}, {images[1].shape}; expected "
            f"{dc.imgs[0].shape}, {dc.imgs[1].shape}"
        )

    components = []
    for image_index, image in enumerate(images):
        image_t = torch.as_tensor(
            image[..., None],
            device=dc._device,
            dtype=torch.float32,
        )
        corrected = apply_correction_to_dataset(
            dc,
            image_t,
            image_index=image_index,
            mode="bilinear",
            chunk_size=1,
            output_dtype=torch.float32,
            output_device=dc._device,
        )[..., 0]
        if image_index == 1:
            rot_k = _rot90_to_image0_frame(dc, image_index=1)
            if rot_k:
                corrected = torch.rot90(corrected, k=rot_k, dims=(0, 1))
        components.append(corrected)

    merged = (components[0] + components[1]) * 0.5

    return {
        "corrected_image": merged.detach().cpu().numpy().astype(np.float32),
        "corrected_image_0": components[0].detach().cpu().numpy().astype(np.float32),
        "corrected_image_1": components[1].detach().cpu().numpy().astype(np.float32),
    }


def compute_vdf(
    ds_4d,
    chunk_rows: int | None = None,
) -> np.ndarray:
    """Compute a virtual dark-field image from a 4D-STEM dataset.

    Averages over the detector dimensions to produce a 2D scan image.
    Accepts ``np.ndarray`` / ``np.memmap`` (CPU path) or ``torch.Tensor``
    on device (computes the reduction in place, then syncs back as a
    small float32 numpy array).  The torch path is preferred when the
    dataset is already on device: avoids a multi-GB device-to-host transfer
    just to compute a megabyte-scale summary.

    Parameters
    ----------
    ds_4d : np.ndarray or torch.Tensor, shape ``(H, W, det_h, det_w)``
        4D-STEM dataset.  numpy can be a ``np.memmap``.
    chunk_rows : int or None
        Number of scan rows to process at a time (numpy path only).
        ``None`` loads everything at once (fastest for in-memory).

    Returns
    -------
    np.ndarray, shape ``(H, W)``, dtype float32
    """
    if isinstance(ds_4d, torch.Tensor):
        H, W = ds_4d.shape[:2]
        det_pixels = 1
        for d in range(2, ds_4d.ndim):
            det_pixels *= ds_4d.shape[d]
        # Widen the accumulator just enough to avoid overflow in the
        # per-pixel detector sum.  int8/int16/uint8 fit safely in int32
        # for det_pixels up to ~65k; wider integer inputs need int64.
        if torch.is_floating_point(ds_4d):
            sum_dtype = torch.float64
        elif ds_4d.dtype in (torch.int8, torch.int16, torch.uint8):
            sum_dtype = torch.int32
        else:
            sum_dtype = torch.int64
        # torch's .sum(dtype=...) materializes a widened copy of the
        # full input in some builds; chunking caps the transient at one
        # row-block instead of the whole dataset.
        if chunk_rows is None:
            # Cap transient at ~1 GB per chunk in the chosen accumulator.
            bytes_per_row = W * det_pixels * sum_dtype.itemsize
            chunk_rows = max(1, int(1e9 / bytes_per_row))
        totals = torch.empty(H, W, dtype=torch.float32, device=ds_4d.device)
        for i in range(0, H, chunk_rows):
            j = min(i + chunk_rows, H)
            chunk = ds_4d[i:j].reshape(j - i, W, det_pixels)
            totals[i:j] = chunk.sum(dim=2, dtype=sum_dtype).to(torch.float32) / det_pixels
        return totals.cpu().numpy()

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
    mode: str = "bilinear",
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

    drift = dc.drift_field(idx).to(device=device, dtype=torch.float32)
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
    # Default the output to the input's device so the pipeline stays
    # in place; explicit ``output_device`` overrides.
    if use_external_output:
        target = torch.device("cpu")
    elif output_device is not None:
        target = torch.device(output_device)
        if target.type == "cuda":
            target = device
    elif isinstance(ds_4d, torch.Tensor) and ds_4d.is_cuda:
        target = device
    else:
        target = torch.device("cpu")
    return_numpy = (
        use_external_output
        or (is_numpy and output_device is None)
    )

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

        # Integer casts truncate. Round first or low-count detector pixels
        # collapse to zero after interpolation.
        is_int = isinstance(out_dt, torch.dtype) and not out_dt.is_floating_point
        if is_int:
            warped_cast = warped.round().clamp_(
                torch.iinfo(out_dt).min, torch.iinfo(out_dt).max)
        else:
            warped_cast = warped
        if use_external_output:
            out_flat[:, :, start:end] = (
                warped_cast.cpu().numpy().astype(_out_np_dtype)
            )
        else:
            internal_output[:, :, start:end] = warped_cast.to(
                device=target, dtype=out_dt,
            )

    if use_external_output:
        return output

    result = internal_output.reshape(original_shape)
    if return_numpy:
        return result.cpu().numpy() if result.is_cuda else result.numpy()
    return result


def generate_corrected_4dstem_collection(
    dc: "DriftCorrection",
    *,
    mode: str = "bilinear",
    chunk_size: int | None = None,
    merge: bool = True,
    verbose: bool = False,
    output_0: np.ndarray | None = None,
    output_1: np.ndarray | None = None,
    output_dtype: torch.dtype | np.dtype | str | None = None,
    output_device: str | torch.device | None = None,
) -> CorrectionResult:
    """Internal worker for 0/90 4D-STEM collection correction.

    Applies the learned scan-derived drift fields to both stored 4D-STEM
    datasets, orients corrected dataset 1 into dataset 0's display frame,
    and optionally merges the corrected diffraction patterns.
    """
    datasets = dc._datasets
    if dc._datasets_consumed:
        raise RuntimeError(
            "Raw datasets were already released to free device memory "
            "during a prior generate_corrected call. Construct a new "
            "DriftCorrection to re-correct."
        )
    if len(datasets) < 2:
        raise ValueError(
            f"Need at least 2 datasets for scan collection correction, "
            f"got {len(datasets)}"
        )

    # When inputs are device-resident, release each raw dataset as soon as
    # its corrected output exists; otherwise we hold four full datasets
    # simultaneously, which exceeds device memory for multi-GB scan collections.
    inputs_on_device = (
        isinstance(datasets[0], torch.Tensor) and datasets[0].is_cuda
        and isinstance(datasets[1], torch.Tensor) and datasets[1].is_cuda
    )

    corrected_4dstem_0 = apply_correction_to_dataset(
        dc, None, image_index=0, mode=mode, chunk_size=chunk_size,
        output_dtype=output_dtype, output_device=output_device,
        output=output_0, verbose=verbose,
    )
    if inputs_on_device:
        dc._datasets[0] = None
        torch.cuda.empty_cache()
    corrected_4dstem_1 = apply_correction_to_dataset(
        dc, None, image_index=1, mode=mode, chunk_size=chunk_size,
        output_dtype=output_dtype, output_device=output_device,
        output=output_1, verbose=verbose,
    )
    if inputs_on_device:
        dc._datasets[1] = None
        dc._datasets_consumed = True
        torch.cuda.empty_cache()

    rot_k = _rot90_to_image0_frame(dc, image_index=1)
    if rot_k:
        if isinstance(corrected_4dstem_1, torch.Tensor):
            corrected_4dstem_1 = torch.rot90(
                corrected_4dstem_1, k=rot_k, dims=(0, 1),
            )
        else:
            corrected_4dstem_1 = np.rot90(
                corrected_4dstem_1, k=rot_k, axes=(0, 1),
            ).copy()

    corrected_4dstem = None
    if merge:
        if corrected_4dstem_0.shape != corrected_4dstem_1.shape:
            raise ValueError(
                f"Cannot merge: corrected_4dstem_0 shape {corrected_4dstem_0.shape} "
                f"!= corrected_4dstem_1 shape {corrected_4dstem_1.shape}. "
                f"Scan collection must have compatible scan dimensions "
                f"after correction and scan-angle rotation."
            )
        if isinstance(corrected_4dstem_0, torch.Tensor):
            if corrected_4dstem_0.is_floating_point():
                corrected_4dstem = (corrected_4dstem_0 + corrected_4dstem_1) * 0.5
            else:
                # Chunked integer merge over rows. Avoids promoting full
                # dataset to float32 (would be 4x the input bytes; a 19 GB
                # uint16 dataset becomes 76 GB transient, which OOMs on 96 GB GPU).
                # int32 sum fits in 2× input bytes per row chunk, then //2 + cast.
                corrected_4dstem = torch.empty_like(corrected_4dstem_0)
                Hm = corrected_4dstem_0.shape[0]
                row_block = max(1, min(32, Hm))
                for r0 in range(0, Hm, row_block):
                    r1 = min(r0 + row_block, Hm)
                    a = corrected_4dstem_0[r0:r1].to(torch.int32)
                    a += corrected_4dstem_1[r0:r1].to(torch.int32)
                    a >>= 1  # divide by 2 (round-toward-zero for non-negative ints)
                    corrected_4dstem[r0:r1] = a.clamp_(
                        0, torch.iinfo(corrected_4dstem_0.dtype).max
                    ).to(corrected_4dstem_0.dtype)
                    del a
        else:
            corrected_4dstem = np.empty_like(corrected_4dstem_0, dtype=np.float32)
            np.add(corrected_4dstem_0, corrected_4dstem_1, out=corrected_4dstem, dtype=np.float32)
            corrected_4dstem *= 0.5

    # Extract raw VDFs from the stored alignment images. The scan collection
    # reference is the scalar channel correction implied by the learned scan
    # drift fields, not an external ground truth.
    alignment_vdf_0 = np.asarray(dc.imgs[0].array)
    alignment_vdf_1 = np.asarray(dc.imgs[1].array)
    scalar_corrected_vdf = correct_virtual_images(
        dc,
        alignment_vdf_0,
        alignment_vdf_1,
    )["corrected_image"]

    return CorrectionResult(
        corrected_4dstem=corrected_4dstem,
        corrected_4dstem_0=corrected_4dstem_0,
        corrected_4dstem_1=corrected_4dstem_1,
        drift=dc,
        raw_vdf_0=alignment_vdf_0,
        raw_vdf_1=alignment_vdf_1,
        scalar_corrected_vdf=scalar_corrected_vdf,
    )


def view_corrected_vdfs(
    dc: "DriftCorrection",
    *,
    image_index: int = 1,
    df_inner_factor: float = 1.5,
    chunk_rows: int = 16,
    show: bool = True,
    cmap: str = "magma",
    **imshow_kwargs,
):
    """Compute drift-corrected BF + DF VDFs from the 4D-STEM dataset.

    Fits the probe circle on the mean DP, then re-integrates the dataset under
    a BF disk mask and an annulus DF mask (radius > ``df_inner_factor * R``),
    and warps each into the corrected scan frame.

    All reductions stay on GPU; only the small mean DP and final 2D VDFs
    move to host.

    Returns
    -------
    bf_corrected, df_corrected : np.ndarray of shape (scan_h, scan_w)
    """
    if not dc._is_4dstem_collection:
        raise RuntimeError(
            "view_corrected_vdfs requires a 4D-STEM collection DriftCorrection.")
    if dc._datasets_consumed or dc._datasets[image_index] is None:
        raise RuntimeError(
            f"Raw dataset for image {image_index} was released. Construct a new "
            f"DriftCorrection to compute VDFs.")

    from quantem.core.utils.diffractive_imaging_utils import fit_probe_circle
    from quantem.imaging.drift_align import backward_warp

    ds = dc._datasets[image_index]
    if not isinstance(ds, torch.Tensor):
        ds = torch.as_tensor(ds, device=dc._device)

    H, W, det_h, det_w = ds.shape
    ds_flat = ds.view(H, W, det_h * det_w)

    # Stream the mean DP and the BF/DF VDFs in one chunked pass over scan rows.
    # Single int64 promote per chunk avoids the 72 GB transient that
    # ds.sum(dtype=int64) would allocate up front.
    mean_dp_acc = torch.zeros(det_h * det_w, dtype=torch.int64, device=ds.device)
    bf_vdf = torch.zeros(H, W, dtype=torch.float32, device=ds.device)
    df_vdf = torch.zeros_like(bf_vdf)

    yy, xx = torch.meshgrid(
        torch.arange(det_h, device=ds.device, dtype=torch.float32),
        torch.arange(det_w, device=ds.device, dtype=torch.float32),
        indexing='ij',
    )
    # Need probe geometry before per-chunk masking, but probe fit needs the
    # mean DP, so do mean DP in pass 1 then VDFs in pass 2.
    for r0 in range(0, H, chunk_rows):
        r1 = min(r0 + chunk_rows, H)
        mean_dp_acc += ds_flat[r0:r1].to(torch.int64).sum(dim=(0, 1))
    mean_dp = (mean_dp_acc.view(det_h, det_w).float() / (H * W))
    yc, xc, radius = fit_probe_circle(mean_dp.cpu().numpy(), show=False)

    r_sq = (yy - yc) ** 2 + (xx - xc) ** 2
    bf_idx = (r_sq < radius ** 2).flatten().nonzero().squeeze(-1)
    df_idx = (r_sq > (df_inner_factor * radius) ** 2).flatten().nonzero().squeeze(-1)

    for r0 in range(0, H, chunk_rows):
        r1 = min(r0 + chunk_rows, H)
        chunk = ds_flat[r0:r1].to(torch.int64)
        bf_vdf[r0:r1] = chunk[..., bf_idx].sum(dim=-1).float()
        df_vdf[r0:r1] = chunk[..., df_idx].sum(dim=-1).float()

    bf_raw = bf_vdf.cpu().numpy()
    df_raw = df_vdf.cpu().numpy()
    drift = dc.drift_field(image_index)
    bf_corrected = backward_warp(bf_vdf, drift=drift, mode='bicubic').cpu().numpy()
    df_corrected = backward_warp(df_vdf, drift=drift, mode='bicubic').cpu().numpy()

    if show:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle
        fig, axes = plt.subplots(1, 5, figsize=(25, 5))
        axes[0].imshow(mean_dp.cpu().numpy(), cmap=cmap, **imshow_kwargs)
        axes[0].add_patch(Circle((xc, yc), radius, fc='none', ec='cyan', lw=2,
                                  label=f'BF R={radius:.1f}'))
        axes[0].add_patch(Circle((xc, yc), df_inner_factor * radius, fc='none',
                                  ec='yellow', lw=2,
                                  label=f'DF inner {df_inner_factor}R'))
        axes[0].set_title(f'mean DP — probe R={radius:.1f}px')
        axes[0].legend(loc='upper right', fontsize=8)
        axes[1].imshow(bf_raw, cmap=cmap, **imshow_kwargs)
        axes[1].set_title(f'BF uncorrected (image {image_index})')
        axes[2].imshow(bf_corrected, cmap=cmap, **imshow_kwargs)
        axes[2].set_title(f'BF corrected (image {image_index})')
        axes[3].imshow(df_raw, cmap=cmap, **imshow_kwargs)
        axes[3].set_title(f'DF uncorrected (r > {df_inner_factor}R)')
        axes[4].imshow(df_corrected, cmap=cmap, **imshow_kwargs)
        axes[4].set_title(f'DF corrected (r > {df_inner_factor}R)')
        for ax in axes:
            ax.set_xticks([]); ax.set_yticks([])
        plt.tight_layout()
        plt.show()

    return bf_corrected, df_corrected


def _dataset_to_np(dataset):
    if isinstance(dataset, torch.Tensor):
        return dataset.cpu().numpy() if dataset.is_cuda else dataset.numpy()
    return np.asarray(dataset)


def _sample_dp(cube_np, drift_t, r, c):
    """Bilinear-sample a DP at the drift-corrected source ``(r-dr, c-dc)``."""
    dr = float(drift_t[0, r] if drift_t.ndim == 2 else drift_t[0, r, c])
    dc_off = float(drift_t[1, r] if drift_t.ndim == 2 else drift_t[1, r, c])
    src_r, src_c = r - dr, c - dc_off
    H, W = cube_np.shape[:2]
    r0 = max(0, min(int(np.floor(src_r)), H - 2))
    c0 = max(0, min(int(np.floor(src_c)), W - 2))
    fr, fc = src_r - r0, src_c - c0
    dp = (
        (1 - fr) * (1 - fc) * cube_np[r0,     c0    ].astype(np.float32)
        + fr     * (1 - fc) * cube_np[r0 + 1, c0    ].astype(np.float32)
        + (1 - fr) * fc     * cube_np[r0,     c0 + 1].astype(np.float32)
        + fr     * fc       * cube_np[r0 + 1, c0 + 1].astype(np.float32)
    )
    return dp, (dr, dc_off)


def view_corrected_dp(
    dc: "DriftCorrection",
    *,
    scan_positions: list[tuple[int, int]] | tuple[int, int] | None = None,
    image_index: int = 1,
    show: bool = True,
    cmap: str = "magma",
    log_scale: bool = False,
    **imshow_kwargs,
):
    """Sanity-check drift correction on one or more diffraction patterns.

    For each scan position, pulls the raw DP and bilinear-samples the
    drift-corrected DP. Confirms the learned drift shifts real DPs (not
    noise) without paying for the full-dataset warp.

    Parameters
    ----------
    dc : DriftCorrection
        Must be a 4D-STEM collection correction (raises otherwise).
    scan_positions : (row, col), list of (row, col), or None
        One or more scan-frame indices to probe. Defaults to the brightest
        VDF pixel of the reference dataset.
    image_index : {0, 1}
        Which dataset to pull from (0 = reference, 1 = target).
    show : bool
        If True, plots VDF + raw + corrected + |diff| (one row per position).
    cmap : str
        matplotlib colormap (default 'magma').
    log_scale : bool
        Use log scale on DP panels (default False).
    **imshow_kwargs
        Forwarded to ``ax.imshow`` (e.g. vmin, vmax, interpolation).

    Returns
    -------
    list of (dp_raw, dp_corrected, (dr, dc)) tuples, one per position.
    """
    if not dc._is_4dstem_collection:
        raise RuntimeError(
            "view_corrected_dp requires a 4D-STEM collection DriftCorrection.")
    if dc._datasets_consumed or dc._datasets[image_index] is None:
        raise RuntimeError(
            f"Raw dataset for image {image_index} was released. Construct a new "
            f"DriftCorrection to view DPs.")

    if scan_positions is None:
        vdf_ref = np.asarray(dc.imgs[0].array)
        r_pick, c_pick = map(int, np.unravel_index(int(vdf_ref.argmax()), vdf_ref.shape))
        positions = [(r_pick, c_pick)]
    elif isinstance(scan_positions, tuple) and len(scan_positions) == 2 and np.isscalar(scan_positions[0]):
        positions = [(int(scan_positions[0]), int(scan_positions[1]))]
    else:
        positions = [(int(r), int(c)) for r, c in scan_positions]

    target_cube = _dataset_to_np(dc._datasets[image_index])
    target_drift = dc.drift_field(image_index)
    show_ref = image_index != 0 and dc._datasets[0] is not None
    ref_cube = _dataset_to_np(dc._datasets[0]) if show_ref else None
    ref_drift = dc.drift_field(0) if show_ref else None

    # 90° rotation between scan_direction[0] and scan_direction[image_index]
    # → image_index's scan grid is rotated. To probe the SAME physical point
    # as image 0's (r, c), transform coords through the inverse rotation.
    sd = dc.scan_direction_degrees
    rot_k = _rot90_to_image0_frame(dc, image_index=image_index)
    H_t, W_t = target_cube.shape[:2]
    H_r, W_r = (ref_cube.shape[:2] if ref_cube is not None else (H_t, W_t))

    def rot_pos(r, c, k):
        """Rotate (r, c) on H_r×W_r grid by k*90° CCW into H_t×W_t."""
        for _ in range(k % 4):
            r, c = c, H_r - 1 - r
        return r, c

    results = []
    for r, c in positions:
        # ref samples in image 0 frame at (r, c)
        dp_ref = _sample_dp(ref_cube, ref_drift, r, c)[0] if show_ref else None
        # target samples in image_index frame at the rotated position
        rt, ct = rot_pos(r, c, (-rot_k) % 4)
        dp_target, drift_offset = _sample_dp(target_cube, target_drift, rt, ct)
        # Rotate target DP back so its display orientation matches image 0
        dp_target_display = np.rot90(dp_target, k=rot_k) if rot_k else dp_target
        results.append((dp_ref, dp_target_display, drift_offset))

    if show:
        import matplotlib.pyplot as plt
        import matplotlib.patheffects as path_effects
        from matplotlib.colors import LogNorm
        from quantem.imaging.drift_align import backward_warp
        # Show image 0's corrected VDF (probe positions are in image 0 frame).
        vdf_t = torch.as_tensor(dc.imgs_t[0], dtype=torch.float32,
                                 device=dc._device)
        vdf_corrected = backward_warp(vdf_t, drift=dc.drift_field(0),
                                       mode='bicubic').cpu().numpy()

        n_rows = len(positions)
        n_cols = 3 if show_ref else 2
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows),
                                 squeeze=False)
        for row_i, ((r_pick, c_pick), (dp_ref, dp_cor, (dr, dc_off))) in enumerate(zip(positions, results)):
            src_r, src_c = r_pick - dr, c_pick - dc_off
            dp_norm = LogNorm(vmin=max(dp_cor.min(), 1e-3)) if log_scale else None
            ax_row = axes[row_i]
            ax_row[0].imshow(vdf_corrected, cmap=cmap, **imshow_kwargs)
            for j, (rj, cj) in enumerate(positions):
                ax_row[0].scatter(cj, rj, s=140, marker='o', facecolors='red',
                                   edgecolors='white', linewidths=1.5, zorder=5)
                ax_row[0].annotate(f'P{j}', (cj, rj), color='white',
                                    fontsize=10, ha='left', va='bottom',
                                    xytext=(8, 4), textcoords='offset points',
                                    path_effects=[path_effects.withStroke(
                                        linewidth=2, foreground='black')])
            ax_row[0].set_title(f'corrected VDF — P{row_i}=({r_pick},{c_pick})')
            col = 1
            if dp_ref is not None:
                ax_row[col].imshow(dp_ref, cmap=cmap, norm=dp_norm, **imshow_kwargs)
                ax_row[col].set_title(f'corrected DP image 0 at ({r_pick},{c_pick})')
                col += 1
            ax_row[col].imshow(dp_cor, cmap=cmap, norm=dp_norm, **imshow_kwargs)
            ax_row[col].set_title(f'corrected DP image {image_index} at ({src_r:.1f},{src_c:.1f})')
            for ax in ax_row:
                ax.set_xticks([]); ax.set_yticks([])
        plt.tight_layout()
        plt.show()
    return results if len(results) > 1 else results[0]
