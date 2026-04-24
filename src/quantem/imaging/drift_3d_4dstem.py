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
    corrected_a, corrected_b : np.ndarray | torch.Tensor
        Per-side drift-corrected datasets, scan-axis-leading layout.
        Type matches the input (numpy when the source arrays were numpy,
        torch when ``output_device`` was specified or inputs were tensors).
    vdf_a, vdf_b : np.ndarray
        Pre-correction virtual-detector images, ``(scan_h, scan_w)``.
    merged : np.ndarray | torch.Tensor | None
        Half-sum of corrected_a + corrected_b in dataset A's frame.
        Always a *distinct* array/tensor from ``corrected_a``.
        ``None`` when ``merge=False`` (heavy-data callers that defer
        the merge to disk).
    drift : DriftCorrection
        Back-reference for plotting / introspection (``result.drift.knots``,
        ``result.drift.print_drift_stats()``, etc.).
    """
    corrected_a: np.ndarray | torch.Tensor
    corrected_b: np.ndarray | torch.Tensor
    vdf_a: np.ndarray
    vdf_b: np.ndarray
    drift: object  # DriftCorrection — typed via TYPE_CHECKING above
    merged: np.ndarray | torch.Tensor | None = field(default=None)


def compute_vdf(
    ds_4d,
    chunk_rows: int | None = None,
) -> np.ndarray:
    """Compute a virtual dark-field image from a 4D-STEM dataset.

    Averages over the detector dimensions to produce a 2D scan image.
    Accepts ``np.ndarray`` / ``np.memmap`` (CPU path) or ``torch.Tensor``
    on device (computes the reduction in place, then syncs back as a
    small float32 numpy array).  The torch path is preferred when the
    cube is already on device: avoids a multi-GB device-to-host transfer
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
        # row-block instead of the whole cube.
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
    mode: str = "bilinear",
    chunk_size: int | None = None,
    merge: bool = True,
    verbose: bool = False,
    output_a: np.ndarray | None = None,
    output_b: np.ndarray | None = None,
    output_dtype: torch.dtype | np.dtype | str | None = None,
    output_device: str | torch.device | None = None,
) -> PairedCorrectionResult:
    """Internal worker for the 4D-STEM branch of ``generate_corrected``.

    Applies the correction to both stored datasets, rotates the second
    into the first scan's coordinate frame, and optionally merges them.
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
            f"Need at least 2 datasets for paired correction, "
            f"got {len(datasets)}"
        )

    # When inputs are device-resident, release each raw cube as soon as
    # its corrected output exists; otherwise we hold four full cubes
    # simultaneously, which exceeds device memory for paired multi-GB scans.
    inputs_on_device = (
        isinstance(datasets[0], torch.Tensor) and datasets[0].is_cuda
        and isinstance(datasets[1], torch.Tensor) and datasets[1].is_cuda
    )

    corrected_a = apply_correction_to_dataset(
        dc, None, image_index=0, mode=mode, chunk_size=chunk_size,
        output_dtype=output_dtype, output_device=output_device,
        output=output_a, verbose=verbose,
    )
    if inputs_on_device:
        dc._datasets[0] = None
        torch.cuda.empty_cache()
    corrected_b = apply_correction_to_dataset(
        dc, None, image_index=1, mode=mode, chunk_size=chunk_size,
        output_dtype=output_dtype, output_device=output_device,
        output=output_b, verbose=verbose,
    )
    if inputs_on_device:
        dc._datasets[1] = None
        dc._datasets_consumed = True
        torch.cuda.empty_cache()

    # Rotate dataset B into dataset A's coordinate frame.
    sd = dc.scan_direction_degrees
    delta = float((sd[1] - sd[0]) % 360)
    rot_k = round(delta / 90) % 4
    if rot_k != 0:
        if isinstance(corrected_b, torch.Tensor):
            # rot90 returns a strided view; add_ below reads it directly,
            # so we skip the full-size .contiguous() write.
            corrected_b = torch.rot90(corrected_b, k=rot_k, dims=(0, 1))
        else:
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
        if isinstance(corrected_a, torch.Tensor):
            merged = (corrected_a.float() + corrected_b.float()) * 0.5
            if corrected_a.is_floating_point():
                merged = merged.to(corrected_a.dtype)
            else:
                merged = merged.round().clamp_(
                    0, torch.iinfo(corrected_a.dtype).max
                ).to(corrected_a.dtype)
        else:
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


def view_corrected_vdfs(
    dc: "DriftCorrection",
    *,
    image_index: int = 1,
    df_inner_factor: float = 1.5,
    chunk_rows: int = 32,
    show: bool = True,
    cmap: str = "magma",
    **imshow_kwargs,
):
    """Compute drift-corrected BF + DF VDFs from the 4D-STEM cube.

    Fits the probe circle on the mean DP, then re-integrates the cube under
    a BF disk mask and an annulus DF mask (radius > ``df_inner_factor * R``),
    and warps each into the corrected scan frame.

    All reductions stay on GPU; only the small mean DP and final 2D VDFs
    move to host.

    Returns
    -------
    bf_corrected, df_corrected : np.ndarray of shape (scan_h, scan_w)
    """
    if not dc.is_paired_4dstem:
        raise RuntimeError(
            "view_corrected_vdfs requires a paired 4D-STEM DriftCorrection.")
    if dc._datasets_consumed or dc._datasets[image_index] is None:
        raise RuntimeError(
            f"Raw cube for image {image_index} was released. Construct a new "
            f"DriftCorrection to compute VDFs.")

    from quantem.core.utils.diffractive_imaging_utils import fit_probe_circle
    from quantem.imaging.drift_align import backward_warp

    cube = dc._datasets[image_index]
    if not isinstance(cube, torch.Tensor):
        cube = torch.as_tensor(cube, device=dc._device)

    H, W, det_h, det_w = cube.shape
    mean_dp = cube.float().mean(dim=(0, 1))
    yc, xc, radius = fit_probe_circle(mean_dp.cpu().numpy(), show=False)

    yy, xx = torch.meshgrid(
        torch.arange(det_h, device=cube.device, dtype=torch.float32),
        torch.arange(det_w, device=cube.device, dtype=torch.float32),
        indexing='ij',
    )
    r_sq = (yy - yc) ** 2 + (xx - xc) ** 2
    bf_idx = (r_sq < radius ** 2).flatten().nonzero().squeeze(-1)
    df_idx = (r_sq > (df_inner_factor * radius) ** 2).flatten().nonzero().squeeze(-1)

    cube_flat = cube.view(H, W, det_h * det_w)
    bf_vdf = torch.zeros(H, W, device=cube.device, dtype=torch.float32)
    df_vdf = torch.zeros_like(bf_vdf)
    for r0 in range(0, H, chunk_rows):
        r1 = min(r0 + chunk_rows, H)
        chunk = cube_flat[r0:r1].to(torch.int64)
        bf_vdf[r0:r1] = chunk[..., bf_idx].sum(dim=-1).float()
        df_vdf[r0:r1] = chunk[..., df_idx].sum(dim=-1).float()

    drift = dc.drift_field(image_index)
    bf_corrected = backward_warp(bf_vdf, drift=drift, mode='bicubic').cpu().numpy()
    df_corrected = backward_warp(df_vdf, drift=drift, mode='bicubic').cpu().numpy()

    if show:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(mean_dp.cpu().numpy(), cmap=cmap, **imshow_kwargs)
        axes[0].add_patch(Circle((xc, yc), radius, fc='none', ec='cyan', lw=2,
                                  label=f'BF R={radius:.1f}'))
        axes[0].add_patch(Circle((xc, yc), df_inner_factor * radius, fc='none',
                                  ec='yellow', lw=2,
                                  label=f'DF inner {df_inner_factor}R'))
        axes[0].set_title(f'mean DP — probe R={radius:.1f}px')
        axes[0].legend(loc='upper right', fontsize=8)
        axes[1].imshow(bf_corrected, cmap=cmap, **imshow_kwargs)
        axes[1].set_title('corrected BF')
        axes[2].imshow(df_corrected, cmap=cmap, **imshow_kwargs)
        axes[2].set_title(f'corrected DF (r > {df_inner_factor}R)')
        for ax in axes:
            ax.set_xticks([]); ax.set_yticks([])
        plt.tight_layout()
        plt.show()

    return bf_corrected, df_corrected


def _cube_to_np(cube):
    if isinstance(cube, torch.Tensor):
        return cube.cpu().numpy() if cube.is_cuda else cube.numpy()
    return np.asarray(cube)


def _sample_dp(cube_np, drift_t, r, c):
    """Bilinear-sample a DP from cube at the drift-corrected source ``(r-dr, c-dc)``."""
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
    noise) without paying for the full-cube warp.

    Parameters
    ----------
    dc : DriftCorrection
        Must be a paired-4DSTEM correction (raises otherwise).
    scan_positions : (row, col), list of (row, col), or None
        One or more scan-frame indices to probe. Defaults to the brightest
        VDF pixel of the reference cube.
    image_index : {0, 1}
        Which cube to pull from (0 = reference, 1 = target).
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
    if not dc.is_paired_4dstem:
        raise RuntimeError(
            "view_corrected_dp requires a paired 4D-STEM DriftCorrection.")
    if dc._datasets_consumed or dc._datasets[image_index] is None:
        raise RuntimeError(
            f"Raw cube for image {image_index} was released. Construct a new "
            f"DriftCorrection to view DPs.")

    if scan_positions is None:
        vdf_ref = np.asarray(dc.imgs[0].array)
        r_pick, c_pick = map(int, np.unravel_index(int(vdf_ref.argmax()), vdf_ref.shape))
        positions = [(r_pick, c_pick)]
    elif isinstance(scan_positions, tuple) and len(scan_positions) == 2 and np.isscalar(scan_positions[0]):
        positions = [(int(scan_positions[0]), int(scan_positions[1]))]
    else:
        positions = [(int(r), int(c)) for r, c in scan_positions]

    target_cube = _cube_to_np(dc._datasets[image_index])
    target_drift = dc.drift_field(image_index)
    show_ref = image_index != 0 and dc._datasets[0] is not None
    ref_cube = _cube_to_np(dc._datasets[0]) if show_ref else None
    ref_drift = dc.drift_field(0) if show_ref else None

    results = []
    for r, c in positions:
        dp_target, drift_offset = _sample_dp(target_cube, target_drift, r, c)
        dp_ref = _sample_dp(ref_cube, ref_drift, r, c)[0] if show_ref else None
        results.append((dp_ref, dp_target, drift_offset))

    if show:
        import matplotlib.pyplot as plt
        import matplotlib.patheffects as path_effects
        from matplotlib.colors import LogNorm
        from quantem.imaging.drift_align import backward_warp
        vdf_t = torch.as_tensor(dc.imgs_t[image_index], dtype=torch.float32,
                                 device=dc._device)
        vdf_corrected = backward_warp(vdf_t, drift=target_drift,
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
