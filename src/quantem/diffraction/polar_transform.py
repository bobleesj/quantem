from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from tqdm import tqdm

from quantem.core.datastructures.dataset4dstem import Dataset4dstem
from quantem.core.datastructures.polar4dstem import Polar4dstem
from quantem.core.utils.utils import to_numpy

# Standard DPs use (row, col) convention. Polar coordinates use (phi, r_pix),
# grid_sample's grid tensor requires them to be ordered (col, row)
# but is noted where the call occures

_MEAN_DP_CHUNK_BYTES = 432 * 1024 * 1024
_ORIGIN_CHUNK_BYTES = 112 * 1024 * 1024
_FLOAT32_EXACT_INTEGER_LIMIT = 2**24


def _is_torch_tensor(arr: Any) -> bool:
    """Array is a torch tensor (any device: CPU, CUDA, MPS, ...)."""
    return isinstance(arr, torch.Tensor)


def _polar_step(
    dp_f: torch.Tensor,
    row_origins: torch.Tensor,
    col_origins: torch.Tensor,
    base_col_norm: torch.Tensor,
    base_row_norm: torch.Tensor,
    col_norm_scale: float,
    row_norm_scale: float,
) -> torch.Tensor:
    """Per-batch polar-sample inner step: build the sampling grid and run
    grid_sample. ``dp_f`` must already be float32 with shape ``(B, 1, H, W)`` so
    the compiled variant stays pure-float and works on MPS.
    """
    grid_col = base_col_norm.unsqueeze(0) + (col_origins * col_norm_scale - 1.0)[:, None, None]
    grid_row = base_row_norm.unsqueeze(0) + (row_origins * row_norm_scale - 1.0)[:, None, None]
    grids = torch.stack([grid_col, grid_row], dim=-1)
    return F.grid_sample(
        dp_f, grids, mode="bilinear", padding_mode="zeros", align_corners=True
    ).squeeze(1)


_polar_step_compiled = torch.compile(_polar_step, mode="reduce-overhead", dynamic=False)


def mean_dp_torch(
    array_4d: Any,
    n_row: int,
    n_col: int,
    device: str,
    *,
    chunk_bytes: int = _MEAN_DP_CHUNK_BYTES,
) -> NDArray:
    """Mean DP. Integer inputs use a chunked int64 accumulator; floats use ``mean()``.

    Already-on-device torch tensors reduce in place; numpy inputs stage to ``device``
    in int64 chunks bounded by ``chunk_bytes``.
    """
    if _is_torch_tensor(array_4d):
        device = str(array_4d.device)
    flat = array_4d.reshape(-1, n_row, n_col)
    n_pos = flat.shape[0]
    is_int = (
        not flat.is_floating_point() if isinstance(flat, torch.Tensor)
        else np.issubdtype(flat.dtype, np.integer)
    )
    if not is_int:
        if isinstance(flat, torch.Tensor):
            return flat.mean(dim=0).to(torch.float32).cpu().numpy()
        return flat.mean(axis=0).astype(np.float32)
    bytes_per_position = 8 * n_row * n_col
    chunk_positions = max(1, chunk_bytes // bytes_per_position)
    sum_t = torch.zeros((n_row, n_col), dtype=torch.int64, device=device)
    for start in range(0, n_pos, chunk_positions):
        end = min(start + chunk_positions, n_pos)
        sum_t += torch.asarray(flat[start:end], dtype=torch.int64, device=device).sum(dim=0)
    # Divide on host in float64 to keep precision; MPS lacks float64 support.
    return (sum_t.cpu().to(torch.float64) / float(n_pos)).to(torch.float32).numpy()


def _array_chunk_to_device_float32(chunk: NDArray, device: str) -> torch.Tensor:
    """Move an array chunk to the torch device as float32."""
    try:
        return torch.asarray(chunk, dtype=torch.float32, device=device)
    except (TypeError, RuntimeError):
        chunk_np = np.ascontiguousarray(chunk, dtype=np.float32)
        return torch.from_numpy(chunk_np).to(device)


@torch.inference_mode()
def auto_origin_id(
    data: Dataset4dstem | NDArray | torch.Tensor | Any,
    *,
    ellipse_params: tuple[float, float, float] | None = None,
    num_annular_bins: int = 180,
    radial_min: float = 0.0,
    radial_max: float | None = None,
    radial_step: float = 2.0,
    two_fold_rotation_symmetry: bool = False,
    batch_size: int = 48,
    local_margin: int = 25,
    show_progress: bool = True,
) -> NDArray:
    """
    Automatic diffraction center finding by minimizing angular intensity
    variation in the polar transform. A correctly centered diffraction
    pattern has uniform intensity along each ring, so the center that
    minimizes the angular standard deviation is the true beam center.

    Uses a coarse-to-fine search on the mean diffraction pattern to find
    a global center, then refines per scan position to account for descan
    across the scan.

    Parameters
    ----------
    data : Dataset4dstem | numpy.ndarray | torch.Tensor
        A 4D-STEM dataset (or 2D wrapped as 4D). When ``data.array`` (or
        the raw input) is a torch tensor, the entire computation stays
        on that tensor's device and no host staging happens. Cupy and
        numpy inputs are accepted via ``Dataset4dstem.from_array``,
        which auto-converts cupy → torch (dlpack, zero-copy).
    ellipse_params : tuple or None
        Ellipse parameters (a, b, theta_deg) for distortion correction
    num_annular_bins : int
        Number of angular bins for the final polar transform
    radial_min : float
        Minimum radius in pixels
    radial_max : float or None
        Maximum radius in pixels
    radial_step : float
        Radial step size in pixels for the search polar grid
    two_fold_rotation_symmetry : bool
        If True, use only 0 to pi range for angles
    batch_size : int
        Number of scan positions evaluated per coarse-stage kernel call.
        Larger values reduce per-iteration overhead but use more memory.
    local_margin : int
        Half-width (in pixels) of the search window used to refine each
        scan position's origin. After the global center is found on the
        mean DP, each DP's origin is searched within a
        ``(2*local_margin+1)`` square window centered on the global
        origin. Set this large enough to cover the worst-case descan
        drift across the scan.
    show_progress : bool
        If True, show a tqdm progress bar during per-scan origin refinement.

    Returns
    -------
    origin_array : np.ndarray
        Array of shape (scan_row, scan_col, 2) containing (row, col) origin
        estimates in pixels.
    """
    raw_array = data.array if hasattr(data, "array") else data
    if len(raw_array.shape) == 2:
        n_row, n_col = raw_array.shape
        scan_row, scan_col = 1, 1
    elif len(raw_array.shape) == 4:
        scan_row, scan_col, n_row, n_col = raw_array.shape
    else:
        raise ValueError(
            f" Got array with shape {raw_array.shape}."
            "To use auto_origin_id, pass a 2D or 4DSTEM dataset."
        )

    origin_shape = (scan_row, scan_col, 2)
    # first get COM of mean DP because it gives a robust rough center
    array_4d = raw_array if raw_array.ndim == 4 else raw_array[None, None, :, :]
    if _is_torch_tensor(array_4d):
        # Torch input: compute follows the tensor's device.
        device = str(array_4d.device)
    else:
        # NumPy input → CPU torch. NumPy is a host-only container; defaulting
        # compute to CPU keeps numerical output bit-stable with Karen's tutorial
        # baseline. Users who want GPU compute pass a torch tensor or cupy
        # array to ``Dataset4dstem.from_array`` instead.
        device = "cpu"
        array_4d = torch.from_numpy(np.ascontiguousarray(array_4d)).to(device)
    mean_dp_np = mean_dp_torch(array_4d, n_row, n_col, device)
    total_intensity = mean_dp_np.sum()
    row_grid, col_grid = np.mgrid[0:n_row, 0:n_col]
    com_row = int(round(float((row_grid * mean_dp_np).sum() / total_intensity)))
    com_col = int(round(float((col_grid * mean_dp_np).sum() / total_intensity)))
    # Radial max of the search polar grid, so dp_mean search candidates
    # (at ±global_margin from COM) stay within image bounds
    # in-image. Single pos candidates further from COM might be out of bounds
    # and are masked with [safe_low, safe_high_*] if so
    # (zero-padded samples would otherwise produce a falsely low score)
    global_margin = 20
    safe_radial_max = float(
        min(
            com_row - global_margin,
            (n_row - 1) - (com_row + global_margin),
            com_col - global_margin,
            (n_col - 1) - (com_col + global_margin),
        )
    )
    if radial_max is not None:
        safe_radial_max = min(safe_radial_max, float(radial_max))
    if safe_radial_max <= radial_min:
        safe_radial_max = radial_min + radial_step
    safe_low = int(np.ceil(safe_radial_max))
    safe_high_row = n_row - 1 - safe_low
    safe_high_col = n_col - 1 - safe_low
    # Internal search-grid resolution. Balancing speed against robustness
    search_n_phi = 18
    local_coarse_step = 5
    offset_row, offset_col, _, radial_bins = _build_polar_sampling_offsets(
        ellipse_params,
        search_n_phi,
        radial_min,
        safe_radial_max,
        radial_step,
        two_fold_rotation_symmetry,
        device,
    )
    n_r = radial_bins.numel()
    min_r_idx = int(np.floor(0.1 * n_r))
    max_r_idx = int(np.ceil(0.9 * n_r))
    score_offset_row = offset_row[:, min_r_idx:max_r_idx]
    score_offset_col = offset_col[:, min_r_idx:max_r_idx]
    score_n_r = score_offset_col.shape[1]
    # Normalize only the radial band used for scoring. This keeps the objective
    # identical while avoiding grid_sample work for radial bins that are discarded.
    col_norm_scale = 2.0 / (n_col - 1)
    row_norm_scale = 2.0 / (n_row - 1)
    base_col_norm = score_offset_col * col_norm_scale
    base_row_norm = score_offset_row * row_norm_scale
    col_origin_norm = torch.arange(n_col, dtype=torch.float32, device=device) * col_norm_scale - 1.0
    row_origin_norm = torch.arange(n_row, dtype=torch.float32, device=device) * row_norm_scale - 1.0
    # Mean-DP global center search: coarse → fine, masking candidates
    # whose polar grid would extend OOB at each step.
    mean_dp_batch = torch.from_numpy(mean_dp_np).to(device)[None, None]
    # Coarse: step=4 over ±global_margin around the COM
    rows, cols, grids = _build_candidate_grids(
        base_col_norm,
        base_row_norm,
        com_row,
        com_col,
        global_margin,
        n_row,
        n_col,
        col_norm_scale,
        row_norm_scale,
        device,
        step=4,
        col_origin_norm=col_origin_norm,
        row_origin_norm=row_origin_norm,
    )
    scores = _angular_std_scores(mean_dp_batch, grids, 0, score_n_r)
    valid = (
        (rows >= safe_low) & (rows <= safe_high_row) & (cols >= safe_low) & (cols <= safe_high_col)
    )
    scores.masked_fill_(~valid, float("inf"))
    best = scores.argmin().item()
    coarse_row, coarse_col = int(rows[best].item()), int(cols[best].item())
    # Fine: step=1 over ±6 around the coarse winner
    rows, cols, grids = _build_candidate_grids(
        base_col_norm,
        base_row_norm,
        coarse_row,
        coarse_col,
        6,
        n_row,
        n_col,
        col_norm_scale,
        row_norm_scale,
        device,
        step=1,
        col_origin_norm=col_origin_norm,
        row_origin_norm=row_origin_norm,
    )
    scores = _angular_std_scores(mean_dp_batch, grids, 0, score_n_r)
    valid = (
        (rows >= safe_low) & (rows <= safe_high_row) & (cols >= safe_low) & (cols <= safe_high_col)
    )
    scores.masked_fill_(~valid, float("inf"))
    best = scores.argmin().item()
    global_row, global_col = int(rows[best].item()), int(cols[best].item())

    # Per-scan-position refinement (coarse → medium → fine) for descan
    # medium and fine search per-DP around the previous winner
    coarse_rows, coarse_cols, coarse_grids = _build_candidate_grids(
        base_col_norm,
        base_row_norm,
        global_row,
        global_col,
        local_margin,
        n_row,
        n_col,
        col_norm_scale,
        row_norm_scale,
        device,
        step=local_coarse_step,
        col_origin_norm=col_origin_norm,
        row_origin_norm=row_origin_norm,
    )
    coarse_valid = (
        (coarse_rows >= safe_low)
        & (coarse_rows <= safe_high_row)
        & (coarse_cols >= safe_low)
        & (coarse_cols <= safe_high_col)
    )
    if not bool(coarse_valid.all().item()):
        coarse_rows = coarse_rows[coarse_valid]
        coarse_cols = coarse_cols[coarse_valid]
        coarse_grids = coarse_grids[coarse_valid]
    n_coarse = coarse_grids.shape[0]
    # Per-DP relative offsets used by the medium and fine stages
    med_rel = torch.arange(
        -local_coarse_step, local_coarse_step + 1, 2, dtype=torch.long, device=device
    )
    med_drow, med_dcol = (m.reshape(-1) for m in torch.meshgrid(med_rel, med_rel, indexing="ij"))
    fine_rel = torch.arange(-1, 2, dtype=torch.long, device=device)
    fine_drow, fine_dcol = (
        m.reshape(-1) for m in torch.meshgrid(fine_rel, fine_rel, indexing="ij")
    )
    # Input is already a torch tensor on device (numpy callers were converted
    # upfront). Use a native-dtype view; per-batch float32 cast is tiny.
    flat_dps_t = array_4d.reshape(-1, n_row, n_col)
    n_pos = flat_dps_t.shape[0]
    origin_flat_t = torch.empty((n_pos, 2), dtype=torch.int32, device=device)
    med_grid_buffer = torch.empty(
        (
            min(batch_size, n_pos),
            med_drow.numel(),
            base_col_norm.shape[0],
            base_col_norm.shape[1],
            2,
        ),
        dtype=base_col_norm.dtype,
        device=device,
    )
    fine_grid_buffer = torch.empty(
        (
            min(batch_size, n_pos),
            fine_drow.numel(),
            base_col_norm.shape[0],
            base_col_norm.shape[1],
            2,
        ),
        dtype=base_col_norm.dtype,
        device=device,
    )

    def refine(dp_batch, current_row, current_col, drow, dcol, grid_buffer):
        """scores candidates per DP and return the best(row, col) per DP. Invalid (out of bounds) candidates are masked."""
        n_cands = drow.numel()
        cand_rows = (current_row[:, None] + drow[None, :]).clamp(0, n_row - 1)
        cand_cols = (current_col[:, None] + dcol[None, :]).clamp(0, n_col - 1)
        grids = grid_buffer[: dp_batch.shape[0], :n_cands]
        grids[..., 0] = base_col_norm[None, None, :, :] + col_origin_norm[cand_cols][
            :, :, None, None
        ]
        grids[..., 1] = base_row_norm[None, None, :, :] + row_origin_norm[cand_rows][
            :, :, None, None
        ]
        grids = grids.reshape(
            dp_batch.shape[0], n_cands, base_col_norm.shape[0] * base_col_norm.shape[1], 2
        )
        polars = F.grid_sample(
            dp_batch, grids, mode="bilinear", padding_mode="zeros", align_corners=True
        )
        scores = (
            polars.squeeze(1)
            .view(dp_batch.shape[0], n_cands, *base_col_norm.shape)
            .var(dim=2, correction=1)
            .sqrt()
            .sum(dim=2)
        )
        valid = (
            (cand_rows >= safe_low)
            & (cand_rows <= safe_high_row)
            & (cand_cols >= safe_low)
            & (cand_cols <= safe_high_col)
        )
        scores.masked_fill_(~valid, float("inf"))
        best = scores.argmin(dim=1)
        return (
            cand_rows.gather(1, best[:, None]).squeeze(1),
            cand_cols.gather(1, best[:, None]).squeeze(1),
        )

    pbar = (
        tqdm(
            total=n_pos,
            desc="Finding origin for each scan position",
            mininterval=0.5,
        )
        if show_progress
        else None
    )
    # Data is already on device; iterate in flat batches.
    for start in range(0, n_pos, batch_size):
        end = min(start + batch_size, n_pos)
        bsz = end - start
        dp_b = flat_dps_t[start:end]
        if dp_b.dtype != torch.float32:
            dp_b = dp_b.to(torch.float32)
        dp_b = dp_b.unsqueeze(1)
        # Coarse (shared grids): broadcast B DPs across n_coarse candidate
        # grids in one grid_sample call by stacking DPs in the channel dim
        # and stride-0 expanding along the candidate dim
        polars_coarse = F.grid_sample(
            dp_b.transpose(0, 1).expand(n_coarse, bsz, n_row, n_col),
            coarse_grids,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        scores_coarse = polars_coarse.var(dim=2, correction=1).sqrt().sum(dim=2)
        best_coarse = scores_coarse.argmin(dim=0)  # best candidate per DP
        current_row, current_col = coarse_rows[best_coarse], coarse_cols[best_coarse]
        # Medium: per-DP search around the coarse winner
        current_row, current_col = refine(
            dp_b, current_row, current_col, med_drow, med_dcol, med_grid_buffer
        )
        # Fine: per-DP ±1 around the medium winner
        current_row, current_col = refine(
            dp_b, current_row, current_col, fine_drow, fine_dcol, fine_grid_buffer
        )
        origin_flat_t[start:end, 0] = current_row
        origin_flat_t[start:end, 1] = current_col
        if pbar is not None:
            pbar.update(bsz)
    if pbar is not None:
        pbar.close()
    return origin_flat_t.cpu().numpy().astype(float, copy=False).reshape(origin_shape)


def polar_transform(
    data: Dataset4dstem,
    origin_array: NDArray | torch.Tensor | None = None,
    ellipse_params: tuple[float, float, float] | None = None,
    num_annular_bins: int = 180,
    radial_min: float = 0.0,
    radial_max: float | None = None,
    radial_step: float = 1.0,
    two_fold_rotation_symmetry: bool = False,
    name: str | None = None,
    signal_units: str | None = None,
    scan_pos: tuple[int, int] | None = None,
    batch_size: int = 1024,
) -> Polar4dstem | torch.Tensor:
    """Re-sample each diffraction pattern onto a polar grid centered on ``origin_array``.

    Parameters
    ----------
    data : Dataset4dstem | numpy.ndarray | torch.Tensor
        4D-STEM dataset (or raw 4D array). Torch-backed ``data.array``
        stays on its tensor's device; cupy is converted zero-copy to
        torch in ``Dataset4dstem.from_array``. NumPy-backed data is
        uploaded to CPU torch for bit-stable parity with the legacy
        numpy baseline.
    origin_array : ndarray | torch.Tensor | None
        Per-scan origin in pixel coordinates, shape ``(scan_row, scan_col, 2)``,
        ``(2,)`` for a broadcast, or None for image center.
    batch_size : int
        Scan positions per ``grid_sample`` call. The compiled fast path fires
        on shape-stable full batches.
    """
    raw_array = data.array if hasattr(data, "array") else data
    if raw_array.ndim != 4:
        raise ValueError(
            f"Found array with shape: {raw_array.shape}. "
            "polar_transform requires a 4D-STEM dataset (ndim=4)."
        )
    scan_row, scan_col, n_row, n_col = raw_array.shape
    input_is_numpy = not _is_torch_tensor(raw_array)
    if input_is_numpy:
        # NumPy input → CPU torch. Keeps output bit-stable with the numpy baseline.
        # Users opt into GPU by passing a torch / cupy array into
        # ``Dataset4dstem.from_array``.
        device = "cpu"
        raw_array = torch.from_numpy(np.ascontiguousarray(raw_array)).to(device)
    else:
        device = str(raw_array.device)

    # Standardize origin_array input
    if isinstance(origin_array, torch.Tensor):
        origin_array = to_numpy(origin_array)
    origin_array = np.asarray(origin_array) if origin_array is not None else None
    if origin_array is None:
        center = np.array([(n_row - 1) / 2.0, (n_col - 1) / 2.0], dtype=float)
        origins = np.broadcast_to(center, (scan_row, scan_col, 2)).copy()
    elif origin_array.shape == (2,):
        origins = np.empty((scan_row, scan_col, 2), dtype=float)
        origins[...] = origin_array
    elif origin_array.shape == (scan_row, scan_col, 2):
        origins = origin_array
    else:
        raise ValueError(
            f" Got {origin_array.shape}. "
            "origin_array must have shape None, (2,) or (scan_row, scan_col, 2)."
        )

    # If scan_pos is provided, compute polar transform only for that position
    if scan_pos is not None:
        i_row, i_col = scan_pos
        dp = raw_array[i_row, i_col].to(torch.float32)
        r0 = float(origins[i_row, i_col, 0])
        c0 = float(origins[i_row, i_col, 1])
        # Clamp radial range to image bounds for this origin
        if radial_max is None:
            radial_max_eff = float(min(r0, (n_row - 1) - r0, c0, (n_col - 1) - c0))
        else:
            radial_max_eff = float(radial_max)
        if radial_max_eff <= radial_min:
            radial_max_eff = radial_min + radial_step
        # Build offsets, translate to this origin, normalize for grid_sample
        offset_row, offset_col, _, _ = _build_polar_sampling_offsets(
            ellipse_params,
            num_annular_bins,
            radial_min,
            radial_max_eff,
            radial_step,
            two_fold_rotation_symmetry,
            device,
        )
        col_norm = 2.0 * (offset_col + c0) / (n_col - 1) - 1.0
        row_norm = 2.0 * (offset_row + r0) / (n_row - 1) - 1.0
        # grid_sample requires (col, row) ordering in the last dim
        grid = torch.stack([col_norm, row_norm], dim=-1).unsqueeze(0)  # (1, n_phi, n_r, 2)
        polar2d = F.grid_sample(
            dp[None, None],
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        return polar2d.squeeze(0).squeeze(0)  # (n_phi, n_r)

    # Use the global minimum safe radius across all origins so every scan
    # position maps to the same-size polar grid (required for a uniform 4D output)
    if radial_max is None:
        r_row_pos = origins[:, :, 0]
        r_row_neg = (n_row - 1) - origins[:, :, 0]
        r_col_pos = origins[:, :, 1]
        r_col_neg = (n_col - 1) - origins[:, :, 1]
        radial_max_eff_array = np.minimum.reduce([r_row_pos, r_row_neg, r_col_pos, r_col_neg])
        radial_max = float(max(radial_max_eff_array.min(), radial_min + radial_step))

    # Build origin-independent polar offsets ONCE. Only the per-origin shift
    # changes from one scan position to the next, so we can reuse these.
    offset_row, offset_col, phi_bins, radial_bins = _build_polar_sampling_offsets(
        ellipse_params,
        num_annular_bins,
        radial_min,
        float(radial_max),
        radial_step,
        two_fold_rotation_symmetry,
        device,
    )
    n_phi = phi_bins.numel()
    n_r = radial_bins.numel()
    radial_max_eff = float(radial_max)

    # Pre-normalize offsets into grid_sample's [-1, 1] coordinate convention
    col_norm_scale = 2.0 / (n_col - 1)
    row_norm_scale = 2.0 / (n_row - 1)
    base_col_norm = offset_col * col_norm_scale  # (n_phi, n_r)
    base_row_norm = offset_row * row_norm_scale  # (n_phi, n_r)

    # Flatten scan dims so we can iterate in flat batches
    n_pos = scan_row * scan_col
    dp_view = raw_array.reshape(n_pos, n_row, n_col)
    origins_t = torch.from_numpy(
        np.ascontiguousarray(origins.reshape(n_pos, 2), dtype=np.float32)
    ).to(device)

    out_t = torch.empty((n_pos, n_phi, n_r), dtype=torch.float32, device=device)
    # The compile cache only pays off when launch overhead is the bottleneck;
    # use it on accelerator backends, run eager on CPU.
    step_fn = _polar_step_compiled if dp_view.device.type != "cpu" else _polar_step
    for start in tqdm(range(0, n_pos, batch_size), desc="Polar transform"):
        end = min(start + batch_size, n_pos)
        bsz = end - start
        row_origins = origins_t[start:end, 0]
        col_origins = origins_t[start:end, 1]

        if bsz == batch_size:
            # Fast path: shape-stable batches reuse the cached compile.
            dp_f = dp_view[start:end].to(torch.float32).unsqueeze(1)
            out_t[start:end] = step_fn(
                dp_f,
                row_origins,
                col_origins,
                base_col_norm,
                base_row_norm,
                col_norm_scale,
                row_norm_scale,
            )
        else:
            # Last partial batch: skip the compile cache (different shape).
            grid_col = (
                base_col_norm.unsqueeze(0) + (col_origins * col_norm_scale - 1.0)[:, None, None]
            )
            grid_row = (
                base_row_norm.unsqueeze(0) + (row_origins * row_norm_scale - 1.0)[:, None, None]
            )
            grids = torch.stack([grid_col, grid_row], dim=-1)
            dp_batch = dp_view[start:end]
            if dp_batch.dtype != torch.float32:
                dp_batch = dp_batch.to(torch.float32)
            polars = F.grid_sample(
                dp_batch.unsqueeze(1),
                grids,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            )
            out_t[start:end] = polars.squeeze(1)

    # Build the shared metadata (works regardless of array backend)
    phi_range = np.pi if two_fold_rotation_symmetry else 2.0 * np.pi
    phi_step_deg = (phi_range / float(n_phi)) * (180.0 / np.pi)
    sampling = np.zeros(4, dtype=float)
    origin = np.zeros(4, dtype=float)
    sampling[0:2] = np.asarray(data.sampling)[0:2]
    sampling[2] = phi_step_deg
    sampling[3] = float(np.asarray(data.sampling)[-1]) * radial_step
    origin[0:2] = np.asarray(data.origin)[0:2]
    origin[2] = 0.0
    origin[3] = radial_min * float(np.asarray(data.sampling)[-1])
    units = [data.units[0], data.units[1], "deg", data.units[-1]]
    metadata = dict(getattr(data, "metadata", {}))
    metadata.update(
        {
            "polar_radial_min": float(radial_min),
            "polar_radial_max": float(radial_max_eff),
            "polar_radial_step": float(radial_step),
            "polar_num_annular_bins": int(n_phi),
            "polar_two_fold_rotation_symmetry": bool(two_fold_rotation_symmetry),
            "polar_ellipticity": tuple(ellipse_params) if ellipse_params is not None else None,
        }
    )

    out_t = out_t.reshape(scan_row, scan_col, n_phi, n_r)
    if input_is_numpy:
        # Caller passed numpy; return a regular Polar4dstem with numpy backing.
        return Polar4dstem(
            array=out_t.cpu().numpy(),
            name=name if name is not None else f"{data.name}_polar",
            origin=origin,
            sampling=sampling,
            units=units,
            signal_units=signal_units if signal_units is not None else data.signal_units,
            metadata=metadata,
            origin_array=origins,
            _token=Polar4dstem._token,
        )

    # Real-time path: keep the polar tensor on device. Construct Polar4dstem
    # directly via __new__ because Dataset.from_array rejects non-numpy.
    # ``_array`` is the Polar4dstem/Dataset internal storage attribute.
    result = Polar4dstem.__new__(Polar4dstem)
    result._array = out_t
    result._name = name if name is not None else f"{data.name}_polar"
    result._origin = origin
    result._sampling = sampling
    result._units = units
    result._signal_units = signal_units if signal_units is not None else data.signal_units
    result._metadata = metadata
    result._file_path = None
    result.origin_array = origins
    return result


def _polar_to_cartesian_offsets(
    phi: torch.Tensor,
    r_pix: torch.Tensor,
    ellipse_params: tuple[float, float, float] | None,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert polar (phi, r_pix) grids to Cartesian (row, col) pixel offsets
    from the origin, optionally correcting for elliptical distortion.

    Returns ``(offset_row, offset_col)`` where
    ``col_offset = r_pix * cos(phi)`` and ``row_offset = r_pix * sin(phi)``.
    """
    if ellipse_params is None:
        offset_col = r_pix * torch.cos(phi)
        offset_row = r_pix * torch.sin(phi)
    else:
        if len(ellipse_params) != 3:
            raise ValueError("ellipse_params must be (a, b, theta_deg).")
        a, b, theta_deg = ellipse_params
        theta = torch.deg2rad(torch.tensor(theta_deg, dtype=torch.float32, device=device))
        # Rotate into the ellipse frame, scale by a/b to undo the distortion,
        # then rotate back so sampling follows the true circular rings
        alpha = phi - theta
        u = (a / b) * r_pix * torch.cos(alpha)
        v_prime = r_pix * torch.sin(alpha)
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)
        offset_col = u * cos_t - v_prime * sin_t
        offset_row = u * sin_t + v_prime * cos_t
    return offset_row, offset_col


def _build_polar_sampling_offsets(
    ellipse_params: tuple[float, float, float] | None,
    num_annular_bins: int,
    radial_min: float,
    radial_max_eff: float,
    radial_step: float,
    two_fold_rotation_symmetry: bool,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build origin-independent Cartesian (row, col) offsets for a polar
    sampling grid.

    Returns ``(offset_row, offset_col, phi_bins, radial_bins)`` where
    ``offset_row`` and ``offset_col`` have shape ``(n_phi, n_r)`` and
    represent pixel displacements from an arbitrary origin.
    """
    if radial_step <= 0:
        raise ValueError(f"Got radial_step = {radial_step}. radial_step must be > 0.")
    if num_annular_bins < 1:
        raise ValueError("num_annular_bins must be >= 1.")

    radial_bins = torch.arange(
        radial_min, radial_max_eff, radial_step, dtype=torch.float32, device=device
    )
    if radial_bins.numel() == 0:
        radial_bins = torch.tensor([radial_min], dtype=torch.float32, device=device)
    phi_range = torch.pi if two_fold_rotation_symmetry else 2.0 * torch.pi
    # Drop the last endpoint because 0 and 2pi (or pi) are the same angle
    phi_bins = torch.linspace(
        0.0, phi_range, num_annular_bins + 1, dtype=torch.float32, device=device
    )[:-1]
    phi_grid, r_pix_grid = torch.meshgrid(phi_bins, radial_bins, indexing="ij")
    # Compute offsets relative to origin (0,0) so they can be reused
    # for any candidate origin by simple translation
    offset_row, offset_col = _polar_to_cartesian_offsets(
        phi_grid, r_pix_grid, ellipse_params, device
    )
    return offset_row, offset_col, phi_bins, radial_bins


def _build_candidate_grids(
    base_col_norm: torch.Tensor,
    base_row_norm: torch.Tensor,
    center_row: int,
    center_col: int,
    margin: int,
    n_row: int,
    n_col: int,
    col_norm_scale: float,
    row_norm_scale: float,
    device: str = "cpu",
    step: int = 1,
    col_origin_norm: torch.Tensor | None = None,
    row_origin_norm: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a batch of normalized sampling grids, one per candidate origin
    pixel in a search window around (center_row, center_col). Candidates are
    produced in a single batched tensor so that they can be evaluated
    simultaneously by ``_angular_std_scores``.

    Parameters
    ----------
    base_col_norm, base_row_norm : torch.Tensor of shape (n_phi, n_r)
        Polar sampling offsets, already expressed in ``grid_sample``'s
        normalized [-1, 1] coordinates, relative to origin (0, 0)
    center_row, center_col : int
        Center of the candidate search window
    margin : int
        Half-width of the search window in pixels
    n_row, n_col : int
        Diffraction-pattern image dimensions
    col_norm_scale, row_norm_scale : float
        Conversion factor from an offset in pixel units to the equivalent
        offset in ``grid_sample``'s normalized coordinates

    Returns
    -------
    row_flat, col_flat : torch.Tensor of shape (N,)
        Candidate origin positions
    grids : torch.Tensor of shape (N, n_phi, n_r, 2)
        Stacked sampling grids ready for ``F.grid_sample`` (ordered ``(col, row)`` )
    """
    # Enumerate all pixel positions in the search window, clamped to image bounds
    rows = torch.arange(
        max(0, center_row - margin),
        min(n_row, center_row + margin + 1),
        step,
        dtype=torch.long,
        device=device,
    )
    cols = torch.arange(
        max(0, center_col - margin),
        min(n_col, center_col + margin + 1),
        step,
        dtype=torch.long,
        device=device,
    )
    row_grid, col_grid = torch.meshgrid(rows, cols, indexing="ij")
    row_flat, col_flat = row_grid.reshape(-1), col_grid.reshape(-1)
    # Shift the pre-computed polar offsets to each candidate origin,
    # converting to grid_sample's [-1, 1] normalized coordinates
    grids = torch.empty(
        (row_flat.numel(), base_col_norm.shape[0], base_col_norm.shape[1], 2),
        dtype=base_col_norm.dtype,
        device=device,
    )
    if col_origin_norm is None:
        col_shift = col_flat.float() * col_norm_scale - 1.0
    else:
        col_shift = col_origin_norm[col_flat]
    if row_origin_norm is None:
        row_shift = row_flat.float() * row_norm_scale - 1.0
    else:
        row_shift = row_origin_norm[row_flat]
    grids[..., 0] = base_col_norm.unsqueeze(0) + col_shift[:, None, None]
    grids[..., 1] = base_row_norm.unsqueeze(0) + row_shift[:, None, None]
    # grid_sample requires (col, row) ordering in the last dim
    return row_flat, col_flat, grids


def _angular_std_scores(
    dp_batch: torch.Tensor,
    grids: torch.Tensor,
    min_r_idx: int,
    max_r_idx: int,
) -> torch.Tensor:
    """Score candidate origins by angular std over a mid-radius band.
    Lower scores indicate better centering."""
    n = grids.shape[0]
    # Sample the diffraction pattern at each candidate's polar grid positions
    polars = F.grid_sample(
        dp_batch.expand(n, -1, -1, -1),
        grids,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    # A correctly centered pattern has uniform intensity along each ring,
    # so the angular std is minimized at the true center
    region = polars.squeeze(1)[:, :, min_r_idx:max_r_idx]
    return region.var(dim=1, correction=1).sqrt().sum(dim=1)
