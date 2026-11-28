"""Per-scanline non-rigid drift optimization."""
import math
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import quantem.imaging.drift.plot as drift_plot
from quantem.imaging.drift.core import knots as drift_knots
from quantem.imaging.drift.core.warping import warp_and_translate


def _grid_sample_mse(
    grid_row: torch.Tensor,
    grid_col: torch.Tensor,
    ref_t: torch.Tensor,
    target_batch: torch.Tensor,
) -> torch.Tensor:
    """Shared tail: stack the (col, row) grid, sample, and return the MSE.

    Inlined into both compiled kernels (`@torch.compile` follows the call).
    Pulling the grid_sample + MSE out of the two K-paths means the only
    difference between K=1 and K>=2 kernels is grid construction.

    The MSE is averaged over both the batch (N images) and the spatial dims,
    so each image's gradient is scaled by 1/N relative to a per-image solve.
    Adam's adaptive step size absorbs the constant rescale; LBFGS line search
    rescales itself.
    """
    grid = torch.stack([grid_col, grid_row], dim=-1)
    warped = F.grid_sample(
        ref_t, grid, mode='bilinear', align_corners=True, padding_mode='border')[:, 0]
    return ((warped - target_batch) ** 2).mean()


def _grid_sample_ncc(
    grid_row: torch.Tensor,
    grid_col: torch.Tensor,
    ref_t: torch.Tensor,
    target_batch: torch.Tensor,
    ref_coverage_t: torch.Tensor | None = None,
) -> torch.Tensor:
    """Warp ``ref_t`` and return coverage-weighted ``1 - mean NCC``.

    When ``ref_t`` is 2-channel ``(N, 2, H, W)`` (image + 1-px border-zeroed
    coverage), both channels are sampled in a single fused ``F.grid_sample``
    call with ``padding_mode='border'`` (giving exact 0 coverage outside the
    canvas). Reference coverage is detached before reduction so knot motion is
    never rewarded merely for changing the valid area.
    """
    grid = torch.stack([grid_col, grid_row], dim=-1)
    if ref_coverage_t is None:
        sampled = F.grid_sample(
            ref_t, grid, mode='bilinear', align_corners=True, padding_mode='border'
        )
        warped = sampled[:, 0]
        coverage = sampled[:, 1].detach().clamp(0.0, 1.0)
    else:
        warped = F.grid_sample(
            ref_t, grid, mode='bilinear', align_corners=True, padding_mode='border'
        )[:, 0]
        coverage = F.grid_sample(
            ref_coverage_t[:, None],
            grid,
            mode='bilinear',
            align_corners=True,
            padding_mode='zeros',
        )[:, 0].detach().clamp(0.0, 1.0)
    # (N, H, W) → (N, H*W), with weighted means/norms on measured pixels.
    w = warped.reshape(warped.shape[0], -1)
    t = target_batch.reshape(target_batch.shape[0], -1)
    mask = coverage.reshape(coverage.shape[0], -1)
    count = mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
    w0 = w - (w * mask).sum(dim=-1, keepdim=True) / count
    t0 = t - (t * mask).sum(dim=-1, keepdim=True) / count
    num = (mask * w0 * t0).sum(dim=-1)
    den = (
        (mask * w0.square()).sum(dim=-1).clamp_min(1e-12).sqrt()
        * (mask * t0.square()).sum(dim=-1).clamp_min(1e-12).sqrt()
    )
    ncc = (num / den).mean()
    return 1.0 - ncc


def _grid_sample_local_ncc(
    grid_row: torch.Tensor,
    grid_col: torch.Tensor,
    ref_t: torch.Tensor,
    target_batch: torch.Tensor,
    ref_coverage_t: torch.Tensor | None = None,
    tile_grid: int = 4,
) -> torch.Tensor:
    """Warp ``ref_t`` and return hybrid ``0.45 * global + 0.55 * tile`` masked ``1 - NCC``.

    Normalizes mean and variance independently inside each valid ``tile_grid × tile_grid``
    region (excluding low-coverage or flat/vacuum tiles via NaN-safe variance guards),
    preventing bright macro blobs from starving darker crystal areas and giving all four
    edges equal optimization weight on MPS and CUDA.
    """
    grid = torch.stack([grid_col, grid_row], dim=-1)
    if ref_coverage_t is None:
        sampled = F.grid_sample(
            ref_t, grid, mode='bilinear', align_corners=True, padding_mode='border'
        )
        warped = sampled[:, 0]
        coverage = sampled[:, 1].detach().clamp(0.0, 1.0)
    else:
        warped = F.grid_sample(
            ref_t, grid, mode='bilinear', align_corners=True, padding_mode='border'
        )[:, 0]
        coverage = F.grid_sample(
            ref_coverage_t[:, None],
            grid,
            mode='bilinear',
            align_corners=True,
            padding_mode='zeros',
        )[:, 0].detach().clamp(0.0, 1.0)
    N, H, W = warped.shape
    w = warped.reshape(N, -1)
    t = target_batch.reshape(N, -1)
    mask = coverage.reshape(N, -1)
    count = mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
    w0 = w - (w * mask).sum(dim=-1, keepdim=True) / count
    t0 = t - (t * mask).sum(dim=-1, keepdim=True) / count
    num = (mask * w0 * t0).sum(dim=-1)
    global_w_var = (mask * w0.square()).sum(dim=-1)
    global_t_var = (mask * t0.square()).sum(dim=-1)
    den = global_w_var.clamp_min(1e-12).sqrt() * global_t_var.clamp_min(1e-12).sqrt()
    loss_global = 1.0 - (num / den).mean()

    if tile_grid <= 1 or H < tile_grid * 16 or W < tile_grid * 16:
        return loss_global

    th = H // tile_grid
    tw = W // tile_grid
    Hu = th * tile_grid
    Wu = tw * tile_grid
    T = tile_grid * tile_grid
    # Zero-copy 5D views (N, tile_grid, th, tile_grid, tw) reduced over (2, 4)
    wt = warped[:, :Hu, :Wu].reshape(N, tile_grid, th, tile_grid, tw)
    tt = target_batch[:, :Hu, :Wu].reshape(N, tile_grid, th, tile_grid, tw)
    mt = coverage[:, :Hu, :Wu].reshape(N, tile_grid, th, tile_grid, tw)

    t_cnt = mt.sum(dim=(2, 4), keepdim=True)
    t_cnt_safe = t_cnt.clamp_min(1.0)
    wt0 = wt - (wt * mt).sum(dim=(2, 4), keepdim=True) / t_cnt_safe
    tt0 = tt - (tt * mt).sum(dim=(2, 4), keepdim=True) / t_cnt_safe
    wt_var = (mt * wt0.square()).sum(dim=(2, 4)).reshape(N, T)
    tt_var = (mt * tt0.square()).sum(dim=(2, 4)).reshape(N, T)
    # Scale-invariant tile variance guard (works across 1e-6 weak-phase to 1e+3 HAADF while rejecting flat vacuum)
    w_var_floor = (1e-4 * global_w_var[:, None] / T).clamp_min(1e-24)
    t_var_floor = (1e-4 * global_t_var[:, None] / T).clamp_min(1e-24)
    valid = (
        (t_cnt.reshape(N, T) >= 0.20 * (th * tw))
        & (wt_var > w_var_floor)
        & (tt_var > t_var_floor)
    ).to(warped.dtype)
    t_num = (mt * wt0 * tt0).sum(dim=(2, 4)).reshape(N, T)
    t_den = wt_var.clamp_min(1e-24).sqrt() * tt_var.clamp_min(1e-24).sqrt()
    t_ncc = (t_num / t_den) * valid
    # Average valid tiles per image first, then across images (consistent with loss_global)
    valid_per_img = valid.sum(dim=-1)
    img_has_tiles = valid_per_img >= 1.0
    per_img_tile_loss = torch.where(
        img_has_tiles,
        1.0 - t_ncc.sum(dim=-1) / valid_per_img.clamp_min(1.0),
        1.0 - (num / den),
    )
    loss_tiles = per_img_tile_loss.mean()
    return 0.45 * loss_global + 0.55 * loss_tiles


def _find_blackwell_ptxas() -> str | None:
    """Locate a CUDA >= 12.8 ptxas binary capable of targeting Blackwell (sm_100/sm_120)."""
    import os
    import re
    import shutil
    import subprocess
    import sys
    from pathlib import Path

    candidates: list[str] = []
    for path in (
        os.environ.get("TRITON_PTXAS_BLACKWELL_PATH"),
        shutil.which("ptxas-blackwell"),
        shutil.which("ptxas"),
        str(Path(sys.executable).resolve().parent / "ptxas"),
        os.path.join(sys.prefix, "bin", "ptxas"),
    ):
        if path and path not in candidates and os.path.isfile(path) and os.access(path, os.X_OK):
            candidates.append(path)
    for path in candidates:
        if os.path.basename(path) == "ptxas-blackwell":
            return path
        try:
            proc = subprocess.run(
                [path, "--version"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            out = (proc.stdout or "") + (proc.stderr or "")
            match = re.search(r"release\s+(\d+)\.(\d+)", out)
            if match and (int(match.group(1)), int(match.group(2))) >= (12, 8):
                return path
        except Exception:
            continue
    return None


def _inductor_supported() -> bool:
    """Return True only when torch.compile(mode='default') can compile cleanly on this device."""
    import os

    if os.environ.get("TORCH_COMPILE_DISABLE", "").strip().lower() in ("1", "true", "yes"):
        return False
    dynamo = getattr(torch, "_dynamo", None)
    if dynamo is not None and getattr(getattr(dynamo, "config", None), "disable", False):
        return False
    if not torch.cuda.is_available():
        return False
    try:
        major, _ = torch.cuda.get_device_capability()
        if major >= 10:
            ptxas_path = _find_blackwell_ptxas()
            if ptxas_path is None:
                return False
            os.environ.setdefault("TRITON_PTXAS_BLACKWELL_PATH", ptxas_path)
            try:
                import triton.knobs as _tk
                if hasattr(_tk, "nvidia") and hasattr(_tk.nvidia, "ptxas_blackwell"):
                    _tk.nvidia.ptxas_blackwell = ptxas_path
            except Exception:
                pass
        @torch.compile(mode="default", dynamic=False)
        def _probe(x: torch.Tensor) -> torch.Tensor:
            return (x + 1.0) * 2.0

        _ = _probe(torch.ones(4, device="cuda"))
    except Exception:
        return False
    return True


_USE_INDUCTOR = _inductor_supported()


def _maybe_compile(fn):
    """Use leak-free mode='default' TorchInductor only when supported; eager otherwise."""
    if _USE_INDUCTOR:
        return torch.compile(mode="default", dynamic=False)(fn)
    return fn


@_maybe_compile
def _compiled_loss_fn_single(
    knots_batch: torch.Tensor,
    ref_t: torch.Tensor,
    target_batch: torch.Tensor,
    row_scan_offsets: torch.Tensor,
    col_scan_offsets: torch.Tensor,
    row_scale: float,
    col_scale: float,
) -> torch.Tensor:
    """Fused K=1 forward pass: knot anchor + scan_fast walk → MSE.

    ``knots_batch`` shape ``(N, 2, num_rows, 1)``.  Each scanline has one
    anchor knot and the per-pixel canvas position is filled in by adding
    the precomputed ``scan_fast`` walk.  ``correct_nonrigid`` selects this
    kernel when ``K == 1`` and :func:`_compiled_loss_fn_multi` for ``K > 1``.
    """
    grid_row = (knots_batch[:, 0, :, :] + row_scan_offsets[:, None, :]) * row_scale - 1.0
    grid_col = (knots_batch[:, 1, :, :] + col_scan_offsets[:, None, :]) * col_scale - 1.0
    return _grid_sample_mse(grid_row, grid_col, ref_t, target_batch)


@_maybe_compile
def _compiled_loss_fn_multi(
    knots_batch: torch.Tensor,
    ref_t: torch.Tensor,
    target_batch: torch.Tensor,
    seg_idx: torch.Tensor,
    seg_frac: torch.Tensor,
    row_scale: float,
    col_scale: float,
) -> torch.Tensor:
    """Fused K-knot forward pass with linear knot interpolation along scanline.

    ``knots_batch`` shape ``(N, 2, num_rows, K)`` with ``K >= 2``.
    ``seg_idx`` (long, shape ``(num_cols,)``) and ``seg_frac`` (shape
    ``(num_cols,)``) precompute, per output column, which adjacent knot
    pair to interpolate and the local fraction.  Both are constant for
    the lifetime of the optimization, so we lift them out of the loop.
    """
    knot_lo = knots_batch[:, :, :, seg_idx]
    knot_hi = knots_batch[:, :, :, seg_idx + 1]
    interp = knot_lo + (knot_hi - knot_lo) * seg_frac[None, None, None, :]
    grid_row = interp[:, 0] * row_scale - 1.0
    grid_col = interp[:, 1] * col_scale - 1.0
    return _grid_sample_mse(grid_row, grid_col, ref_t, target_batch)


@_maybe_compile
def _compiled_loss_fn_single_ncc(
    knots_batch: torch.Tensor,
    ref_t: torch.Tensor,
    target_batch: torch.Tensor,
    row_scan_offsets: torch.Tensor,
    col_scan_offsets: torch.Tensor,
    row_scale: float,
    col_scale: float,
) -> torch.Tensor:
    """K=1 path with fused 2-channel ``1 - NCC`` loss (brightness-invariant)."""
    grid_row = (knots_batch[:, 0, :, :] + row_scan_offsets[:, None, :]) * row_scale - 1.0
    grid_col = (knots_batch[:, 1, :, :] + col_scan_offsets[:, None, :]) * col_scale - 1.0
    return _grid_sample_ncc(
        grid_row, grid_col, ref_t, target_batch
    )


@_maybe_compile
def _compiled_loss_fn_multi_ncc(
    knots_batch: torch.Tensor,
    ref_t: torch.Tensor,
    target_batch: torch.Tensor,
    seg_idx: torch.Tensor,
    seg_frac: torch.Tensor,
    row_scale: float,
    col_scale: float,
) -> torch.Tensor:
    """K>=2 path with fused 2-channel ``1 - NCC`` loss."""
    knot_lo = knots_batch[:, :, :, seg_idx]
    knot_hi = knots_batch[:, :, :, seg_idx + 1]
    interp = knot_lo + (knot_hi - knot_lo) * seg_frac[None, None, None, :]
    grid_row = interp[:, 0] * row_scale - 1.0
    grid_col = interp[:, 1] * col_scale - 1.0
    return _grid_sample_ncc(
        grid_row, grid_col, ref_t, target_batch
    )


@_maybe_compile
def _compiled_loss_fn_single_local_ncc(
    knots_batch: torch.Tensor,
    ref_t: torch.Tensor,
    target_batch: torch.Tensor,
    row_scan_offsets: torch.Tensor,
    col_scan_offsets: torch.Tensor,
    row_scale: float,
    col_scale: float,
    tile_grid: int,
) -> torch.Tensor:
    """K=1 path with fused 2-channel hybrid global + tile-local ``1 - NCC`` loss."""
    grid_row = (knots_batch[:, 0, :, :] + row_scan_offsets[:, None, :]) * row_scale - 1.0
    grid_col = (knots_batch[:, 1, :, :] + col_scan_offsets[:, None, :]) * col_scale - 1.0
    return _grid_sample_local_ncc(
        grid_row, grid_col, ref_t, target_batch, tile_grid=tile_grid
    )


@_maybe_compile
def _compiled_loss_fn_multi_local_ncc(
    knots_batch: torch.Tensor,
    ref_t: torch.Tensor,
    target_batch: torch.Tensor,
    seg_idx: torch.Tensor,
    seg_frac: torch.Tensor,
    row_scale: float,
    col_scale: float,
    tile_grid: int,
) -> torch.Tensor:
    """K>=2 path with fused 2-channel hybrid global + tile-local ``1 - NCC`` loss."""
    knot_lo = knots_batch[:, :, :, seg_idx]
    knot_hi = knots_batch[:, :, :, seg_idx + 1]
    interp = knot_lo + (knot_hi - knot_lo) * seg_frac[None, None, None, :]
    grid_row = interp[:, 0] * row_scale - 1.0
    grid_col = interp[:, 1] * col_scale - 1.0
    return _grid_sample_local_ncc(
        grid_row, grid_col, ref_t, target_batch, tile_grid=tile_grid
    )


def _optimize_knots_adam(
    ref_batch, target_batch, knots_batch,
    loss_fn, loss_args,
    optimizer, steps, grad_mask=None,
):
    """Run ``steps`` of Adam on a batched knot tensor against ``loss_fn``."""
    ref_t = ref_batch[:, None] if ref_batch.ndim == 3 else ref_batch
    for _ in range(steps):
        optimizer.zero_grad()
        loss = loss_fn(knots_batch, ref_t, target_batch, *loss_args)
        loss.backward()
        if grad_mask is not None:
            knots_batch.grad.mul_(grad_mask)
        optimizer.step()


def _optimize_knots_lbfgs(
    ref_batch, target_batch, knots_batch,
    loss_fn, loss_args,
    optimizer, grad_mask=None,
):
    """Run one LBFGS outer step (line search re-evaluates the closure several times)."""
    ref_t = ref_batch[:, None] if ref_batch.ndim == 3 else ref_batch
    def closure():
        optimizer.zero_grad()
        loss = loss_fn(knots_batch, ref_t, target_batch, *loss_args)
        loss.backward()
        if grad_mask is not None:
            knots_batch.grad.mul_(grad_mask)
        return loss
    optimizer.step(closure)


def sobel_gradient_magnitude(
    images: torch.Tensor,
    pre_smooth: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Compute per-image Sobel gradient magnitude with optional Gaussian pre-smooth.

    Returns ``(N, H, W)`` z-score-normalized per image so each output has
    zero mean and unit variance — removes gain/offset sensitivity for
    cross-detector loss comparisons.
    """
    img = images[:, None]  # (N, 1, H, W) for conv2d
    if pre_smooth > 0:
        ks = max(3, int(6 * pre_smooth) | 1)  # odd kernel size
        x = torch.arange(ks, dtype=dtype, device=device) - ks // 2
        g = torch.exp(-0.5 * (x / max(pre_smooth, 1e-6)) ** 2)
        g = g / g.sum()
        pad_h = ks // 2
        img = F.pad(img, (pad_h, pad_h, 0, 0), mode='reflect')
        img = F.conv2d(img, g.reshape(1, 1, 1, -1))
        img = F.pad(img, (0, 0, pad_h, pad_h), mode='reflect')
        img = F.conv2d(img, g.reshape(1, 1, -1, 1))
    sobel_column = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        dtype=dtype, device=device).reshape(1, 1, 3, 3)
    sobel_row = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
        dtype=dtype, device=device).reshape(1, 1, 3, 3)
    img_pad = F.pad(img, (1, 1, 1, 1), mode='reflect')
    gradient_column = F.conv2d(img_pad, sobel_column)
    gradient_row = F.conv2d(img_pad, sobel_row)
    grad_mag = (gradient_row**2 + gradient_column**2).sqrt()[:, 0]
    mean = grad_mag.mean(dim=(-2, -1), keepdim=True)
    std = grad_mag.std(dim=(-2, -1), keepdim=True).clamp(min=1e-8)
    return (grad_mag - mean) / std


def _project_nonfolding_slope(
    knots_batch: torch.Tensor,
    init_batch: torch.Tensor,
    max_slope: float,
) -> None:
    """Directional non-folding Jacobian projection along slow- and fast-scan axes.

    For a scan with unit slow-scan direction ``s_hat``, the raster Jacobian determinant
    is ``det(J) = 1 + g_par`` where ``g_par = (Δdisp/Δrow) · s_hat``. Positive stretch
    (``g_par > 0``) and transverse shear (``g_perp`` perpendicular to ``s_hat``) never
    fold the raster; only compression ``g_par < -max_slope`` reduces ``det(J)`` below
    ``1 - max_slope``. Bounding compression strictly while keeping stretch and shear
    generously bounded preserves real piezo/thermal drift without folding.
    Preserves each scanline's mean displacement via vectorized center-outward cumsum.
    """
    if max_slope <= 0.0:
        raise ValueError(f"max_slope must be positive, got {max_slope!r}")
    H = int(knots_batch.shape[2])
    K = int(knots_batch.shape[3])
    if H <= 1:
        return
    with torch.no_grad():
        disp = knots_batch - init_batch
        # Infer each scan's unit slow-scan vector directly from init_batch (works for any angle)
        init_step = init_batch[:, :, 1:, :1] - init_batch[:, :, :-1, :1]
        s_vec = init_step.mean(dim=2, keepdim=True)  # (N, 2, 1, 1)
        s_hat = s_vec / torch.norm(s_vec, dim=1, keepdim=True).clamp_min(1e-8)
        f_hat = torch.stack([-s_hat[:, 1], s_hat[:, 0]], dim=1)

        dd = disp[:, :, 1:, :] - disp[:, :, :-1, :]  # (N, 2, H-1, K)
        g_par = (dd * s_hat).sum(dim=1, keepdim=True)
        g_perp = (dd * f_hat).sum(dim=1, keepdim=True)
        # Unified physical stretch/shear caps scale symmetrically with max_slope (v28 consensus).
        max_stretch = max(0.55, 1.2 * max_slope)
        max_shear = min(0.45, 0.85 * max_slope)
        g_par_c = g_par.clamp(-max_slope, max_stretch)
        g_perp_c = g_perp.clamp(-max_shear, max_shear)
        # Run vectorized clamp+cumsum unconditionally on GPU (avoids blocking torch.equal syncs)
        dd_clamped = g_par_c * s_hat + g_perp_c * f_hat
        mid = H // 2
        out = disp.clone()
        out[:, :, mid + 1 :, :] = disp[:, :, mid : mid + 1, :] + torch.cumsum(
            dd_clamped[:, :, mid:, :], dim=2
        )
        out[:, :, :mid, :] = disp[:, :, mid : mid + 1, :] - torch.flip(
            torch.cumsum(torch.flip(dd_clamped[:, :, :mid, :], dims=[2]), dim=2),
            dims=[2],
        )
        out = out + (disp.mean(dim=2, keepdim=True) - out.mean(dim=2, keepdim=True))
        disp = out

        # When K > 1, also bound fast-axis compression and shear along dim=3
        if K > 1:
            init_f = init_batch[:, :, :, 1:] - init_batch[:, :, :, :-1]
            fast_ref = torch.norm(init_f, dim=1, keepdim=True).clamp_min(1.0)
            uf = init_f / fast_ref
            up = torch.stack([-uf[:, 1], uf[:, 0]], dim=1)
            df = disp[:, :, :, 1:] - disp[:, :, :, :-1]
            f_par = (df * uf).sum(dim=1, keepdim=True)
            f_perp = (df * up).sum(dim=1, keepdim=True)
            # Tighter fast-axis bounds ensure det(J) = (1+g_par)*(1+f_par/fast_ref) - g_perp*(f_perp/fast_ref) stays well > 0
            f_comp_cap = min(max_slope, 0.45) * fast_ref
            f_stretch_cap = min(max_stretch, 0.45) * fast_ref
            f_shear_cap = min(max_shear, 0.35) * fast_ref
            f_par_c = f_par.clamp(-f_comp_cap, f_stretch_cap)
            f_perp_c = f_perp.clamp(-f_shear_cap, f_shear_cap)
            df_clamped = f_par_c * uf + f_perp_c * up
            mid_k = K // 2
            out_f = disp.clone()
            out_f[:, :, :, mid_k + 1 :] = disp[:, :, :, mid_k : mid_k + 1] + torch.cumsum(
                df_clamped[:, :, :, mid_k:], dim=3
            )
            out_f[:, :, :, :mid_k] = disp[:, :, :, mid_k : mid_k + 1] - torch.flip(
                torch.cumsum(torch.flip(df_clamped[:, :, :, :mid_k], dims=[3]), dim=3),
                dims=[3],
            )
            out_f = out_f + (disp.mean(dim=3, keepdim=True) - out_f.mean(dim=3, keepdim=True))
            disp = out_f

        knots_batch.copy_(init_batch + disp)


def _regularize_knots(
    knots_batch: torch.Tensor,
    knots_prev: torch.Tensor,
    vander: torch.Tensor | None,
    max_shift_px: float | None,
    sigma_px: float | None,
    step_size: float | None,
    row_cov: torch.Tensor | None = None,
    init_batch: torch.Tensor | None = None,
    max_slope: float | None = None,
) -> None:
    """Apply per-iteration knot regularization (in place on ``knots_batch``).

    Three independent stages, each gated by its parameter being non-None:
        1. Per-knot shift cap: clamp ``|new - prev|`` to ``max_shift_px``
           so the optimizer can't move any knot too far in one outer iter.
        2. Polynomial detrend + Gaussian smooth: keep low-order trends,
           smooth the residual along the scan-line dimension. When ``row_cov``
           is supplied, uses coverage-weighted polynomial detrending so
           un-overlapped border rows smoothly extrapolate the interior trend
           instead of hooking backwards.
        3. Step-size blend: ``new = prev + step_size · (new - prev)``,
           optionally followed by non-folding slow-axis slope projection.
    """
    # Knots are 4D ``(N, 2, R, K)`` everywhere — K=1 just has trailing 1.
    num_images, _, num_rows_knot, K = knots_batch.shape
    with torch.no_grad():
        if max_shift_px is not None:
            shift = knots_batch - knots_prev
            dist = torch.norm(shift, dim=1, keepdim=True)
            scale_factor = torch.clamp(max_shift_px / dist.clamp(min=1e-8), max=1.0)
            knots_batch.copy_(knots_prev + shift * scale_factor)
        if sigma_px is not None and sigma_px > 0 and vander is not None:
            if row_cov is not None:
                eye = 1e-6 * torch.eye(
                    vander.shape[1], device=vander.device, dtype=vander.dtype
                )
                # Detrend displacement relative to init_batch so even trend_order=0 preserves the scan ramp
                base_batch = init_batch if init_batch is not None else torch.zeros_like(knots_batch)
                # Vectorized, sync-free coverage-weighted detrend + smooth + border extrapolation.
                # Smoothly tapers border rows with <55% shared-FOV coverage so uncovered edges
                # linearly extrapolate the interior drift instead of hooking backwards.
                cov_w = ((row_cov - 0.15) / 0.45).clamp(0.0, 1.0)  # (N, R)
                enough_cov = (cov_w > 0.1).sum(dim=1, keepdim=True) >= (vander.shape[1] + 2)
                rw = torch.where(enough_cov, cov_w.clamp(min=1e-4), row_cov.clamp(min=0.02))
                vw = vander[None, :, :] * rw[:, :, None]  # (N, R, P)
                disp = knots_batch - base_batch  # (N, 2, R, K)
                kf = disp.permute(0, 2, 1, 3).reshape(num_images, num_rows_knot, -1)  # (N, R, 2*K)
                coef = torch.linalg.solve(
                    torch.matmul(vw.transpose(1, 2), vander[None]) + eye[None],
                    torch.matmul(vw.transpose(1, 2), kf),
                )  # (N, P, 2*K)
                trend = torch.matmul(vander[None], coef).transpose(1, 2)  # (N, 2*K, R)
                res = (kf.transpose(1, 2) - trend) * cov_w[:, None, :]  # (N, 2*K, R)
                sm = drift_knots.gaussian_smooth_1d(
                    res.reshape(-1, num_rows_knot), sigma_px
                ).reshape(num_images, -1, num_rows_knot)
                combined = sm + trend  # (N, 2*K, R)
                # Sync-free linear extrapolation of leading/trailing low-coverage rows
                r_idx = torch.arange(num_rows_knot, device=combined.device, dtype=torch.int64)
                is_cov = cov_w >= 0.45  # (N, R)
                has_enough = is_cov.sum(dim=1) >= 16  # (N,)
                i0 = torch.where(is_cov, r_idx[None, :], num_rows_knot).amin(dim=1)  # (N,)
                i1 = torch.where(is_cov, r_idx[None, :], -1).amax(dim=1)  # (N,)
                span = ((i1 - i0) // 6).clamp(min=6, max=24)  # (N,)
                xs_mean = (span.to(combined.dtype) - 1.0) * 0.5  # (N,)
                zero_s = torch.zeros(1, device=combined.device, dtype=combined.dtype)
                # Leading rows (r < i0)
                in_lead = (r_idx[None, :] >= i0[:, None]) & (r_idx[None, :] < (i0 + span)[:, None])
                xs_lead = (r_idx[None, :] - i0[:, None]).to(combined.dtype)
                xs_c_lead = torch.where(in_lead, xs_lead - xs_mean[:, None], zero_s)
                slope_lead = (combined * xs_c_lead[:, None, :]).sum(dim=2) / (
                    xs_c_lead.pow(2).sum(dim=1, keepdim=True) + 1e-8
                )
                val_i0 = combined.gather(
                    2, i0.clamp(0, num_rows_knot - 1)[:, None, None].expand(-1, combined.shape[1], 1)
                )
                extrap_lead = val_i0 + slope_lead[:, :, None] * xs_lead[:, None, :]
                do_lead = (has_enough & (i0 > 0))[:, None] & (r_idx[None, :] < i0[:, None])
                combined = torch.where(
                    do_lead[:, None, :],
                    cov_w[:, None, :] * combined + (1.0 - cov_w[:, None, :]) * extrap_lead,
                    combined,
                )
                # Trailing rows (r > i1)
                tail_start = i1 - span + 1
                in_tail = (r_idx[None, :] >= tail_start[:, None]) & (r_idx[None, :] <= i1[:, None])
                xs_tail_local = (r_idx[None, :] - tail_start[:, None]).to(combined.dtype)
                xs_c_tail = torch.where(in_tail, xs_tail_local - xs_mean[:, None], zero_s)
                slope_tail = (combined * xs_c_tail[:, None, :]).sum(dim=2) / (
                    xs_c_tail.pow(2).sum(dim=1, keepdim=True) + 1e-8
                )
                val_i1 = combined.gather(
                    2, i1.clamp(0, num_rows_knot - 1)[:, None, None].expand(-1, combined.shape[1], 1)
                )
                idx_tail = (r_idx[None, :] - i1[:, None]).to(combined.dtype)
                extrap_tail = val_i1 + slope_tail[:, :, None] * idx_tail[:, None, :]
                do_tail = (has_enough & (i1 < num_rows_knot - 1))[:, None] & (r_idx[None, :] > i1[:, None])
                combined = torch.where(
                    do_tail[:, None, :],
                    cov_w[:, None, :] * combined + (1.0 - cov_w[:, None, :]) * extrap_tail,
                    combined,
                )
                knots_batch.copy_(
                    base_batch + combined.reshape(num_images, 2, K, num_rows_knot).permute(0, 1, 3, 2)
                )
            else:
                # Smooth/detrend along the row axis.  Treat each (axis, intra-row
                # knot) slot as an independent series along rows by moving the row
                # dim last and flattening the leading channels.
                knots_flat = knots_batch.permute(0, 1, 3, 2).reshape(-1, num_rows_knot).T
                if vander.device.type == "mps":
                    # MPS does not implement lstsq. The normalized polynomial basis
                    # has at most four columns, so its full-rank system is small.
                    if vander.shape[0] < vander.shape[1]:
                        coefficients = torch.linalg.lstsq(
                            vander.cpu(), knots_flat.cpu()
                        ).solution.to(vander.device)
                    else:
                        normal_matrix = vander.T @ vander
                        coefficients = torch.linalg.solve(
                            normal_matrix, vander.T @ knots_flat
                        )
                else:
                    coefficients = torch.linalg.lstsq(vander, knots_flat).solution
                trend = (vander @ coefficients).T
                residual = knots_flat.T - trend
                smoothed = drift_knots.gaussian_smooth_1d(residual, sigma_px)
                knots_batch.copy_(
                    (smoothed + trend)
                    .reshape(num_images, 2, K, num_rows_knot)
                    .permute(0, 1, 3, 2))
        if step_size is not None:
            knots_batch.copy_(knots_prev + (knots_batch - knots_prev) * step_size)
        if max_slope is not None and init_batch is not None:
            _project_nonfolding_slope(knots_batch, init_batch, max_slope)


def setup_loss_kernel(
    self,
    K: int,
    canvas_shape: tuple[int, int],
    loss: str = "mse",
    tile_grid: int = 4,
):
    """Pick the K-aware compiled loss kernel and precompute its constants.

    Returns ``(loss_fn, loss_args)`` ready to be passed to
    ``_optimize_knots_adam`` / ``_optimize_knots_lbfgs``.

    K=1 path: scan_fast walk offsets per image (``row_scan_offsets`` /
    ``col_scan_offsets``).  K>=2 path: per-output-column segment indices
    + local fractions for the linear knot interpolation.  Constants are
    lifted out of the inner Adam / LBFGS loop so the compiled kernel
    sees them as static.

    ``loss`` selects the scalar: ``"mse"`` / ``"gradient_mse"`` use MSE
    kernels; ``"ncc"`` uses ``1 - mean NCC``; ``"local_ncc"`` uses hybrid
    global + ``tile_grid × tile_grid`` tile-local NCC.
    """
    device, dtype = self._device, self._dtype
    num_images = self.shape[0]
    row_scale = 2.0 / (canvas_shape[0] - 1)
    col_scale = 2.0 / (canvas_shape[1] - 1)
    use_ncc = loss == "ncc"
    use_local_ncc = loss == "local_ncc"
    if K > 1:
        # Identical to _transform_coordinates_multi_knot's geometry so
        # apply_correction downstream inverts it cleanly.
        num_cols = self.imgs[0].shape[1]
        t = torch.linspace(0, 1, num_cols, dtype=dtype, device=device) * (K - 1)
        seg_idx = torch.clamp(t.long(), max=K - 2)
        seg_frac = t - seg_idx.to(dtype)
        if use_local_ncc:
            return (_compiled_loss_fn_multi_local_ncc, (seg_idx, seg_frac, row_scale, col_scale, int(tile_grid)))
        fn = (
            _compiled_loss_fn_multi_ncc
            if use_ncc
            else _compiled_loss_fn_multi
        )
        return (fn, (seg_idx, seg_frac, row_scale, col_scale))
    # K=1: same scan-position vector projects onto row/col via scan_fast.
    u_t = [
        torch.as_tensor(self.u_per_image[i], dtype=dtype, device=device)
        for i in range(num_images)
    ]
    row_scan_offsets = torch.stack(
        [
            u_t[i] * (self.scan_fast[i][0] * (self.imgs[i].shape[0] - 1))
            for i in range(num_images)
        ]
    )
    col_scan_offsets = torch.stack(
        [
            u_t[i] * (self.scan_fast[i][1] * (self.imgs[i].shape[1] - 1))
            for i in range(num_images)
        ]
    )
    if use_local_ncc:
        return (
            _compiled_loss_fn_single_local_ncc,
            (row_scan_offsets, col_scan_offsets, row_scale, col_scale, int(tile_grid)),
        )
    fn = (
        _compiled_loss_fn_single_ncc
        if use_ncc
        else _compiled_loss_fn_single
    )
    return (fn, (row_scan_offsets, col_scan_offsets, row_scale, col_scale))


def correct_nonrigid(
    self,
    *,
    num_knots: int | None = None,
    optimizer: str = "adam",
    num_refine_cycles: int = 16,
    knot_smoothing_sigma: float = 8.0,
    update_fraction: float | None = 0.8,
    trend_order: int = 1,
    max_image_shift: float = 32.0,
    optimizer_steps: int | None = 30,
    learning_rate: float | None = None,
    max_knot_step: float | None = None,
    fixed_scans: list[int] | None = None,
    loss: str = "auto",
    edge_smoothing_sigma: float = 1.0,
    lowpass: float = 0.0,
    cost_taper: int = 0,
    early_stop_patience: int = 3,
    early_stop_rtol: float = 1e-4,
    min_iterations: int = 4,
    tile_grid: int = 4,
    coverage_weighted: bool = False,
    max_slope: float | None = None,
    solve_translation: bool | None = None,
    deadline: float | None = None,
    show_combined: bool = True,
    show_scans: bool = False,
    show_knots: bool = True,
    show_knot_plot: bool = False,
    show_report: bool = False,
    verbose: bool = True,
):
    """Remove residual scanline drift after affine or strip correction.

    The optimizer updates scanline knots against the other corrected scans or
    a fixed reference. Smooth knot regularization suppresses scanline jitter
    while retaining drift that changes gradually across the acquisition.

    Prefer :meth:`correct_strip` first when residual after affine is a
    smooth function of slow-scan position (deceleration / bend), especially
    for reference-mode EDS on pure lattices - free nonrigid with a large
    ``max_image_shift`` can lattice-alias. On atomic lattices keep
    ``max_image_shift`` small (e.g. 2).

    Calls are cumulative. On an atomic lattice, begin with
    ``max_image_shift=1`` or ``2`` and strong regularization (8-16 rows), then
    accept the stage only when common-mask NCC and the RGB overlay improve.
    In reference mode, the affine/strip solution already establishes the
    external reference frame, so nonrigid iterations preserve that frame
    instead of re-solving a potentially lattice-aliased global translation.

    Parameters
    ----------
    num_knots : int or None, default None
        Knots along every fast-scan line. ``None`` keeps the current layout;
        use more than one when residual drift changes within a scanline. The
        existing affine or strip field is preserved before non-rigid
        optimization begins.
    optimizer : str, default "adam"
        ``"adam"`` (first-order momentum) or ``"lbfgs"`` (quasi-Newton
        with strong-Wolfe line search). Adam is the fastest default for
        ≤1024 px images. LBFGS auto-scales the step size and is preferred
        for ≥2048 px or when the drift magnitude is unknown. Don't normalize
        inputs to [0, 1] when using LBFGS - strong-Wolfe needs absolute
        gradient magnitude and silently returns step=0 on unit-variance images.
    num_refine_cycles : int, default 16
        Outer iterations for alternating reference build + knot update.
    knot_smoothing_sigma : float, default 8.0
        Gaussian smoothing sigma for knot regularization. 4-12 typical
        for STEM data; smaller = finer per-row correction.
    update_fraction : float, default 0.8
        Step size for knot updates (0-1, lower = more conservative).
    trend_order : int, default 1
        Polynomial order for trend removal in knot regularization.
    max_image_shift : float, default 32.0
        Maximum shift for translation alignment between iterations.
        Adam's auto-``learning_rate`` derives from this - set close to your expected
        drift bound, otherwise Adam silently under-converges.
    optimizer_steps : int or None, default 30
        Inner optimizer steps per refinement cycle. ``None`` selects 30
        for Adam and 20 for LBFGS.
    learning_rate : float or None, default None
        Adam learning rate. When ``None``, it is derived as
        ``max_image_shift / (num_refine_cycles * optimizer_steps * 4)``.
        Adam's ``m/sqrt(v)`` update self-normalizes the gradient, so each
        step moves a knot by ~``learning_rate`` pixels regardless of intensity scale.
        Override only when you know the actual drift magnitude.
    fixed_scans : list[int] or None, default None
        Indices of images whose knots are frozen (their mean becomes the
        target for every moving image). Use ``[0]`` for single-sided
        alignment with a HAADF reference. Auto-set to ``[0]`` in
        reference mode (when the constructor was called with a 2-D ref +
        ≥3-D drifted dataset).
    loss : str, default "auto"
        Most drift workflows compare **similar detectors** (0°/90° HAADF,
        survey HAADF vs EDS-session HAADF, etc.). ``"auto"`` follows that
        typical case and resolves to ``"ncc"`` (coverage-masked normalized
        cross-correlation), so the train loss matches regional NCC reports.
        Explicit choices:

        - ``"ncc"`` - coverage-masked ``1 - mean NCC``; brightness / offset
          invariant; excludes unmeasured canvas fill. Default via ``"auto"``
          for similar-detector pairs.
        - ``"mse"`` - raw-intensity MSE when both images share the same
          detector and nearly identical gain (no intensity bias).
        - ``"gradient_mse"`` - Sobel-gradient MSE after pre-smooth +
          per-image z-score; only needed for **dissimilar / cross-detector**
          pairs (e.g. HAADF + virtual dark field).

    edge_smoothing_sigma : float, default 1.0
        Gaussian sigma applied before Sobel when ``loss="gradient_mse"``.
        Set to 0 to disable. Ignored for ``"mse"`` / ``"ncc"``.
    early_stop_patience : int, default 3
        Stop when ``patience`` consecutive iterations show no improvement.
        Set to ``num_refine_cycles`` to disable.
    early_stop_rtol : float, default 1e-4
        Minimum relative improvement to count as progress.
    min_iterations : int, default 4
        Floor before early stopping can trigger.
    show_combined, show_scans : bool
        Display knobs forwarded to the plot helpers.
    show_report : bool, default False
        Print all completed regional NCC checkpoints, including the
        nonrigid result.
    verbose : bool, default True
        Show refinement-cycle progress with the current and best mean
        absolute alignment error. Set to ``False`` for compact notebooks.

    Returns
    -------
    Self
        For method chaining.

    Examples
    --------
    Similar-detector HAADF pair (``loss="auto"`` → NCC):

    >>> dc = DriftCorrection(im0, im1, scan_direction_degrees=[0, 90])
    >>> dc.correct_affine(show_combined=False)
    >>> dc.correct_nonrigid()  # auto → ncc

    Atomic lattice: keep the shift tiny so nonrigid cannot lattice-alias:

    >>> dc.correct_nonrigid(max_image_shift=2)

    Let residual motion vary along the fast-scan direction:

    >>> dc.correct_nonrigid(num_knots=6, max_image_shift=2)

    Dissimilar / cross-detector only (HAADF + VDF):

    >>> dc.correct_nonrigid(loss="gradient_mse", knot_smoothing_sigma=8.0)

    Notes
    -----
    Plotting refreshes the cached warped images lazily. ``corrected()`` builds
    its result directly from the fitted knots.
    """
    if not hasattr(self, "knots"):
        raise RuntimeError("No knots found. Call .preprocess() before running alignment.")
    # Reloaded tensors start on the host; move solve state to the selected
    # device before optimization.
    self.imgs_t = [t.to(self._device) for t in self.imgs_t]
    self.knots = [k.to(self._device) for k in self.knots]
    for attr in ("_knots_after_affine", "_knots_after_strip", "_initial_knots"):
        snapshot = getattr(self, attr, None)
        if snapshot is not None:
            setattr(self, attr, [k.to(self._device) for k in snapshot])
    if num_knots is not None:
        drift_knots.resize_scanline_knots(self, num_knots)
    if loss == "auto":
        loss = "ncc"
    valid_losses = ("mse", "gradient_mse", "ncc", "local_ncc")
    if loss not in valid_losses:
        raise ValueError(f"loss must be one of {valid_losses!r} or 'auto', got {loss!r}")
    uses_cov_loss = loss in ("ncc", "local_ncc")
    if optimizer == "lbfgs" and self._normalized:
        import warnings

        warnings.warn(
            "normalize=True + LBFGS can cause silent convergence failure. "
            "Wolfe line search may return step=0 on unit-variance images. "
            "Consider using optimizer='adam' or normalize=False.",
            UserWarning,
            stacklevel=2,
        )
    # Reference-mode auto-anchors the reference image (index 0).
    if fixed_scans is None and self._reference_mode:
        fixed_scans = [0]
    fixed_set = frozenset(fixed_scans) if fixed_scans is not None else frozenset()
    moving_indices = [i for i in range(self.shape[0]) if i not in fixed_set]
    if fixed_set and not moving_indices:
        raise ValueError(
            f"All {self.shape[0]} images are in fixed_scans - nothing left to "
            "optimize. fixed_scans must leave at least one moving image."
        )
    device = self._device
    dtype = self._dtype
    num_images = self.shape[0]
    canvas_shape = (self.shape[1], self.shape[2])
    K_per_image = {self.knots[i].shape[2] for i in range(num_images)}
    if len(K_per_image) > 1:
        raise ValueError(f"All images must use the same number of knots, got {K_per_image}")
    K = K_per_image.pop()
    # Knots are 4D throughout the optimizer - K=1 just keeps the trailing 1
    # so downstream code (regularizer, warp, sync) doesn't branch on shape.
    knots_batch = (
        torch.stack([self.knots[i] for i in range(num_images)]).detach().requires_grad_(True)
    )
    init_knots_attr = getattr(self, "_initial_knots", None)
    init_batch = (
        torch.stack([init_knots_attr[i] for i in range(num_images)]).detach()
        if init_knots_attr is not None and (max_slope is not None or coverage_weighted)
        else None
    )
    num_rows_knot = knots_batch.shape[2]
    # a reloaded (AutoSerialize) object leaves imgs_t on CPU; alignment
    # tensors must live on the solve device or Sobel/warp kernels mismatch
    target_batch = torch.stack(self.imgs_t).to(self._device)
    loss_fn, loss_args = setup_loss_kernel(self, K, canvas_shape, loss=loss, tile_grid=tile_grid)
    optimizer_steps = (
        optimizer_steps if optimizer_steps is not None else (30 if optimizer == "adam" else 20)
    )
    if optimizer == "adam":
        # Auto-derive lr so the total movement budget covers a quarter
        # of max_image_shift. The safety factor of 4 (not 2) prevents
        # over-shooting at small image sizes where actual drift is well
        # below max_image_shift; at large sizes the same factor still
        # converges because the loss surface is smoother. See the `lr`
        # parameter docstring for the full rationale.
        adam_lr = (
            learning_rate
            if learning_rate is not None
            else max_image_shift / (num_refine_cycles * optimizer_steps * 4)
        )
        torch_optimizer = torch.optim.Adam([knots_batch], lr=adam_lr, fused=True)
    elif optimizer == "lbfgs":
        torch_optimizer = torch.optim.LBFGS(
            [knots_batch], lr=1.0, max_iter=optimizer_steps, line_search_fn="strong_wolfe"
        )
    else:
        raise ValueError(f"optimizer must be 'adam' or 'lbfgs', got {optimizer!r}")
    if knot_smoothing_sigma is not None and knot_smoothing_sigma > 0:
        x_knot = torch.arange(num_rows_knot, dtype=dtype, device=device)
        x_norm = (x_knot - x_knot.mean()) / x_knot.std()
        vander = torch.stack([x_norm**p for p in range(trend_order + 1)], dim=1)
    else:
        vander = None
    # For gradient_mse, warp the edge-filtered images instead of the
    # raw ones. Knots are spatial transforms independent of image
    # content, so optimizing in gradient space yields the same drift
    # field while being robust to intensity/contrast differences.
    # imgs_t_override threads Sobel images through warp_and_translate
    # without mutating self.imgs_t (which would silently corrupt
    # apply_correction, visualization, and error metrics afterwards).
    if loss == "gradient_mse":
        sobel_batch = sobel_gradient_magnitude(
            target_batch, edge_smoothing_sigma, device, dtype
        )
        imgs_t_override = [sobel_batch[i] for i in range(num_images)]
        target_batch = sobel_batch
    else:
        imgs_t_override = None
    # Affine (and an optional preceding strip stage) already establishes
    # the absolute frame in reference mode. Re-solving a global shift here
    # can hop by one lattice period after an otherwise good correction.
    # Mutual multi-scan mode still solves translation every outer step unless disabled.
    solve_global_translation = (not self._reference_mode) if solve_translation is None else bool(solve_translation)
    need_weights = uses_cov_loss or coverage_weighted
    warp_result = warp_and_translate(
        self,
        max_image_shift,
        upsample_factor=8,
        knots_batch=knots_batch,
        solve_translation=solve_global_translation,
        fixed_indices=fixed_set,
        imgs_t_override=imgs_t_override,
        return_weights=need_weights,
        lowpass=lowpass,
        ramp=cost_taper,
        sync_cpu=False,
    )
    if need_weights:
        warped_t, coverage_weights_t = warp_result
    else:
        warped_t = warp_result
        coverage_weights_t = None
    # Build a boolean mask on device to zero fixed gradients efficiently.
    # knots_batch is always 4D ``(N, 2, R, K)``; broadcast over (2, R, K).
    fixed_grad_mask = None
    if fixed_set:
        fixed_grad_mask = torch.ones(num_images, 1, 1, 1, dtype=dtype, device=device)
        for idx in fixed_set:
            fixed_grad_mask[idx] = 0.0
    error_buffer = []
    best_error = float("inf")
    best_knots = knots_batch.detach().clone()
    patience_counter = 0
    # Smart Adam step cap: when max_knot_step and lr are both set, extra steps past
    # ~1.8 * max_knot_step / lr get clipped away by _regularize_knots.
    eff_adam_steps = optimizer_steps
    if optimizer == "adam" and max_knot_step is not None and adam_lr > 0:
        eff_adam_steps = min(
            int(optimizer_steps),
            max(8, int(math.ceil(1.8 * float(max_knot_step) / float(adam_lr)))),
        )
    import time as _time_mod
    pbar = tqdm(
        range(num_refine_cycles),
        desc=f"Solving nonrigid drift ({optimizer})",
        disable=not verbose,
    )
    for iter_idx in pbar:
        if deadline is not None and iter_idx > 0 and _time_mod.monotonic() >= deadline:
            break
        # Build the reference under no_grad: arithmetic on warped_t (an
        # inference tensor) would otherwise return an autograd-tracked
        # leaf, and the optimizer would build a graph through it.
        with torch.no_grad():
            if fixed_set:
                # Fixed images define the reference for all moving images.
                # .contiguous() avoids stride-0 expand recompiles in torch.compile.
                fixed_mean = warped_t[sorted(fixed_set)].mean(0)
                ref_batch = fixed_mean[None].expand(num_images, -1, -1).contiguous()
            else:
                warped_sum = warped_t.sum(0)
                ref_batch = (warped_sum[None] - warped_t) / (num_images - 1)
            knots_prev = knots_batch.detach().clone()
            ref_coverage_batch = None
            row_cov = None
            if need_weights:
                # ``bilinear_kde_batch`` considers weights >= 1e-3 covered.
                # Reuse that exact ramp so the optimizer and renderer agree
                # about which reference pixels are measured rather than fill.
                coverage_batch = (coverage_weights_t / 1e-3).clamp(0.0, 1.0)
                if fixed_set:
                    fixed_coverage = coverage_batch[sorted(fixed_set)].amin(0)
                    ref_coverage_batch = fixed_coverage[None].expand(
                        num_images, -1, -1
                    ).contiguous()
                else:
                    ref_coverage_batch = torch.stack(
                        [
                            coverage_batch[
                                [j for j in range(num_images) if j != i]
                            ].amin(0)
                            for i in range(num_images)
                        ]
                    )
            if uses_cov_loss:
                cov_bordered = ref_coverage_batch.clone()
                cov_bordered[:, 0, :] = 0.0
                cov_bordered[:, -1, :] = 0.0
                cov_bordered[:, :, 0] = 0.0
                cov_bordered[:, :, -1] = 0.0
                ref_opt = torch.stack([ref_batch, cov_bordered], dim=1)
            else:
                ref_opt = ref_batch[:, None]
            cycle_loss_args = loss_args
            cycle_grad_mask = fixed_grad_mask
            if coverage_weighted and ref_coverage_batch is not None:
                if K == 1:
                    r_off, c_off, r_sc, c_sc = loss_args[:4]
                    gr0 = (knots_batch[:, 0, :, :] + r_off[:, None, :]) * r_sc - 1.0
                    gc0 = (knots_batch[:, 1, :, :] + c_off[:, None, :]) * c_sc - 1.0
                else:
                    s_idx, s_frac, r_sc, c_sc = loss_args[:4]
                    lo = knots_batch[:, :, :, s_idx]
                    hi = knots_batch[:, :, :, s_idx + 1]
                    ip = lo + (hi - lo) * s_frac[None, None, None, :]
                    gr0 = ip[:, 0] * r_sc - 1.0
                    gc0 = ip[:, 1] * c_sc - 1.0
                g0 = torch.stack([gc0, gr0], dim=-1)
                row_cov = F.grid_sample(
                    ref_coverage_batch[:, None], g0,
                    mode='bilinear', align_corners=True, padding_mode='zeros',
                )[:, 0].mean(dim=-1)
                rw_mask = ((row_cov - 0.10) / 0.40).clamp(0.0, 1.0)[:, None, :, None]
                cycle_grad_mask = rw_mask if fixed_grad_mask is None else (rw_mask * fixed_grad_mask)
        # Regularization alters the loss surface between outer iters, so
        # stale momentum / curvature history would push knots the wrong way.
        torch_optimizer.state.clear()
        if optimizer == "adam":
            _optimize_knots_adam(
                ref_opt,
                target_batch,
                knots_batch,
                loss_fn,
                cycle_loss_args,
                torch_optimizer,
                eff_adam_steps,
                grad_mask=cycle_grad_mask,
            )
        else:
            _optimize_knots_lbfgs(
                ref_opt,
                target_batch,
                knots_batch,
                loss_fn,
                cycle_loss_args,
                torch_optimizer,
                grad_mask=cycle_grad_mask,
            )
        _regularize_knots(
            knots_batch,
            knots_prev,
            vander,
            max_knot_step,
            knot_smoothing_sigma,
            update_fraction,
            row_cov=row_cov,
            init_batch=init_batch,
            max_slope=max_slope,
        )
        # Restore fixed knots - regularization is a global smooth that
        # would subtly shift them via polynomial detrend + Gaussian blur.
        if fixed_set:
            with torch.no_grad():
                for idx in fixed_set:
                    knots_batch[idx] = knots_prev[idx]
        del ref_batch, ref_opt, ref_coverage_batch
        warp_result = warp_and_translate(
            self,
            max_image_shift,
            upsample_factor=8,
            knots_batch=knots_batch,
            solve_translation=solve_global_translation,
            fixed_indices=fixed_set,
            imgs_t_override=imgs_t_override,
            return_weights=need_weights,
            lowpass=lowpass,
            ramp=cost_taper,
            sync_cpu=False,
        )
        if need_weights:
            warped_t, coverage_weights_t = warp_result
        else:
            warped_t = warp_result
        # Per-iter error stays on GPU; monitor common-coverage masked 1-NCC when
        # optimizing NCC/local_ncc so early stopping matches the objective.
        images_mean = warped_t.mean(dim=0)
        iter_error = torch.mean(torch.abs(warped_t - images_mean[None]), dim=(1, 2))
        error_buffer.append(iter_error)
        if uses_cov_loss and coverage_weights_t is not None and num_images == 2:
            cm = (coverage_weights_t >= 1e-3).all(dim=0)
            cm_f = cm.to(warped_t.dtype)
            cnt = cm_f.sum().clamp_min(1.0)
            w0m = warped_t[0] - (warped_t[0] * cm_f).sum() / cnt
            w1m = warped_t[1] - (warped_t[1] * cm_f).sum() / cnt
            ncc_mon = (cm_f * w0m * w1m).sum() / (
                (cm_f * w0m.square()).sum().clamp_min(1e-12).sqrt()
                * (cm_f * w1m.square()).sum().clamp_min(1e-12).sqrt()
            )
            current_error = float((1.0 - ncc_mon).item())
        else:
            current_error = float(iter_error.mean().item())
        if current_error < best_error * (1 - early_stop_rtol):
            best_error = current_error
            best_knots = knots_batch.detach().clone()
            patience_counter = 0
        else:
            patience_counter += 1
        pbar.set_postfix_str(
            f"error={current_error:.4g}, best={best_error:.4g}",
            refresh=verbose,
        )
        if iter_idx >= min_iterations - 1 and patience_counter >= early_stop_patience:
            pbar.set_postfix_str(
                f"converged at cycle {iter_idx + 1}, "
                f"error={current_error:.4g}, best={best_error:.4g}",
                refresh=verbose,
            )
            break
    # Sync best knots back (never return post-patience degraded knots); leave
    # imgs_warped lazy so callers that never plot avoid GPU→CPU copies.
    knots_final = best_knots if best_error < float("inf") else knots_batch.detach()
    for img_idx in range(num_images):
        self.knots[img_idx][...] = knots_final[img_idx]
    self._images_warped_stale = True
    self._max_image_shift_cached = max_image_shift
    torch_optimizer.state.clear()
    knots_batch.grad = None
    del warped_t, coverage_weights_t, target_batch, knots_batch, best_knots
    if error_buffer:
        # Transfer and append the full convergence history once.
        errors_np = torch.stack(error_buffer).cpu().numpy()  # (num_iterations, num_images)
        mode_col = np.full((len(errors_np), 1), 2.0)
        mean_col = errors_np.mean(axis=1, keepdims=True)
        new_rows = np.hstack((mode_col, mean_col, errors_np))
        if not hasattr(self, "error_track"):
            self.error_track = new_rows
        else:
            self.error_track = np.vstack((self.error_track, new_rows))

    drift_plot.show_after_step(
        self,
        "non-rigid",
        show_combined=show_combined,
        show_scans=show_scans,
        show_knots=show_knots,
    )
    if show_knot_plot:
        self.plot_knots()
    if show_report:
        print(self.report().to_string())
    return self
