"""Optimization primitives for nonrigid drift correction.

Pure module functions: no instance state, no I/O. ``DriftCorrection.align_nonrigid``
calls into here for the per-step Adam / LBFGS work, the fused loss kernel,
the Sobel gradient signal used by ``loss="gradient_mse"``, and the
post-step knot regularizer.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from quantem.imaging.drift_knot import gaussian_smooth_1d


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


@torch.compile(mode="reduce-overhead", dynamic=False)
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
    the precomputed ``scan_fast`` walk.  ``align_nonrigid`` selects this
    kernel when ``K == 1`` and :func:`_compiled_loss_fn_multi` for ``K > 1``.
    """
    grid_row = (knots_batch[:, 0, :, :] + row_scan_offsets[:, None, :]) * row_scale - 1.0
    grid_col = (knots_batch[:, 1, :, :] + col_scan_offsets[:, None, :]) * col_scale - 1.0
    return _grid_sample_mse(grid_row, grid_col, ref_t, target_batch)


@torch.compile(mode="reduce-overhead", dynamic=False)
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


def _optimize_knots_adam(
    ref_batch, target_batch, knots_batch,
    loss_fn, loss_args,
    optimizer, adam_steps, grad_mask=None,
):
    """Run ``adam_steps`` of Adam on a batched knot tensor against ``loss_fn``."""
    ref_t = ref_batch[:, None]
    for _ in range(adam_steps):
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
    ref_t = ref_batch[:, None]
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
    sx = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        dtype=dtype, device=device).reshape(1, 1, 3, 3)
    sy = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
        dtype=dtype, device=device).reshape(1, 1, 3, 3)
    img_pad = F.pad(img, (1, 1, 1, 1), mode='reflect')
    gx = F.conv2d(img_pad, sx)
    gy = F.conv2d(img_pad, sy)
    grad_mag = (gx ** 2 + gy ** 2).sqrt()[:, 0]
    mean = grad_mag.mean(dim=(-2, -1), keepdim=True)
    std = grad_mag.std(dim=(-2, -1), keepdim=True).clamp(min=1e-8)
    return (grad_mag - mean) / std


def _regularize_knots(
    knots_batch: torch.Tensor,
    knots_prev: torch.Tensor,
    vander: torch.Tensor | None,
    max_shift_px: float | None,
    sigma_px: float | None,
    step_size: float | None,
) -> None:
    """Apply per-iteration knot regularization (in place on ``knots_batch``).

    Three independent stages, each gated by its parameter being non-None:
        1. Per-knot shift cap: clamp ``|new - prev|`` to ``max_shift_px``
           so the optimizer can't move any knot too far in one outer iter.
        2. Polynomial detrend + Gaussian smooth: keep low-order trends,
           smooth the residual along the scan-line dimension. Removes
           high-frequency optimizer wobble while preserving the drift signal.
        3. Step-size blend: ``new = prev + step_size · (new - prev)``,
           under-relaxes the update for stability across outer iterations.
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
            # Smooth/detrend along the row axis.  Treat each (axis, intra-row
            # knot) slot as an independent series along rows by moving the row
            # dim last and flattening the leading channels.
            knots_flat = knots_batch.permute(0, 1, 3, 2).reshape(-1, num_rows_knot).T
            coefs, _, _, _ = torch.linalg.lstsq(vander, knots_flat)
            trend = (vander @ coefs).T
            residual = knots_flat.T - trend
            smoothed = gaussian_smooth_1d(residual, sigma_px)
            knots_batch.copy_(
                (smoothed + trend)
                .reshape(num_images, 2, K, num_rows_knot)
                .permute(0, 1, 3, 2))
        if step_size is not None:
            knots_batch.copy_(knots_prev + (knots_batch - knots_prev) * step_size)
