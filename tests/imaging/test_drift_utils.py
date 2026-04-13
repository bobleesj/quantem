"""Parity tests for drift_utils.py torch functions.

Tests the core building blocks against numpy/scipy equivalents.
The full pipeline is covered by frozen baselines in test_drift.py.
"""

import numpy as np
import pytest
import torch
from scipy.ndimage import gaussian_filter

from quantem.core.utils.imaging_utils import bilinear_kde
from quantem.imaging.drift_utils import (
    _parabolic_peak_2d,
    _parabolic_sub_pixel,
    _symmetric_pad,
    backward_warp,
    bilinear_kde_batch,
    cross_corr_batch,
    fourier_shift_warp,
    gaussian_smooth_1d,
    gaussian_smooth_batch,
    initialize_scanline_knots,
)


# ---------------------------------------------------------------------------
# High-level: cross-correlation and warping
# ---------------------------------------------------------------------------


def test_cross_corr_zero_cost_for_identical():
    """Identical images must produce near-zero MAE after alignment.

    This validates the full sub-pixel pipeline: FFT cross-correlation →
    parabolic peak → DFT upsample → Fourier shift. If any step has a
    bias (like the 0.5 px center-index bug), identical images will show
    a nonzero cost from the spurious shift.
    """
    rng = np.random.default_rng(42)
    image = rng.random((64, 64)).astype(np.float32)
    reference = torch.tensor(image)[None]
    cost = cross_corr_batch(reference, reference.clone(), upsample_factor=8)
    assert cost.item() < 1e-6


@pytest.mark.parametrize("shift_row,shift_col", [(0, 0), (3, -5), (7, 2), (2.3, -1.7)])
def test_parabolic_peak_2d_known_shift(shift_row, shift_col):
    """Parabolic peak refinement must recover known shifts from Fourier-shifted images.

    At zero shift, this catches the negative-float-rounding bug where
    (-4.5e-8) % N = N instead of 0. At nonzero shifts, it verifies
    the periodic wrapping, stencil extraction, and sub-pixel precision.
    """
    rng = np.random.default_rng(42)
    num_pixels = 64
    image = rng.random((num_pixels, num_pixels)).astype(np.float64)
    # Fourier shift creates a perfect sub-pixel-accurate shifted image
    k_row = np.fft.fftfreq(num_pixels)[:, None]
    k_col = np.fft.fftfreq(num_pixels)[None, :]
    shifted = np.real(np.fft.ifft2(
        np.fft.fft2(image) * np.exp(-2j * np.pi * (k_row * shift_row + k_col * shift_col))
    ))
    reference = torch.tensor(image, dtype=torch.float32)[None]
    moving = torch.tensor(shifted, dtype=torch.float32)[None]
    cross_corr = torch.fft.ifft2(torch.fft.fft2(reference) * torch.fft.fft2(moving).conj()).real
    peak_flat = cross_corr.flatten(1).argmax(dim=1)
    peak_row = peak_flat // num_pixels
    peak_col = peak_flat % num_pixels
    batch_idx = torch.arange(1)
    refined_row, refined_col = _parabolic_peak_2d(
        cross_corr, peak_row, peak_col, num_pixels, num_pixels, batch_idx)
    # Cross-correlation finds the negative shift, wrapped to [0, N)
    expected_row = (-shift_row) % num_pixels
    expected_col = (-shift_col) % num_pixels
    # Parabolic gives ~0.1 px precision on sub-pixel shifts, exact on integer
    tolerance = 0.15
    assert abs(refined_row.item() - expected_row) < tolerance, f"Row: expected {expected_row}, got {refined_row.item()}"
    assert abs(refined_col.item() - expected_col) < tolerance, f"Col: expected {expected_col}, got {refined_col.item()}"


@pytest.mark.parametrize("scale", [1, 2])
def test_bilinear_kde_matches_numpy(scale):
    """Torch batched KDE scatter must match numpy bilinear_kde.

    This is the core warping operation: scatter source pixels onto a
    canvas with bilinear weights, smooth, and normalize. If this diverges,
    the affine grid search scores candidates differently and picks
    wrong drift vectors. Tested at two scales to catch size-dependent bugs.
    """
    rng = np.random.default_rng(42)
    num_rows_in = 32 * scale
    num_cols_in = 32 * scale
    num_rows_out = 40 * scale
    num_cols_out = 40 * scale
    kde_sigma = 0.5
    pad_value = 100.0
    source_image = rng.random((num_rows_in, num_cols_in)).astype(np.float32)
    # Fractional offsets (not integer) to exercise the bilinear weight split —
    # 4.3 means each pixel lands 0.3 of the way between grid points
    row_coords = (np.arange(num_rows_in)[:, None] + 4.3 * scale
                  + np.zeros((1, num_cols_in))).astype(np.float32)
    col_coords = (np.zeros((num_rows_in, 1))
                  + np.arange(num_cols_in)[None, :] + 4.7 * scale).astype(np.float32)
    expected = bilinear_kde(
        row_coords, col_coords, source_image,
        (num_rows_out, num_cols_out), kde_sigma, pad_value,
    )
    result, _ = bilinear_kde_batch(
        torch.tensor(row_coords)[None],
        torch.tensor(col_coords)[None],
        torch.tensor(source_image),
        (num_rows_out, num_cols_out),
        kde_sigma, pad_value,
    )
    np.testing.assert_allclose(
        result[0].numpy(), expected.astype(np.float32), atol=1e-5,
        err_msg=f"Warped image mismatch at scale={scale}",
    )


def test_initialize_scanline_knots_single_knot_centers_footprint():
    """Single-knot init should place a vertical anchor line at the scan start edge.

    For a 0 deg scan, the anchors are vertically arranged at constant column,
    while the full scanline footprint spans the canvas width symmetrically
    around the padded center once the fast-scan direction is applied.
    """
    input_shape = (4, 6)
    output_shape = (8, 10)
    scan_fast = np.array([0.0, 1.0])
    scan_slow = np.array([1.0, 0.0])

    knots = initialize_scanline_knots(
        input_shape=input_shape,
        output_shape=output_shape,
        scan_fast=scan_fast,
        scan_slow=scan_slow,
        number_knots=1,
    )

    expected_rows = np.arange(2.0, 6.0)
    expected_col = np.full(input_shape[0], 2.0)
    np.testing.assert_allclose(knots[0, :, 0], expected_rows)
    np.testing.assert_allclose(knots[1, :, 0], expected_col)


# ---------------------------------------------------------------------------
# Mid-level: smoothing and padding that the KDE depends on
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sigma", [0.5, 2.0])
def test_gaussian_smooth_matches_scipy(sigma):
    """Torch separable Gaussian must match scipy.ndimage.gaussian_filter.

    The KDE normalization step (values / counts) amplifies any smoothing
    mismatch. Tested at sigma=0.5 (tight, 3-pixel kernel) and sigma=2.0
    (wide, 9-pixel kernel) to cover both regimes.
    """
    rng = np.random.default_rng(42)
    image = rng.random((64, 64)).astype(np.float32)
    expected = gaussian_filter(image, sigma).astype(np.float32)
    result = gaussian_smooth_batch(torch.tensor(image)[None], sigma)[0].numpy()
    np.testing.assert_allclose(result, expected, atol=1e-5)


@pytest.mark.parametrize("sigma", [0.5, 2.0, 16.0])
def test_gaussian_smooth_1d_matches_scipy(sigma):
    """Torch 1D Gaussian must match scipy.ndimage.gaussian_filter on 1D signal.

    Used in nonrigid regularization to smooth knot residuals. sigma=16 is
    the default regularization_sigma_px — tests the exact kernel size used
    in production. If this diverges, the polynomial-detrend + smooth
    regularization produces different knot positions.
    """
    rng = np.random.default_rng(42)
    signal = rng.random(128).astype(np.float32)
    expected = gaussian_filter(signal, sigma).astype(np.float32)
    result = gaussian_smooth_1d(torch.tensor(signal)[None], sigma)[0].numpy()
    np.testing.assert_allclose(result, expected, atol=1e-5)


def test_symmetric_pad_matches_numpy():
    """Torch symmetric padding must match np.pad(mode='symmetric').

    This is critical for parity: scipy's gaussian_filter uses symmetric
    (reflect-with-edge-repeat) boundaries. If our torch padding differs,
    the smoothed KDE images diverge at canvas edges and the frozen
    baselines in test_drift.py break.
    """
    rng = np.random.default_rng(42)
    image = rng.random((8, 10)).astype(np.float32)
    pad_rows, pad_cols = 3, 4
    expected = np.pad(image, ((pad_rows, pad_rows), (pad_cols, pad_cols)), mode="symmetric")
    result = _symmetric_pad(torch.tensor(image)[None, None], pad_rows, pad_cols)[0, 0].numpy()
    np.testing.assert_allclose(result, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# Low-level: sub-pixel math primitives
# ---------------------------------------------------------------------------


def test_parabolic_sub_pixel_exact():
    """Parabolic fit on y = -(x - offset)^2 must recover the exact offset.

    This is the sub-pixel refinement used to center the DFT upsample
    window. If the offset is wrong, the upsampled patch misses the
    true correlation peak and the shift estimate degrades.
    """
    for offset in [0.0, 0.3, -0.4, 0.49]:
        val_m1 = torch.tensor([-((-1 - offset) ** 2)])
        val_0 = torch.tensor([-(0 - offset) ** 2])
        val_p1 = torch.tensor([-(1 - offset) ** 2])
        result = _parabolic_sub_pixel(val_m1, val_0, val_p1)
        assert abs(result.item() - offset) < 1e-6, f"Failed for offset={offset}"


# ---------------------------------------------------------------------------
# backward_warp: grid_sample inverse of bilinear_kde_batch
# ---------------------------------------------------------------------------


def test_backward_warp_identity():
    """Zero drift and zero rigid shift must return the input unchanged."""
    rng = np.random.default_rng(42)
    image = torch.tensor(rng.random((64, 64)).astype(np.float32))
    result = backward_warp(image, drift=(0.0, 0.0), rigid_shift=(0.0, 0.0))
    np.testing.assert_allclose(result.numpy(), image.numpy(), atol=1e-5)


def test_backward_warp_pure_translation():
    """A rigid shift with zero drift should translate the image.

    Shift a smooth image by a known amount and verify the center region
    matches the expected translated content.
    """
    n = 64
    rng = np.random.default_rng(42)
    from scipy.ndimage import gaussian_filter as gf
    image_np = gf(rng.random((n, n)).astype(np.float32), sigma=3)
    image = torch.tensor(image_np)

    shift_row, shift_col = 3.0, -2.0
    result = backward_warp(image, drift=(0.0, 0.0), rigid_shift=(shift_row, shift_col))

    # Fourier shift for ground truth
    k_row = np.fft.fftfreq(n)[:, None]
    k_col = np.fft.fftfreq(n)[None, :]
    expected = np.real(np.fft.ifft2(
        np.fft.fft2(image_np)
        * np.exp(-2j * np.pi * (k_row * shift_row + k_col * shift_col))
    )).astype(np.float32)

    # Compare center region (avoid border effects from grid_sample vs Fourier wrapping)
    c = 10
    np.testing.assert_allclose(
        result.numpy()[c:-c, c:-c], expected[c:-c, c:-c], atol=0.02,
    )


def test_backward_warp_batch():
    """Batch dim should work: (N, H, W) input returns (N, H, W) output."""
    rng = np.random.default_rng(42)
    images = torch.tensor(rng.random((3, 32, 32)).astype(np.float32))
    result = backward_warp(images, drift=(0.01, -0.02))
    assert result.shape == (3, 32, 32)


def test_backward_warp_reverses_known_drift():
    """Apply a known column drift then undo it; center region must match original.

    This is the key correctness test: forward-drift an image by shifting
    each scanline, then call backward_warp with the same drift rate.
    The center region of the round-tripped image should match the original.
    """
    n = 128
    rng = np.random.default_rng(42)
    from scipy.ndimage import gaussian_filter as gf
    original = gf(rng.random((n, n)).astype(np.float32), sigma=4)

    drift_col = 0.05  # 0.05 px/line → 6.4 px total
    offset = np.arange(n) - (n - 1) / 2

    # Forward drift: shift each scanline's columns
    drifted = np.zeros_like(original)
    for r in range(n):
        shift = drift_col * offset[r]
        # sub-pixel shift via Fourier
        k = np.fft.fftfreq(n)
        row_fft = np.fft.fft(original[r])
        drifted[r] = np.real(np.fft.ifft(row_fft * np.exp(-2j * np.pi * k * shift)))

    corrected = backward_warp(
        torch.tensor(drifted), drift=(0.0, drift_col),
    ).numpy()

    # Check center region matches original (avoid edges where content is lost)
    c = 15
    np.testing.assert_allclose(
        corrected[c:-c, c:-c], original[c:-c, c:-c], atol=0.05,
    )


# ---------------------------------------------------------------------------
# fourier_shift_warp
# ---------------------------------------------------------------------------


def test_fourier_shift_warp_identity():
    """Zero drift + zero rigid shift should return the input unchanged."""
    img = torch.randn(64, 64, dtype=torch.float32)
    out = fourier_shift_warp(img, drift=(0.0, 0.0))
    torch.testing.assert_close(out, img, atol=1e-5, rtol=1e-5)


def test_fourier_shift_warp_reverses_known_drift():
    """Fourier warp should recover a Fourier-drifted image with near-zero error.

    Unlike backward_warp (bilinear/bicubic), the column-direction shift
    is exact — the only error comes from row-direction linear mixing.
    """
    n = 128
    rng = np.random.default_rng(42)
    from scipy.ndimage import gaussian_filter as gf
    original = gf(rng.random((n, n)).astype(np.float32), sigma=4)

    drift_col = 0.05  # 0.05 px/line → 6.4 px total
    offset = np.arange(n) - (n - 1) / 2

    # Forward drift: shift each scanline's columns via Fourier (exact)
    drifted = np.zeros_like(original)
    for r in range(n):
        shift = drift_col * offset[r]
        k = np.fft.fftfreq(n)
        row_fft = np.fft.fft(original[r])
        drifted[r] = np.real(np.fft.ifft(row_fft * np.exp(-2j * np.pi * k * shift)))

    corrected = fourier_shift_warp(
        torch.tensor(drifted), drift=(0.0, drift_col),
    ).numpy()

    # Column-only drift → Fourier correction should be near-exact
    c = 15
    np.testing.assert_allclose(
        corrected[c:-c, c:-c], original[c:-c, c:-c], atol=0.01,
    )


def test_fourier_shift_warp_beats_bilinear_on_high_freq():
    """Fourier roundtrip preserves high-frequency content better than bilinear.

    Drift a high-freq image, correct with each method, compare to original.
    Fourier should be more accurate because it doesn't attenuate frequencies.
    """
    n = 256
    x = np.arange(n, dtype=np.float32)
    # Integer cycles for perfect periodicity (avoids Fourier boundary artifacts)
    cycles = int(0.4 * n)  # 102 full cycles
    img = np.sin(2 * np.pi * cycles / n * x)[None, :].repeat(n, axis=0).astype(np.float32)

    drift_col = 0.03  # 0.03 px/line → ~3.8 px total
    offset = np.arange(n) - (n - 1) / 2

    # Forward drift: shift each scanline's columns via Fourier (exact)
    drifted = np.zeros_like(img)
    for r in range(n):
        shift = drift_col * offset[r]
        k = np.fft.fftfreq(n)
        row_fft = np.fft.fft(img[r])
        drifted[r] = np.real(np.fft.ifft(row_fft * np.exp(-2j * np.pi * k * shift)))

    drifted_t = torch.tensor(drifted)
    bilinear_out = backward_warp(drifted_t, drift=(0.0, drift_col), mode="bilinear").numpy()
    fourier_out = fourier_shift_warp(drifted_t, drift=(0.0, drift_col)).numpy()

    c = 20
    err_bilinear = np.sqrt(np.mean((bilinear_out[c:-c, c:-c] - img[c:-c, c:-c]) ** 2))
    err_fourier = np.sqrt(np.mean((fourier_out[c:-c, c:-c] - img[c:-c, c:-c]) ** 2))
    assert err_fourier < err_bilinear, (
        f"Fourier ({err_fourier:.6f}) should beat bilinear ({err_bilinear:.6f})"
    )
