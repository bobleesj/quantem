"""Parity + analytic tests for the core/ building blocks.

Tests the geometry and math primitives against numpy/scipy equivalents, or
against a hand-computable analytic answer where no reference implementation
exists. The full pipeline is covered by frozen baselines in
test_drift_simulations.py.
"""

import numpy as np
import pytest
import torch
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter

from quantem.core.utils.imaging_utils import bilinear_kde
from quantem.imaging.drift.apply import largest_rectangle, warped_stack
from quantem.imaging.drift.core.knots import (
    DriftKnot,
    _symmetric_pad,
    bilinear_kde_batch,
    gaussian_smooth_1d,
    gaussian_smooth_batch,
    initialize_scanline_knots,
    resize_scanline_knots,
)
from quantem.imaging.drift.core.warping import (
    _parabolic_peak_2d,
    _parabolic_sub_pixel,
    backward_warp,
    backward_warp_grid_search,
    cross_corr_batch,
    fixed_overlap_ncc,
    translate_align,
    translate_align_pair_batch,
    warp_and_translate,
)
from quantem.imaging.drift.correction import DriftCorrection

# ---------------------------------------------------------------------------
# High-level: cross-correlation and warping
# ---------------------------------------------------------------------------


def test_diagnose_affine_reports_actual_regions_without_changing_knots():
    """Regional diagnostics should measure the delivered images without mutation."""
    rng = np.random.default_rng(20260802)
    reference = rng.random((64, 64), dtype=np.float32)
    moving = np.roll(reference, (3, -4), axis=(0, 1))
    correction = DriftCorrection.from_images(
        reference,
        moving,
        scan_direction_degrees=(0.0, 0.0),
        device="cpu",
    )
    correction.preprocess(
        padding_fraction=0.25,
        normalize=False,
        show_combined=False,
        show_scans=False,
        show_knots=False,
        verbose=False,
    )
    warped_stack(correction)
    knots = [value.clone() for value in correction.knots]
    smoothing_sigma = correction.kde_sigma
    warped = correction.imgs_warped.array.copy()
    warped_stale = correction._images_warped_stale
    warped_fingerprint = correction._warped_fingerprint

    figure, regions = correction.diagnose_affine(
        stage=None,
        smoothing_sigma=1.5,
    )

    assert regions.shape[0] == 4
    assert len(figure.axes) == 16
    assert figure.axes[0].get_title().startswith("Scan 0")
    assert "Residual difference" in figure.axes[3].get_title()
    assert np.all(regions["current_ncc"] < 0.1)
    assert np.all(regions["mean_absolute_difference"] > 0)
    for before, after in zip(knots, correction.knots, strict=True):
        torch.testing.assert_close(before, after)
    assert correction.kde_sigma == smoothing_sigma
    assert correction._images_warped_stale == warped_stale
    assert correction._warped_fingerprint == warped_fingerprint
    np.testing.assert_array_equal(correction.imgs_warped.array, warped)
    plt.close(figure)


def test_affine_region_correction_transfers_to_full_pair():
    """Trusted-region affine should correct the complete pair without notebook math."""
    rng = np.random.default_rng(20260803)
    reference = rng.random((128, 128), dtype=np.float32)
    moving_global = np.roll(reference, (3, -4), axis=(0, 1))
    moving = np.ascontiguousarray(np.rot90(moving_global, k=-1))
    correction = DriftCorrection.from_images(
        reference,
        moving,
        scan_direction_degrees=(0.0, -90.0),
        device="cpu",
    )

    correction.correct_affine(
        region="top_left",
        region_smoothing_sigma=0.5,
        max_image_shift=8,
        show_combined=False,
        show_scans=False,
        show_knots=False,
        verbose=False,
    )
    figure, regions = correction.diagnose_affine(
        smoothing_sigma=0.5,
    )

    assert correction.affine_search_info["strategy"] == "trusted_region"
    assert correction.affine_search_info["full_image_num_knots"] == 1
    assert correction.affine_search_info["trusted_region"] == "top_left"
    assert correction.affine_search_info["trusted_region_bounds_row_column"] == [
        0,
        64,
        0,
        64,
    ]
    assert all(knots.shape[-1] == 1 for knots in correction.knots)
    assert np.isfinite(regions["current_ncc"]).all()
    assert regions.loc[regions["region"] == "top left", "region_role"].item() == (
        "trusted affine fit"
    )
    figure, axes = correction.plot_combined(
        stage=("initial", "affine"),
        show_knots=True,
    )
    assert all(not axis.lines for axis in axes)
    plt.close(figure)
    figure, axis = correction.plot_combined(stage="nonrigid", show_knots=True)
    assert len(axis.lines) == 2
    plt.close(figure)


def test_affine_accepts_explicit_row_column_region_bounds():
    """Scientists should be able to fit a feature that crosses quadrant bounds."""
    rng = np.random.default_rng(20260804)
    reference = rng.random((128, 128), dtype=np.float32)
    moving = np.ascontiguousarray(
        np.rot90(np.roll(reference, (2, -3), axis=(0, 1)), k=-1)
    )
    correction = DriftCorrection.from_images(
        reference,
        moving,
        scan_direction_degrees=(0.0, -90.0),
        device="cpu",
    )

    correction.correct_affine(
        region=(8, 72, 16, 80),
        region_smoothing_sigma=0.5,
        max_image_shift=8,
        show_combined=False,
        show_scans=False,
        show_knots=False,
        verbose=False,
    )

    assert correction.affine_search_info["trusted_region"] == "custom"
    assert correction.affine_search_info["trusted_region_bounds_row_column"] == [
        8,
        72,
        16,
        80,
    ]
    corrected = correction.corrected()
    assert corrected.array.ndim == 2


def test_affine_rejects_region_bounds_outside_the_image():
    """Invalid bounds should identify the image shape and requested region."""
    image = np.ones((32, 32), dtype=np.float32)
    correction = DriftCorrection.from_images(
        image,
        image.copy(),
        scan_direction_degrees=(0.0, 0.0),
        device="cpu",
    )

    with pytest.raises(ValueError, match="outside the image shape"):
        correction.correct_affine(region=(0, 40, 0, 32))


def test_coverage_mask_uses_full_multiple_knot_interpolation():
    """Multi-knot diagnostics need the measured footprint of the full field."""
    rng = np.random.default_rng(20260810)
    image = rng.random((48, 48), dtype=np.float32)
    correction = DriftCorrection.from_images(
        image,
        image.copy(),
        scan_direction_degrees=(0.0, 0.0),
        device="cpu",
    )
    correction.preprocess(
        padding_fraction=0.25,
        num_knots=3,
        show_combined=False,
        show_scans=False,
        show_knots=False,
        verbose=False,
    )

    mask = correction.coverage_mask()

    assert mask.shape == image.shape
    assert mask.dtype == bool
    assert mask.mean() > 0.95


def test_resize_scanline_knots_preserves_affine_field():
    """Choosing non-rigid flexibility must not change the affine correction."""
    rng = np.random.default_rng(20260810)
    image = rng.random((48, 48), dtype=np.float32)
    correction = DriftCorrection.from_images(
        image,
        image.copy(),
        scan_direction_degrees=(0.0, 90.0),
        device="cpu",
    )
    correction.preprocess(
        padding_fraction=0.25,
        num_knots=1,
        show_combined=False,
        show_scans=False,
        show_knots=False,
        verbose=False,
    )
    row_ramp = torch.linspace(-2, 3, 48)
    for knots in correction.knots:
        knots[1, :, 0] += row_ramp
    correction._knots_after_affine = [value.clone() for value in correction.knots]
    before = warp_and_translate(
        correction,
        max_image_shift=None,
        solve_translation=False,
    )

    resize_scanline_knots(correction, 6)
    after = warp_and_translate(
        correction,
        max_image_shift=None,
        solve_translation=False,
    )

    assert all(value.shape[-1] == 6 for value in correction.knots)
    torch.testing.assert_close(before, after, rtol=2e-5, atol=2e-5)


def test_diagnose_nonrigid_compares_counts_without_changing_correction():
    """A knot-count study should return evidence and preserve its affine input."""
    rng = np.random.default_rng(20260810)
    reference = rng.random((32, 32), dtype=np.float32)
    moving = np.roll(reference, (1, -1), axis=(0, 1))
    correction = DriftCorrection.from_images(
        reference,
        moving,
        scan_direction_degrees=(0.0, 90.0),
        device="cpu",
    )
    correction.preprocess(
        padding_fraction=0.25,
        num_knots=1,
        show_combined=False,
        show_scans=False,
        show_knots=False,
        verbose=False,
    )
    correction._knots_after_affine = [value.clone() for value in correction.knots]
    warped_stack(correction)
    before = [value.clone() for value in correction.knots]
    warped = correction.imgs_warped.array.copy()
    warped_stale = correction._images_warped_stale
    warped_fingerprint = correction._warped_fingerprint

    figure, metrics = correction.diagnose_nonrigid(
        num_knots=(1, 2),
        num_refine_cycles=1,
        optimizer_steps=1,
        learning_rate=0.01,
        knot_smoothing_sigma=0,
        max_image_shift=2,
        loss="mse",
        early_stop_patience=1,
        min_iterations=1,
        verbose=False,
    )

    assert metrics["num_knots"].tolist() == [1, 2]
    assert {
        "common_ncc",
        "yellow",
        "fast_roughness_px",
        "seconds",
    }.issubset(metrics.columns)
    assert len(figure.axes) == 8
    for expected, actual in zip(before, correction.knots, strict=True):
        torch.testing.assert_close(expected, actual)
    assert correction._images_warped_stale == warped_stale
    assert correction._warped_fingerprint == warped_fingerprint
    np.testing.assert_array_equal(correction.imgs_warped.array, warped)
    plt.close(figure)


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


def test_fixed_overlap_ncc_ignores_padding_and_recovers_shift():
    """Measured overlap must choose the image shift instead of padded borders."""
    rng = np.random.default_rng(20260802)
    reference = rng.random((48, 48)).astype(np.float32)
    moving = np.full_like(reference, np.median(reference))
    moving[5:, :-7] = reference[:-5, 7:]
    canvas = np.full((2, 64, 64), np.median(reference), dtype=np.float32)
    canvas[0, 8:56, 8:56] = reference
    canvas[1, 8:56, 8:56] = moving

    cost, shifts, gain = fixed_overlap_ncc(
        torch.from_numpy(canvas[:1]),
        torch.from_numpy(canvas[1:]),
        (48, 48),
        10,
    )

    torch.testing.assert_close(shifts[0], torch.tensor([-5.0, 7.0]))
    assert cost[0] < 1e-5
    assert gain[0] > 0.5


def test_translate_align_pair_batch_matches_sequential_solver():
    """Candidate-batched pair translations preserve sequential shifts."""
    rng = np.random.default_rng(20260731)
    pairs = []
    for shift_row, shift_col in ((0.0, 0.0), (3.0, -5.0), (1.3, 2.7)):
        reference = rng.random((64, 64)).astype(np.float32)
        freq_row = np.fft.fftfreq(64)[:, None]
        freq_col = np.fft.fftfreq(64)[None, :]
        moving = np.fft.ifft2(
            np.fft.fft2(reference)
            * np.exp(
                -2j
                * np.pi
                * (freq_row * shift_row + freq_col * shift_col)
            )
        ).real.astype(np.float32)
        pairs.append(np.stack((reference, moving)))
    pairs_t = torch.as_tensor(np.stack(pairs))
    expected = torch.stack(
        [translate_align(pair, 8, 16.0) for pair in pairs_t]
    )
    actual = translate_align_pair_batch(pairs_t, 8, 16.0)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=0.0)


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
    the default smooth_sigma — tests the exact kernel size used
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
# backward_warp_grid_search
# ---------------------------------------------------------------------------

def test_backward_warp_grid_search_finds_known_centered_drift():
    """backward_warp_grid_search should find the correct drift rate
    for a centered drift model within 1 coarse step."""
    from scipy.ndimage import map_coordinates
    rng = np.random.default_rng(42)
    N = 128
    ref = gaussian_filter(rng.random((N, N)), sigma=2.0).astype(np.float32)
    true_row, true_col = 0.04, -0.06
    center = (N - 1) / 2.0
    rr, cc = np.meshgrid(
        np.arange(N, dtype=np.float32),
        np.arange(N, dtype=np.float32),
        indexing="ij",
    )
    offset = (np.arange(N, dtype=np.float32) - center)[:, None]
    drifted = map_coordinates(
        ref,
        [rr + true_row * offset, cc + true_col * offset],
        order=1, mode="nearest",
    ).astype(np.float32)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ref_t = torch.tensor(ref, device=device)
    mov_t = torch.tensor(drifted, device=device)

    step = 0.02
    axis = np.arange(-5, 6) * step
    rg, cg = np.meshgrid(axis, axis, indexing="ij")
    mask = rg**2 + cg**2 <= (5 * step)**2
    candidates = torch.tensor(
        np.vstack((rg[mask], cg[mask])).T, dtype=torch.float32, device=device)

    best_idx, costs = backward_warp_grid_search(
        ref_t, mov_t, candidates, upsample_factor=8, max_image_shift=32)

    est = candidates[best_idx].cpu().numpy()
    assert abs(est[0] - true_row) <= step, (
        f"Row drift error {abs(est[0] - true_row):.4f} exceeds step {step}")
    assert abs(est[1] - true_col) <= step, (
        f"Col drift error {abs(est[1] - true_col):.4f} exceeds step {step}")


def test_backward_warp_grid_search_true_drift_beats_zero():
    """Cost at true drift should be lower than cost at zero drift."""
    from scipy.ndimage import map_coordinates
    rng = np.random.default_rng(123)
    N = 64
    ref = gaussian_filter(rng.random((N, N)), sigma=2.0).astype(np.float32)
    true_row, true_col = 0.05, 0.03
    center = (N - 1) / 2.0
    rr, cc = np.meshgrid(
        np.arange(N, dtype=np.float32),
        np.arange(N, dtype=np.float32),
        indexing="ij",
    )
    offset = (np.arange(N, dtype=np.float32) - center)[:, None]
    drifted = map_coordinates(
        ref,
        [rr + true_row * offset, cc + true_col * offset],
        order=1, mode="nearest",
    ).astype(np.float32)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ref_t = torch.tensor(ref, device=device)
    mov_t = torch.tensor(drifted, device=device)

    candidates = torch.tensor([
        [0.0, 0.0],
        [true_row, true_col],
    ], dtype=torch.float32, device=device)

    best_idx, costs = backward_warp_grid_search(
        ref_t, mov_t, candidates, upsample_factor=8, max_image_shift=32)

    costs_np = costs.cpu().numpy()
    assert best_idx == 1, "True drift should have lower cost than zero drift"
    assert costs_np[1] < costs_np[0], (
        f"True drift cost ({costs_np[1]:.6f}) should be lower than "
        f"zero drift cost ({costs_np[0]:.6f})"
    )


def test_backward_warp_grid_search_shows_progress_for_multiple_chunks(capsys):
    """A labeled affine search should report progress only when chunked."""
    generator = torch.Generator().manual_seed(42)
    image = torch.rand((16, 16), generator=generator)
    candidates = torch.tensor([[0.0, 0.0], [0.01, 0.0]])

    backward_warp_grid_search(
        image,
        image,
        candidates,
        upsample_factor=2,
        max_image_shift=4,
        chunk_size=1,
        progress_desc="Affine coarse search",
    )

    assert "Affine coarse search" in capsys.readouterr().err


def test_backward_warp_grid_search_hides_progress_for_single_chunk(capsys):
    """A one-pass affine search should not flash a zero-to-complete bar."""
    generator = torch.Generator().manual_seed(42)
    image = torch.rand((16, 16), generator=generator)
    candidates = torch.tensor([[0.0, 0.0], [0.01, 0.0]])

    backward_warp_grid_search(
        image,
        image,
        candidates,
        upsample_factor=2,
        max_image_shift=4,
        chunk_size=2,
        progress_desc="Affine coarse search",
    )

    assert "Affine coarse search" not in capsys.readouterr().err


# ──────────────────────────────────────────────────────────────────────
# DriftKnot — direct unit tests of the K-aware dispatch class
# ──────────────────────────────────────────────────────────────────────


def _build_geometry(K, scan_fast=(0.0, 1.0), scan_slow=(1.0, 0.0),
                    input_shape=(64, 64), seed=0):
    """Build a DriftKnot from initial knots for the given K (no drift)."""
    H, W = input_shape
    fast = np.asarray(scan_fast, dtype=np.float32)
    slow = np.asarray(scan_slow, dtype=np.float32)
    knots_np = initialize_scanline_knots(
        input_shape=input_shape,
        output_shape=input_shape,
        scan_fast=fast,
        scan_slow=slow,
        number_knots=K,
    )
    knots = torch.tensor(knots_np, dtype=torch.float32)
    geom = DriftKnot(
        knots,
        torch.tensor(fast, dtype=torch.float32),
        torch.tensor(slow, dtype=torch.float32),
        input_shape,
    )
    return geom


def test_knot_geometry_to_canvas_K1_K2_match_at_initial_knots():
    """Square image: K=1 walk and K=2 lerp at default knot grid produce
    the same per-pixel canvas coordinates (scanline endpoints align)."""
    geom1 = _build_geometry(K=1, input_shape=(64, 64))
    geom2 = _build_geometry(K=2, input_shape=(64, 64))
    r1, c1 = geom1.to_canvas()
    r2, c2 = geom2.to_canvas()
    torch.testing.assert_close(r1, r2, atol=1e-5, rtol=0)
    torch.testing.assert_close(c1, c2, atol=1e-5, rtol=0)


def test_knot_geometry_multi_knot_to_canvas_endpoints_at_knots():
    """K=3 to_canvas at fast_fraction=0/0.5/1 returns exactly the K knot positions."""
    geom = _build_geometry(K=3, input_shape=(8, 5))  # K-1=2 segments → fractions 0.0/0.5/1.0
    r, c = geom.to_canvas()  # (H, W) each
    # cols 0, 2, 4 correspond to knots 0, 1, 2 (linspace 0..1 with W=5)
    for col, k in [(0, 0), (2, 1), (4, 2)]:
        torch.testing.assert_close(r[:, col], geom.knots[0, :, k], atol=1e-5, rtol=0)
        torch.testing.assert_close(c[:, col], geom.knots[1, :, k], atol=1e-5, rtol=0)


def test_knot_geometry_drift_raw_K1_per_row_shape():
    """K=1 drift_raw returns (2, H) per-row shifts."""
    geom = _build_geometry(K=1, input_shape=(32, 32))
    initial = geom.knots.clone()
    geom.knots = initial + torch.tensor([0.5, 1.0])[:, None, None]  # uniform shift
    out = geom.drift_raw(initial)
    assert out.shape == (2, 32)


def test_knot_geometry_drift_raw_K2_per_pixel_shape():
    """K>=2 drift_raw returns (2, H, W) per-pixel drift."""
    geom = _build_geometry(K=2, input_shape=(32, 24))
    initial = geom.knots.clone()
    geom.knots = initial + torch.tensor([0.5, 1.0])[:, None, None]
    out = geom.drift_raw(initial)
    assert out.shape == (2, 32, 24)


def test_knot_geometry_apply_affine_shift_centered_at_middle_row():
    """apply_affine_shift adds drift_vec * (i - (H-1)/2) per scanline; row (H-1)/2 stays put."""
    geom = _build_geometry(K=1, input_shape=(11, 11))
    snapshot = geom.knots.clone()
    drift_vec = torch.tensor([1.0, 2.0])
    geom.apply_affine_shift(drift_vec)
    delta = geom.knots - snapshot
    # The centered scanline (idx 5 for H=11) should not move
    assert torch.allclose(delta[:, 5, 0], torch.zeros(2), atol=1e-6)
    # Endpoints should shift by drift_vec * ±5
    torch.testing.assert_close(delta[:, 0, 0], -5.0 * drift_vec, atol=1e-5, rtol=0)
    torch.testing.assert_close(delta[:, -1, 0], 5.0 * drift_vec, atol=1e-5, rtol=0)


def test_knot_geometry_warp_to_canvas_round_trip():
    """warp_to_canvas of an image at zero drift returns a recognizable copy on the canvas."""
    np.random.seed(0)
    H = W = 32
    image = torch.tensor(np.random.rand(H, W).astype(np.float32))
    geom = _build_geometry(K=1, input_shape=(H, W))
    canvas_shape = (H, W)
    warped, weights = geom.warp_to_canvas(image, canvas_shape, kde_sigma=0.5, pad_value=0.0)
    assert warped.shape == canvas_shape
    assert weights.shape == canvas_shape
    # Center region (where weights are highest) should resemble the input intensity range
    cy, cx = H // 2, W // 2
    assert weights[cy, cx] > 0.5  # adequate coverage at center
    assert warped[cy, cx].item() > 0.0


def _interp_with_drift(scan_fast, scan_slow, input_shape, drift_canvas):
    """Build a DriftKnot whose ``drift(initial)`` equals ``drift_canvas``.

    The pure rotation tests below feed synthetic deltas directly so we can
    pin the canvas → raw Jacobian without running a full preprocess pipeline.
    """
    H = drift_canvas.shape[1]
    initial = torch.zeros(2, H, 1, dtype=drift_canvas.dtype)
    knots = initial.clone()
    knots[:, :, 0] = drift_canvas
    interp = DriftKnot(
        knots, torch.tensor(scan_fast, dtype=drift_canvas.dtype),
        torch.tensor(scan_slow, dtype=drift_canvas.dtype), input_shape)
    return interp, initial


def test_drift_raw_identity_for_zero_angle():
    """0° scan: drift_raw returns the canvas drift unchanged."""
    drift_canvas = torch.tensor([[1.0, 2.0], [3.0, 4.0]])  # (2, 2)
    interp, initial = _interp_with_drift(
        scan_fast=[0.0, 1.0], scan_slow=[1.0, 0.0],
        input_shape=(2, 2), drift_canvas=drift_canvas)
    out = interp.drift_raw(initial)
    torch.testing.assert_close(out[0], drift_canvas[0], atol=1e-6, rtol=0)
    torch.testing.assert_close(out[1], drift_canvas[1], atol=1e-6, rtol=0)


def test_drift_raw_rotation_for_90():
    """scan_direction=90°: canvas drift rotates by 90°."""
    drift_canvas = torch.tensor([[5.0], [3.0]])  # δr=5, δc=3
    interp, initial = _interp_with_drift(
        scan_fast=[-1.0, 0.0], scan_slow=[0.0, 1.0],
        input_shape=(64, 64), drift_canvas=drift_canvas)
    out = interp.drift_raw(initial)
    torch.testing.assert_close(out[0], torch.tensor([3.0]), atol=1e-5, rtol=0)
    torch.testing.assert_close(out[1], torch.tensor([-5.0]), atol=1e-5, rtol=0)


def test_drift_raw_rotation_for_neg90():
    """scan_direction=-90°: canvas drift rotates by -90°."""
    drift_canvas = torch.tensor([[5.0], [3.0]])
    interp, initial = _interp_with_drift(
        scan_fast=[1.0, 0.0], scan_slow=[0.0, -1.0],
        input_shape=(64, 64), drift_canvas=drift_canvas)
    out = interp.drift_raw(initial)
    torch.testing.assert_close(out[0], torch.tensor([-3.0]), atol=1e-5, rtol=0)
    torch.testing.assert_close(out[1], torch.tensor([5.0]), atol=1e-5, rtol=0)


def test_drift_raw_nonsquare_90_uses_alpha():
    """Non-square scans carry the aspect-ratio Jacobian factor — without the
    alpha term the row drift would be returned unscaled."""
    scan_h, scan_w = 128, 64
    alpha = (scan_h - 1) / (scan_w - 1)
    drift_canvas = torch.tensor([[5.0], [3.0]])
    interp, initial = _interp_with_drift(
        scan_fast=[-1.0, 0.0], scan_slow=[0.0, 1.0],
        input_shape=(scan_h, scan_w), drift_canvas=drift_canvas)
    out = interp.drift_raw(initial)
    torch.testing.assert_close(out[0], torch.tensor([3.0]), atol=1e-5, rtol=0)
    torch.testing.assert_close(out[1], torch.tensor([-5.0 / alpha]), atol=1e-5, rtol=0)


# ---------------------------------------------------------------------------
# Analytic crop-footprint geometry: crop_slices derives from KNOT FOOTPRINTS,
# not cross-correlation. The crop is the largest axis-aligned rectangle
# inside the AND of every scan's coverage footprint, where each footprint is
# rasterized purely from the solved knots (scanline origins + fast axis).
# Every case here has a hand-computable answer.
# ---------------------------------------------------------------------------


class _MockDC:
    def __init__(self, canvas, scan, knots_list, fasts):
        class _A:
            pass
        self.imgs_warped = _A()
        self.imgs_warped.array = np.zeros((len(knots_list),) + canvas)
        self.imgs = [np.zeros(scan) for _ in knots_list]
        self.knots = [torch.tensor(k)[:, :, None].float() for k in knots_list]
        self.scan_fast = fasts
        self._reference_mode = False

    coverage_mask = DriftCorrection.coverage_mask
    crop_slices = DriftCorrection.crop_slices


def _orthogonal_pair(origin=50, rows=100):
    k0 = np.stack([np.arange(rows) + origin, np.full(rows, origin)])
    k1 = np.stack([np.full(rows, origin), np.arange(rows) + origin])
    return k0, k1


def test_largest_rectangle_exact():
    mask = np.zeros((40, 60), dtype=bool)
    mask[5:25, 10:50] = True
    assert largest_rectangle(mask) == (5, 25, 10, 50)


def test_full_overlap_crop_is_full_scan_minus_guard():
    k0, k1 = _orthogonal_pair()
    dc = _MockDC((200, 200), (100, 100), [k0, k1], [(0.0, 1.0), (1.0, 0.0)])
    rows, cols = dc.crop_slices()
    # full 100 px overlap minus the 4 px bilinear guard per edge
    assert rows == slice(4, 96) and cols == slice(4, 96)


def test_shifted_scan_crop_matches_geometry():
    k0, _ = _orthogonal_pair()
    n = 100
    k1 = np.stack([np.full(n, 70), np.arange(n) + 60])  # scan 1 at +20 rows, +10 cols
    dc = _MockDC((200, 200), (100, 100), [k0, k1], [(0.0, 1.0), (1.0, 0.0)])
    rows, cols = dc.crop_slices()
    # overlap rows 20..100, cols 10..100 in the scan frame, minus 4 px guards
    assert rows == slice(24, 96)
    assert cols == slice(14, 96)


def test_three_scan_collection():
    k0, k1 = _orthogonal_pair()
    n = 100
    k2 = np.stack([np.arange(n) + 60, np.full(n, 55)])  # third scan, +10 rows +5 cols
    dc = _MockDC((200, 200), (100, 100), [k0, k1, k2],
                 [(0.0, 1.0), (1.0, 0.0), (0.0, 1.0)])
    rows, cols = dc.crop_slices()
    assert rows.start >= 14 and cols.start >= 9   # third scan tightens the window
    assert rows.stop <= 96 and cols.stop <= 96
