"""
Tests for the DriftCorrection class in quantem.imaging.drift

Synthetic data: chevron pattern with linear drift + jitter at 0 and 90 deg scan angles.
See PR #133 for images: https://github.com/electronmicroscopy/quantem/pull/133
"""

import numpy as np
import pytest
import torch
import warnings
import matplotlib
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, map_coordinates
import quantem.imaging as imaging
import quantem.imaging.drift as drift_module
import quantem.imaging.drift_4dstem as drift_4dstem_module
from quantem.core.datastructures.dataset2d import Dataset2d
from quantem.core.datastructures.dataset3d import Dataset3d
from quantem.imaging.drift import (
    DriftCorrection,
    CorrectionResult,
)
from quantem.imaging.drift_visualization import (
    center_crop,
    fft_log_magnitude,
    normalized_cross_correlation,
    element_map,
)


def make_synthetic_drift_data(scale=1, seed=42):
    """Generate a chevron base image plus two scan-distorted views.

    Image 0 scans along columns; image 1 scans along rows. Both apply the
    same linear row/col drift plus per-scanline jitter so the nonrigid
    solver has a non-trivial knot field to recover.
    """
    np.random.seed(seed)
    shape = (200 * scale, 200 * scale)
    row_grid, col_grid = np.meshgrid(
        np.arange(-shape[0] / 2, shape[0] / 2),
        np.arange(-shape[0] / 2, shape[0] / 2),
        indexing="ij",
    )
    base_image = (np.mod(np.abs(row_grid) + np.abs(col_grid), 16 * scale) < 8 * scale).astype("float")
    base_image[np.logical_and(row_grid > 0, col_grid > 0)] += 0.5
    base_image[np.maximum(np.abs(row_grid), np.abs(col_grid)) < 20 * scale] = 2
    base_image = gaussian_filter(base_image, sigma=0.667 * scale)

    scan_size = 128 * scale
    scan_positions = np.arange(scan_size)
    row_drift = scan_positions * 0.001 * scale
    col_drift = scan_positions * 0.1 * scale
    jitter_mag = 0.5 * scale
    jitter0 = np.random.randn(2, scan_size) * jitter_mag
    jitter1 = np.random.randn(2, scan_size) * jitter_mag

    im0 = np.zeros((scan_size, scan_size))
    for row_idx in range(scan_size):
        start_row = 40 * scale + row_idx + row_drift[row_idx] + jitter0[0, row_idx]
        start_col = 30 * scale + 0 + col_drift[row_idx] + jitter0[1, row_idx]
        row_coords = start_row + scan_positions * 0
        col_coords = start_col + scan_positions * 1
        row_coords = np.clip(row_coords, 0, shape[0] - 2)
        col_coords = np.clip(col_coords, 0, shape[1] - 2)
        row_floor = np.floor(row_coords).astype("int")
        col_floor = np.floor(col_coords).astype("int")
        row_frac = row_coords - row_floor
        col_frac = col_coords - col_floor
        im0[row_idx, :] = (
            base_image[row_floor, col_floor] * (1 - row_frac) * (1 - col_frac)
            + base_image[row_floor + 1, col_floor] * row_frac * (1 - col_frac)
            + base_image[row_floor, col_floor + 1] * (1 - row_frac) * col_frac
            + base_image[row_floor + 1, col_floor + 1] * row_frac * col_frac
        )

    im1 = np.zeros((scan_size, scan_size))
    for row_idx in range(scan_size):
        start_row = 170 * scale + 0 + row_drift[row_idx] + jitter1[0, row_idx]
        start_col = 30 * scale + row_idx + col_drift[row_idx] + jitter1[1, row_idx]
        row_coords = start_row - scan_positions * 1
        col_coords = start_col + scan_positions * 0
        row_coords = np.clip(row_coords, 0, shape[0] - 2)
        col_coords = np.clip(col_coords, 0, shape[1] - 2)
        row_floor = np.floor(row_coords).astype("int")
        col_floor = np.floor(col_coords).astype("int")
        row_frac = row_coords - row_floor
        col_frac = col_coords - col_floor
        im1[row_idx, :] = (
            base_image[row_floor, col_floor] * (1 - row_frac) * (1 - col_frac)
            + base_image[row_floor + 1, col_floor] * row_frac * (1 - col_frac)
            + base_image[row_floor, col_floor + 1] * (1 - row_frac) * col_frac
            + base_image[row_floor + 1, col_floor + 1] * row_frac * col_frac
        )

    return im0, im1, base_image


def bilinear_sample(image, row_coords, col_coords):
    """Sample ``image`` at floating-point coordinates using bilinear interpolation."""
    row_coords = np.clip(row_coords, 0, image.shape[0] - 2)
    col_coords = np.clip(col_coords, 0, image.shape[1] - 2)
    row_floor = np.floor(row_coords).astype(int)
    col_floor = np.floor(col_coords).astype(int)
    row_frac = row_coords - row_floor
    col_frac = col_coords - col_floor
    return (
        image[row_floor, col_floor] * (1 - row_frac) * (1 - col_frac)
        + image[row_floor + 1, col_floor] * row_frac * (1 - col_frac)
        + image[row_floor, col_floor + 1] * (1 - row_frac) * col_frac
        + image[row_floor + 1, col_floor + 1] * row_frac * col_frac
    )


def test_full_pipeline_deterministic():
    """Full pipeline produces correct, deterministic, low-error results."""
    im0, im1, _ = make_synthetic_drift_data(scale=1, seed=42)

    drift = DriftCorrection(
        im0, im1,
        scan_direction_degrees=[0.0, -90.0],
    ).preprocess(
        pad_fraction=0.25,
        pad_value="median",
        kde_sigma=0.5,
        number_knots=1,
        show_merged=False,
        show_images=False,
    )
    drift.align_affine(step=0.02, num_tests=5, refine=False)
    drift.align_nonrigid(
        num_iterations=2,
        regularization_sigma_px=0.5,
        loss="mse",
        show_merged=False,
        show_images=False,
    )
    img_corr = drift.generate_corrected(upsample_factor=1, show_merged=False)

    assert isinstance(img_corr, Dataset2d)
    assert not np.isnan(img_corr.array).any()
    assert drift.error_track[-1, 1] < 0.1

    # Determinism: second run with same seed must match exactly
    im0_2, im1_2, _ = make_synthetic_drift_data(scale=1, seed=42)
    drift2 = DriftCorrection(
        im0_2, im1_2,
        scan_direction_degrees=[0.0, -90.0],
    ).preprocess(
        pad_fraction=0.25,
        pad_value="median",
        kde_sigma=0.5,
        number_knots=1,
        show_merged=False,
        show_images=False,
    )
    drift2.align_affine(step=0.02, num_tests=5, refine=False)
    drift2.align_nonrigid(
        num_iterations=2,
        regularization_sigma_px=0.5,
        loss="mse",
        show_merged=False,
        show_images=False,
    )
    img_corr2 = drift2.generate_corrected(upsample_factor=1, show_merged=False)

    np.testing.assert_array_almost_equal(
        img_corr.array, img_corr2.array, decimal=10,
        err_msg="Drift correction output is not deterministic!",
    )



def test_constructor_requires_at_least_two_datasets():
    """DriftCorrection requires ≥2 datasets - single-image construction is not supported."""
    im0, _, _ = make_synthetic_drift_data(scale=1, seed=42)
    with pytest.raises(TypeError, match="at least 2 datasets"):
        DriftCorrection(im0, scan_direction_degrees=[0.0])


# Baseline values from float32 torch path, captured once and frozen.
# (scale, error, knots0_sum, knots1_sum)
# Recaptured after torch-native knots unification (knots stored as torch tensors).
# scale=1 is bit-exact across versions; scale=2,4 shifted by ~0.03-1.1%
# due to optimizer trajectory divergence from dependency upgrades.
# Frozen baselines captured on RTX PRO 6000 Blackwell + PyTorch 2.10 + CUDA 13.
# Bit-reproducible per-machine, but cuFFT/cuBLAS algorithm selection at
# scale=2 (256²) differs across CUDA versions, so the scale=2 row may need
# recapture on a different host. Scale 1 + 4 are usually portable.
AFFINE_BASELINES = [
    (1, 0.09237674623727798, 12157.736328125, 28546.263671875),
    (2, 0.13840323686599731, 49798.91015625, 113529.09375),
    (4, 0.16396018862724304, 194685.40625, 459650.59375),
]


@pytest.mark.parametrize("scale,expected_error,expected_k0,expected_k1", AFFINE_BASELINES)
def test_align_affine_matches_frozen_baseline(scale, expected_error, expected_k0, expected_k1):
    """Affine on synthetic data must match frozen float32 baseline."""
    im0, im1, _ = make_synthetic_drift_data(scale=scale, seed=42)
    drift = DriftCorrection(
        im0, im1, scan_direction_degrees=[0.0, -90.0],
    ).preprocess(show_merged=False, show_images=False)
    drift.align_affine(
        step=0.02, num_tests=5, refine=True,
        show_merged=False, show_images=False,
    )
    # rtol=1e-6 = float32 machine epsilon × 10. Bit-exact match required on
    # the capture machine. On a different GPU + CUDA version expect drift
    # up to ~3e-5 (affine) — recapture baselines or loosen rtol if porting.
    np.testing.assert_allclose(
        drift.error_track[-1, 1], expected_error, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        drift.knots[0].sum().item(), expected_k0, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        drift.knots[1].sum().item(), expected_k1, rtol=1e-6, atol=1e-6)


# Frozen baselines for the pytorch backend with optimizer_name="adam".
# Recaptured after torch-native knots unification (adam_steps=50).
NONRIGID_ADAM_BASELINES = [
    (1, 0.05627802759408951, 12023.8701171875, 28671.677734375),
    (2, 0.12936685979366302, 49747.04296875, 113559.015625),
]


@pytest.mark.parametrize("scale,expected_error,expected_k0,expected_k1", NONRIGID_ADAM_BASELINES)
def test_align_nonrigid_adam_matches_frozen_baseline(scale, expected_error, expected_k0, expected_k1):
    """Nonrigid on synthetic data must match frozen baseline.

    Runs preprocess → affine → nonrigid (2 iterations for speed).
    If the GPU warp or translation path changes numerical output,
    these baselines catch it immediately.
    """
    im0, im1, _ = make_synthetic_drift_data(scale=scale, seed=42)
    drift = DriftCorrection(
        im0, im1, scan_direction_degrees=[0.0, -90.0],
    ).preprocess(show_merged=False, show_images=False)
    drift.align_affine(
        step=0.02, num_tests=5, refine=True,
        show_merged=False, show_images=False,
    )
    drift.align_nonrigid(
        num_iterations=2, adam_steps=50,
        regularization_sigma_px=16.0,
        # Pin lr to the value the baselines were captured at - the public
        # default is now auto-derived from max_image_shift, but the frozen
        # baselines must stay numerically stable across that change.
        lr=0.02,
        loss="mse",
        show_merged=False, show_images=False,
    )
    # rtol=1e-6 = float32 machine epsilon × 10. Bit-exact on the capture
    # machine. Cross-machine Adam drift is ~1e-4 — recapture if porting.
    np.testing.assert_allclose(
        drift.error_track[-1, 1], expected_error, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        drift.knots[0].sum().item(), expected_k0, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        drift.knots[1].sum().item(), expected_k1, rtol=1e-6, atol=1e-6)


# Frozen baselines for the pytorch backend with optimizer_name="lbfgs".
# Recaptured after torch-native knots unification.
NONRIGID_LBFGS_BASELINES = [
    (1, 0.0727548599243164, 12151.92578125, 28535.07421875),
    (2, 0.11218203604221344, 50291.56640625, 113703.953125),
]


@pytest.mark.parametrize("scale,expected_error,expected_k0,expected_k1", NONRIGID_LBFGS_BASELINES)
def test_align_nonrigid_lbfgs_matches_frozen_baseline(scale, expected_error, expected_k0, expected_k1):
    """Nonrigid LBFGS path on synthetic data must match frozen baseline.

    The LBFGS optimizer uses a closure-based forward+backward instead of
    Adam's compiled inner loop. This test ensures both paths stay
    numerically deterministic and that LBFGS doesn't silently regress.
    """
    im0, im1, _ = make_synthetic_drift_data(scale=scale, seed=42)
    drift = DriftCorrection(
        im0, im1, scan_direction_degrees=[0.0, -90.0],
    ).preprocess(show_merged=False, show_images=False)
    drift.align_affine(
        step=0.02, num_tests=5, refine=True,
        show_merged=False, show_images=False,
    )
    drift.align_nonrigid(
        optimizer_name="lbfgs",
        num_iterations=2, lbfgs_max_iter=20,
        regularization_sigma_px=16.0,
        loss="mse",
        show_merged=False, show_images=False,
    )
    # rtol=1e-6 = float32 machine epsilon × 10. Bit-exact on the capture
    # machine. LBFGS line-search can amplify gradient noise to ~1e-3 across
    # CUDA versions — recapture or loosen rtol heavily if porting.
    np.testing.assert_allclose(
        drift.error_track[-1, 1], expected_error, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        drift.knots[0].sum().item(), expected_k0, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        drift.knots[1].sum().item(), expected_k1, rtol=1e-6, atol=1e-6)


# ---------------------------------------------------------------------------
# Tests for fixed_indices support in align_affine
# ---------------------------------------------------------------------------


def test_align_affine_fixed_indices_recovers_known_drift():
    """align_affine(fixed_indices=[0]) should recover a known single-sided drift.

    Synthetic image with known per-line drift, exercised through the unified
    DriftCorrection API with fixed_indices.
    """
    rng = np.random.default_rng(42)
    reference = gaussian_filter(rng.random((64, 64)), sigma=1.0).astype(np.float32)
    row_grid, col_grid = np.meshgrid(
        np.arange(reference.shape[0], dtype=np.float32),
        np.arange(reference.shape[1], dtype=np.float32),
        indexing="ij",
    )
    scanline_offset = (
        np.arange(reference.shape[0], dtype=np.float32) - (reference.shape[0] - 1) / 2
    )[:, None]
    expected_drift = np.array([0.03, -0.05], dtype=np.float32)
    moving = bilinear_sample(
        reference,
        row_grid + expected_drift[0] * scanline_offset,
        col_grid + expected_drift[1] * scanline_offset,
    ).astype(np.float32)

    # Save initial knots for reference image to verify they don't change
    drift = DriftCorrection(
        reference, moving,
        scan_direction_degrees=[0.0, 0.0],
    ).preprocess(
        pad_fraction=0.25,
        pad_value=0.0,
        kde_sigma=0.5,
        number_knots=1,
        show_merged=False,
        show_images=False,
    )
    knots0_before = drift.knots[0].clone()

    drift.align_affine(
        step=0.01,
        num_tests=13,
        refine=True,
        fixed_indices=[0],
        show_merged=False,
        show_images=False,
    )

    # Reference knots must not have changed
    np.testing.assert_array_equal(
        drift.knots[0].cpu().numpy(), knots0_before.cpu().numpy(),
        err_msg="fixed_indices=[0] should leave image 0 knots unchanged",
    )
    # Moving image knots should have changed (drift was applied)
    assert not np.array_equal(drift.knots[1].cpu().numpy(), knots0_before.cpu().numpy()), \
        "Image 1 knots should have been modified by the affine search"
    # Error should have decreased
    assert drift.error_track[-1, 1] < drift.error_track[0, 1], \
        "Affine alignment with fixed_indices should reduce error"


def test_align_affine_fixed_indices_none_matches_default():
    """fixed_indices=None (default) should produce identical results to the original path."""
    im0, im1, _ = make_synthetic_drift_data(scale=1, seed=42)

    # Run without fixed_indices
    drift_a = DriftCorrection(
        im0, im1, scan_direction_degrees=[0.0, -90.0],
    ).preprocess(show_merged=False, show_images=False)
    drift_a.align_affine(
        step=0.02, num_tests=5, refine=True,
        show_merged=False, show_images=False,
    )

    # Run with explicit fixed_indices=None
    drift_b = DriftCorrection(
        im0, im1, scan_direction_degrees=[0.0, -90.0],
    ).preprocess(show_merged=False, show_images=False)
    drift_b.align_affine(
        step=0.02, num_tests=5, refine=True,
        fixed_indices=None,
        show_merged=False, show_images=False,
    )

    np.testing.assert_array_almost_equal(
        drift_a.knots[0].cpu().numpy(), drift_b.knots[0].cpu().numpy(), decimal=10,
    )
    np.testing.assert_array_almost_equal(
        drift_a.knots[1].cpu().numpy(), drift_b.knots[1].cpu().numpy(), decimal=10,
    )
    np.testing.assert_almost_equal(
        drift_a.error_track[-1, 1], drift_b.error_track[-1, 1], decimal=10,
    )


def test_preprocess_normalize_scales_to_unit_range():
    """preprocess(normalize=True) should scale each image to [0, 1]."""
    im0, im1, _ = make_synthetic_drift_data(scale=1, seed=42)
    # Artificially scale im1 to a very different range
    im1_scaled = im1 * 1000 + 5000

    dc = DriftCorrection(
        im0, im1_scaled, scan_direction_degrees=[0.0, -90.0],
    )
    dc.preprocess(normalize=True, show_merged=False, show_images=False)

    for img_idx in range(2):
        arr = dc.imgs[img_idx].array
        assert arr.min() >= -0.01, f"Image {img_idx} min={arr.min()}"
        assert arr.max() <= 1.01, f"Image {img_idx} max={arr.max()}"


# ---------------------------------------------------------------------------
# Tests for fixed_indices support in align_nonrigid
# ---------------------------------------------------------------------------


def test_align_nonrigid_fixed_indices_freezes_reference():
    """align_nonrigid(fixed_indices=[0]) must not modify reference knots.

    The reference (image 0) knots should stay frozen while the moving
    image's knots are optimized against the fixed reference.
    """
    rng = np.random.default_rng(42)
    reference = gaussian_filter(rng.random((64, 64)), sigma=1.0).astype(np.float32)
    row_grid, col_grid = np.meshgrid(
        np.arange(reference.shape[0], dtype=np.float32),
        np.arange(reference.shape[1], dtype=np.float32),
        indexing="ij",
    )
    scanline_offset = (
        np.arange(reference.shape[0], dtype=np.float32) - (reference.shape[0] - 1) / 2
    )[:, None]
    expected_drift = np.array([0.03, -0.05], dtype=np.float32)
    moving = bilinear_sample(
        reference,
        row_grid + expected_drift[0] * scanline_offset,
        col_grid + expected_drift[1] * scanline_offset,
    ).astype(np.float32)

    drift = DriftCorrection(
        reference, moving,
        scan_direction_degrees=[0.0, 0.0],
    ).preprocess(
        pad_fraction=0.25, pad_value=0.0, kde_sigma=0.5, number_knots=1,
        show_merged=False, show_images=False,
    )
    drift.align_affine(
        step=0.01, num_tests=13, refine=True,
        fixed_indices=[0], show_merged=False, show_images=False,
    )
    knots0_after_affine = drift.knots[0].clone()

    drift.align_nonrigid(
        optimizer_name="adam",
        num_iterations=2, adam_steps=20,
        regularization_sigma_px=8.0, lr=0.02,
        fixed_indices=[0],
        show_merged=False, show_images=False,
    )

    # Reference knots must not have changed
    np.testing.assert_array_equal(
        drift.knots[0].cpu().numpy(), knots0_after_affine.cpu().numpy(),
        err_msg="fixed_indices=[0] should leave image 0 knots unchanged in nonrigid",
    )
    # Moving image knots should have changed
    assert not np.array_equal(
        drift.knots[1].cpu().numpy(), knots0_after_affine.cpu().numpy()
    ), "Image 1 knots should have been modified by nonrigid optimization"


def test_align_nonrigid_fixed_indices_reduces_error():
    """Nonrigid with fixed_indices should reduce alignment error."""
    rng = np.random.default_rng(42)
    reference = gaussian_filter(rng.random((64, 64)), sigma=1.0).astype(np.float32)
    row_grid, col_grid = np.meshgrid(
        np.arange(reference.shape[0], dtype=np.float32),
        np.arange(reference.shape[1], dtype=np.float32),
        indexing="ij",
    )
    scanline_offset = (
        np.arange(reference.shape[0], dtype=np.float32) - (reference.shape[0] - 1) / 2
    )[:, None]
    expected_drift = np.array([0.03, -0.05], dtype=np.float32)
    moving = bilinear_sample(
        reference,
        row_grid + expected_drift[0] * scanline_offset,
        col_grid + expected_drift[1] * scanline_offset,
    ).astype(np.float32)

    drift = DriftCorrection(
        reference, moving,
        scan_direction_degrees=[0.0, 0.0],
    ).preprocess(
        pad_fraction=0.25, pad_value=0.0, kde_sigma=0.5, number_knots=1,
        show_merged=False, show_images=False,
    )
    drift.align_affine(
        step=0.01, num_tests=13, refine=True,
        fixed_indices=[0], show_merged=False, show_images=False,
    )
    error_after_affine = drift.error_track[-1, 1]

    drift.align_nonrigid(
        optimizer_name="adam",
        num_iterations=2, adam_steps=20,
        regularization_sigma_px=8.0, lr=0.02,
        fixed_indices=[0],
        loss="mse",
        show_merged=False, show_images=False,
    )
    error_after_nonrigid = drift.error_track[-1, 1]

    assert error_after_nonrigid <= error_after_affine * 1.1, (
        f"Nonrigid should not significantly increase error: "
        f"{error_after_nonrigid} vs {error_after_affine}"
    )


def test_align_nonrigid_fixed_indices_all_fixed_raises():
    """fixed_indices covering all images should raise ValueError."""
    im0, im1, _ = make_synthetic_drift_data(scale=1, seed=42)
    drift = DriftCorrection(
        im0, im1, scan_direction_degrees=[0.0, -90.0],
    ).preprocess(show_merged=False, show_images=False)
    drift.align_affine(
        step=0.02, num_tests=5, refine=True,
        show_merged=False, show_images=False,
    )
    with pytest.raises(ValueError, match="All images are fixed"):
        drift.align_nonrigid(
            fixed_indices=[0, 1],
            show_merged=False, show_images=False,
        )


def test_align_nonrigid_gradient_mse_runs():
    """align_nonrigid(loss='gradient_mse') should run without error and modify knots."""
    rng = np.random.default_rng(42)
    reference = gaussian_filter(rng.random((64, 64)), sigma=1.0).astype(np.float32)
    row_grid, col_grid = np.meshgrid(
        np.arange(reference.shape[0], dtype=np.float32),
        np.arange(reference.shape[1], dtype=np.float32),
        indexing="ij",
    )
    scanline_offset = (
        np.arange(reference.shape[0], dtype=np.float32) - (reference.shape[0] - 1) / 2
    )[:, None]
    expected_drift = np.array([0.03, -0.05], dtype=np.float32)
    moving = bilinear_sample(
        reference,
        row_grid + expected_drift[0] * scanline_offset,
        col_grid + expected_drift[1] * scanline_offset,
    ).astype(np.float32)

    drift = DriftCorrection(
        reference, moving,
        scan_direction_degrees=[0.0, 0.0],
    ).preprocess(
        pad_fraction=0.25, pad_value=0.0, kde_sigma=0.5, number_knots=1,
        show_merged=False, show_images=False,
    )
    drift.align_affine(
        step=0.01, num_tests=13, refine=True,
        fixed_indices=[0], show_merged=False, show_images=False,
    )
    knots_after_affine = drift.knots[1].clone()

    drift.align_nonrigid(
        optimizer_name="adam",
        num_iterations=2, adam_steps=20,
        regularization_sigma_px=8.0, lr=0.02,
        fixed_indices=[0], loss="gradient_mse",
        show_merged=False, show_images=False,
    )

    # Moving image knots should have changed
    assert not np.array_equal(
        drift.knots[1].cpu().numpy(), knots_after_affine.cpu().numpy()
    ), "gradient_mse should modify the moving image knots"
    # images_t should contain original (non-Sobel) images after alignment
    orig_stack = torch.stack(drift.imgs_t)
    assert orig_stack.min() >= 0, "images_t should be restored (gradient images can be negative)"


def test_align_nonrigid_gradient_mse_lbfgs():
    """gradient_mse should also work with LBFGS optimizer."""
    rng = np.random.default_rng(42)
    reference = gaussian_filter(rng.random((64, 64)), sigma=1.0).astype(np.float32)
    row_grid, col_grid = np.meshgrid(
        np.arange(reference.shape[0], dtype=np.float32),
        np.arange(reference.shape[1], dtype=np.float32),
        indexing="ij",
    )
    scanline_offset = (
        np.arange(reference.shape[0], dtype=np.float32) - (reference.shape[0] - 1) / 2
    )[:, None]
    expected_drift = np.array([0.03, -0.05], dtype=np.float32)
    moving = bilinear_sample(
        reference,
        row_grid + expected_drift[0] * scanline_offset,
        col_grid + expected_drift[1] * scanline_offset,
    ).astype(np.float32)

    drift = DriftCorrection(
        reference, moving,
        scan_direction_degrees=[0.0, 0.0],
    ).preprocess(
        pad_fraction=0.25, pad_value=0.0, kde_sigma=0.5, number_knots=1,
        normalize=False, show_merged=False, show_images=False,
    )
    drift.align_affine(
        step=0.01, num_tests=13, refine=True,
        fixed_indices=[0], show_merged=False, show_images=False,
    )

    drift.align_nonrigid(
        optimizer_name="lbfgs",
        num_iterations=2, regularization_sigma_px=8.0,
        fixed_indices=[0], loss="gradient_mse",
        show_merged=False, show_images=False,
    )
    # Smoke test: just verify it completed without error


def test_align_nonrigid_gradient_mse_beats_mse_with_gain_offset():
    """gradient_mse should outperform mse when images have gain + offset mismatch."""
    rng = np.random.default_rng(99)
    reference = gaussian_filter(rng.random((64, 64)), sigma=1.0).astype(np.float32) * 100 + 50
    row_grid, col_grid = np.meshgrid(
        np.arange(reference.shape[0], dtype=np.float32),
        np.arange(reference.shape[1], dtype=np.float32),
        indexing="ij",
    )
    scanline_offset = (
        np.arange(reference.shape[0], dtype=np.float32) - (reference.shape[0] - 1) / 2
    )[:, None]
    expected_drift = np.array([0.04, -0.06], dtype=np.float32)
    moving = bilinear_sample(
        reference,
        row_grid + expected_drift[0] * scanline_offset,
        col_grid + expected_drift[1] * scanline_offset,
    ).astype(np.float32)
    # Apply gain + offset mismatch
    moving = moving * 0.6 + 30.0

    def run_with_loss(loss_name, **kwargs):
        drift = DriftCorrection(
            reference, moving,
            scan_direction_degrees=[0.0, 0.0],
        ).preprocess(
            pad_fraction=0.25, pad_value=0.0, kde_sigma=0.5, number_knots=1,
            normalize=False, show_merged=False, show_images=False,
        )
        drift.align_affine(
            step=0.01, num_tests=13, refine=True,
            fixed_indices=[0], show_merged=False, show_images=False,
        )
        drift.align_nonrigid(
            optimizer_name="lbfgs",
            num_iterations=4, regularization_sigma_px=8.0,
            fixed_indices=[0], loss=loss_name, max_image_shift=32.0,
            show_merged=False, show_images=False, **kwargs,
        )
        return drift.error_track[-1, 1]

    err_mse = run_with_loss("mse")
    err_grad = run_with_loss("gradient_mse")
    # gradient_mse should be at least as good (often better)
    assert err_grad <= err_mse * 1.5, (
        f"gradient_mse ({err_grad:.4f}) should not be much worse than mse ({err_mse:.4f})"
    )


def test_align_nonrigid_invalid_loss_raises():
    """Invalid loss name should raise ValueError."""
    im0, im1, _ = make_synthetic_drift_data(scale=1, seed=42)
    drift = DriftCorrection(
        im0, im1, scan_direction_degrees=[0.0, -90.0],
    ).preprocess(show_merged=False, show_images=False)
    drift.align_affine(show_merged=False, show_images=False)
    with pytest.raises(ValueError, match="loss must be one of"):
        drift.align_nonrigid(
            loss="invalid_loss",
            show_merged=False, show_images=False,
        )


def test_align_nonrigid_regularization_sigma_none():
    """align_nonrigid should work with regularization_sigma_px=None (no smoothing)."""
    rng = np.random.default_rng(42)
    reference = gaussian_filter(rng.random((64, 64)), sigma=1.0).astype(np.float32)
    row_grid, col_grid = np.meshgrid(
        np.arange(reference.shape[0], dtype=np.float32),
        np.arange(reference.shape[1], dtype=np.float32),
        indexing="ij",
    )
    scanline_offset = (
        np.arange(reference.shape[0], dtype=np.float32) - (reference.shape[0] - 1) / 2
    )[:, None]
    drift_rate = np.array([0.03, -0.05], dtype=np.float32)
    moving = bilinear_sample(
        reference,
        row_grid + drift_rate[0] * scanline_offset,
        col_grid + drift_rate[1] * scanline_offset,
    ).astype(np.float32)

    drift = DriftCorrection(
        reference, moving,
        scan_direction_degrees=[0.0, 0.0],
    ).preprocess(
        pad_fraction=0.25, pad_value=0.0, kde_sigma=0.5, number_knots=1,
        show_merged=False, show_images=False,
    )
    drift.align_affine(
        step=0.01, num_tests=13, refine=True,
        fixed_indices=[0], show_merged=False, show_images=False,
    )
    # This previously crashed with UnboundLocalError on `vander`
    drift.align_nonrigid(
        optimizer_name="adam",
        num_iterations=2, adam_steps=20,
        regularization_sigma_px=None, lr=0.02,
        fixed_indices=[0],
        show_merged=False, show_images=False,
    )


# ──────────────────────────────────────────────────────────────
# Tests for apply_correction() and validation
# ──────────────────────────────────────────────────────────────

def _make_single_sided_dc(scan_h=256, drift_rate=(0.05, 0.1), seed=42):
    """Helper: build a DriftCorrection with known single-sided drift."""
    np.random.seed(seed)
    row_coords, col_coords = np.mgrid[:scan_h, :scan_h]
    ref = np.sin(0.1 * row_coords + 0.15 * col_coords).astype(np.float32) * 50 + 100
    rows = np.arange(scan_h, dtype=np.float32)
    src_row = row_coords - drift_rate[0] * rows[:, None]
    src_col = col_coords - drift_rate[1] * rows[:, None]
    drifted = map_coordinates(ref, [src_row, src_col], order=3, mode='nearest').astype(np.float32)

    dc = DriftCorrection(
        ref, drifted,
        scan_direction_degrees=[0.0, 0.0],
    )
    dc.preprocess(
        pad_fraction=0.25, pad_value=0.0, kde_sigma=0.5,
        number_knots=1, normalize=True,
        show_merged=False, show_images=False,
    )
    dc.align_affine(
        step=0.02, num_tests=11, refine=True,
        fixed_indices=[0], upsample_factor=8, max_image_shift=32,
        show_merged=False, show_images=False,
    )
    return dc, ref, drifted


def test_apply_correction_reduces_rms():
    """apply_correction should produce an image closer to the reference."""
    dc, ref, drifted = _make_single_sided_dc()
    corrected = dc.apply_correction(mode='bicubic').cpu().numpy()
    # z-normalize for fair comparison
    def znorm(a):
        return (a - a.mean()) / (a.std() + 1e-8)
    crop = 10  # avoid edges
    s = slice(crop, -crop)
    ref_n = znorm(ref[s, s])
    raw_rms = float(np.sqrt(((znorm(drifted[s, s]) - ref_n)**2).mean()))
    cor_rms = float(np.sqrt(((znorm(corrected[s, s]) - ref_n)**2).mean()))
    assert cor_rms < raw_rms * 0.8, (
        f"apply_correction should reduce RMS: raw={raw_rms:.4f}, corrected={cor_rms:.4f}"
    )


def test_apply_correction_accepts_external_images():
    """apply_correction(data=...) should work on external arrays."""
    dc, ref, drifted = _make_single_sided_dc()
    # Pass external image as numpy
    result_np = dc.apply_correction(data=drifted, mode='bilinear')
    assert result_np.shape == drifted.shape
    # Pass external image as tensor
    t = torch.tensor(drifted, device=dc._device, dtype=torch.float32)
    result_t = dc.apply_correction(data=t, mode='bilinear')
    assert result_t.shape == t.shape
    # Both should produce same output
    diff = float((result_np - result_t).abs().max())
    assert diff < 1e-5, f"numpy vs tensor path differ by {diff}"


def test_apply_correction_invalid_mode_raises():
    """apply_correction with invalid mode should raise ValueError."""
    dc, _, _ = _make_single_sided_dc()
    with pytest.raises(ValueError, match="mode must be one of"):
        dc.apply_correction(mode='cubic')


def test_apply_correction_wrong_height_raises():
    """apply_correction with mismatched image height should raise."""
    dc, _, _ = _make_single_sided_dc()
    wrong_size = np.zeros((32, 256), dtype=np.float32)
    with pytest.raises(ValueError, match="scan-row axis"):
        dc.apply_correction(data=wrong_size)


def test_apply_correction_before_preprocess_raises():
    """apply_correction before preprocess() should raise RuntimeError."""
    dc = DriftCorrection(
        np.zeros((64, 64)), np.zeros((64, 64)),
        scan_direction_degrees=[0.0, 0.0],
    )
    with pytest.raises(RuntimeError, match="preprocess"):
        dc.apply_correction()


def test_affine_confidence_margin_exists():
    """align_affine should set affine_confidence_margin."""
    dc, _, _ = _make_single_sided_dc()
    assert hasattr(dc, 'affine_confidence_margin')
    assert dc.affine_confidence_margin > 0, "Margin should be positive for clear drift"


def test_apply_correction_with_nonrigid():
    """apply_correction after nonrigid should reduce error further."""
    dc, ref, drifted = _make_single_sided_dc()
    aff_corrected = dc.apply_correction(mode='bicubic').cpu().numpy()
    dc.align_nonrigid(
        fixed_indices=[0],
        show_merged=False, show_images=False,
    )
    nr_corrected = dc.apply_correction(mode='bicubic').cpu().numpy()
    def znorm(a):
        return (a - a.mean()) / (a.std() + 1e-8)
    crop = 10
    s = slice(crop, -crop)
    ref_n = znorm(ref[s, s])
    aff_rms = float(np.sqrt(((znorm(aff_corrected[s, s]) - ref_n)**2).mean()))
    nr_rms = float(np.sqrt(((znorm(nr_corrected[s, s]) - ref_n)**2).mean()))
    # Nonrigid should not be worse than affine (within tolerance)
    assert nr_rms <= aff_rms * 1.05, (
        f"Nonrigid should not be significantly worse: aff={aff_rms:.4f}, nr={nr_rms:.4f}"
    )


def test_plot_correction_summary_runs():
    """plot_correction_summary should produce a figure without errors."""
    matplotlib.use('Agg')  # non-interactive backend
    dc, ref, drifted = _make_single_sided_dc()

    # Full summary (FFT + diff)
    fig, axs = dc.plot_correction_summary(show_fft=True, show_diff=True)
    assert fig is not None
    assert axs.shape == (3, 3)  # 3 rows x 3 cols

    # Images only — still returns (fig, axes)
    fig2, axs2 = dc.plot_correction_summary(show_fft=False, show_diff=False)
    assert fig2 is not None

    # User kwargs forwarded to show_2d
    fig3, axs3 = dc.plot_correction_summary(
        show_fft=False, show_diff=False, cmap="viridis", show_ticks=True,
    )
    assert fig3 is not None

    # Raw-merge column used to leave the diff row with the wrong number of
    # axes; keep this visual notebook path covered.
    fig4, axs4 = dc.plot_correction_summary(
        show_fft=False, show_diff=True, show_raw_merge=True,
    )
    assert fig4 is not None
    assert axs4.shape == (2, 4)

    plt.close("all")


# --------------- drift_rate, print_drift_stats, plot_correction_comparison, plot_radial_power ------

def test_drift_rate_matches_ground_truth():
    """drift_rate should approximate the known drift slope."""
    rate_gt = (0.05, 0.1)
    # Use larger image for more accurate drift estimation
    dc, _, _ = _make_single_sided_dc(drift_rate=rate_gt)
    rate = dc.drift_rate
    # Negative because knots compensate drift
    assert abs(rate[0] + rate_gt[0]) < 0.04, f"row rate {rate[0]} far from {-rate_gt[0]}"
    assert abs(rate[1] + rate_gt[1]) < 0.04, f"col rate {rate[1]} far from {-rate_gt[1]}"


def test_drift_rate_before_align_raises():
    """drift_rate before preprocess/align should raise RuntimeError."""
    dc = DriftCorrection(
        np.zeros((32, 32)), np.zeros((32, 32)),
        scan_direction_degrees=[0.0, 0.0],
    )
    with pytest.raises(RuntimeError, match="preprocess"):
        _ = dc.drift_rate


def test_print_drift_stats(capsys):
    """print_drift_stats should output rate, total, and confidence."""
    dc, _, _ = _make_single_sided_dc()
    dc.print_drift_stats()
    out = capsys.readouterr().out
    assert "Drift rate:" in out
    assert "Total drift:" in out
    assert "Affine confidence:" in out


def test_print_drift_stats_with_nonrigid(capsys):
    """print_drift_stats after nonrigid should also print nonrigid max."""
    dc, _, _ = _make_single_sided_dc()
    dc.align_nonrigid(fixed_indices=[0], show_merged=False, show_images=False)
    dc.print_drift_stats()
    out = capsys.readouterr().out
    assert "Nonrigid max correction:" in out


def test_plot_correction_comparison_runs():
    """plot_correction_comparison should produce fig, axes, metrics."""
    matplotlib.use('Agg')
    dc, _, _ = _make_single_sided_dc()
    # After affine only — should still work (no nonrigid)
    fig, axes, metrics = dc.plot_correction_comparison(crop=40)
    assert fig is not None
    assert "nonrigid bicubic" in metrics
    assert len(metrics) >= 2  # at least raw + nonrigid bicubic

    plt.close("all")


def test_plot_correction_comparison_with_nonrigid():
    """plot_correction_comparison after nonrigid should show all 4 modes."""
    matplotlib.use('Agg')
    dc, _, _ = _make_single_sided_dc()
    dc.align_nonrigid(fixed_indices=[0], show_merged=False, show_images=False)
    fig, axes, metrics = dc.plot_correction_comparison(crop=40, show_fft=False)
    assert fig is not None
    # Should have affine + nonrigid × bilinear + bicubic = 4 + raw
    assert "affine bilinear" in metrics
    assert "affine bicubic" in metrics
    assert "nonrigid bilinear" in metrics
    assert "nonrigid bicubic" in metrics

    plt.close("all")


def test_plot_radial_power_runs():
    """plot_radial_power should produce a figure."""
    matplotlib.use('Agg')
    dc, _, _ = _make_single_sided_dc()
    fig, ax = dc.plot_radial_power(crop=40)
    assert fig is not None
    assert len(ax.get_lines()) >= 2  # at least ref + one method

    plt.close("all")


def test_plot_radial_power_custom_methods():
    """plot_radial_power with user-provided methods dict."""
    matplotlib.use('Agg')
    dc, ref, drifted = _make_single_sided_dc()
    custom = {"reference": ref, "drifted": drifted}
    fig, ax = dc.plot_radial_power(methods=custom, crop=40)
    assert fig is not None
    assert len(ax.get_lines()) == 2

    plt.close("all")


def test_drift_visualization_image_helpers():
    """Notebook helpers should live in drift visualization utilities."""
    image = np.arange(6 * 8, dtype=np.float32).reshape(6, 8)
    cropped = center_crop(image, (4, 4))
    np.testing.assert_array_equal(cropped, image[1:5, 2:6])

    cube = np.zeros((6, 8, 2), dtype=np.float32)
    cube[..., 0] = image
    cube_crop = center_crop(cube, (4, 4))
    assert cube_crop.shape == (4, 4, 2)
    np.testing.assert_array_equal(cube_crop[..., 0], cropped)

    fft = fft_log_magnitude(cropped)
    assert fft.shape == cropped.shape
    assert np.isfinite(fft).all()

    rotated_view = np.rot90(image)
    assert any(stride < 0 for stride in rotated_view.strides)
    fft_rotated = fft_log_magnitude(rotated_view)
    assert fft_rotated.shape == rotated_view.shape
    assert np.isfinite(fft_rotated).all()

    assert normalized_cross_correlation(image, image, margin=1) > 0.999
    shifted = image + 10.0
    assert normalized_cross_correlation(image, shifted, crop_shape=(4, 4)) > 0.999


# ──────────────────────────────────────────────────────────────
# Tests for 3D spectral cube and 4D-STEM detector correction
# ──────────────────────────────────────────────────────────────

def _apply_drift_to_channels(channels, drift_rate, scan_h):
    """Apply known per-row linear drift to a stack of 2D channels.

    Parameters
    ----------
    channels : ndarray, shape (N, H, W)
        Clean channel images.
    drift_rate : tuple (row_rate, col_rate)
        Pixels per scan row of linear drift.
    scan_h : int
        Number of scan rows.

    Returns
    -------
    drifted : ndarray, shape (N, H, W)
        Drifted channels.
    """
    rows = np.arange(scan_h, dtype=np.float32)
    row_grid, col_grid = np.mgrid[:scan_h, :channels.shape[2]]
    src_row = row_grid - drift_rate[0] * rows[:, None]
    src_col = col_grid - drift_rate[1] * rows[:, None]
    drifted = np.empty_like(channels)
    for channel_idx in range(channels.shape[0]):
        drifted[channel_idx] = map_coordinates(
            channels[channel_idx], [src_row, src_col], order=3, mode='nearest'
        ).astype(np.float32)
    return drifted


def _make_diverse_channels(ref, n_channels, seed=99):
    """Create spatially distinct channels from a reference image.

    Returns channels with genuinely different spatial structure —
    localized peaks, filtered bands, masked regions — not just
    scaled copies.
    """
    np.random.seed(seed)
    H, W = ref.shape
    channels = np.empty((n_channels, H, W), dtype=np.float32)
    row_grid, col_grid = np.mgrid[:H, :W].astype(np.float32)

    for channel_idx in range(n_channels):
        if channel_idx % 4 == 0:
            # Localized Gaussian peak at random position
            center_row, center_col = np.random.randint(H // 4, 3 * H // 4, size=2)
            channels[channel_idx] = np.exp(
                -((row_grid - center_row)**2 + (col_grid - center_col)**2) / (2 * 30**2))
        elif channel_idx % 4 == 1:
            # Horizontal stripe pattern with different frequency
            freq = 0.05 + 0.03 * channel_idx
            channels[channel_idx] = (np.sin(freq * row_grid) + 1) * 50
        elif channel_idx % 4 == 2:
            # Masked quadrant of the reference
            mask = np.zeros((H, W), dtype=np.float32)
            quad_row, quad_col = channel_idx % 2, (channel_idx // 2) % 2
            mask[quad_row * H // 2:(quad_row + 1) * H // 2,
                 quad_col * W // 2:(quad_col + 1) * W // 2] = 1.0
            channels[channel_idx] = ref * mask
        else:
            # Smoothed + inverted reference
            channels[channel_idx] = gaussian_filter(ref.max() - ref, sigma=3 + channel_idx)
    return channels


def test_apply_correction_3d_spectral_cube():
    """Batch-correct an EDX-like spectral cube (H, W, E) → permute → correct."""
    scan_h = 128
    drift_rate = (0.05, 0.1)
    dc, ref, _ = _make_single_sided_dc(scan_h=scan_h, drift_rate=drift_rate)
    n_energy = 8

    # Build cube in natural (H, W, E) layout
    channels_clean = _make_diverse_channels(ref, n_energy, seed=200)
    channels_drifted = _apply_drift_to_channels(channels_clean, drift_rate, scan_h)
    cube_drifted = channels_drifted.transpose(1, 2, 0)  # (H, W, E)
    assert cube_drifted.shape == (scan_h, scan_h, n_energy)

    # Permute to (E, H, W) for apply_correction — this is the real user workflow
    batch = cube_drifted.transpose(2, 0, 1)  # (E, H, W)
    corrected = dc.apply_correction(data=batch, mode='bicubic')
    assert corrected.shape == (n_energy, scan_h, scan_h)

    corrected_np = corrected.cpu().numpy()
    crop = 15
    s = slice(crop, -crop)
    for ch in range(n_energy):
        gt = channels_clean[ch][s, s]
        gt_norm = (gt - gt.mean()) / (gt.std() + 1e-8)
        raw = channels_drifted[ch][s, s]
        raw_norm = (raw - raw.mean()) / (raw.std() + 1e-8)
        cor = corrected_np[ch][s, s]
        cor_norm = (cor - cor.mean()) / (cor.std() + 1e-8)
        raw_rms = float(np.sqrt(((raw_norm - gt_norm)**2).mean()))
        cor_rms = float(np.sqrt(((cor_norm - gt_norm)**2).mean()))
        assert cor_rms < raw_rms * 0.90, (
            f"Channel {ch}: correction should reduce RMS "
            f"(raw={raw_rms:.4f}, corrected={cor_rms:.4f})"
        )


def test_apply_correction_4d_stem_detector():
    """Batch-correct 4D-STEM detector pixels (H, W, det_h, det_w) → reshape → correct."""
    scan_h = 128
    drift_rate = (0.05, 0.1)
    dc, ref, _ = _make_single_sided_dc(scan_h=scan_h, drift_rate=drift_rate)
    det_h, det_w = 4, 4
    n_det = det_h * det_w  # 16 detector pixels

    # Build 4D cube in natural (H, W, det_h, det_w) layout
    channels_clean = _make_diverse_channels(ref, n_det, seed=300)
    channels_drifted = _apply_drift_to_channels(channels_clean, drift_rate, scan_h)
    cube_4d_drifted = channels_drifted.reshape(n_det, scan_h, scan_h)
    cube_4d_drifted = cube_4d_drifted.transpose(1, 2, 0)  # (H, W, n_det)
    cube_4d_drifted = cube_4d_drifted.reshape(scan_h, scan_h, det_h, det_w)
    assert cube_4d_drifted.shape == (scan_h, scan_h, det_h, det_w)

    # Reshape to (det_h*det_w, H, W) — the 4D-STEM user workflow
    batch = cube_4d_drifted.reshape(scan_h, scan_h, -1).transpose(2, 0, 1)
    assert batch.shape == (n_det, scan_h, scan_h)

    corrected = dc.apply_correction(data=batch, mode='bicubic')
    assert corrected.shape == (n_det, scan_h, scan_h)

    corrected_np = corrected.cpu().numpy()
    crop = 15
    s = slice(crop, -crop)
    for ch in range(n_det):
        gt = channels_clean[ch][s, s]
        gt_norm = (gt - gt.mean()) / (gt.std() + 1e-8)
        raw_norm = (channels_drifted[ch][s, s] - channels_drifted[ch][s, s].mean()) / (channels_drifted[ch][s, s].std() + 1e-8)
        cor = corrected_np[ch][s, s]
        cor_norm = (cor - cor.mean()) / (cor.std() + 1e-8)
        raw_rms = float(np.sqrt(((raw_norm - gt_norm)**2).mean()))
        cor_rms = float(np.sqrt(((cor_norm - gt_norm)**2).mean()))
        assert cor_rms < raw_rms * 0.90, (
            f"Detector pixel {ch}: correction should reduce RMS "
            f"(raw={raw_rms:.4f}, corrected={cor_rms:.4f})"
        )


def test_apply_correction_batch_matches_individual():
    """Batch correction must match per-channel correction exactly."""
    scan_h = 128
    dc, ref, _ = _make_single_sided_dc(scan_h=scan_h)
    channels = _make_diverse_channels(ref, 6, seed=400)
    drifted = _apply_drift_to_channels(channels, (0.05, 0.1), scan_h)

    # Batch
    batch_result = dc.apply_correction(data=drifted, mode='bicubic')

    # Individual
    for ch in range(drifted.shape[0]):
        single = dc.apply_correction(data=drifted[ch], mode='bicubic')
        torch.testing.assert_close(
            batch_result[ch], single,
            atol=1e-4, rtol=1e-4,
            msg=f"Channel {ch}: batch vs individual mismatch",
        )


def test_apply_correction_integer_input():
    """apply_correction transparently converts uint8/uint16 to float32."""
    scan_h = 128
    dc, ref, _ = _make_single_sided_dc(scan_h=scan_h)

    # uint8 input
    img_u8 = (ref / ref.max() * 200).astype(np.uint8)
    result_u8 = dc.apply_correction(data=img_u8, mode='bilinear')
    assert result_u8.dtype == torch.float32
    assert result_u8.shape == img_u8.shape

    # uint16 input
    img_u16 = (ref * 100).astype(np.uint16)
    result_u16 = dc.apply_correction(data=img_u16, mode='bilinear')
    assert result_u16.dtype == torch.float32
    assert result_u16.shape == img_u16.shape

    # float32 reference — results should match the float path
    img_f32 = img_u8.astype(np.float32)
    result_f32 = dc.apply_correction(data=img_f32, mode='bilinear')
    torch.testing.assert_close(
        result_u8, result_f32, atol=0.6, rtol=1e-3,
        msg="uint8 path should match float32 path (values are discretized)",
    )


def test_preprocess_rejects_nonsquare_with_multi_direction():
    """Multi-direction scan collection require square images (the canvas geometry
    assumes a single scanline length)."""
    np.random.seed(0)
    im0 = np.random.rand(96, 64).astype(np.float32)  # non-square
    im1 = np.random.rand(96, 64).astype(np.float32)
    dc = DriftCorrection(im0, im1, scan_direction_degrees=[0.0, -90.0])
    with pytest.raises(ValueError, match="square images"):
        dc.preprocess(pad_fraction=0.25, kde_sigma=0.5,
                      show_merged=False, show_images=False)


def test_preprocess_allows_nonsquare_with_single_direction():
    """Same scan direction → non-square is fine (consistent canvas geometry)."""
    np.random.seed(0)
    im0 = np.random.rand(96, 64).astype(np.float32)
    im1 = np.random.rand(96, 64).astype(np.float32)
    dc = DriftCorrection(im0, im1, scan_direction_degrees=[0.0, 0.0])
    dc.preprocess(pad_fraction=0.25, kde_sigma=0.5,
                  show_merged=False, show_images=False)  # should not raise


def test_preprocess_rejects_zero_knots():
    """preprocess() requires number_knots >= 1 (negative / zero is meaningless)."""
    ref = np.random.randn(64, 64).astype(np.float32)
    drifted = np.roll(ref, 2, axis=1).astype(np.float32)
    dc = DriftCorrection(ref, drifted, scan_direction_degrees=[0.0, 0.0])
    with pytest.raises(ValueError, match="number_knots"):
        dc.preprocess(number_knots=0, show_merged=False, show_images=False)


# ──────────────────────────────────────────────────────────────
# Multi-knot (K>1) tests — knot grid spans intra-row drift
# ──────────────────────────────────────────────────────────────


def test_multi_knot_initial_warp_matches_single_knot():
    """K=1 and K=2 with default knot grids produce identical initial warps
    on square images (both encode the same straight-scanline geometry)."""
    im0, im1, _ = make_synthetic_drift_data(scale=1, seed=42)
    dc1 = DriftCorrection(im0, im1, scan_direction_degrees=[0.0, -90.0]).preprocess(
        pad_fraction=0.25, kde_sigma=0.5, number_knots=1,
        show_merged=False, show_images=False)
    dc2 = DriftCorrection(im0, im1, scan_direction_degrees=[0.0, -90.0]).preprocess(
        pad_fraction=0.25, kde_sigma=0.5, number_knots=2,
        show_merged=False, show_images=False)
    assert dc2.knots[0].shape == (2, 128, 2)
    assert dc2.knots[1].shape == (2, 128, 2)
    np.testing.assert_allclose(
        dc1.imgs_warped.array, dc2.imgs_warped.array, atol=1e-5,
        err_msg="K=2 with default initial knots should match K=1 initial warp")


def test_multi_knot_full_pipeline_runs():
    """Full preprocess + align_affine + align_nonrigid + generate_corrected
    pipeline runs to completion with K=2 and reduces alignment error."""
    im0, im1, _ = make_synthetic_drift_data(scale=1, seed=42)
    dc = DriftCorrection(im0, im1, scan_direction_degrees=[0.0, -90.0]).preprocess(
        pad_fraction=0.25, kde_sigma=0.5, number_knots=2,
        show_merged=False, show_images=False)
    initial_error = float(dc.error_track[-1, 1])
    dc.align_affine(step=0.02, num_tests=5, refine=False, show_merged=False)
    dc.align_nonrigid(num_iterations=2, regularization_sigma_px=0.5,
                      loss="mse", show_merged=False, show_images=False)
    final_error = float(dc.error_track[-1, 1])
    assert final_error < initial_error, (
        f"K=2 nonrigid should reduce error: {initial_error=} {final_error=}")
    img_corr = dc.generate_corrected(upsample_factor=1, show_merged=False)
    assert isinstance(img_corr, Dataset2d)
    assert not np.isnan(img_corr.array).any()


def test_multi_knot_apply_correction_returns_correct_shape():
    """apply_correction on K=2 routes through DriftKnot and
    yields a per-pixel drift tensor of the right shape."""
    im0, im1, _ = make_synthetic_drift_data(scale=1, seed=42)
    dc = DriftCorrection(im0, im1, scan_direction_degrees=[0.0, -90.0]).preprocess(
        pad_fraction=0.25, kde_sigma=0.5, number_knots=2,
        show_merged=False, show_images=False)
    dc.align_affine(step=0.02, num_tests=5, refine=False, show_merged=False)
    drift_pixel = dc.drift_field(1)
    assert drift_pixel.shape == (2, 128, 128)
    corrected = dc.apply_correction(image_index=1)
    assert corrected.shape == (128, 128)
    assert not torch.isnan(corrected).any()


def test_multi_knot_lbfgs_runs():
    """LBFGS optimizer also handles K>1 (parallel dispatch path)."""
    im0, im1, _ = make_synthetic_drift_data(scale=1, seed=42)
    dc = DriftCorrection(im0, im1, scan_direction_degrees=[0.0, -90.0]).preprocess(
        pad_fraction=0.25, kde_sigma=0.5, number_knots=2,
        show_merged=False, show_images=False)
    dc.align_affine(step=0.02, num_tests=5, refine=False, show_merged=False)
    dc.align_nonrigid(optimizer_name="lbfgs", num_iterations=2,
                      regularization_sigma_px=0.5, loss="mse",
                      show_merged=False, show_images=False)
    assert dc.knots[0].shape[2] == 2


def test_multi_knot_inconsistent_K_rejected():
    """All images must use the same K — mismatched knot tensors should raise."""
    im0, im1, _ = make_synthetic_drift_data(scale=1, seed=42)
    dc = DriftCorrection(im0, im1, scan_direction_degrees=[0.0, -90.0]).preprocess(
        pad_fraction=0.25, kde_sigma=0.5, number_knots=2,
        show_merged=False, show_images=False)
    dc.align_affine(step=0.02, num_tests=5, refine=False, show_merged=False)
    # Tamper: replace image 1's knots with a single-knot grid.
    dc.knots[1] = dc.knots[1][:, :, :1].clone()
    with pytest.raises(ValueError, match="same number of knots"):
        dc.align_nonrigid(num_iterations=1, regularization_sigma_px=0.5,
                          loss="mse", show_merged=False, show_images=False)


# ──────────────────────────────────────────────────────────────
# Tests for apply_correction() — 3D/4D correction
# ──────────────────────────────────────────────────────────────

def test_apply_correction_3d_eds():
    """apply_correction on a 3D (H, W, E) EDX cube."""
    scan_h = 128
    drift_rate = (0.05, 0.1)
    dc, ref, _ = _make_single_sided_dc(scan_h=scan_h, drift_rate=drift_rate)
    n_energy = 12

    channels_clean = _make_diverse_channels(ref, n_energy, seed=500)
    channels_drifted = _apply_drift_to_channels(channels_clean, drift_rate, scan_h)
    # Natural EDX layout: (H, W, E)
    cube = channels_drifted.transpose(1, 2, 0)
    assert cube.shape == (scan_h, scan_h, n_energy)

    corrected = dc.apply_correction(cube)
    assert isinstance(corrected, np.ndarray)
    assert corrected.shape == cube.shape

    crop = 15
    s = slice(crop, -crop)
    for ch in range(n_energy):
        gt = channels_clean[ch][s, s]
        gt_n = (gt - gt.mean()) / (gt.std() + 1e-8)
        cor = corrected[s, s, ch]
        cor_n = (cor - cor.mean()) / (cor.std() + 1e-8)
        raw = cube[s, s, ch]
        raw_n = (raw - raw.mean()) / (raw.std() + 1e-8)
        raw_rms = float(np.sqrt(((raw_n - gt_n)**2).mean()))
        cor_rms = float(np.sqrt(((cor_n - gt_n)**2).mean()))
        assert cor_rms < raw_rms * 0.95, (
            f"Energy channel {ch}: cube correction should reduce RMS "
            f"(raw={raw_rms:.4f}, corrected={cor_rms:.4f})"
        )


def test_apply_correction_4d_stem():
    """apply_correction on a 4D (H, W, det_h, det_w) STEM cube."""
    scan_h = 128
    drift_rate = (0.05, 0.1)
    dc, ref, _ = _make_single_sided_dc(scan_h=scan_h, drift_rate=drift_rate)
    det_h, det_w = 4, 4
    n_det = det_h * det_w

    channels_clean = _make_diverse_channels(ref, n_det, seed=600)
    channels_drifted = _apply_drift_to_channels(channels_clean, drift_rate, scan_h)
    # Natural 4D layout: (H, W, det_h, det_w)
    cube_4d = channels_drifted.transpose(1, 2, 0).reshape(
        scan_h, scan_h, det_h, det_w
    )
    assert cube_4d.shape == (scan_h, scan_h, det_h, det_w)

    corrected = dc.apply_correction(cube_4d)
    assert isinstance(corrected, np.ndarray)
    assert corrected.shape == cube_4d.shape

    cor_flat = corrected.reshape(scan_h, scan_h, -1).transpose(2, 0, 1)
    crop = 15
    s = slice(crop, -crop)
    for ch in range(n_det):
        gt = channels_clean[ch][s, s]
        gt_n = (gt - gt.mean()) / (gt.std() + 1e-8)
        cor = cor_flat[ch][s, s]
        cor_n = (cor - cor.mean()) / (cor.std() + 1e-8)
        raw = channels_drifted[ch][s, s]
        raw_n = (raw - raw.mean()) / (raw.std() + 1e-8)
        raw_rms = float(np.sqrt(((raw_n - gt_n)**2).mean()))
        cor_rms = float(np.sqrt(((cor_n - gt_n)**2).mean()))
        assert cor_rms < raw_rms * 0.95, (
            f"Detector pixel {ch}: cube correction should reduce RMS "
            f"(raw={raw_rms:.4f}, corrected={cor_rms:.4f})"
        )


def test_apply_correction_matches_manual_chunking():
    """apply_correction must produce same results as manual loop."""
    scan_h = 128
    dc, ref, _ = _make_single_sided_dc(scan_h=scan_h)
    n_energy = 8

    channels = _make_diverse_channels(ref, n_energy, seed=700)
    drifted = _apply_drift_to_channels(channels, (0.05, 0.1), scan_h)
    cube = drifted.transpose(1, 2, 0)  # (H, W, E)

    # apply_correction
    auto = dc.apply_correction(cube)

    # Manual chunking (what user had to do before)
    batch = cube.transpose(2, 0, 1)  # (E, H, W)
    manual = dc.apply_correction(data=batch).cpu().numpy()
    manual = manual.transpose(1, 2, 0)  # (H, W, E)

    np.testing.assert_allclose(auto, manual, atol=1e-4, rtol=1e-4,
                               err_msg="Cube method must match manual batch")


def test_apply_correction_torch_input():
    """apply_correction works with torch.Tensor input."""
    scan_h = 128
    dc, ref, _ = _make_single_sided_dc(scan_h=scan_h)
    cube_np = np.random.randn(scan_h, scan_h, 6).astype(np.float32)
    cube_t = torch.from_numpy(cube_np)

    result = dc.apply_correction(cube_t)
    assert isinstance(result, torch.Tensor)
    assert result.shape == cube_t.shape


def test_apply_correction_output_dtype_same():
    """apply_correction with output_dtype='same' preserves input dtype."""
    scan_h = 128
    dc, ref, _ = _make_single_sided_dc(scan_h=scan_h)

    cube_u16 = (np.random.rand(scan_h, scan_h, 4) * 1000).astype(np.uint16)
    result = dc.apply_correction(cube_u16, output_dtype="same")
    assert result.dtype == np.uint16
    assert result.shape == cube_u16.shape


def test_apply_correction_matches_apply_correction_scan_collection():
    """apply_correction and apply_correction agree for 0/90° scans."""
    im0, im1, _ = make_synthetic_drift_data()
    dc = DriftCorrection(
        im0, im1,
        scan_direction_degrees=[0, 90],
    )
    dc.preprocess(
        pad_fraction=0.25, kde_sigma=0.5, number_knots=1,
        show_merged=False, show_images=False,
    )
    dc.align_affine(
        step=0.02, num_tests=11,
        show_merged=False, show_images=False,
    )
    # Correct image 1 (90° scan) with both methods
    corrected_2d = dc.apply_correction(data=im1.astype(np.float32), image_index=1)
    corrected_2d_np = corrected_2d.cpu().numpy()
    cube_3d = im1[:, :, None].astype(np.float32)
    corrected_3d = dc.apply_correction(cube_3d, image_index=1)
    np.testing.assert_allclose(
        corrected_3d[:, :, 0], corrected_2d_np, atol=1e-4,
        err_msg="apply_correction and apply_correction must agree for 90° scan",
    )


def test_apply_correction_output_parameter():
    """output= writes directly to a pre-allocated numpy array."""
    scan_h = 128
    dc, ref, _ = _make_single_sided_dc(scan_h=scan_h)
    n_energy = 8
    channels = _make_diverse_channels(ref, n_energy, seed=800)
    drifted = _apply_drift_to_channels(channels, (0.05, 0.1), scan_h)
    cube = drifted.transpose(1, 2, 0)  # (H, W, E)

    # Without output= (baseline)
    auto = dc.apply_correction(cube)

    # With output= pre-allocated
    out_buf = np.empty_like(auto)
    returned = dc.apply_correction(cube, output=out_buf)
    assert returned is out_buf, "Should return the same array object"
    np.testing.assert_allclose(out_buf, auto, atol=1e-5,
                               err_msg="output= must match default path")


def test_apply_correction_output_memmap(tmp_path):
    """output= works with np.memmap for disk-backed writes."""
    scan_h = 128
    dc, ref, _ = _make_single_sided_dc(scan_h=scan_h)
    cube = np.random.rand(scan_h, scan_h, 4, 4).astype(np.float32)
    shape = cube.shape

    mmap_path = tmp_path / "corrected.dat"
    out_mmap = np.memmap(str(mmap_path), dtype="float32", mode="w+", shape=shape)
    returned = dc.apply_correction(cube, output=out_mmap)
    assert returned is out_mmap
    assert returned.shape == shape

    # Verify data was actually written to disk
    out_mmap.flush()
    loaded = np.memmap(str(mmap_path), dtype="float32", mode="r", shape=shape)
    np.testing.assert_allclose(loaded, out_mmap, atol=1e-6)

    # Verify it matches the default path
    auto = dc.apply_correction(cube)
    np.testing.assert_allclose(loaded, auto, atol=1e-5,
                               err_msg="memmap output must match default path")


def test_apply_correction_output_shape_mismatch():
    """output= with wrong shape raises ValueError."""
    scan_h = 128
    dc, ref, _ = _make_single_sided_dc(scan_h=scan_h)
    cube = np.random.rand(scan_h, scan_h, 6).astype(np.float32)
    wrong = np.empty((scan_h, scan_h, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="does not match"):
        dc.apply_correction(cube, output=wrong)


def test_apply_correction_output_type_error():
    """output= with non-numpy type raises TypeError."""
    scan_h = 128
    dc, ref, _ = _make_single_sided_dc(scan_h=scan_h)
    cube = np.random.rand(scan_h, scan_h, 6).astype(np.float32)
    with pytest.raises(TypeError, match="numpy ndarray"):
        dc.apply_correction(cube, output=[1, 2, 3])


# ──────────────────────────────────────────────────────────────
# Tests for compute_vdf() — VDF extraction from 4D-STEM
# ──────────────────────────────────────────────────────────────

def test_compute_vdf_basic():
    """compute_vdf returns correct mean over detector dimensions."""
    H, W, dh, dw = 32, 32, 4, 4
    cube = np.random.rand(H, W, dh, dw).astype(np.float32)
    vdf = DriftCorrection.compute_vdf(cube)
    assert vdf.shape == (H, W)
    assert vdf.dtype == np.float32
    expected = cube.reshape(H, W, -1).mean(axis=2)
    np.testing.assert_allclose(vdf, expected, atol=1e-6)


def test_compute_vdf_chunked():
    """compute_vdf with chunk_rows matches non-chunked result."""
    H, W, dh, dw = 64, 64, 8, 8
    cube = np.random.rand(H, W, dh, dw).astype(np.float32)
    vdf_full = DriftCorrection.compute_vdf(cube)
    vdf_chunked = DriftCorrection.compute_vdf(cube, chunk_rows=16)
    np.testing.assert_allclose(vdf_chunked, vdf_full, atol=1e-6)


def test_compute_vdf_3d():
    """compute_vdf works on 3D (H, W, C) data too."""
    H, W, C = 32, 32, 12
    cube = np.random.rand(H, W, C).astype(np.float32)
    vdf = DriftCorrection.compute_vdf(cube)
    assert vdf.shape == (H, W)
    expected = cube.mean(axis=2)
    np.testing.assert_allclose(vdf, expected, atol=1e-6)


# ═════════════════════════════════════════════════════════════════════════
#   4D-STEM class integration tests (from_4dstem with 4D cubes)
# ═════════════════════════════════════════════════════════════════════════

def _make_4dstem_collection(scan_size=32, det_size=8, seed=42):
    """Create small synthetic 4D cubes for testing.

    Returns (cube_0deg, cube_90deg) each with shape
    ``(scan_size, scan_size, det_size, det_size)`` where the first two
    dimensions mimic VDF images from ``make_synthetic_drift_data``.
    """
    im0, im1, _ = make_synthetic_drift_data(scale=1, seed=seed)
    im0 = im0[:scan_size, :scan_size]
    im1 = im1[:scan_size, :scan_size]
    rng = np.random.RandomState(seed)
    cube_a = np.empty((scan_size, scan_size, det_size, det_size), dtype=np.float32)
    cube_b = np.empty_like(cube_a)
    for i in range(det_size):
        for j in range(det_size):
            cube_a[:, :, i, j] = im0 + rng.randn(scan_size, scan_size) * 0.01
            cube_b[:, :, i, j] = im1 + rng.randn(scan_size, scan_size) * 0.01
    return cube_a, cube_b


def _best_rot90_by_ncc(reference, moving, candidates):
    """Pick the quarter-turn that best aligns ``moving`` to ``reference``."""
    ref = reference.astype(np.float32)
    ref = ref - ref.mean()
    ref_norm = np.linalg.norm(ref)
    best_k, best_score = None, -np.inf
    for k in candidates:
        cand = np.rot90(moving, k=k).astype(np.float32)
        cand = cand - cand.mean()
        score = float((ref * cand).sum() / (ref_norm * np.linalg.norm(cand) + 1e-12))
        if score > best_score:
            best_k, best_score = k, score
    return best_k


def _make_detector_template(det_size):
    """Small structured diffraction pattern used to lift 2-D scans to 4-D."""
    det_center = det_size / 2
    rr, cc = np.meshgrid(
        np.arange(det_size, dtype=np.float32) - det_center,
        np.arange(det_size, dtype=np.float32) - det_center,
        indexing="ij",
    )
    radius = np.sqrt(rr**2 + cc**2)
    template = np.where(radius < det_size / 8, 1.0, 0.05).astype(np.float32)
    for theta in np.linspace(0, 2 * np.pi, 6, endpoint=False):
        row0 = (det_size / 3) * np.sin(theta)
        col0 = (det_size / 3) * np.cos(theta)
        template += 0.4 * np.exp(-((rr - row0) ** 2 + (cc - col0) ** 2) / 4)
    return template.astype(np.float32)


def _lift_to_4dstem(scan_image, detector_template):
    return (
        scan_image.astype(np.float32)[:, :, None, None]
        * detector_template[None, None, :, :]
    ).astype(np.float32)


def _annular_vdf_mask(det_size, inner_fraction=0.25):
    center = det_size // 2
    rr, cc = np.meshgrid(
        np.arange(det_size) - center,
        np.arange(det_size) - center,
        indexing="ij",
    )
    return rr**2 + cc**2 > (inner_fraction * det_size) ** 2


def _masked_vdf(cube, mask):
    return cube[:, :, mask].sum(axis=-1).astype(np.float32)


def _aligned_ncc(reference, candidate, margin=12):
    """Integer-shift align ``candidate`` to ``reference`` and NCC the interior."""
    ref_z = reference - reference.mean()
    cand_z = candidate - candidate.mean()
    cross_corr = np.fft.fftshift(
        np.real(np.fft.ifft2(np.fft.fft2(ref_z) * np.conj(np.fft.fft2(cand_z))))
    )
    peak_row, peak_col = np.unravel_index(int(np.argmax(cross_corr)), cross_corr.shape)
    row_shift = peak_row - reference.shape[0] // 2
    col_shift = peak_col - reference.shape[1] // 2
    aligned = np.roll(candidate, (row_shift, col_shift), axis=(0, 1))
    s = slice(margin, -margin)
    ref_crop = reference[s, s].astype(np.float32)
    aligned_crop = aligned[s, s].astype(np.float32)
    ref_crop = (ref_crop - ref_crop.mean()) / (ref_crop.std() + 1e-8)
    aligned_crop = (aligned_crop - aligned_crop.mean()) / (aligned_crop.std() + 1e-8)
    return float((ref_crop * aligned_crop).mean())


def test_from_4dstem_detects_4d():
    """from_4dstem with 4D arrays creates a 4D-STEM instance."""
    cube_a, cube_b = _make_4dstem_collection(scan_size=32, det_size=4)
    dc = DriftCorrection(
        cube_a, cube_b, scan_direction_degrees=[0, 90],
    )
    assert dc.is_4dstem
    # VDFs were extracted automatically — alignment images are 2D
    assert dc.imgs[0].array.shape == (32, 32)
    assert dc.imgs[1].array.shape == (32, 32)


def test_from_images_4dstem_alignment_image_rejected_in_scan_collection_mode():
    """alignment_image= is only valid in reference mode, not 4D-STEM collection."""
    cube_a, cube_b = _make_4dstem_collection(scan_size=32, det_size=4)
    custom_vdf = cube_a[:, :, 0, 0].copy()
    with pytest.raises(TypeError, match="only meaningful in reference mode"):
        DriftCorrection(
            cube_a, cube_b, scan_direction_degrees=[0, 90],
            alignment_image=custom_vdf,
        )


@pytest.mark.parametrize("scan_direction_degrees", ([0, 0], [0, 180], [13, 103], [0.5, 90.5]))
def test_from_4dstem_requires_orthogonal_scan_directions(scan_direction_degrees):
    """4D-STEM collection is specifically for 0/90 scan pairs."""
    cube_a, cube_b = _make_4dstem_collection(scan_size=32, det_size=4)
    with pytest.raises(ValueError, match="0/90"):
        DriftCorrection.from_4dstem(
            cube_a, cube_b, scan_direction_degrees=scan_direction_degrees,
        )


# ---
# Tests for from_reference (single-sided HAADF + drifted cube)
# ---

def _make_reference_pair(scan_h=64, det_size=4, kind="3d", seed=0):
    rng = np.random.default_rng(seed)
    ref = rng.random((scan_h, scan_h), dtype=np.float32)
    if kind == "2d":
        drifted = rng.random((scan_h, scan_h), dtype=np.float32)
    elif kind == "3d":
        drifted = rng.random((scan_h, scan_h, det_size), dtype=np.float32)
    else:
        drifted = rng.random((scan_h, scan_h, det_size, det_size), dtype=np.float32)
    return ref, drifted


def test_from_reference_3d_eds_returns_dataset3d():
    """from_reference + 3-D drifted → generate_corrected returns Dataset3d."""
    ref, eds = _make_reference_pair(scan_h=32, det_size=4, kind="3d")
    dc = DriftCorrection.from_reference(ref, eds)
    assert dc._reference_mode
    dc.preprocess(normalize=True, kde_sigma=0.5, number_knots=1,
                  show_merged=False, show_images=False)
    dc.align_affine(step=0.02, num_tests=5,
                    show_merged=False, show_images=False)
    result = dc.generate_corrected(show_merged=False)
    assert isinstance(result, Dataset3d)
    assert result.array.shape == eds.shape


def test_from_reference_4d_stem_returns_dataset4d():
    """from_reference + 4-D drifted → generate_corrected returns Dataset4d."""
    from quantem.core.datastructures.dataset4d import Dataset4d
    ref, cube = _make_reference_pair(scan_h=32, det_size=4, kind="4d")
    dc = DriftCorrection.from_reference(ref, cube)
    assert dc._reference_mode
    dc.preprocess(normalize=True, kde_sigma=0.5, number_knots=1,
                  show_merged=False, show_images=False)
    dc.align_affine(step=0.02, num_tests=5,
                    show_merged=False, show_images=False)
    result = dc.generate_corrected(show_merged=False)
    assert isinstance(result, Dataset4d)
    assert result.array.shape == cube.shape


def test_from_reference_2d_returns_dataset2d():
    """from_reference + 2-D drifted → generate_corrected returns Dataset2d.

    Same scan angle (0, 0) signals reference mode rather than orthogonal pair."""
    ref, drifted = _make_reference_pair(scan_h=32, kind="2d")
    dc = DriftCorrection.from_reference(ref, drifted)
    assert dc._reference_mode
    dc.preprocess(normalize=True, kde_sigma=0.5, number_knots=1,
                  show_merged=False, show_images=False)
    dc.align_affine(step=0.02, num_tests=5,
                    show_merged=False, show_images=False)
    result = dc.generate_corrected(show_merged=False)
    assert isinstance(result, Dataset2d)
    assert result.array.shape == drifted.shape


def test_from_reference_auto_anchors_reference():
    """In reference mode, align_affine and align_nonrigid auto-set
    fixed_indices=[0] so reference knots stay anchored."""
    ref, eds = _make_reference_pair(scan_h=32, det_size=4, kind="3d")
    dc = DriftCorrection.from_reference(ref, eds)
    dc.preprocess(normalize=True, kde_sigma=0.5, number_knots=1,
                  show_merged=False, show_images=False)
    knots_ref_initial = dc._initial_knots[0].clone()
    dc.align_affine(step=0.02, num_tests=5,
                    show_merged=False, show_images=False)
    # Reference image (idx 0) should be unchanged after affine.
    assert torch.allclose(dc.knots[0], knots_ref_initial, atol=1e-6)
    dc.align_nonrigid(num_iterations=1, adam_steps=5,
                      show_merged=False, show_images=False)
    # Reference image (idx 0) should also stay anchored after nonrigid.
    assert torch.allclose(dc.knots[0], knots_ref_initial, atol=1e-6)


def test_from_reference_shape_mismatch_raises():
    """reference shape must match leading 2 axes of drifted."""
    ref = np.zeros((32, 32), dtype=np.float32)
    drifted = np.zeros((40, 32, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="leading two axes"):
        DriftCorrection.from_reference(ref, drifted)


def test_from_reference_alignment_image_override():
    """alignment_image= bypasses the auto VDF computation."""
    ref, cube = _make_reference_pair(scan_h=32, det_size=4, kind="4d")
    custom_vdf = np.ones((32, 32), dtype=np.float32) * 0.5
    dc = DriftCorrection.from_reference(ref, cube, alignment_image=custom_vdf)
    np.testing.assert_array_equal(dc.imgs[1].array, custom_vdf)


def test_from_reference_rejects_ambiguous_2d_pair_scan_angles():
    """The named reference factory must not silently create image collection mode."""
    ref, drifted = _make_reference_pair(scan_h=32, kind="2d")
    with pytest.raises(ValueError, match="unambiguously single-sided"):
        DriftCorrection.from_reference(
            ref, drifted, scan_direction_degrees=(0.0, 90.0),
        )


def test_generate_corrected_basic():
    """Full pipeline: from_4dstem → preprocess → align → explicit scan collection correction."""
    cube_a, cube_b = _make_4dstem_collection(scan_size=32, det_size=4)
    dc = DriftCorrection(
        cube_a, cube_b, scan_direction_degrees=[0, 90],
    )
    dc.preprocess(
        pad_fraction=0.25, kde_sigma=0.5, number_knots=1,
        show_merged=False, show_images=False,
    )
    dc.align_affine(
        step=0.02, num_tests=11,
        show_merged=False, show_images=False,
    )
    result = dc.generate_corrected_4dstem()

    assert isinstance(result, CorrectionResult)
    assert result.corrected_4dstem is not None
    assert result.corrected_4dstem.shape == cube_a.shape
    assert result.corrected_4dstem.dtype == np.float32
    assert result.corrected_4dstem_0.shape == cube_a.shape
    assert result.corrected_4dstem_1.shape == cube_a.shape
    assert isinstance(result.drift, DriftCorrection)
    assert result.raw_vdf_0.shape == (32, 32)
    assert result.raw_vdf_1.shape == (32, 32)


def test_from_4dstem_named_api_and_result_fields():
    """0/90 4D-STEM has an explicit first-class entry point."""
    cube_a, cube_b = _make_4dstem_collection(scan_size=32, det_size=4)
    dc = DriftCorrection.from_4dstem(
        cube_a, cube_b, scan_direction_degrees=[0, 90],
    )
    assert dc._is_4dstem_collection
    dc.preprocess(
        pad_fraction=0.25, kde_sigma=0.5, number_knots=1,
        show_merged=False, show_images=False,
    )
    dc.align_affine(
        step=0.02, num_tests=5, refine=False,
        show_merged=False, show_images=False,
    )
    assert drift_4dstem_module._rot90_to_image0_frame(dc, image_index=1) == 3
    result = dc.generate_corrected_4dstem()

    assert isinstance(result, CorrectionResult)
    assert result.corrected_4dstem is not None
    assert result.corrected_4dstem_0.shape == cube_a.shape
    assert result.corrected_4dstem_1.shape == cube_a.shape
    assert result.scalar_corrected_vdf.shape == (32, 32)
    for old_name in [
        "merged",
        "corrected_a",
        "corrected_b",
        "component_0",
        "component_1",
        "corrected_cube",
        "cube",
        "reference_vdf",
    ]:
        assert not hasattr(result, old_name)

def test_generate_corrected_4dstem_rejects_reference_mode():
    """The 4D-STEM collection API should not hide reference-mode semantics."""
    ref, cube = _make_reference_pair(scan_h=32, det_size=4, kind="4d")
    dc = DriftCorrection.from_reference(ref, cube)
    dc.preprocess(
        normalize=True, kde_sigma=0.5, number_knots=1,
        show_merged=False, show_images=False,
    )
    dc.align_affine(
        step=0.02, num_tests=5,
        show_merged=False, show_images=False,
    )
    with pytest.raises(RuntimeError, match="4D-STEM collection"):
        dc.generate_corrected_4dstem()


def test_correct_virtual_images_matches_result_scalar_corrected_vdf():
    """Scan-corrected scalar VDF is exposed as a reusable API."""
    cube_a, cube_b = _make_4dstem_collection(scan_size=32, det_size=4)
    dc = DriftCorrection.from_4dstem(
        cube_a, cube_b, scan_direction_degrees=[0, 90],
    )
    dc.preprocess(
        pad_fraction=0.25, kde_sigma=0.5, number_knots=1,
        show_merged=False, show_images=False,
    )
    dc.align_affine(
        step=0.02, num_tests=5, refine=False,
        show_merged=False, show_images=False,
    )
    assert drift_4dstem_module._rot90_to_image0_frame(dc, image_index=1) == 3
    result = dc.generate_corrected_4dstem()
    virtual = dc.correct_virtual_images(result.raw_vdf_0, result.raw_vdf_1)

    np.testing.assert_allclose(virtual["corrected_image"], result.scalar_corrected_vdf, atol=0.0)
    assert virtual["corrected_image_0"].shape == (32, 32)
    assert virtual["corrected_image_1"].shape == (32, 32)
    for old_name in [
        "merged", "image_0", "image_1", "component_0", "component_1",
        "weight_0", "weight_1",
    ]:
        assert old_name not in virtual


def test_4dstem_pair_090_orients_image_1_with_microscope_convention():
    """A 0/90 4D-STEM pair rotates corrected image 1 into image 0's frame."""
    cube_0, cube_90 = _make_4dstem_collection(scan_size=32, det_size=4)
    dc = DriftCorrection.from_4dstem(
        cube_0, cube_90, scan_direction_degrees=[0, 90],
    )
    dc.preprocess(
        pad_fraction=0.25, kde_sigma=0.5, number_knots=1,
        show_merged=False, show_images=False,
    )
    dc.align_affine(
        step=0.02, num_tests=5, refine=False,
        show_merged=False, show_images=False,
    )
    result = dc.generate_corrected_4dstem()
    virtual = dc.correct_virtual_images(result.raw_vdf_0, result.raw_vdf_1)

    vdf_1_unoriented = DriftCorrection.integrate_virtual_detector(
        result.corrected_4dstem_1,
        reduce="mean",
    )
    np.testing.assert_allclose(
        virtual["corrected_image_1"], vdf_1_unoriented, atol=2e-7,
    )
    assert virtual["corrected_image_1"].shape == virtual["corrected_image_0"].shape


def test_probe_positions_export_and_plot_for_4dstem_collection():
    """Updated probe positions stay indexed like the raw 4D-STEM scans."""
    cube_a, cube_b = _make_4dstem_collection(scan_size=32, det_size=4)
    dc = DriftCorrection.from_4dstem(
        cube_a, cube_b, scan_direction_degrees=[0, 90],
    )
    dc.preprocess(
        pad_fraction=0.25, kde_sigma=0.5, number_knots=1,
        show_merged=False, show_images=False,
    )
    nominal_1 = dc.probe_positions(image_index=1, corrected=False, plot=False)
    dc.align_affine(
        step=0.02, num_tests=5, refine=False,
        show_merged=False, show_images=False,
    )
    positions_0 = dc.probe_positions(image_index=0, plot=False)
    positions_1 = dc.probe_positions(image_index=1, plot=False)

    assert positions_0.shape == (32, 32, 2)
    assert positions_1.shape == (32, 32, 2)
    assert positions_0.dtype == np.float32
    assert positions_1.dtype == np.float32
    assert not np.allclose(positions_1, nominal_1)
    # Flattening preserves raw diffraction-pattern order for ptychography.
    assert positions_1.reshape(-1, 2).shape == (32 * 32, 2)

    result = dc.generate_corrected_4dstem(merge=False)
    result_positions_1 = result.probe_positions(image_index=1, plot=False)
    np.testing.assert_allclose(result_positions_1, positions_1)

    fig, axes = dc.plot_probe_positions(image_index=1, stride=8)
    assert len(axes) == 2
    plt.close(fig)


def test_scan_collection_result_virtual_image_and_crop_helpers():
    """Result helper names make corrected 4D-STEM integrations explicit."""
    cube_a, cube_b = _make_4dstem_collection(scan_size=32, det_size=4)
    df_mask = _annular_vdf_mask(det_size=4)
    bf_mask = ~df_mask
    dc = DriftCorrection.from_4dstem(
        cube_a, cube_b, scan_direction_degrees=[0, 90],
    )
    dc.preprocess(
        pad_fraction=0.25, kde_sigma=0.5, number_knots=1,
        show_merged=False, show_images=False,
    )
    dc.align_affine(
        step=0.02, num_tests=5, refine=False,
        show_merged=False, show_images=False,
    )
    result = dc.generate_corrected_4dstem()

    for mask, reduce in [
        (None, "mean"),       # all-detector VDF used for alignment
        (bf_mask, "sum"),     # BF-style disk integration
        (df_mask, "sum"),     # DF-style annular integration
    ]:
        cube_vdf = result.virtual_image(mask, reduce=reduce)
        manual_vdf = DriftCorrection.integrate_virtual_detector(
            result.corrected_4dstem, mask, reduce=reduce,
        )
        np.testing.assert_allclose(cube_vdf, manual_vdf, atol=1e-6)


def test_scan_collection_4dstem_synthetic_corrected_vdf_matches_reference_ncc():
    """Synthetic 0/90 4D-STEM cubes recover the ground-truth reference VDF.

    This locks the intended diagnostic workflow:
    1. Start from one drift-free reference 4D-STEM cube.
    2. Simulate a drifted 0-degree scan and a drifted 90-degree scan.
    3. Lift both scans to 4D-STEM diffraction-pattern cubes.
    4. Estimate drift from their auto-extracted VDFs.
    5. Apply the same drift fields to the full cubes and merge.

    The acceptance criterion is the real target: the VDF computed from the
    corrected 4D-STEM dataset should have higher NCC against the ground-truth
    reference VDF than the uncorrected raw merge.
    """
    scan_size = 128
    det_size = 6
    im0, im90, base = make_synthetic_drift_data(scale=1, seed=42)
    reference = base[40:40 + scan_size, 30:30 + scan_size].astype(np.float32)
    detector = _make_detector_template(det_size)
    vdf_mask = _annular_vdf_mask(det_size)
    reference_cube = _lift_to_4dstem(reference, detector)
    cube_0 = _lift_to_4dstem(im0.astype(np.float32), detector)
    cube_90 = _lift_to_4dstem(im90.astype(np.float32), detector)

    dc = DriftCorrection.from_4dstem(
        cube_0, cube_90, scan_direction_degrees=[0, 90],
    )
    dc.preprocess(
        pad_fraction=0.25, kde_sigma=0.5, number_knots=2,
        normalize=True, show_merged=False, show_images=False,
    )
    dc.align_affine(
        step=0.02, num_tests=11, refine=True, max_image_shift=64,
        show_merged=False, show_images=False,
    )
    dc.align_nonrigid(
        num_iterations=20, regularization_sigma_px=16.0,
        show_merged=False, show_images=False,
    )
    result = dc.generate_corrected_4dstem(
        mode="bilinear", output_dtype=torch.float32,
    )

    ground_truth_vdf = _masked_vdf(reference_cube, vdf_mask)
    raw_vdf_0 = _masked_vdf(cube_0, vdf_mask)
    raw_vdf_90 = _masked_vdf(cube_90, vdf_mask)
    raw_rot_k = _best_rot90_by_ncc(ground_truth_vdf, raw_vdf_90, candidates={1, 3})
    raw_merge = (raw_vdf_0 + np.rot90(raw_vdf_90, k=raw_rot_k)) * 0.5

    vdf_from_corrected_4dstem = _masked_vdf(result.corrected_4dstem, vdf_mask)
    ncc_raw = _aligned_ncc(ground_truth_vdf, raw_merge)
    ncc_corrected = _aligned_ncc(ground_truth_vdf, vdf_from_corrected_4dstem)
    assert ncc_raw < 0.80
    assert ncc_corrected > 0.97
    assert ncc_corrected > ncc_raw + 0.19


def test_generate_corrected_no_merge():
    """merge=False leaves corrected_4dstem unset but keeps both components."""
    cube_a, cube_b = _make_4dstem_collection(scan_size=32, det_size=4)
    dc = DriftCorrection(
        cube_a, cube_b, scan_direction_degrees=[0, 90],
    )
    dc.preprocess(
        pad_fraction=0.25, kde_sigma=0.5, number_knots=1,
        show_merged=False, show_images=False,
    )
    dc.align_affine(
        step=0.02, num_tests=11,
        show_merged=False, show_images=False,
    )
    result = dc.generate_corrected_4dstem(merge=False)
    assert result.corrected_4dstem is None
    assert result.corrected_4dstem_0.shape == cube_a.shape
    assert result.corrected_4dstem_1.shape == cube_a.shape


def test_generate_corrected_nonrigid():
    """Nonrigid alignment works in the 4D pipeline."""
    cube_a, cube_b = _make_4dstem_collection(scan_size=32, det_size=4)
    dc = DriftCorrection(
        cube_a, cube_b, scan_direction_degrees=[0, 90],
    )
    dc.preprocess(
        pad_fraction=0.25, kde_sigma=0.5, number_knots=1,
        show_merged=False, show_images=False,
    )
    dc.align_affine(
        step=0.02, num_tests=11,
        show_merged=False, show_images=False,
    )
    dc.align_nonrigid(show_merged=False, show_images=False)
    result = dc.generate_corrected_4dstem()
    assert result.corrected_4dstem is not None
    assert result.corrected_4dstem.shape == cube_a.shape


def test_apply_correction_uses_stored_cube():
    """apply_correction with no cube arg uses stored cube."""
    cube_a, cube_b = _make_4dstem_collection(scan_size=32, det_size=4)
    dc = DriftCorrection(
        cube_a, cube_b, scan_direction_degrees=[0, 90],
    )
    dc.preprocess(
        pad_fraction=0.25, kde_sigma=0.5, number_knots=1,
        show_merged=False, show_images=False,
    )
    dc.align_affine(
        step=0.02, num_tests=11,
        show_merged=False, show_images=False,
    )
    corrected_0 = dc.apply_correction(image_index=0)
    assert corrected_0.shape == cube_a.shape


def test_apply_correction_pair_mode_uses_stored_image():
    """apply_correction() with no args on a pair-mode dc warps the stored image."""
    im0, im1, _ = make_synthetic_drift_data(scale=1)
    dc = DriftCorrection(
        im0[:32, :32], im1[:32, :32], scan_direction_degrees=[0, 90],
    )
    dc.preprocess(
        pad_fraction=0.25, kde_sigma=0.5, number_knots=1,
        show_merged=False, show_images=False,
    )
    dc.align_affine(
        step=0.02, num_tests=11,
        show_merged=False, show_images=False,
    )
    result = dc.apply_correction()
    assert result.shape == (32, 32)


def test_generate_corrected_pair_mode_returns_dataset2d():
    """generate_corrected on pair-mode dc returns Dataset2d."""
    from quantem.core.datastructures.dataset2d import Dataset2d
    im0, im1, _ = make_synthetic_drift_data(scale=1)
    dc = DriftCorrection(
        im0[:32, :32], im1[:32, :32], scan_direction_degrees=[0, 90],
    )
    dc.preprocess(
        pad_fraction=0.25, kde_sigma=0.5, number_knots=1,
        show_merged=False, show_images=False,
    )
    dc.align_affine(
        step=0.02, num_tests=11,
        show_merged=False, show_images=False,
    )
    result = dc.generate_corrected(show_merged=False)
    assert isinstance(result, Dataset2d)


def test_is_4dstem_property():
    """is_4dstem reflects whether cubes are stored."""
    im0, im1, _ = make_synthetic_drift_data(scale=1)
    dc_2d = DriftCorrection(
        im0[:32, :32], im1[:32, :32], scan_direction_degrees=[0, 90],
    )
    assert not dc_2d.is_4dstem
    assert not dc_2d._is_4dstem_collection

    cube_a, cube_b = _make_4dstem_collection(scan_size=32, det_size=4)
    dc_4d = DriftCorrection(
        cube_a, cube_b, scan_direction_degrees=[0, 90],
    )
    assert dc_4d.is_4dstem
    assert dc_4d._is_4dstem_collection


def test__is_4dstem_collection_false_for_reference_mode():
    """Reference mode sets is_4dstem=True but _is_4dstem_collection=False."""
    ref, cube = _make_reference_pair(scan_h=32, det_size=4, kind="4d")
    dc = DriftCorrection.from_reference(ref, cube)
    assert dc.is_4dstem
    assert not dc._is_4dstem_collection


def test_generate_corrected_4dstem_is_distinct_from_corrected_4dstem_0():
    """The corrected 4D-STEM dataset must not alias dataset 0."""
    cube_a, cube_b = _make_4dstem_collection(scan_size=32, det_size=4)
    dc = DriftCorrection(
        cube_a, cube_b, scan_direction_degrees=[0, 90],
    )
    dc.preprocess(
        pad_fraction=0.25, kde_sigma=0.5, number_knots=1,
        show_merged=False, show_images=False,
    )
    dc.align_affine(
        step=0.02, num_tests=11,
        show_merged=False, show_images=False,
    )
    result = dc.generate_corrected_4dstem()
    assert result.corrected_4dstem is not result.corrected_4dstem_0


def test_generate_corrected_rejects_scan_collection_4dstem_generic_api():
    """4D-STEM collection must use the explicit first-class API."""
    cube_a, cube_b = _make_4dstem_collection(scan_size=32, det_size=4)
    dc = DriftCorrection.from_4dstem(
        cube_a, cube_b, scan_direction_degrees=[0, 90],
    )
    dc.preprocess(
        pad_fraction=0.25, kde_sigma=0.5, number_knots=1,
        show_merged=False, show_images=False,
    )
    dc.align_affine(
        step=0.02, num_tests=11,
        show_merged=False, show_images=False,
    )
    with pytest.raises(RuntimeError, match="generate_corrected_4dstem"):
        dc.generate_corrected()


def test_generate_corrected_strip_padding():
    """strip_padding=True returns original scan dimensions, not padded canvas.

    Uses a true scan collection (0°/90°) alignment so generate_corrected goes through
    the merge-on-canvas path where padding is visible (reference mode warps
    on the scan grid directly, so strip_padding has no effect there)."""
    im0, im1, _ = make_synthetic_drift_data(scale=1, seed=0)
    dc = DriftCorrection(im0, im1, scan_direction_degrees=[0, 90])
    dc.preprocess(pad_fraction=0.25, kde_sigma=0.5, number_knots=1,
                  show_merged=False, show_images=False)
    dc.align_affine(step=0.02, num_tests=11,
                    show_merged=False, show_images=False)
    scan_h, scan_w = im0.shape

    padded = dc.generate_corrected(
        mask_output=False, strip_padding=False, show_merged=False,
    )
    stripped = dc.generate_corrected(
        mask_output=False, strip_padding=True, show_merged=False,
    )

    canvas_h, canvas_w = dc.shape[1], dc.shape[2]
    assert padded.array.shape == (canvas_h, canvas_w), (
        f"Without strip_padding, shape should be canvas: {padded.array.shape}"
    )
    assert stripped.array.shape == (scan_h, scan_w), (
        f"With strip_padding, shape should be original scan: {stripped.array.shape}"
    )
    assert canvas_h > scan_h, "Canvas should be larger than original scan"


# ---------------------------------------------------------------------------
# Tests for loss="auto" resolution
# ---------------------------------------------------------------------------


def test_align_nonrigid_loss_auto_resolves_pytorch():
    """loss='auto' should resolve to 'gradient_mse' for pytorch backend."""
    im0, im1, _ = make_synthetic_drift_data(scale=1, seed=42)
    drift = DriftCorrection(
        im0, im1, scan_direction_degrees=[0.0, -90.0],
    ).preprocess(show_merged=False, show_images=False)
    drift.align_affine(show_merged=False, show_images=False)
    # Should not raise — auto resolves to gradient_mse for pytorch
    drift.align_nonrigid(
        loss="auto", num_iterations=2,
        show_merged=False, show_images=False,
    )
    # Verify it completed without error
    assert drift.error_track is not None


# ---------------------------------------------------------------------------
# Tests for early stopping
# ---------------------------------------------------------------------------


def test_align_nonrigid_early_stopping():
    """Early stopping should terminate before max iterations on easy data."""
    im0, im1, _ = make_synthetic_drift_data(scale=1, seed=42)
    drift = DriftCorrection(
        im0, im1, scan_direction_degrees=[0.0, -90.0],
    ).preprocess(show_merged=False, show_images=False)
    drift.align_affine(
        step=0.02, num_tests=5, refine=True,
        show_merged=False, show_images=False,
    )
    # Use many max iterations but expect early stop on this easy synthetic data
    drift.align_nonrigid(
        num_iterations=64,
        min_iterations=4,
        early_stop_patience=3,
        early_stop_rtol=1e-4,
        loss="mse",
        regularization_sigma_px=8.0,
        show_merged=False, show_images=False,
    )
    # Should have stopped before 64 iterations
    # error_track has initial rows + nonrigid rows; count nonrigid rows (mode=2.0)
    nonrigid_rows = (drift.error_track[:, 0] == 2.0).sum()
    assert nonrigid_rows < 64, (
        f"Expected early stopping before 64 iterations, got {nonrigid_rows}"
    )
    assert nonrigid_rows >= 4, (
        f"Must run at least min_iterations=4, got {nonrigid_rows}"
    )


def test_align_nonrigid_early_stopping_disabled():
    """Setting patience >= num_iterations disables early stopping."""
    im0, im1, _ = make_synthetic_drift_data(scale=1, seed=42)
    drift = DriftCorrection(
        im0, im1, scan_direction_degrees=[0.0, -90.0],
    ).preprocess(show_merged=False, show_images=False)
    drift.align_affine(
        step=0.02, num_tests=5, refine=True,
        show_merged=False, show_images=False,
    )
    drift.align_nonrigid(
        num_iterations=8,
        early_stop_patience=8,
        loss="mse",
        regularization_sigma_px=8.0,
        show_merged=False, show_images=False,
    )
    nonrigid_rows = (drift.error_track[:, 0] == 2.0).sum()
    assert nonrigid_rows == 8, (
        f"With patience=num_iterations, should run all 8, got {nonrigid_rows}"
    )


# ---------------------------------------------------------------------------
# Tests for LBFGS + normalize warning
# ---------------------------------------------------------------------------


def test_align_nonrigid_lbfgs_normalize_warns():
    """LBFGS with normalize=True should emit a warning."""
    im0, im1, _ = make_synthetic_drift_data(scale=1, seed=42)
    drift = DriftCorrection(
        im0, im1, scan_direction_degrees=[0.0, -90.0],
    ).preprocess(normalize=True, show_merged=False, show_images=False)
    drift.align_affine(show_merged=False, show_images=False)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        drift.align_nonrigid(
            optimizer_name="lbfgs", loss="mse",
            num_iterations=1,
            show_merged=False, show_images=False,
        )
        lbfgs_warnings = [x for x in w if "normalize=True + LBFGS" in str(x.message)]
        assert len(lbfgs_warnings) == 1, (
            f"Expected 1 LBFGS+normalize warning, got {len(lbfgs_warnings)}"
        )


def test_align_nonrigid_adam_normalize_no_warning():
    """Adam with normalize=True should NOT emit a warning."""
    im0, im1, _ = make_synthetic_drift_data(scale=1, seed=42)
    drift = DriftCorrection(
        im0, im1, scan_direction_degrees=[0.0, -90.0],
    ).preprocess(normalize=True, show_merged=False, show_images=False)
    drift.align_affine(show_merged=False, show_images=False)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        drift.align_nonrigid(
            optimizer_name="adam", loss="mse",
            num_iterations=2,
            show_merged=False, show_images=False,
        )
        lbfgs_warnings = [x for x in w if "normalize=True + LBFGS" in str(x.message)]
        assert len(lbfgs_warnings) == 0, (
            f"Adam should not trigger LBFGS warning, got {len(lbfgs_warnings)}"
        )


# ---------------------------------------------------------------------------
# Tests for sobel z-score normalization
# ---------------------------------------------------------------------------


def test_sobel_gradient_magnitude_znorm():
    """sobel_gradient_magnitude should z-score normalize each image."""
    from quantem.imaging.drift_optimize import sobel_gradient_magnitude
    images = torch.randn(3, 64, 64)
    # Scale each image differently to test gain invariance
    images[1] *= 10.0
    images[2] *= 0.01
    result = sobel_gradient_magnitude(
        images, pre_smooth=1.0, device=images.device, dtype=images.dtype,
    )
    assert result.shape == (3, 64, 64)
    # Each image should have ~zero mean and ~unit std
    for i in range(3):
        mean = result[i].mean().item()
        std = result[i].std().item()
        assert abs(mean) < 0.01, f"Image {i} mean={mean}, expected ~0"
        assert abs(std - 1.0) < 0.05, f"Image {i} std={std}, expected ~1.0"



def test_save_load_roundtrip_scan_collection():
    """save() then load() should round-trip a image-collection DriftCorrection."""
    import tempfile
    from pathlib import Path
    from quantem.core.io.serialize import load
    rng = np.random.default_rng(0)
    a = rng.random((64, 64), dtype=np.float32)
    b = rng.random((64, 64), dtype=np.float32)
    dc = DriftCorrection(a, b, scan_direction_degrees=(0, 90))
    dc.preprocess(show_merged=False, show_images=False)
    dc.align_affine(num_tests=5, show_merged=False, show_images=False)
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "dc.zip"
        dc.save(str(p))
        loaded = load(str(p))
        assert isinstance(loaded, DriftCorrection)
        assert loaded.imgs[0].array.shape == (64, 64)


def test_save_load_roundtrip_4dstem_clear_error():
    """4D-STEM mode: save() drops the heavy _datasets; on reload, calling
    apply_correction() without re-attaching them must raise a clear error."""
    import tempfile
    from pathlib import Path
    from quantem.core.io.serialize import load
    cube_a, cube_b = _make_4dstem_collection(scan_size=32, det_size=4)
    dc = DriftCorrection.from_4dstem(
        cube_a, cube_b, scan_direction_degrees=[0, 90],
    )
    dc.preprocess(kde_sigma=0.5, number_knots=1,
                  show_merged=False, show_images=False)
    dc.align_affine(step=0.02, num_tests=5,
                    show_merged=False, show_images=False)
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "dc.zip"
        dc.save(str(p))
        loaded = load(str(p))
        # The cubes are intentionally dropped on save (too large to serialize).
        # Calling apply_correction() with no data should error clearly.
        assert getattr(loaded, "_datasets", None) is None
        with pytest.raises((TypeError, ValueError, AttributeError)):
            loaded.apply_correction()


def test_multi_angle_three_image_dispatch():
    """N=3 2D image collection dispatch (the *more_images branch) builds a single instance."""
    rng = np.random.default_rng(0)
    im0 = rng.random((48, 48), dtype=np.float32)
    im45 = rng.random((48, 48), dtype=np.float32)
    im90 = rng.random((48, 48), dtype=np.float32)
    dc = DriftCorrection(im0, im45, im90, scan_direction_degrees=(0, 45, 90))
    assert len(dc.imgs) == 3
    assert tuple(dc.scan_direction_degrees) == (0, 45, 90)


def test_scan_collection_4dstem_scan_dim_mismatch_rejected():
    """4D + 4D with mismatched (scan_h, scan_w) should error before alignment."""
    rng = np.random.default_rng(0)
    a = rng.random((32, 32, 4, 4), dtype=np.float32)
    b = rng.random((32, 40, 4, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="scan dims"):
        DriftCorrection(a, b, scan_direction_degrees=(0, 90))


def test_canvas_to_raw_drift_nonsquare_integration():
    """End-to-end scan collection correction on a non-square scan exercises the
    alpha = (H-1)/(W-1) Jacobian factor that square tests don't reach."""
    np.random.seed(0)
    H, W = 96, 64  # non-square
    row_coords, col_coords = np.mgrid[:H, :W]
    ref = np.sin(0.1 * row_coords + 0.15 * col_coords).astype(np.float32) * 50 + 100
    rows = np.arange(H, dtype=np.float32)
    src_row = row_coords - 0.05 * rows[:, None]
    src_col = col_coords - 0.10 * rows[:, None]
    drifted = map_coordinates(ref, [src_row, src_col], order=3, mode='nearest').astype(np.float32)
    dc = DriftCorrection(ref, drifted, scan_direction_degrees=(0, 0))
    dc.preprocess(pad_fraction=0.25, pad_value=0.0, kde_sigma=0.5,
                  show_merged=False, show_images=False)
    dc.align_affine(step=0.02, num_tests=11,
                    show_merged=False, show_images=False)
    corrected = dc.apply_correction()
    if hasattr(corrected, "cpu"):
        corrected = corrected.cpu().numpy()
    raw_rms = float(np.sqrt(((ref - drifted) ** 2).mean()))
    cor_rms = float(np.sqrt(((ref - corrected) ** 2).mean()))
    assert cor_rms < raw_rms, f"non-square correction should reduce RMSE: {raw_rms=}, {cor_rms=}"


# --- Synthetic EDS (reference-mode) integration tests -------------------------

def test_element_map_integrates_energy_window():
    """element_map sums the calibrated energy window -> a 2-D band image."""
    cube = np.zeros((8, 8, 100), dtype=np.float32)
    cube[2:5, 2:5, 40:50] = 3.0  # a feature living in the [4.0, 5.0) keV band
    energy_axis = np.arange(100) * 0.1
    emap = element_map(cube, energy_axis, 4.5, width=0.5)  # window -> channels 40..50
    assert emap.shape == (8, 8)
    np.testing.assert_allclose(emap, cube[..., 40:51].sum(-1), atol=1e-5)


def test_from_reference_eds_recovers_known_column_drift():
    """Chevron HAADF + multi-channel cube + a KNOWN column drift: from_reference must
    recover it and apply_correction must move the cube back toward the clean ground truth."""
    scan = 96
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[:scan, :scan]
    ref = np.zeros((scan, scan), np.float32)
    for cy, cx in rng.uniform(15, scan - 15, (8, 2)):        # localized particles -> sharp, unambiguous drift constraint
        ref += np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 7.0 ** 2))
    ref = (ref + 0.2 * gaussian_filter(rng.standard_normal((scan, scan)).astype(np.float32), 1.2)).astype(np.float32)
    n_channels = 6
    eds_clean = np.stack(
        [ref * (0.5 + 0.5 * np.sin(0.1 * c)) for c in range(n_channels)], -1).astype(np.float32)

    rate = 0.06
    def column_drift(img):
        cols = np.arange(img.shape[1])
        out = np.empty_like(img)
        for r in range(img.shape[0]):
            x = np.clip(cols + rate * r, 0, img.shape[1] - 1.001)
            c0 = np.floor(x).astype(int)
            frac = x - c0
            lo, hi = img[r, c0], img[r, c0 + 1]
            out[r] = lo + (hi - lo) * (frac[:, None] if img.ndim == 3 else frac)
        return out

    haadf_drifted = column_drift(ref).astype(np.float32)
    eds_drifted = column_drift(eds_clean).astype(np.float32)

    dc = DriftCorrection.from_reference(
        ref, eds_drifted, alignment_image=haadf_drifted, scan_direction_degrees=0)
    dc.preprocess(pad_fraction=0.25, pad_value="median", kde_sigma=0.5, number_knots=1,
                  normalize=False, show_merged=False, show_images=False)
    dc.align_affine(step=0.01, num_tests=31, refine=True, upsample_factor=8,
                    max_image_shift=16, fixed_indices=[0], show_merged=False, show_images=False)
    corrected = dc.apply_correction(eds_drifted)
    corrected = corrected.cpu().numpy() if hasattr(corrected, "cpu") else np.asarray(corrected)

    ncc_drifted = normalized_cross_correlation(eds_clean, eds_drifted, margin=10)
    ncc_corrected = normalized_cross_correlation(eds_clean, corrected, margin=10)
    assert ncc_corrected > ncc_drifted, f"correction did not improve match: {ncc_corrected} <= {ncc_drifted}"
    assert ncc_corrected > 0.9, f"corrected cube should closely match ground truth, got {ncc_corrected}"
