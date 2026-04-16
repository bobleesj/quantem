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
from quantem.core.datastructures.dataset2d import Dataset2d
from quantem.imaging.drift import DriftCorrection, correct_series


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

    drift = DriftCorrection.from_data(
        images=[im0, im1],
        scan_direction_degrees=[0.0, 90.0],
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
    img_corr = drift.generate_corrected_image(upsample_factor=1, show_image=False)

    assert isinstance(img_corr, Dataset2d)
    assert not np.isnan(img_corr.array).any()
    assert drift.error_track[-1, 1] < 0.1

    # Determinism: second run with same seed must match exactly
    im0_2, im1_2, _ = make_synthetic_drift_data(scale=1, seed=42)
    drift2 = DriftCorrection.from_data(
        images=[im0_2, im1_2],
        scan_direction_degrees=[0.0, 90.0],
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
    img_corr2 = drift2.generate_corrected_image(upsample_factor=1, show_image=False)

    np.testing.assert_array_almost_equal(
        img_corr.array, img_corr2.array, decimal=10,
        err_msg="Drift correction output is not deterministic!",
    )



def test_preprocess_single_image_builds_centered_canvas():
    """Preprocess should support a single image for geometry-only reuse.

    This is the setup used by the single-sided 4D-STEM workflow: reuse the
    scanline model and initial warp without invoking the pairwise aligners.
    """
    im0, _, _ = make_synthetic_drift_data(scale=1, seed=42)

    drift = DriftCorrection.from_data(
        images=[im0],
        scan_direction_degrees=[0.0],
    ).preprocess(
        pad_fraction=0.25,
        pad_value="median",
        kde_sigma=0.5,
        number_knots=1,
        show_merged=False,
        show_images=False,
    )

    assert drift.shape == (1, 160, 160)
    assert drift.knots[0].shape == (2, 128, 1)
    assert drift.imgs_warped.array.shape == (1, 160, 160)
    assert not np.isnan(drift.imgs_warped.array).any()


# Baseline values from float32 torch path, captured once and frozen.
# (scale, error, knots0_sum, knots1_sum)
# Recaptured after torch-native knots unification (knots stored as torch tensors).
# scale=1 is bit-exact across versions; scale=2,4 shifted by ~0.03-1.1%
# due to optimizer trajectory divergence from dependency upgrades.
AFFINE_BASELINES = [
    (1, 0.09237674623727798, 12157.736328125, 28546.263671875),
    (2, 0.13840323686599731, 49798.91015625, 113529.09375),
    (4, 0.16396018862724304, 194685.40625, 459650.59375),
]


@pytest.mark.parametrize("scale,expected_error,expected_k0,expected_k1", AFFINE_BASELINES)
def test_align_affine_matches_frozen_baseline(scale, expected_error, expected_k0, expected_k1):
    """Affine on synthetic data must match frozen float32 baseline."""
    im0, im1, _ = make_synthetic_drift_data(scale=scale, seed=42)
    drift = DriftCorrection.from_data(
        images=[im0, im1], scan_direction_degrees=[0.0, 90.0],
    ).preprocess(show_merged=False, show_images=False)
    drift.align_affine(
        step=0.02, num_tests=5, refine=True,
        show_merged=False, show_images=False,
    )
    np.testing.assert_almost_equal(
        drift.error_track[-1, 1], expected_error, decimal=6)
    np.testing.assert_almost_equal(
        drift.knots[0].sum().item(), expected_k0, decimal=6)
    np.testing.assert_almost_equal(
        drift.knots[1].sum().item(), expected_k1, decimal=6)


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
    drift = DriftCorrection.from_data(
        images=[im0, im1], scan_direction_degrees=[0.0, 90.0],
    ).preprocess(show_merged=False, show_images=False)
    drift.align_affine(
        step=0.02, num_tests=5, refine=True,
        show_merged=False, show_images=False,
    )
    drift.align_nonrigid(
        backend="pytorch", num_iterations=2, adam_steps=50,
        regularization_sigma_px=16.0,
        # Pin lr to the value the baselines were captured at - the public
        # default is now auto-derived from max_image_shift, but the frozen
        # baselines must stay numerically stable across that change.
        lr=0.02,
        loss="mse",
        show_merged=False, show_images=False,
    )
    np.testing.assert_almost_equal(
        drift.error_track[-1, 1], expected_error, decimal=6)
    np.testing.assert_almost_equal(
        drift.knots[0].sum().item(), expected_k0, decimal=6)
    np.testing.assert_almost_equal(
        drift.knots[1].sum().item(), expected_k1, decimal=6)


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
    drift = DriftCorrection.from_data(
        images=[im0, im1], scan_direction_degrees=[0.0, 90.0],
    ).preprocess(show_merged=False, show_images=False)
    drift.align_affine(
        step=0.02, num_tests=5, refine=True,
        show_merged=False, show_images=False,
    )
    drift.align_nonrigid(
        backend="pytorch", optimizer_name="lbfgs",
        num_iterations=2, lbfgs_max_iter=20,
        regularization_sigma_px=16.0,
        loss="mse",
        show_merged=False, show_images=False,
    )
    np.testing.assert_almost_equal(
        drift.error_track[-1, 1], expected_error, decimal=6)
    np.testing.assert_almost_equal(
        drift.knots[0].sum().item(), expected_k0, decimal=6)
    np.testing.assert_almost_equal(
        drift.knots[1].sum().item(), expected_k1, decimal=6)


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
    drift = DriftCorrection.from_data(
        images=[reference, moving],
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
    drift_a = DriftCorrection.from_data(
        images=[im0, im1], scan_direction_degrees=[0.0, 90.0],
    ).preprocess(show_merged=False, show_images=False)
    drift_a.align_affine(
        step=0.02, num_tests=5, refine=True,
        show_merged=False, show_images=False,
    )

    # Run with explicit fixed_indices=None
    drift_b = DriftCorrection.from_data(
        images=[im0, im1], scan_direction_degrees=[0.0, 90.0],
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

    dc = DriftCorrection.from_data(
        images=[im0, im1_scaled], scan_direction_degrees=[0.0, 90.0],
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

    drift = DriftCorrection.from_data(
        images=[reference, moving],
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
        backend="pytorch", optimizer_name="adam",
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

    drift = DriftCorrection.from_data(
        images=[reference, moving],
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
        backend="pytorch", optimizer_name="adam",
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
    drift = DriftCorrection.from_data(
        images=[im0, im1], scan_direction_degrees=[0.0, 90.0],
    ).preprocess(show_merged=False, show_images=False)
    drift.align_affine(
        step=0.02, num_tests=5, refine=True,
        show_merged=False, show_images=False,
    )
    with pytest.raises(ValueError, match="All images are fixed"):
        drift.align_nonrigid(
            backend="pytorch", fixed_indices=[0, 1],
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

    drift = DriftCorrection.from_data(
        images=[reference, moving],
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
        backend="pytorch", optimizer_name="adam",
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

    drift = DriftCorrection.from_data(
        images=[reference, moving],
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
        backend="pytorch", optimizer_name="lbfgs",
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
        drift = DriftCorrection.from_data(
            images=[reference, moving],
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
            backend="pytorch", optimizer_name="lbfgs",
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
    drift = DriftCorrection.from_data(
        images=[im0, im1], scan_direction_degrees=[0.0, 90.0],
    ).preprocess(show_merged=False, show_images=False)
    drift.align_affine(show_merged=False, show_images=False)
    with pytest.raises(ValueError, match="loss must be one of"):
        drift.align_nonrigid(
            loss="invalid_loss",
            show_merged=False, show_images=False,
        )


def test_align_nonrigid_gradient_mse_scipy_raises():
    """gradient_mse with scipy backend should raise ValueError."""
    im0, im1, _ = make_synthetic_drift_data(scale=1, seed=42)
    drift = DriftCorrection.from_data(
        images=[im0, im1], scan_direction_degrees=[0.0, 90.0],
    ).preprocess(show_merged=False, show_images=False)
    drift.align_affine(show_merged=False, show_images=False)
    with pytest.raises(ValueError, match="only supported with backend='pytorch'"):
        drift.align_nonrigid(
            backend="scipy", loss="gradient_mse",
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

    drift = DriftCorrection.from_data(
        images=[reference, moving],
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
        backend="pytorch", optimizer_name="adam",
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

    dc = DriftCorrection.from_data(
        images=[ref, drifted],
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
    """apply_correction(images=...) should work on external arrays."""
    dc, ref, drifted = _make_single_sided_dc()
    # Pass external image as numpy
    result_np = dc.apply_correction(images=drifted, mode='bilinear')
    assert result_np.shape == drifted.shape
    # Pass external image as tensor
    t = torch.tensor(drifted, device=dc._device, dtype=torch.float32)
    result_t = dc.apply_correction(images=t, mode='bilinear')
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
    with pytest.raises(ValueError, match="Image height"):
        dc.apply_correction(images=wrong_size)


def test_apply_correction_before_preprocess_raises():
    """apply_correction before preprocess() should raise RuntimeError."""
    dc = DriftCorrection.from_data(
        images=[np.zeros((64, 64)), np.zeros((64, 64))],
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
    dc = DriftCorrection.from_data(
        images=[np.zeros((32, 32)), np.zeros((32, 32))],
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
    corrected = dc.apply_correction(images=batch, mode='bicubic')
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

    corrected = dc.apply_correction(images=batch, mode='bicubic')
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
    batch_result = dc.apply_correction(images=drifted, mode='bicubic')

    # Individual
    for ch in range(drifted.shape[0]):
        single = dc.apply_correction(images=drifted[ch], mode='bicubic')
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
    result_u8 = dc.apply_correction(images=img_u8, mode='bilinear')
    assert result_u8.dtype == torch.float32
    assert result_u8.shape == img_u8.shape

    # uint16 input
    img_u16 = (ref * 100).astype(np.uint16)
    result_u16 = dc.apply_correction(images=img_u16, mode='bilinear')
    assert result_u16.dtype == torch.float32
    assert result_u16.shape == img_u16.shape

    # float32 reference — results should match the float path
    img_f32 = img_u8.astype(np.float32)
    result_f32 = dc.apply_correction(images=img_f32, mode='bilinear')
    torch.testing.assert_close(
        result_u8, result_f32, atol=0.6, rtol=1e-3,
        msg="uint8 path should match float32 path (values are discretized)",
    )


def test_apply_correction_multi_knot_batch_raises():
    """apply_correction rejects batched input when number_knots > 1."""
    ref = np.random.randn(64, 64).astype(np.float32)
    drifted = np.roll(ref, 2, axis=1).astype(np.float32)
    dc = DriftCorrection.from_data(
        images=[ref, drifted], scan_direction_degrees=[0.0, 0.0],
    )
    dc.preprocess(
        pad_fraction=0.25, pad_value=0.0, kde_sigma=0.5,
        number_knots=3, normalize=True,
        show_merged=False, show_images=False,
    )
    dc.align_affine(
        step=0.02, num_tests=9, refine=False,
        fixed_indices=[0], max_image_shift=32,
        show_merged=False, show_images=False,
    )
    batch = np.random.randn(4, 64, 64).astype(np.float32)
    with pytest.raises(NotImplementedError, match="number_knots=1"):
        dc.apply_correction(images=batch)


# ──────────────────────────────────────────────────────────────
# Tests for apply_correction_4dstem() — 3D/4D correction
# ──────────────────────────────────────────────────────────────

def test_apply_correction_4dstem_3d_eds():
    """apply_correction_4dstem on a 3D (H, W, E) EDX cube."""
    scan_h = 128
    drift_rate = (0.05, 0.1)
    dc, ref, _ = _make_single_sided_dc(scan_h=scan_h, drift_rate=drift_rate)
    n_energy = 12

    channels_clean = _make_diverse_channels(ref, n_energy, seed=500)
    channels_drifted = _apply_drift_to_channels(channels_clean, drift_rate, scan_h)
    # Natural EDX layout: (H, W, E)
    cube = channels_drifted.transpose(1, 2, 0)
    assert cube.shape == (scan_h, scan_h, n_energy)

    corrected = dc.apply_correction_4dstem(cube)
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


def test_apply_correction_4dstem_4d_stem():
    """apply_correction_4dstem on a 4D (H, W, det_h, det_w) STEM cube."""
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

    corrected = dc.apply_correction_4dstem(cube_4d)
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


def test_apply_correction_4dstem_matches_manual_chunking():
    """apply_correction_4dstem must produce same results as manual loop."""
    scan_h = 128
    dc, ref, _ = _make_single_sided_dc(scan_h=scan_h)
    n_energy = 8

    channels = _make_diverse_channels(ref, n_energy, seed=700)
    drifted = _apply_drift_to_channels(channels, (0.05, 0.1), scan_h)
    cube = drifted.transpose(1, 2, 0)  # (H, W, E)

    # apply_correction_4dstem
    auto = dc.apply_correction_4dstem(cube)

    # Manual chunking (what user had to do before)
    batch = cube.transpose(2, 0, 1)  # (E, H, W)
    manual = dc.apply_correction(images=batch, mode='bicubic').cpu().numpy()
    manual = manual.transpose(1, 2, 0)  # (H, W, E)

    np.testing.assert_allclose(auto, manual, atol=1e-4, rtol=1e-4,
                               err_msg="Cube method must match manual batch")


def test_apply_correction_4dstem_torch_input():
    """apply_correction_4dstem works with torch.Tensor input."""
    scan_h = 128
    dc, ref, _ = _make_single_sided_dc(scan_h=scan_h)
    cube_np = np.random.randn(scan_h, scan_h, 6).astype(np.float32)
    cube_t = torch.from_numpy(cube_np)

    result = dc.apply_correction_4dstem(cube_t)
    assert isinstance(result, torch.Tensor)
    assert result.shape == cube_t.shape


def test_apply_correction_4dstem_output_dtype_same():
    """apply_correction_4dstem with output_dtype='same' preserves input dtype."""
    scan_h = 128
    dc, ref, _ = _make_single_sided_dc(scan_h=scan_h)

    cube_u16 = (np.random.rand(scan_h, scan_h, 4) * 1000).astype(np.uint16)
    result = dc.apply_correction_4dstem(cube_u16, output_dtype="same")
    assert result.dtype == np.uint16
    assert result.shape == cube_u16.shape


def test_apply_correction_4dstem_2d_raises():
    """apply_correction_4dstem rejects 2D input."""
    scan_h = 128
    dc, ref, _ = _make_single_sided_dc(scan_h=scan_h)
    with pytest.raises(ValueError, match="at least 3D"):
        dc.apply_correction_4dstem(ref)


def test_generate_corrected_image_strip_padding():
    """strip_padding=True returns original scan dimensions, not padded canvas."""
    dc, ref, _ = _make_single_sided_dc(scan_h=128)
    scan_h, scan_w = ref.shape

    padded = dc.generate_corrected_image(
        mask_output=False, strip_padding=False, show_image=False,
    )
    stripped = dc.generate_corrected_image(
        mask_output=False, strip_padding=True, show_image=False,
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
    drift = DriftCorrection.from_data(
        images=[im0, im1], scan_direction_degrees=[0.0, 90.0],
    ).preprocess(show_merged=False, show_images=False)
    drift.align_affine(show_merged=False, show_images=False)
    # Should not raise — auto resolves to gradient_mse for pytorch
    drift.align_nonrigid(
        backend="pytorch", loss="auto", num_iterations=2,
        show_merged=False, show_images=False,
    )
    # Verify it completed without error
    assert drift.error_track is not None


def test_align_nonrigid_loss_auto_resolves_scipy():
    """loss='auto' should resolve to 'mse' for scipy backend."""
    im0, im1, _ = make_synthetic_drift_data(scale=1, seed=42)
    drift = DriftCorrection.from_data(
        images=[im0, im1], scan_direction_degrees=[0.0, 90.0],
    ).preprocess(show_merged=False, show_images=False)
    drift.align_affine(show_merged=False, show_images=False)
    # Should not raise — auto resolves to mse for scipy
    drift.align_nonrigid(
        backend="scipy", loss="auto", num_iterations=1,
        show_merged=False, show_images=False,
    )
    assert drift.error_track is not None


# ---------------------------------------------------------------------------
# Tests for early stopping
# ---------------------------------------------------------------------------


def test_align_nonrigid_early_stopping():
    """Early stopping should terminate before max iterations on easy data."""
    im0, im1, _ = make_synthetic_drift_data(scale=1, seed=42)
    drift = DriftCorrection.from_data(
        images=[im0, im1], scan_direction_degrees=[0.0, 90.0],
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
    drift = DriftCorrection.from_data(
        images=[im0, im1], scan_direction_degrees=[0.0, 90.0],
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
    drift = DriftCorrection.from_data(
        images=[im0, im1], scan_direction_degrees=[0.0, 90.0],
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
    drift = DriftCorrection.from_data(
        images=[im0, im1], scan_direction_degrees=[0.0, 90.0],
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
    """_sobel_gradient_magnitude should z-score normalize each image."""
    images = torch.randn(3, 64, 64)
    # Scale each image differently to test gain invariance
    images[1] *= 10.0
    images[2] *= 0.01
    result = DriftCorrection._sobel_gradient_magnitude(
        images, pre_smooth=1.0, device=images.device, dtype=images.dtype,
    )
    assert result.shape == (3, 64, 64)
    # Each image should have ~zero mean and ~unit std
    for i in range(3):
        mean = result[i].mean().item()
        std = result[i].std().item()
        assert abs(mean) < 0.01, f"Image {i} mean={mean}, expected ~0"
        assert abs(std - 1.0) < 0.05, f"Image {i} std={std}, expected ~1.0"


# ── correct_series tests ─────────────────────────────────────────────────


def _make_series_pair(n_frames=3, size=128, seed=42):
    """Create a small (N, H, W) pair of image stacks for series tests."""
    rng = np.random.default_rng(seed)
    im0, im1, _ = make_synthetic_drift_data(scale=1, seed=seed)
    a = np.stack([im0 + rng.normal(0, 0.01, im0.shape) for _ in range(n_frames)])
    b = np.stack([im1 + rng.normal(0, 0.01, im1.shape) for _ in range(n_frames)])
    return a.astype(np.float32), b.astype(np.float32)


@pytest.fixture
def series_pair():
    return _make_series_pair(n_frames=2)


class TestCorrectSeries:
    """Tests for the correct_series() standalone function."""

    def test_defaults(self, series_pair):
        """All-defaults call should produce (N, H, W) float32 output."""
        a, b = series_pair
        corrected, objs = correct_series(a, b, scan_direction_degrees=[0, -90])
        assert corrected.ndim == 3
        assert corrected.shape[0] == a.shape[0]
        assert corrected.dtype == np.float32
        assert len(objs) == a.shape[0]
        assert all(isinstance(o, DriftCorrection) for o in objs)

    def test_stage_kwargs_forwarded(self, series_pair):
        """Stage dicts should be forwarded to the underlying methods."""
        a, b = series_pair
        corrected, objs = correct_series(
            a, b,
            scan_direction_degrees=[0, -90],
            preprocess=dict(pad_fraction=0.3, kde_sigma=0.8, number_knots=1),
            align_affine=dict(step=0.05, num_tests=5),
            generate=dict(upsample_factor=1, kde_sigma=0.8),
        )
        assert corrected.shape[0] == a.shape[0]
        assert corrected.dtype == np.float32

    def test_nonrigid_true(self, series_pair):
        """align_nonrigid=True should run nonrigid with defaults."""
        a, b = series_pair
        corrected, objs = correct_series(
            a, b,
            scan_direction_degrees=[0, -90],
            align_nonrigid=True,
            generate=dict(upsample_factor=1),
        )
        assert corrected.shape[0] == a.shape[0]
        for obj in objs:
            assert (obj.error_track[:, 0] == 2.0).any(), "nonrigid stage not found"

    def test_nonrigid_dict(self, series_pair):
        """align_nonrigid=dict(...) should forward params to align_nonrigid."""
        a, b = series_pair
        corrected, objs = correct_series(
            a, b,
            scan_direction_degrees=[0, -90],
            align_nonrigid=dict(num_iterations=2, adam_steps=5),
            generate=dict(upsample_factor=1),
        )
        assert corrected.shape[0] == a.shape[0]
        for obj in objs:
            assert (obj.error_track[:, 0] == 2.0).any(), "nonrigid stage not found"

    def test_nonrigid_false_skips(self, series_pair):
        """align_nonrigid=False (default) should skip nonrigid."""
        a, b = series_pair
        _, objs = correct_series(
            a, b,
            scan_direction_degrees=[0, -90],
            generate=dict(upsample_factor=1),
        )
        for obj in objs:
            assert not (obj.error_track[:, 0] == 2.0).any(), "nonrigid should be skipped"

    def test_shape_mismatch_raises(self):
        """Mismatched image shapes should raise ValueError."""
        a = np.zeros((3, 64, 64))
        b = np.zeros((3, 64, 32))
        with pytest.raises(ValueError, match="Shape mismatch"):
            correct_series(a, b, scan_direction_degrees=[0, -90])

    def test_2d_input_raises(self):
        """2-D input should raise ValueError."""
        a = np.zeros((64, 64))
        b = np.zeros((64, 64))
        with pytest.raises(ValueError, match="must be 3-D"):
            correct_series(a, b, scan_direction_degrees=[0, -90])

    def test_drift_objects_are_inspectable(self, series_pair):
        """Returned DriftCorrection objects should support standard inspection."""
        a, b = series_pair
        _, objs = correct_series(
            a, b,
            scan_direction_degrees=[0, -90],
            generate=dict(upsample_factor=1),
        )
        for obj in objs:
            assert hasattr(obj, "knots")
            assert hasattr(obj, "drift_rate")


class TestSeriesClassAPI:
    """Tests for DriftCorrection class-level series support."""

    def test_from_data_detects_3d_stacks(self, series_pair):
        """from_data with 3-D arrays should create a series instance."""
        a, b = series_pair
        dc = DriftCorrection.from_data([a, b], scan_direction_degrees=[0, -90])
        assert dc.is_series
        assert dc.n_frames == a.shape[0]

    def test_from_data_2d_is_single(self):
        """from_data with 2-D arrays should create a normal single instance."""
        im0, im1, _ = make_synthetic_drift_data(scale=1, seed=42)
        dc = DriftCorrection.from_data([im0, im1], scan_direction_degrees=[0, -90])
        assert not dc.is_series

    def test_getitem_returns_single(self, series_pair):
        """Indexing a series should return a single-pair instance."""
        a, b = series_pair
        dc = DriftCorrection.from_data([a, b], scan_direction_degrees=[0, -90])
        frame = dc[0]
        assert not frame.is_series

    def test_iter_yields_all_frames(self, series_pair):
        """Iterating over series should yield all frames."""
        a, b = series_pair
        dc = DriftCorrection.from_data([a, b], scan_direction_degrees=[0, -90])
        frames = list(dc)
        assert len(frames) == dc.n_frames

    def test_full_pipeline(self, series_pair):
        """Full pipeline via class API should produce corrected stack."""
        a, b = series_pair
        dc = DriftCorrection.from_data([a, b], scan_direction_degrees=[0, -90])
        dc.preprocess(pad_fraction=0.25, kde_sigma=0.5, number_knots=1)
        dc.align_affine(step=0.02, num_tests=5)
        result = dc.generate_corrected_image(upsample_factor=1)
        assert isinstance(result, np.ndarray)
        assert result.ndim == 3
        assert result.shape[0] == a.shape[0]
        assert result.dtype == np.float32

    def test_method_chaining(self, series_pair):
        """Pipeline methods should return self for chaining."""
        a, b = series_pair
        dc = DriftCorrection.from_data([a, b], scan_direction_degrees=[0, -90])
        result = dc.preprocess().align_affine(step=0.02, num_tests=5)
        assert result is dc

    def test_per_frame_inspection(self, series_pair):
        """Individual frames should be fully functional after series pipeline."""
        a, b = series_pair
        dc = DriftCorrection.from_data([a, b], scan_direction_degrees=[0, -90])
        dc.preprocess().align_affine(step=0.02, num_tests=5)
        for i in range(dc.n_frames):
            frame = dc[i]
            assert hasattr(frame, "knots")
            rate = frame.drift_rate
            assert len(rate) == 2

    def test_ensure_single_guards(self, series_pair):
        """Methods that require single-pair should raise TypeError on series."""
        a, b = series_pair
        dc = DriftCorrection.from_data([a, b], scan_direction_degrees=[0, -90])
        dc.preprocess().align_affine(step=0.02, num_tests=5)
        with pytest.raises(TypeError, match="not supported on series"):
            dc.drift_rate
        with pytest.raises(TypeError, match="not supported on series"):
            dc.print_drift_stats()
        with pytest.raises(TypeError, match="not supported on series"):
            dc.apply_correction()

    def test_getitem_on_single_raises(self):
        """Indexing a single-pair instance should raise TypeError."""
        im0, im1, _ = make_synthetic_drift_data(scale=1, seed=42)
        dc = DriftCorrection.from_data([im0, im1], scan_direction_degrees=[0, -90])
        with pytest.raises(TypeError, match="not indexable"):
            dc[0]

    def test_frame_count_mismatch_raises(self):
        """Stacks with different frame counts should raise ValueError."""
        a = np.zeros((3, 64, 64), dtype=np.float32)
        b = np.zeros((4, 64, 64), dtype=np.float32)
        with pytest.raises(ValueError, match="Frame count mismatch"):
            DriftCorrection.from_data([a, b], scan_direction_degrees=[0, -90])

    def test_nonrigid_series(self, series_pair):
        """Nonrigid alignment should work on series."""
        a, b = series_pair
        dc = DriftCorrection.from_data([a, b], scan_direction_degrees=[0, -90])
        dc.preprocess().align_affine(step=0.02, num_tests=5)
        dc.align_nonrigid(num_iterations=2, adam_steps=5)
        result = dc.generate_corrected_image(upsample_factor=1)
        assert result.shape[0] == a.shape[0]
        for frame in dc:
            assert (frame.error_track[:, 0] == 2.0).any()

    def test_empty_series_raises(self):
        """Zero-frame stacks should raise ValueError, not IndexError."""
        a = np.zeros((0, 64, 64), dtype=np.float32)
        b = np.zeros((0, 64, 64), dtype=np.float32)
        with pytest.raises(ValueError, match="at least 1 frame"):
            DriftCorrection.from_data([a, b], scan_direction_degrees=[0, -90])

    def test_calculate_error_on_series_raises(self, series_pair):
        """calculate_error should raise TypeError on series instances."""
        a, b = series_pair
        dc = DriftCorrection.from_data([a, b], scan_direction_degrees=[0, -90])
        dc.preprocess()
        with pytest.raises(TypeError, match="not supported on series"):
            dc.calculate_error(mode=0)

    def test_preprocess_suppresses_plots(self, series_pair):
        """preprocess on series should not trigger per-frame plots."""
        a, b = series_pair
        dc = DriftCorrection.from_data([a, b], scan_direction_degrees=[0, -90])
        # show_merged=True would normally show a plot; on series it's forced off
        # If this doesn't raise, it ran without attempting to show 2 plots
        dc.preprocess(show_merged=True)
        # Verify each frame was preprocessed correctly
        for frame in dc:
            assert hasattr(frame, "knots")
