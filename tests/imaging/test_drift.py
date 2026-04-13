"""
Tests for the DriftCorrection class in quantem.imaging.drift

Synthetic data: chevron pattern with linear drift + jitter at 0 and 90 deg scan angles.
See PR #133 for images: https://github.com/electronmicroscopy/quantem/pull/133
"""

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter
from quantem.core.datastructures.dataset2d import Dataset2d
from quantem.imaging.drift import DriftCorrection, align_affine_single_sided


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
        show_merged=False,
        show_images=False,
    )
    img_corr2 = drift2.generate_corrected_image(upsample_factor=1, show_image=False)

    np.testing.assert_array_almost_equal(
        img_corr.array, img_corr2.array, decimal=10,
        err_msg="Drift correction output is not deterministic!",
    )


def test_align_affine_single_sided_recovers_known_drift():
    """Single-sided affine search should recover a known per-line drift."""
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

    result = align_affine_single_sided(
        reference_image=reference,
        moving_image=moving,
        scan_direction_degrees=0.0,
        pad_fraction=0.25,
        pad_value=0.0,
        kde_sigma=0.5,
        number_knots=1,
        step=0.01,
        num_tests=13,
        refine=True,
        device="cpu",
    )
    baseline = align_affine_single_sided(
        reference_image=reference,
        moving_image=moving,
        scan_direction_degrees=0.0,
        pad_fraction=0.25,
        pad_value=0.0,
        kde_sigma=0.5,
        number_knots=1,
        step=0.0,
        num_tests=1,
        refine=False,
        device="cpu",
    )

    np.testing.assert_allclose(result["best_drift"], expected_drift, atol=0.011)
    assert result["best_cost"] < baseline["best_cost"] * 0.9
    assert result["reference_canvas"].shape == (80, 80)
    assert result["moving_canvas"].shape == (80, 80)


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
    assert drift.images_warped.array.shape == (1, 160, 160)
    assert not np.isnan(drift.images_warped.array).any()


# Baseline values from float32 torch path, captured once and frozen.
# (scale, error, knots0_sum, knots1_sum)
# Recaptured on torch 2.10.0, scipy 1.17.1, numpy 2.4.3 (2026-04-09).
# scale=1 is bit-exact across versions; scale=2,4 shifted by ~0.03-1.1%
# due to optimizer trajectory divergence from dependency upgrades.
AFFINE_BASELINES = [
    (1, 0.09237676858901978, 12157.7373046875, 28546.2626953125),
    (2, 0.13840317726135254, 49798.91015625, 113529.08984375),
    (4, 0.16398872435092926, 194687.2578125, 459648.7421875),
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
        drift.knots[0].sum(), expected_k0, decimal=6)
    np.testing.assert_almost_equal(
        drift.knots[1].sum(), expected_k1, decimal=6)


# Frozen baselines for the pytorch backend with optimizer_name="adam".
# Recaptured on torch 2.10.0, scipy 1.17.1, numpy 2.4.3 (2026-04-09).
NONRIGID_ADAM_BASELINES = [
    (1, 0.05627801641821861, 12023.871063232422, 28671.676582336426),
    (2, 0.1293669193983078, 49747.038246154785, 113559.02951431274),
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
        show_merged=False, show_images=False,
    )
    np.testing.assert_almost_equal(
        drift.error_track[-1, 1], expected_error, decimal=6)
    np.testing.assert_almost_equal(
        drift.knots[0].sum(), expected_k0, decimal=6)
    np.testing.assert_almost_equal(
        drift.knots[1].sum(), expected_k1, decimal=6)


# Frozen baselines for the pytorch backend with optimizer_name="lbfgs".
# Shares _compiled_loss_fn with the Adam path, so this catches regressions
# in either the optimizer dispatch or the shared loss.
# Recaptured on torch 2.10.0, scipy 1.17.1, numpy 2.4.3 (2026-04-09).
NONRIGID_LBFGS_BASELINES = [
    (1, 0.07269975543022156, 12152.98459815979, 28536.15177345276),
    (2, 0.110267274081707, 50195.45083999634, 113757.61969947815),
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
        show_merged=False, show_images=False,
    )
    np.testing.assert_almost_equal(
        drift.error_track[-1, 1], expected_error, decimal=6)
    np.testing.assert_almost_equal(
        drift.knots[0].sum(), expected_k0, decimal=6)
    np.testing.assert_almost_equal(
        drift.knots[1].sum(), expected_k1, decimal=6)


# ---------------------------------------------------------------------------
# Tests for fixed_indices support in align_affine
# ---------------------------------------------------------------------------


def test_align_affine_fixed_indices_recovers_known_drift():
    """align_affine(fixed_indices=[0]) should recover a known single-sided drift.

    Same setup as test_align_affine_single_sided_recovers_known_drift but
    exercised through the unified DriftCorrection API.
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
    knots0_before = drift.knots[0].copy()

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
        drift.knots[0], knots0_before,
        err_msg="fixed_indices=[0] should leave image 0 knots unchanged",
    )
    # Moving image knots should have changed (drift was applied)
    assert not np.array_equal(drift.knots[1], knots0_before), \
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
        drift_a.knots[0], drift_b.knots[0], decimal=10,
    )
    np.testing.assert_array_almost_equal(
        drift_a.knots[1], drift_b.knots[1], decimal=10,
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

    for i in range(2):
        arr = dc.images[i].array
        assert arr.min() >= -0.01, f"Image {i} min={arr.min()}"
        assert arr.max() <= 1.01, f"Image {i} max={arr.max()}"
