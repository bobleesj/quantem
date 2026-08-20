"""Synthetic-data tests: forward model, DriftCorrection workflows, frozen baselines.

Everything here runs on generated data (no real EMD files), so it always
runs, always in seconds. Real-data paper reproduction lives in
test_drift2d.py / test_drift3d.py. EMD metadata/scan_pairs rules live in
test_drift_io.py. Core building-block parity vs numpy/scipy lives in
test_drift_utils.py.
"""

import numpy as np
import pytest
import torch
from scipy.ndimage import gaussian_filter, map_coordinates

import quantem.imaging.drift.core.strip as strip
from quantem.core.datastructures.dataset2d import Dataset2d
from quantem.imaging.drift import (
    CorrectionResult,
    DriftCorrection,
    StripPass,
)
from tests.imaging.drift.simulation_fixture import (
    bilinear_sample,
    make_synthetic_drift_data,
    rotated_scan_positions,
    scan_time_drift_field,
    simulate_drifted_4dstem,
)


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
        padding_fraction=0.25, padding_value=0.0, smoothing_sigma=0.5,
        num_knots=1, normalize=True,
        show_combined=False, show_scans=False,
    )
    dc.correct_affine(
        max_drift_rate=0.10, num_rates=11, refine=True,
        fixed_scans=[0], max_image_shift=32,
        show_combined=False, show_scans=False,
    )
    return dc, ref, drifted


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


# ---------------------------------------------------------------------------
# Forward model: simulate_drifted_4dstem / scan_time_drift_field / rotated
# scan positions match a hand-derivable geometric answer.
# ---------------------------------------------------------------------------


def test_rotated_scan_positions_90_is_counterclockwise():
    positions = rotated_scan_positions((5, 5), 90)
    # At +90 degrees, the raw image appears counterclockwise on display.
    np.testing.assert_allclose(positions[0, 0], [0, 4])
    np.testing.assert_allclose(positions[-1, -1], [4, 0])


def test_rotated_scan_positions_90_displays_back_to_image0():
    image = np.arange(25, dtype=np.float32).reshape(5, 5)
    data = image[..., None]

    simulated = simulate_drifted_4dstem(
        data,
        scan_direction_degrees=90,
        total_drift_px=(0.0, 0.0),
        device="cpu",
    )["data"][..., 0]

    np.testing.assert_array_equal(np.rot90(simulated, k=-1), image)


def test_scan_time_drift_field_is_line_constant_and_hits_total():
    drift = scan_time_drift_field((5, 4), total_drift_px=(0.25, 8.0))

    assert drift.shape == (5, 4, 2)
    np.testing.assert_allclose(drift[0, :, :], 0.0)
    np.testing.assert_allclose(drift[-1, :, 0], 0.25)
    np.testing.assert_allclose(drift[-1, :, 1], 8.0)
    np.testing.assert_allclose(drift[:, 0, :], drift[:, -1, :])


def test_simulate_drifted_4dstem_samples_scan_axes_not_detector_axes():
    clean = np.zeros((4, 4, 2, 3), dtype=np.float32)
    for row in range(4):
        for col in range(4):
            clean[row, col] = 100 * row + 10 * col + np.arange(6).reshape(2, 3)

    simulated = simulate_drifted_4dstem(
        clean,
        scan_direction_degrees=0,
        total_drift_px=(0.0, 1.0),
        device="cpu",
    )
    out = simulated["data"]

    expected = np.empty_like(clean)
    for row in range(4):
        shift = row / 3.0
        for col in range(4):
            src_col = max(col - shift, 0.0)
            c0 = int(np.floor(src_col))
            c1 = min(c0 + 1, 3)
            frac = src_col - c0
            expected[row, col] = clean[row, c0] * (1 - frac) + clean[row, c1] * frac

    np.testing.assert_allclose(out, expected, rtol=2e-7, atol=4e-5)
    assert np.all(out[0, 0] == clean[0, 0])


def test_detector_integration_commutes_with_bilinear_scan_drift():
    clean = np.zeros((5, 5, 2, 3), dtype=np.float32)
    for row in range(5):
        for col in range(5):
            clean[row, col] = 100 * row + 10 * col + np.arange(6).reshape(2, 3)
    detector_mask = np.array([[True, False, True], [False, True, False]])
    drift = scan_time_drift_field((5, 5), total_drift_px=(0.75, 1.5))

    drifted_4dstem = simulate_drifted_4dstem(
        clean,
        drift_field_px=drift,
        device="cpu",
    )["data"]
    vdf_after_4dstem_drift = drifted_4dstem[..., detector_mask].sum(axis=-1)

    clean_vdf = clean[..., detector_mask].sum(axis=-1)
    drifted_vdf = simulate_drifted_4dstem(
        clean_vdf[..., None],
        drift_field_px=drift,
        device="cpu",
    )["data"][..., 0]

    np.testing.assert_allclose(vdf_after_4dstem_drift, drifted_vdf, rtol=2e-7, atol=4e-5)


def test_simulate_drifted_4dstem_can_scan_window_inside_larger_source():
    clean = np.zeros((6, 6, 1), dtype=np.float32)
    for row in range(6):
        for col in range(6):
            clean[row, col, 0] = 100 * row + col

    simulated = simulate_drifted_4dstem(
        clean,
        scan_shape=(4, 4),
        scan_origin_px=(1.0, 1.0),
        scan_direction_degrees=0,
        total_drift_px=(0.0, 0.0),
        device="cpu",
    )["data"][..., 0]

    np.testing.assert_allclose(simulated, clean[1:5, 1:5, 0])


def test_positions_offset_includes_rotation_and_shared_drift():
    drift = scan_time_drift_field((4, 4), total_drift_px=(0.0, 2.0))
    sim0 = simulate_drifted_4dstem(
        np.zeros((4, 4, 1), dtype=np.float32),
        scan_direction_degrees=0,
        drift_field_px=drift,
        device="cpu",
    )
    sim90 = simulate_drifted_4dstem(
        np.zeros((4, 4, 1), dtype=np.float32),
        scan_direction_degrees=90,
        drift_field_px=drift,
        device="cpu",
    )

    np.testing.assert_allclose(sim0["drift_field_px"], sim90["drift_field_px"])
    np.testing.assert_allclose(sim0["positions_offset_px"], -drift)
    expected_90 = rotated_scan_positions((4, 4), 90) - drift - rotated_scan_positions((4, 4), 0)
    np.testing.assert_allclose(sim90["positions_offset_px"], expected_90)


# ---------------------------------------------------------------------------
# DriftCorrection workflows on synthetic data: preprocess/affine/strip/
# nonrigid, from_reference, from_4dstem, corrected(), save/load.
# ---------------------------------------------------------------------------


def test_full_pipeline_deterministic():
    """Full pipeline produces correct, deterministic, low-error results."""
    im0, im1, _ = make_synthetic_drift_data(scale=1, seed=42)

    drift = DriftCorrection(
        im0, im1,
        scan_direction_degrees=[0.0, -90.0],
    ).preprocess(
        padding_fraction=0.25,
        padding_value="median",
        smoothing_sigma=0.5,
        num_knots=1,
        show_combined=False,
        show_scans=False,
    )
    drift.correct_affine(max_drift_rate=0.04, num_rates=5, refine=False)
    drift.correct_nonrigid(
        num_refine_cycles=2,
        knot_smoothing_sigma=0.5,
        loss="mse",
        show_combined=False,
        show_scans=False,
    )
    img_corr = drift.corrected(upsample_factor=1)

    assert isinstance(img_corr, Dataset2d)
    assert not np.isnan(img_corr.array).any()
    assert drift.error_track[-1, 1] < 0.1

    # Determinism: second run with same seed must match exactly
    im0_2, im1_2, _ = make_synthetic_drift_data(scale=1, seed=42)
    drift2 = DriftCorrection(
        im0_2, im1_2,
        scan_direction_degrees=[0.0, -90.0],
    ).preprocess(
        padding_fraction=0.25,
        padding_value="median",
        smoothing_sigma=0.5,
        num_knots=1,
        show_combined=False,
        show_scans=False,
    )
    drift2.correct_affine(max_drift_rate=0.04, num_rates=5, refine=False)
    drift2.correct_nonrigid(
        num_refine_cycles=2,
        knot_smoothing_sigma=0.5,
        loss="mse",
        show_combined=False,
        show_scans=False,
    )
    img_corr2 = drift2.corrected(upsample_factor=1)

    np.testing.assert_array_almost_equal(
        img_corr.array, img_corr2.array, decimal=10,
        err_msg="Drift correction output is not deterministic!",
    )


def test_correct_affine_no_arguments_uses_automatic_pyramid_search():
    """The simplest affine workflow prepares itself and remains auditable."""
    im0, im1, _ = make_synthetic_drift_data(scale=1, seed=42)
    drift = DriftCorrection(
        im0, im1, scan_direction_degrees=[0.0, -90.0],
    )

    drift.correct_affine(show_combined=False, show_scans=False, verbose=False)

    info = drift.affine_search_info
    assert drift.preprocess_info["padding_mode"] == "implicit_auto"
    assert drift.preprocess_info["padding_fraction"] == pytest.approx(0.25)
    assert drift.shape == (2, 160, 160)
    assert info["strategy"] == "automatic_pyramid"
    assert info["downsample_factor"] in {1, 2, 4, 8}
    assert info["refine_downsample_factor"] == 2
    assert info["candidate_evaluations"] > 0
    assert info["broad_candidate_evaluations"] == 49
    assert info["broad_candidate_reuses"] == 0
    assert info["native_candidate_evaluations"] > 0
    assert len(info["drift_rate_row_col"]) == 2
    assert info["max_image_shift"] > 0
    assert info["seconds"] > 0
    assert info["preprocess_seconds"] > 0
    assert info["total_seconds"] >= info["seconds"]
    assert [
        stage["radius"]
        for stage in info["history"]
        if stage["stage"] == "pyramid"
    ] == [0.2]
    assert any(
        stage["stage"] == "delivered_objective"
        for stage in info["history"]
    )


def test_preprocess_auto_padding_covers_rotated_affine_envelope():
    """Automatic padding grows for a rotated scan instead of clipping it."""
    image = np.ones((128, 128), dtype=np.float32)
    drift = DriftCorrection(
        image, image, scan_direction_degrees=[0.0, 45.0],
    ).preprocess(show_combined=False, show_scans=False, verbose=False)

    assert drift.pad_fraction > 0.6
    assert drift.shape[1] == drift.shape[2]
    assert drift.preprocess_info["padding_mode"] == "auto"


def test_report_always_uses_final_common_coverage_mask(monkeypatch):
    """Every reported NCC excludes padding and non-common scan coverage."""
    im0, im1, _ = make_synthetic_drift_data(scale=1, seed=42)
    correction = DriftCorrection(
        im0, im1, scan_direction_degrees=[0.0, -90.0],
    )
    correction.correct_affine(
        max_drift_rate=0.04,
        num_rates=3,
        refine=False,
        show_combined=False,
        show_scans=False,
        verbose=False,
    )
    expected_mask = correction.coverage_mask().copy()
    captured_masks = []
    region_ncc = strip.region_ncc

    def capture_mask(reference, moving, mask, **kwargs):
        captured_masks.append(np.asarray(mask, dtype=bool).copy())
        return region_ncc(reference, moving, mask, **kwargs)

    monkeypatch.setattr(strip, "region_ncc", capture_mask)

    report = correction.report()

    assert captured_masks
    assert all(np.array_equal(mask, expected_mask) for mask in captured_masks)
    assert np.allclose(report["Coverage"], expected_mask.mean())


def test_correct_affine_fixed_scans_recovers_known_drift():
    """correct_affine(fixed_scans=[0]) should recover a known single-sided drift.

    Synthetic image with known per-line drift, exercised through the unified
    DriftCorrection API with fixed_scans.
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
        padding_fraction=0.25,
        padding_value=0.0,
        smoothing_sigma=0.5,
        num_knots=1,
        show_combined=False,
        show_scans=False,
    )
    knots0_before = drift.knots[0].clone()

    drift.correct_affine(
        max_drift_rate=0.06,
        num_rates=13,
        refine=True,
        fixed_scans=[0],
        show_combined=False,
        show_scans=False,
    )

    # Reference knots must not have changed
    np.testing.assert_array_equal(
        drift.knots[0].cpu().numpy(), knots0_before.cpu().numpy(),
        err_msg="fixed_scans=[0] should leave image 0 knots unchanged",
    )
    # Moving image knots should have changed (drift was applied)
    assert not np.array_equal(drift.knots[1].cpu().numpy(), knots0_before.cpu().numpy()), \
        "Image 1 knots should have been modified by the affine search"
    # Error should have decreased
    assert drift.error_track[-1, 1] < drift.error_track[0, 1], \
        "Affine alignment with fixed_scans should reduce error"


def test_correct_strip_moves_moving_keeps_fixed(capsys):
    """correct_strip is on DriftCorrection, freezes fixed side, updates free knots."""
    rng = np.random.default_rng(11)
    reference = gaussian_filter(rng.random((96, 96)), sigma=1.2).astype(np.float32)
    # Extra slow-scan column bend on the bottom half (piecewise residual).
    row_grid, col_grid = np.meshgrid(
        np.arange(reference.shape[0], dtype=np.float32),
        np.arange(reference.shape[1], dtype=np.float32),
        indexing="ij",
    )
    bend = np.where(
        row_grid >= reference.shape[0] * 0.45,
        0.08 * (row_grid - reference.shape[0] * 0.45),
        0.0,
    )
    moving = bilinear_sample(
        reference,
        row_grid,
        col_grid + bend,
    ).astype(np.float32)

    drift = DriftCorrection(
        reference, moving,
        scan_direction_degrees=[0.0, 0.0],
    ).preprocess(
        padding_fraction=0.25, padding_value=0.0, smoothing_sigma=0.5, num_knots=1,
        show_combined=False, show_scans=False,
    )
    drift.correct_affine(
        max_drift_rate=0.10, num_rates=11, refine=True,
        fixed_scans=[0], show_combined=False, show_scans=False,
        max_image_shift=32,
    )
    knots0 = drift.knots[0].detach().clone()
    knots1 = drift.knots[1].detach().clone()

    out = drift.correct_strip(
        num_strips=8,
        max_column_shift=16,
        max_row_shift=4,
        correction_start_fraction=None,
        num_refine_cycles=1,
        fixed_scans=[0],
        show_combined=False,
        show_scans=False,
        show_knots=False,
        show_knot_plot=False,
        verbose=True,
    )
    progress = capsys.readouterr().err
    assert "Solving strip drift" in progress
    assert "strips=8, scan=1" in progress
    assert out is drift
    assert torch.equal(drift.knots[0], knots0), "fixed scan knots must not move"
    assert not torch.equal(drift.knots[1], knots1), "moving scan knots should update"


def test_correct_strip_accepts_heterogeneous_pass_recipe():
    """One public call should run a typed coarse-to-fine strip sequence."""

    rng = np.random.default_rng(22)
    reference = gaussian_filter(
        rng.random((96, 96)), sigma=1.2
    ).astype(np.float32)
    moving = np.roll(reference, shift=2, axis=1)
    drift = DriftCorrection(
        reference, moving, scan_direction_degrees=[0.0, 0.0]
    ).preprocess(
        padding_fraction=0.25,
        padding_value=0.0,
        smoothing_sigma=0.5,
        num_knots=1,
        show_combined=False,
        show_scans=False,
    )
    drift.correct_affine(
        max_drift_rate=0.10,
        num_rates=11,
        refine=True,
        fixed_scans=[0],
        show_combined=False,
        show_scans=False,
        max_image_shift=16,
    )
    recipe = [
        StripPass(
            num_strips=8,
            smoothing_sigma=8.0,
            max_column_shift=8,
            max_row_shift=2,
        ),
        StripPass(
            num_strips=12,
            smoothing_sigma=4.0,
            max_column_shift=3,
            max_row_shift=1,
            update_fraction=0.8,
        ),
    ]
    result = drift.correct_strip(
        passes=recipe,
        fixed_scans=[0],
        show_combined=False,
        show_scans=False,
        show_knots=False,
        verbose=False,
    )
    assert result is drift


def test_strip_pass_requires_two_regions():
    """A strip pass needs two regions to measure slow-scan variation."""
    with pytest.raises(ValueError, match="num_strips must be at least 2"):
        StripPass(num_strips=1)


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


def test_drift_rate_matches_ground_truth():
    """drift_rate should approximate the known drift slope."""
    rate_gt = (0.05, 0.1)
    # Use larger image for more accurate drift estimation
    dc, _, _ = _make_single_sided_dc(drift_rate=rate_gt)
    rate = dc.drift_rate
    # Negative because knots compensate drift
    assert abs(rate[0] + rate_gt[0]) < 0.04, f"row rate {rate[0]} far from {-rate_gt[0]}"
    assert abs(rate[1] + rate_gt[1]) < 0.04, f"col rate {rate[1]} far from {-rate_gt[1]}"


def test_plot_combined_interactive_stages_defaults_to_rgb():
    """The interactive combined view defaults to one RGB panel per stage."""
    im0, im1, _ = make_synthetic_drift_data(scale=1, seed=42)
    dc = DriftCorrection(
        im0, im1,
        scan_direction_degrees=[0.0, -90.0],
    ).preprocess(
        padding_fraction=0.25, padding_value="median", smoothing_sigma=0.5,
        num_knots=1, show_combined=False, show_scans=False,
    )
    dc.correct_affine(max_drift_rate=0.04, num_rates=5, refine=False)
    dc.correct_nonrigid(
        num_refine_cycles=2, knot_smoothing_sigma=0.5, loss="mse",
        show_combined=False, show_scans=False,
    )
    widget = dc.plot_combined(
        stage=("affine", "nonrigid"), interactive=True, width=800,
    )
    # Single row: one RGB panel per stage, affine and non-rigid side by side.
    assert widget.n_images == 2
    assert widget.labels == [
        "Combined: after affine correction (RGB)",
        "Combined: after non-rigid correction (RGB)",
    ]
    assert widget.ncols == 2
    # width=800 must reach the per-panel display width (Show2D `size` trait)
    assert widget.size == 800
    # Combined comparisons default to a clean report surface and a genuinely
    # centered viewport. Show2D treats (0, 0) as the top-left pixel, not as a
    # zero offset, so it must not be forwarded as the default center.
    assert widget.show_controls is False
    assert widget.show_stats is False
    assert widget.zoom_row is None
    assert widget.zoom_col is None
    # panels must be genuinely RGB, not colormapped grayscale
    assert list(widget.is_rgb) == [True, True]
    for combined in widget._rgb_frames:
        assert combined is not None and combined.ndim == 3 and combined.shape[-1] == 3
        # red-green channel structure (default mode="rgb"): blue empty,
        # red (reference) and green (moving) distinct, yellow where aligned
        np.testing.assert_array_equal(combined[..., 2], 0)
        assert not np.allclose(combined[..., 0], combined[..., 1])
    # affine and nonrigid stages must come from different knot snapshots
    assert not np.allclose(widget._rgb_frames[0], widget._rgb_frames[1])
    # rgb=False: plain grayscale merged images, one per stage
    plain = dc.plot_combined(rgb=False, stage=("affine", "nonrigid"), interactive=True)
    assert plain.labels == [
        "Combined: after affine correction",
        "Combined: after non-rigid correction",
    ]
    assert list(plain.is_rgb) == [False, False]
    assert not np.allclose(plain._data[0], plain._data[1])


def test_from_reference_2d_returns_dataset2d():
    """from_reference + 2-D drifted → corrected returns Dataset2d.

    Same scan angle (0, 0) signals reference mode rather than orthogonal pair."""
    ref, drifted = _make_reference_pair(scan_h=32, kind="2d")
    dc = DriftCorrection.from_reference(ref, drifted)
    assert dc._reference_mode
    dc.preprocess(normalize=True, smoothing_sigma=0.5, num_knots=1,
                  show_combined=False, show_scans=False)
    dc.correct_affine(max_drift_rate=0.04, num_rates=5,
                    show_combined=False, show_scans=False)
    result = dc.corrected()
    assert isinstance(result, Dataset2d)
    assert result.array.shape == drifted.shape


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
    dc.preprocess(padding_fraction=0.25, padding_value="median", smoothing_sigma=0.5, num_knots=1,
                  normalize=False, show_combined=False, show_scans=False)
    dc.correct_affine(max_drift_rate=0.15, num_rates=31, refine=True,
                    max_image_shift=16, fixed_scans=[0], show_combined=False, show_scans=False)
    corrected = dc.apply_correction(eds_drifted)
    corrected = corrected.cpu().numpy() if hasattr(corrected, "cpu") else np.asarray(corrected)

    reference_crop = eds_clean[10:-10, 10:-10].ravel()
    ncc_drifted = float(
        np.corrcoef(reference_crop, eds_drifted[10:-10, 10:-10].ravel())[0, 1]
    )
    ncc_corrected = float(
        np.corrcoef(reference_crop, corrected[10:-10, 10:-10].ravel())[0, 1]
    )
    assert ncc_corrected > ncc_drifted, f"correction did not improve match: {ncc_corrected} <= {ncc_drifted}"
    assert ncc_corrected > 0.9, f"corrected cube should closely match ground truth, got {ncc_corrected}"


def test_from_4dstem_named_api_and_result_fields():
    """0/90 4D-STEM has an explicit first-class entry point."""
    cube_a, cube_b = _make_4dstem_collection(scan_size=32, det_size=4)
    dc = DriftCorrection.from_4dstem(
        cube_a, cube_b, scan_direction_degrees=[0, 90],
    )
    dc.preprocess(
        padding_fraction=0.25, smoothing_sigma=0.5, num_knots=1,
        show_combined=False, show_scans=False,
    )
    dc.correct_affine(
        max_drift_rate=0.04, num_rates=5, refine=False,
        show_combined=False, show_scans=False,
    )
    corrected_vdf = dc.corrected_virtual_images(
        dc.imgs[0].array,
        dc.imgs[1].array,
    )
    canvas_vdf = dc.corrected_virtual_images(
        dc.imgs[0].array,
        dc.imgs[1].array,
        output_frame="canvas",
    )
    result = dc.corrected_4dstem()

    assert corrected_vdf["corrected_image"].shape == (32, 32)
    assert corrected_vdf["corrected_image_0"].shape == (32, 32)
    assert corrected_vdf["corrected_image_1"].shape == (32, 32)
    canvas_shape = tuple(dc.shape[-2:])
    assert canvas_vdf["corrected_image"].shape == canvas_shape
    assert canvas_vdf["coverage_image"].shape == canvas_shape
    valid_0 = canvas_vdf["coverage_image_0"] >= 1e-3
    valid_1 = canvas_vdf["coverage_image_1"] >= 1e-3
    contribution_count = valid_0.astype(np.float32) + valid_1.astype(np.float32)
    expected_canvas = np.divide(
        canvas_vdf["corrected_image_0"] * valid_0
        + canvas_vdf["corrected_image_1"] * valid_1,
        contribution_count,
        out=np.zeros(canvas_shape, dtype=np.float32),
        where=contribution_count > 0,
    )
    np.testing.assert_array_equal(canvas_vdf["corrected_image"], expected_canvas)
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


def test_regional_diffraction_patterns_uses_probe_position_membership():
    """Named regions average raw patterns without detector interpolation."""
    cube_a, cube_b = _make_4dstem_collection(scan_size=16, det_size=4)
    dc = DriftCorrection.from_4dstem(
        cube_a,
        cube_b,
        scan_direction_degrees=[0, 90],
    )
    dc.preprocess(
        padding_fraction=0.25,
        smoothing_sigma=0.5,
        num_knots=1,
        show_combined=False,
        show_scans=False,
    )

    regions = {"particle": (8, 8)}
    result = dc.regional_diffraction_patterns(regions, radius_px=2)

    assert result["patterns"].shape == (2, 1, 2, 4, 4)
    assert result["patterns"].dtype == np.float32
    assert result["sample_counts"].shape == (2, 1, 2)
    assert result["region_names"] == ("particle",)
    assert result["stages"] == ("initial", "corrected")
    np.testing.assert_array_equal(result["region_centers_px"], [[8, 8]])

    for stage_index, corrected in enumerate((False, True)):
        for scan_index, cube in enumerate((cube_a, cube_b)):
            positions = dc.probe_positions(
                scan_index,
                corrected=corrected,
                strip_padding=True,
                plot=False,
            )
            mask = (
                (positions[..., 0] - 8) ** 2
                + (positions[..., 1] - 8) ** 2
                <= 2**2
            )
            expected = cube[mask].mean(axis=0, dtype=np.float32)
            np.testing.assert_allclose(
                result["patterns"][stage_index, 0, scan_index],
                expected,
            )
            assert result["sample_counts"][stage_index, 0, scan_index] == mask.sum()


def test_regional_diffraction_patterns_accepts_saved_correction_datasets():
    """A saved correction can consume an explicitly reloaded raw pair."""
    cube_a, cube_b = _make_4dstem_collection(scan_size=16, det_size=4)
    dc = DriftCorrection.from_4dstem(
        cube_a,
        cube_b,
        scan_direction_degrees=[0, 90],
    )
    dc.preprocess(
        padding_fraction=0.25,
        smoothing_sigma=0.5,
        num_knots=1,
        show_combined=False,
        show_scans=False,
    )
    dc._datasets = None

    with pytest.raises(RuntimeError, match="Pass datasets"):
        dc.regional_diffraction_patterns({"support": (8, 8)})

    # Exercise the uint16 path used by the ARINA gold acquisition. The
    # implementation converts only each small selected block before indexing.
    datasets = tuple(
        torch.as_tensor(np.clip(cube, 0, None), dtype=torch.uint16)
        for cube in (cube_a, cube_b)
    )
    result = dc.regional_diffraction_patterns(
        {"support": (8, 8)},
        datasets=datasets,
        radius_px=2,
        stages=("corrected",),
    )

    assert result["patterns"].shape == (1, 1, 2, 4, 4)
    assert np.isfinite(result["patterns"]).all()


@pytest.mark.parametrize(
    ("regions", "radius_px", "stages", "message"),
    [
        ({}, 4, ("initial", "corrected"), "non-empty"),
        ({"bad": (8, 8)}, 0, ("initial", "corrected"), "positive"),
        ({"bad": (8, 8)}, 4, ("affine",), "initial"),
        ({"bad": (8, 8)}, 4, ("corrected", "corrected"), "duplicates"),
    ],
)
def test_regional_diffraction_patterns_validates_scientific_inputs(
    regions,
    radius_px,
    stages,
    message,
):
    """Invalid region geometry and stage names fail with corrective messages."""
    cube_a, cube_b = _make_4dstem_collection(scan_size=16, det_size=4)
    dc = DriftCorrection.from_4dstem(
        cube_a,
        cube_b,
        scan_direction_degrees=[0, 90],
    )
    dc.preprocess(
        padding_fraction=0.25,
        smoothing_sigma=0.5,
        num_knots=1,
        show_combined=False,
        show_scans=False,
    )

    with pytest.raises(ValueError, match=message):
        dc.regional_diffraction_patterns(
            regions,
            radius_px=radius_px,
            stages=stages,
        )


def test_virtual_detector_matches_array_backends():
    """Virtual images preserve masked integer sums across array backends."""
    data = np.arange(3 * 4 * 2 * 3, dtype=np.uint16).reshape(3, 4, 2, 3)
    mask = np.array([[True, False, True], [False, True, False]])
    expected = data[..., mask].sum(axis=-1, dtype=np.uint64).astype(np.float32)

    numpy_image = DriftCorrection.integrate_virtual_detector(
        data,
        mask,
        reduce="sum",
    )
    tensor_image = DriftCorrection.integrate_virtual_detector(
        torch.from_numpy(data),
        mask,
        reduce="sum",
    )
    spectrum_image = DriftCorrection.integrate_virtual_detector(
        data.reshape(3, 4, 6),
        mask,
        reduce="sum",
    )

    np.testing.assert_array_equal(numpy_image, expected)
    np.testing.assert_array_equal(tensor_image, expected)
    np.testing.assert_array_equal(spectrum_image, expected)


def test_corrected_rejects_controls_that_do_not_apply_to_image_pairs():
    """Image correction never silently accepts dataset-only controls."""
    image_0, image_90, _ = make_synthetic_drift_data(scale=1)
    correction = DriftCorrection(
        image_0[:32, :32],
        image_90[:32, :32],
        scan_direction_degrees=(0, 90),
    )
    correction.correct_affine(
        max_drift_rate=0.1,
        num_rates=5,
        refine=False,
        show_combined=False,
        show_scans=False,
        verbose=False,
    )

    with pytest.raises(TypeError, match="mode"):
        correction.corrected(mode="bicubic")
    with pytest.raises(TypeError, match="chunk_size"):
        correction.corrected(chunk_size=8)
    with pytest.raises(ValueError, match="merge=False"):
        correction.corrected(merge=False, strip_padding=True)


def test_generate_corrected_pair_mode_returns_dataset2d():
    """corrected on pair-mode dc returns Dataset2d."""
    im0, im1, _ = make_synthetic_drift_data(scale=1)
    dc = DriftCorrection(
        im0[:32, :32], im1[:32, :32], scan_direction_degrees=[0, 90],
    )
    dc.preprocess(
        padding_fraction=0.25, smoothing_sigma=0.5, num_knots=1,
        show_combined=False, show_scans=False,
    )
    dc.correct_affine(
        max_drift_rate=0.10, num_rates=11,
        show_combined=False, show_scans=False,
    )
    result = dc.corrected()
    assert isinstance(result, Dataset2d)


def test_save_load_roundtrip_scan_collection(tmp_path):
    """AutoSerialize preserves fitted knots and convergence history exactly."""
    from quantem.core.io.serialize import load

    rng = np.random.default_rng(0)
    a = rng.random((64, 64), dtype=np.float32)
    b = rng.random((64, 64), dtype=np.float32)
    dc = DriftCorrection(a, b, scan_direction_degrees=(0, 90))
    dc.preprocess(show_combined=False, show_scans=False)
    dc.correct_affine(max_drift_rate=0.02, num_rates=5, show_combined=False, show_scans=False)
    expected_knots = [k.detach().cpu().numpy().copy() for k in dc.knots]
    expected_errors = np.asarray(dc.error_track).copy()

    path = tmp_path / "dc.zip"
    dc.save(str(path))
    loaded = load(str(path))

    assert isinstance(loaded, DriftCorrection)
    assert loaded.imgs[0].array.shape == (64, 64)
    for expected, recovered in zip(expected_knots, loaded.knots, strict=True):
        np.testing.assert_array_equal(expected, recovered.detach().cpu().numpy())
    np.testing.assert_array_equal(expected_errors, np.asarray(loaded.error_track))


def test_automatic_affine_search_provenance_roundtrips(tmp_path):
    """A saved automatic solve retains its audit trail and chosen rate."""
    from quantem.core.io.serialize import load

    im0, im1, _ = make_synthetic_drift_data(scale=1, seed=42)
    drift = DriftCorrection(
        im0, im1, scan_direction_degrees=(0.0, -90.0),
    ).preprocess(show_combined=False, show_scans=False)
    drift.correct_affine(
        show_combined=False, show_scans=False, verbose=False,
    )
    path = tmp_path / "automatic-affine.zip"
    drift.save(str(path))

    loaded = load(str(path))

    assert loaded.affine_search_info == drift.affine_search_info
    for expected, recovered in zip(drift.knots, loaded.knots, strict=True):
        np.testing.assert_array_equal(
            expected.detach().cpu().numpy(),
            recovered.detach().cpu().numpy(),
        )


# ---------------------------------------------------------------------------
# Frozen numerical baselines: DriftCorrection on the synthetic chevron pair
# must reproduce pinned float32 values. A regression here means the affine
# or nonrigid solve path changed numerically, not just the simulation fixture.
# ---------------------------------------------------------------------------


AFFINE_BASELINES = [
    (1, 0.09237674623727798, 12157.736328125, 28546.263671875),
    # 2026-07-01 pin (kept). scale=2 host band applied in helper below.
    (2, 0.1384429633617401, 49830.90625, 113497.09375),
    (4, 0.16396018862724304, 194685.40625, 459650.59375),
]


NONRIGID_ADAM_BASELINES = [
    (1, 0.05627802759408951, 12023.8701171875, 28671.677734375),
    # 2026-07-01 pin (kept). Host band for scale=2 only; see helper.
    (2, 0.12947677075862885, 49829.3671875, 113481.71875),
]


NONRIGID_LBFGS_BASELINES = [
    (1, 0.0727548599243164, 12151.92578125, 28535.07421875),
    # 2026-07-01 pin (kept). LBFGS error host band is widest (line search).
    (2, 0.11079858243465424, 50327.71484375, 113553.3046875),
]


def _assert_frozen_baseline(scale, actual_error, actual_k0, actual_k1,
                            expected_error, expected_k0, expected_k1, *, kind):
    """Compare to frozen pins without rewriting the gold numbers.

    scale!=2: tight float32-class match (rtol 1e-6).
    scale==2: same pins, wider band for documented FFT/BLAS host drift only.
    Outside that band is a real regression (investigate; do not recapture).
    """
    if scale == 2:
        # Error: LBFGS line-search amplifies noise the most (~1e-2 seen).
        # Knot sums: typically <1e-3 relative on this size.
        err_rtol, err_atol = (2e-2, 1e-3) if kind == "lbfgs" else (3e-3, 1e-4)
        knot_rtol, knot_atol = 2e-3, 1.0
    else:
        err_rtol = knot_rtol = 1e-6
        err_atol = knot_atol = 1e-6
    np.testing.assert_allclose(
        actual_error, expected_error, rtol=err_rtol, atol=err_atol,
        err_msg=(
            f"{kind} scale={scale}: error left the frozen band "
            f"(got {actual_error}, pin {expected_error}). "
            f"Do not rewrite the pin; investigate the algorithm path."
        ),
    )
    np.testing.assert_allclose(
        actual_k0, expected_k0, rtol=knot_rtol, atol=knot_atol,
        err_msg=(
            f"{kind} scale={scale}: knots[0] sum left the frozen band "
            f"(got {actual_k0}, pin {expected_k0})."
        ),
    )
    np.testing.assert_allclose(
        actual_k1, expected_k1, rtol=knot_rtol, atol=knot_atol,
        err_msg=(
            f"{kind} scale={scale}: knots[1] sum left the frozen band "
            f"(got {actual_k1}, pin {expected_k1})."
        ),
    )


@pytest.mark.parametrize("scale,expected_error,expected_k0,expected_k1", AFFINE_BASELINES)
def test_correct_affine_matches_frozen_baseline(scale, expected_error, expected_k0, expected_k1):
    """Affine on synthetic data must match frozen float32 baseline."""
    im0, im1, _ = make_synthetic_drift_data(scale=scale, seed=42)
    drift = DriftCorrection(
        im0, im1, scan_direction_degrees=[0.0, -90.0],
    ).preprocess(show_combined=False, show_scans=False)
    drift.correct_affine(
        max_drift_rate=0.04, num_rates=5, refine=True,
        show_combined=False, show_scans=False,
    )
    _assert_frozen_baseline(
        scale,
        float(drift.error_track[-1, 1]),
        float(drift.knots[0].sum().item()),
        float(drift.knots[1].sum().item()),
        expected_error, expected_k0, expected_k1,
        kind="affine",
    )


@pytest.mark.parametrize("scale,expected_error,expected_k0,expected_k1", NONRIGID_ADAM_BASELINES)
def test_correct_nonrigid_adam_matches_frozen_baseline(scale, expected_error, expected_k0, expected_k1):
    """Nonrigid on synthetic data must match frozen baseline.

    Runs preprocess -> affine -> nonrigid (2 iterations for speed). If the
    GPU warp or translation path changes numerical output, these baselines
    catch it immediately.
    """
    im0, im1, _ = make_synthetic_drift_data(scale=scale, seed=42)
    drift = DriftCorrection(
        im0, im1, scan_direction_degrees=[0.0, -90.0],
    ).preprocess(show_combined=False, show_scans=False)
    drift.correct_affine(
        max_drift_rate=0.04, num_rates=5, refine=True,
        show_combined=False, show_scans=False,
    )
    drift.correct_nonrigid(
        num_refine_cycles=2, optimizer_steps=50,
        knot_smoothing_sigma=16.0,
        # Pin lr to the value the baselines were captured at - the public
        # default is now auto-derived from max_image_shift, but the frozen
        # baselines must stay numerically stable across that change.
        learning_rate=0.02,
        loss="mse",
        show_combined=False, show_scans=False,
    )
    _assert_frozen_baseline(
        scale,
        float(drift.error_track[-1, 1]),
        float(drift.knots[0].sum().item()),
        float(drift.knots[1].sum().item()),
        expected_error, expected_k0, expected_k1,
        kind="adam",
    )


@pytest.mark.parametrize("scale,expected_error,expected_k0,expected_k1", NONRIGID_LBFGS_BASELINES)
def test_correct_nonrigid_lbfgs_matches_frozen_baseline(scale, expected_error, expected_k0, expected_k1):
    """Nonrigid LBFGS path on synthetic data must match frozen baseline.

    The LBFGS optimizer uses a closure-based forward+backward instead of
    Adam's compiled inner loop. This test ensures both paths stay
    numerically deterministic and that LBFGS doesn't silently regress.
    """
    im0, im1, _ = make_synthetic_drift_data(scale=scale, seed=42)
    drift = DriftCorrection(
        im0, im1, scan_direction_degrees=[0.0, -90.0],
    ).preprocess(show_combined=False, show_scans=False)
    drift.correct_affine(
        max_drift_rate=0.04, num_rates=5, refine=True,
        show_combined=False, show_scans=False,
    )
    drift.correct_nonrigid(
        optimizer="lbfgs",
        num_refine_cycles=2, optimizer_steps=20,
        knot_smoothing_sigma=16.0,
        loss="mse",
        show_combined=False, show_scans=False,
    )
    _assert_frozen_baseline(
        scale,
        float(drift.error_track[-1, 1]),
        float(drift.knots[0].sum().item()),
        float(drift.knots[1].sum().item()),
        expected_error, expected_k0, expected_k1,
        kind="lbfgs",
    )
