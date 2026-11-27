"""Scientist-facing API contracts for drift correction."""

import inspect

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.figure import Figure
from scipy.ndimage import gaussian_filter, shift

from quantem.core.datastructures.dataset2d import Dataset2d
from quantem.imaging import DriftCorrection


def _orthogonal_pair(size: int = 64) -> tuple[Dataset2d, Dataset2d]:
    """Return one calibrated synthetic field acquired at 0 and 90 degrees."""
    rng = np.random.default_rng(42)
    image = gaussian_filter(
        rng.normal(size=(size, size)).astype(np.float32),
        sigma=1.2,
    )
    rows, columns = np.indices(image.shape, dtype=np.float32)
    image += 2.0 * np.exp(
        -((rows - 19.0) ** 2 + (columns - 43.0) ** 2) / (2.0 * 4.0**2)
    )
    scans = []
    for array, angle in ((image, 0.0), (np.rot90(image, k=-1).copy(), 90.0)):
        dataset = Dataset2d.from_array(
            array,
            origin=(1.0, 2.0),
            sampling=(0.08, 0.08),
            units=("nm", "nm"),
        )
        dataset.metadata["scan_rotation_deg"] = angle
        scans.append(dataset)
    return scans[0], scans[1]


def test_primary_api_uses_scientist_facing_names():
    """The final PR exposes one concise chain without legacy solve names."""
    preprocess = inspect.signature(DriftCorrection.preprocess).parameters
    translation = inspect.signature(DriftCorrection.align_translation).parameters
    affine = inspect.signature(DriftCorrection.correct_affine).parameters
    nonrigid = inspect.signature(DriftCorrection.correct_nonrigid).parameters

    assert {"padding_fraction", "smoothing_sigma", "num_knots"} <= set(preprocess)
    assert {"max_image_shift", "fixed_scans"} <= set(translation)
    assert {"max_drift_rate", "num_rates", "region"} <= set(affine)
    assert {"num_knots", "num_refine_cycles", "knot_smoothing_sigma"} <= set(
        nonrigid
    )
    for old_name in (
        "pad_fraction",
        "kde_sigma",
        "number_knots",
        "step",
        "num_tests",
        "num_iterations",
    ):
        assert old_name not in preprocess
        assert old_name not in affine
        assert old_name not in nonrigid
    for old_method in (
        "from_data",
        "align_affine",
        "align_nonrigid",
        "generate_corrected",
    ):
        assert not hasattr(DriftCorrection, old_method)


def test_from_images_requires_angles_for_bare_arrays():
    """Bare arrays never receive a silent scan-angle assumption."""
    image = np.zeros((16, 16), dtype=np.float32)
    with pytest.raises(TypeError, match="scan_direction_degrees is required"):
        DriftCorrection.from_images(image, image.copy(), device="cpu")


def test_static_show_returns_one_closed_matplotlib_figure():
    """A bare static show call does not also queue an inline duplicate."""
    scan_0, scan_90 = _orthogonal_pair()
    drift = DriftCorrection.from_images(scan_0, scan_90, device="cpu")
    drift.preprocess(
        padding_fraction=0.25,
        show_combined=False,
        show_scans=False,
        show_knots=False,
        verbose=False,
    )

    figure = drift.show(mode="static", cmap="gray")

    assert isinstance(figure, Figure)
    assert len(figure.axes) == 6
    assert figure.number not in plt.get_fignums()


def test_metadata_driven_affine_workflow_returns_calibrated_dataset():
    """The normal image workflow is short, finite, and calibration preserving."""
    scan_0, scan_90 = _orthogonal_pair()
    drift = DriftCorrection.from_images(scan_0, scan_90, device="cpu")
    drift.preprocess(
        padding_fraction=0.25,
        show_combined=False,
        show_scans=False,
        show_knots=False,
        verbose=False,
    )
    drift.correct_affine(
        max_drift_rate=0.02,
        num_rates=5,
        refine=False,
        max_image_shift=8,
        show_combined=False,
        show_scans=False,
        show_knots=False,
        verbose=False,
    )
    corrected = drift.corrected(
        upsample_factor=1,
        output_frame="input",
        verbose=False,
    )

    assert isinstance(corrected, Dataset2d)
    assert corrected.shape == scan_0.shape
    assert np.isfinite(corrected.array).all()
    np.testing.assert_allclose(corrected.origin, scan_0.origin)
    np.testing.assert_allclose(corrected.sampling, scan_0.sampling)
    assert corrected.units == scan_0.units
    assert len(drift.drift_rate) == 2

    automatic = drift.corrected(upsample_factor=1, verbose=False)
    canvas = drift.corrected(
        upsample_factor=1,
        output_frame="canvas",
        verbose=False,
    )
    assert automatic.shape == tuple(drift.shape[-2:])
    assert canvas.shape == tuple(drift.shape[-2:])


def test_manual_translation_alignment_reduces_global_offset():
    """Manual translation alignment registers scans without fitting drift."""
    rng = np.random.default_rng(7)
    reference = gaussian_filter(
        rng.normal(size=(64, 64)).astype(np.float32),
        sigma=1.5,
    )
    moving = np.rot90(
        shift(
            reference,
            shift=(3.0, -4.0),
            order=1,
            mode="constant",
            cval=float(np.median(reference)),
        ),
        k=1,
    ).astype(np.float32)
    drift = DriftCorrection.from_images(
        reference,
        moving,
        scan_direction_degrees=(0.0, 90.0),
        device="cpu",
    )

    returned = drift.align_translation(
        max_image_shift=8,
        show_combined=False,
        show_scans=False,
        show_knots=False,
        verbose=False,
    )
    initial = drift.corrected(
        stage="initial",
        merge=False,
        output_frame="input",
        verbose=False,
    )
    aligned = drift.corrected(
        merge=False,
        output_frame="input",
        verbose=False,
    )
    interior = np.s_[8:-8, 8:-8]
    initial_error = np.mean(
        np.abs(initial[0].array[interior] - initial[1].array[interior])
    )
    aligned_error = np.mean(
        np.abs(aligned[0].array[interior] - aligned[1].array[interior])
    )

    assert returned is drift
    assert aligned_error < initial_error * 0.6


def test_nonrigid_diagnostic_defines_fast_roughness():
    """The difficult multi-knot diagnostic states exactly what roughness means."""
    doc = DriftCorrection.diagnose_nonrigid.__doc__ or ""
    assert "root-mean-square difference between neighboring knot" in doc
    assert "does not measure image noise" in doc


def test_nonrigid_local_ncc_coverage_weighted_and_max_slope_nan_safe():
    """Tile-local NCC + coverage-weighted non-folding regularization stays finite on flat vacuum."""
    from scipy.ndimage import map_coordinates

    size = 96
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    base = (
        np.sin(2.0 * np.pi * yy / 8.0) * np.sin(2.0 * np.pi * xx / 8.0)
        + 1.2 * np.exp(-((yy - 36.0) ** 2 + (xx - 52.0) ** 2) / 140.0)
        + 0.9 * np.exp(-((yy - 68.0) ** 2 + (xx - 34.0) ** 2) / 110.0)
    ).astype(np.float32)
    # Flat zero-variance vacuum strip exercises NaN-safe variance guards in _grid_sample_local_ncc
    base[:, :16] = 0.0

    t = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    dy0 = 3.5 * t + 1.2 * (t**2 - 1.0 / 3.0)
    dx0 = -2.5 * t
    dy1 = 1.5 * t - 0.8 * (t**2 - 1.0 / 3.0)
    dx1 = 1.8 * t

    im0 = map_coordinates(base, [yy + dy0[:, None], xx + dx0[:, None]], order=3, mode="nearest").astype(np.float32)
    im1_canvas = map_coordinates(base, [yy + dy1[None, :], xx + dx1[None, :]], order=3, mode="nearest").astype(np.float32)
    im1 = np.ascontiguousarray(np.rot90(im1_canvas, 1))

    drift = DriftCorrection.from_images(im0, im1, scan_direction_degrees=(0.0, 90.0), device="cpu")
    drift.preprocess(
        padding_fraction=0.25,
        smoothing_sigma=0.5,
        num_knots=1,
        show_combined=False,
        show_scans=False,
        show_knots=False,
        verbose=False,
    )
    drift.correct_affine(
        show_combined=False,
        show_scans=False,
        show_knots=False,
        verbose=False,
    )
    rep_before = float(drift.report().iloc[-1]["Common NCC"])
    drift.correct_nonrigid(
        num_knots=1,
        max_image_shift=6.0,
        knot_smoothing_sigma=2.0,
        num_refine_cycles=8,
        learning_rate=0.05,
        max_knot_step=0.35,
        trend_order=2,
        loss="local_ncc",
        tile_grid=4,
        coverage_weighted=True,
        max_slope=0.42,
        solve_translation=False,
        show_combined=False,
        show_scans=False,
        show_knots=False,
        verbose=False,
    )
    rep_after = float(drift.report().iloc[-1]["Common NCC"])
    assert all(np.isfinite(k.detach().cpu().numpy()).all() for k in drift.knots)
    assert rep_after >= rep_before
    # Directional non-folding slope projection guarantees det(J) = 1 + g_par >= 1 - max_slope
    for idx in range(2):
        init_k = drift._initial_knots[idx].detach().cpu().numpy()
        delta = drift.knots[idx].detach().cpu().numpy() - init_k
        s_vec = np.mean(np.diff(init_k[:, :, :1], axis=1), axis=1, keepdims=True)
        s_hat = s_vec / max(float(np.linalg.norm(s_vec)), 1e-8)
        dd = np.diff(delta, axis=1)
        g_par = np.sum(dd * s_hat, axis=0)
        assert float(np.min(g_par)) >= -0.43, f"compression exceeded max_slope on scan {idx}: {g_par.min():.4f}"


def test_serpentine_snake_lag_zero_slope_bias():
    """Verify symmetric midpoint estimator recovers true snake lag without tilted-feature bias."""
    from scipy.ndimage import shift as ndi_shift
    from quantem.imaging.drift import serpentine

    size = 128
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    # 30-degree tilted ridges + localized Gaussian features (no snake lag)
    u = xx * np.cos(np.deg2rad(30.0)) + yy * np.sin(np.deg2rad(30.0))
    clean = (
        np.sin(2.0 * np.pi * u / 11.0)
        + 1.4 * np.exp(-((yy - 40.0) ** 2 + (xx - 55.0) ** 2) / 120.0)
        + 1.1 * np.exp(-((yy - 88.0) ** 2 + (xx - 78.0) ** 2) / 90.0)
    ).astype(np.float32)

    # Zero-lag tilted image must NOT report false slope-biased lag
    info_zero = serpentine.estimate_snake_lag(clean, fast_axis=1)
    assert abs(info_zero["lag_px"]) < 0.15, f"False slope bias detected: {info_zero}"

    # Inject true bidirectional lag = +2.4 px (even shifted +1.2, odd shifted -1.2)
    lagged = clean.copy()
    lagged[0::2] = ndi_shift(clean[0::2], shift=(0.0, +1.2), order=3, mode="nearest")
    lagged[1::2] = ndi_shift(clean[1::2], shift=(0.0, -1.2), order=3, mode="nearest")
    info_lag = serpentine.estimate_snake_lag(lagged, fast_axis=1)
    assert info_lag["applied"] is True
    assert abs(info_lag["lag_px"] - 2.4) < 0.25, f"Expected ~2.4 px lag, got {info_lag}"
    fixed = serpentine.decomb_serpentine(lagged, info_lag["lag_px"], fast_axis=1)
    corr_before = float(np.corrcoef(clean[8:-8, 8:-8].ravel(), lagged[8:-8, 8:-8].ravel())[0, 1])
    corr_after = float(np.corrcoef(clean[8:-8, 8:-8].ravel(), fixed[8:-8, 8:-8].ravel())[0, 1])
    assert corr_after > 0.98 and corr_after > corr_before + 0.1


def test_serpentine_2x2_interleaved_split_and_reinterleave():
    """Verify 4-pass serpentine_2x2 pass geometry, DriftCorrection alignment, and full (H,W,2) probes."""
    from quantem.imaging.drift import DriftCorrection, serpentine

    size = 128
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    base = (
        1.5 * np.exp(-((yy - 42.0) ** 2 + (xx - 50.0) ** 2) / 140.0)
        + 1.2 * np.exp(-((yy - 84.0) ** 2 + (xx - 80.0) ** 2) / 110.0)
        + 0.4 * np.sin(2.0 * np.pi * xx / 14.0) * np.cos(2.0 * np.pi * yy / 14.0)
    ).astype(np.float32)

    passes, specs = serpentine.split_interleaved_passes(base, "serpentine_2x2", decomb=False, compensate_subgrid=True)
    assert len(passes) == 4
    assert [sp["offset"] for sp in specs] == [(0, 0), (0, 1), (1, 1), (1, 0)]
    assert [sp["angle_deg"] for sp in specs] == [0.0, -90.0, 180.0, 90.0]
    # Sub-grid compensation aligns all 4 passes to > 0.995 on a drift-free scan
    for i in range(1, 4):
        c = float(np.corrcoef(passes[0][4:-4, 4:-4].ravel(), passes[i][4:-4, 4:-4].ravel())[0, 1])
        assert c > 0.995, f"Sub-grid offset mismatch on pass {i}: corr={c:.4f}"

    rot_passes = [np.rot90(p, k=sp["rot90_k"]).copy() for p, sp in zip(passes, specs)]
    angles = tuple(sp["angle_deg"] for sp in specs)
    dc = DriftCorrection.from_images(*rot_passes, scan_direction_degrees=angles, device="cpu")
    dc.preprocess(padding_fraction=0.25, smoothing_sigma=0.5, num_knots=1, verbose=False)
    dc.correct_affine(show_combined=False, show_scans=False, show_knots=False, verbose=False)
    re = serpentine.reinterleave_probe_positions(dc, specs, base.shape, corrected=True)
    assert re["corrected"].shape == (size, size, 2)
    assert re["min_det_j"] > 0.85
    # Un-rotated pass indexing must match nominal coordinates to < 1.5 px on a zero-drift scan
    re_nom = serpentine.reinterleave_probe_positions(dc, specs, base.shape, corrected=False)
    assert float(np.max(np.abs(re_nom["corrected"] - re_nom["nominal"]))) < 1e-4
    assert float(np.max(np.abs(re["corrected"] - re["nominal"]))) < 1.5
