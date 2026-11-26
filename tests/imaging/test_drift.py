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
