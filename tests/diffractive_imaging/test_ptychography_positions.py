"""Basic ptychography scan-position behavior tests."""

from __future__ import annotations

import numpy as np
import pytest

from quantem.core import config
from quantem.core.datastructures.dataset4dstem import Dataset4dstem
from quantem.diffractive_imaging.dataset_models import PtychographyDatasetRaster
from quantem.diffractive_imaging.ptychography_lite import PtychoLite


@pytest.fixture(autouse=True)
def _cpu_config():
    previous = config.get_device()
    config.set_device("cpu")
    yield
    config.set_device(previous)


def _make_raw_dataset():
    scan_shape = (3, 4)
    detector_shape = (16, 16)
    scan_sampling = (2.0, 3.0)
    reciprocal_sampling = (0.25, 0.5)
    data = np.ones((*scan_shape, *detector_shape), dtype=np.float32)
    return Dataset4dstem.from_array(
        array=data,
        sampling=(*scan_sampling, *reciprocal_sampling),
        units=("A", "A", "A^-1", "A^-1"),
    )


def _make_raster_dataset():
    dset = _make_raw_dataset()
    return PtychographyDatasetRaster.from_dataset4dstem(dset, verbose=0)


def _sample_grid(scan_shape=(3, 4)):
    rows, cols = np.meshgrid(
        np.arange(scan_shape[0], dtype=np.float32),
        np.arange(scan_shape[1], dtype=np.float32),
        indexing="ij",
    )
    return np.stack((rows, cols), axis=-1)


def _expected_regular_raster_positions(pdset, obj_padding_px):
    rows = np.arange(pdset.gpts[0], dtype=np.float32) * pdset.scan_sampling[0]
    cols = np.arange(pdset.gpts[1], dtype=np.float32) * pdset.scan_sampling[1]
    row_grid, col_grid = np.meshgrid(rows, cols, indexing="ij")
    expected = np.stack((row_grid.ravel(), col_grid.ravel()), axis=-1)
    expected[:, 0] /= pdset.obj_sampling[0]
    expected[:, 1] /= pdset.obj_sampling[1]
    expected[:, 0] += obj_padding_px[0]
    expected[:, 1] += obj_padding_px[1]
    return expected


def _expected_probe_positions(pdset, probe_positions_px, obj_padding_px):
    probe_positions = np.asarray(probe_positions_px, dtype=np.float32).reshape(-1, 2)
    expected = np.empty_like(probe_positions)
    expected[:, 0] = probe_positions[:, 0] * pdset.scan_sampling[0] / pdset.obj_sampling[0]
    expected[:, 1] = probe_positions[:, 1] * pdset.scan_sampling[1] / pdset.obj_sampling[1]
    expected[:, 0] += obj_padding_px[0]
    expected[:, 1] += obj_padding_px[1]
    return expected


def _preprocess_positions(pdset, obj_padding_px):
    pdset.preprocess(
        com_fit_function="constant",
        force_com_rotation=0,
        force_com_transpose=False,
        obj_padding_px=obj_padding_px,
        plot_rotation=False,
        plot_com=False,
    )


def test_initial_scan_positions_match_regular_raster_geometry():
    pdset = _make_raster_dataset()
    obj_padding_px = (3, 4)

    _preprocess_positions(pdset, obj_padding_px)

    expected = _expected_regular_raster_positions(pdset, obj_padding_px)
    actual = pdset.scan_positions_px.detach().cpu().numpy()
    initial = pdset.initial_scan_positions_px.detach().cpu().numpy()
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-6)
    np.testing.assert_allclose(initial, expected, rtol=0, atol=1e-6)


def test_probe_positions_px_regular_grid_matches_nominal_raster():
    obj_padding_px = (3, 4)
    nominal = _make_raster_dataset()
    _preprocess_positions(nominal, obj_padding_px)

    explicit = _make_raster_dataset()
    explicit.preprocess(
        com_fit_function="constant",
        force_com_rotation=0,
        force_com_transpose=False,
        obj_padding_px=obj_padding_px,
        probe_positions_px=_sample_grid(),
        plot_rotation=False,
        plot_com=False,
    )

    np.testing.assert_allclose(
        explicit.scan_positions_px.detach().cpu().numpy(),
        nominal.scan_positions_px.detach().cpu().numpy(),
        rtol=0,
        atol=1e-6,
    )


def test_probe_positions_px_fractional_offsets_convert_to_object_pixels():
    pdset = _make_raster_dataset()
    obj_padding_px = (3, 4)
    probe_positions = _sample_grid()
    probe_positions[..., 0] += np.linspace(0.0, 0.5, probe_positions.shape[0])[:, None]
    probe_positions[..., 1] += np.linspace(0.0, 1.25, probe_positions.shape[0])[:, None]

    pdset.preprocess(
        com_fit_function="constant",
        force_com_rotation=0,
        force_com_transpose=False,
        obj_padding_px=obj_padding_px,
        probe_positions_px=probe_positions,
        plot_rotation=False,
        plot_com=False,
    )

    expected = _expected_probe_positions(pdset, probe_positions, obj_padding_px)
    np.testing.assert_allclose(
        pdset.scan_positions_px.detach().cpu().numpy(),
        expected,
        rtol=0,
        atol=1e-6,
    )


def test_scan_positions_px_setter_accepts_fractional_position_updates():
    pdset = _make_raster_dataset()
    obj_padding_px = (3, 4)
    _preprocess_positions(pdset, obj_padding_px)
    nominal = pdset.scan_positions_px.detach().cpu().numpy()
    offsets = np.zeros_like(nominal)
    offsets[:, 0] = np.linspace(0.05, 0.25, nominal.shape[0], dtype=np.float32)
    offsets[:, 1] = np.linspace(-0.25, -0.05, nominal.shape[0], dtype=np.float32)
    adjusted = nominal + offsets

    pdset.scan_positions_px = adjusted

    batch_indices = np.array([0, 5, 11])
    _patch_indices, positions_px, positions_frac, _descan_shifts = pdset.forward(
        batch_indices,
        obj_padding_px,
    )
    expected_positions = adjusted[batch_indices]
    expected_frac = expected_positions - np.round(expected_positions)
    np.testing.assert_allclose(
        positions_px.detach().cpu().numpy(),
        expected_positions,
        rtol=0,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        positions_frac.detach().cpu().numpy(),
        expected_frac,
        rtol=0,
        atol=1e-6,
    )


def test_scan_positions_px_setter_rejects_wrong_shape():
    pdset = _make_raster_dataset()

    with pytest.raises(ValueError, match="scan_positions_px"):
        pdset.scan_positions_px = np.zeros((pdset.num_gpts - 1, 2), dtype=np.float32)


def test_ptycholite_accepts_probe_positions_px():
    pdset = PtychographyDatasetRaster.from_dataset4dstem(_make_raw_dataset(), verbose=0)
    _preprocess_positions(pdset, (0, 0))
    obj_padding_px = (3, 4)
    probe_positions = _sample_grid()
    probe_positions[..., 1] += np.linspace(0.0, 1.0, probe_positions.shape[0])[:, None]

    ptycho = PtychoLite.from_dataset(
        pdset,
        num_slices=1,
        obj_type="complex",
        energy=300e3,
        defocus=10.0,
        semiangle_cutoff=20.0,
        obj_padding_px=obj_padding_px,
        probe_positions_px=probe_positions,
        device="cpu",
        verbose=False,
        rng=0,
    )

    expected = _expected_probe_positions(ptycho.dset, probe_positions, ptycho.obj_padding_px)
    np.testing.assert_allclose(
        ptycho.dset.scan_positions_px.detach().cpu().numpy(),
        expected,
        rtol=0,
        atol=1e-6,
    )
