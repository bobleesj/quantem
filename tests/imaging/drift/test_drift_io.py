"""EMD load + angle rules + crop/show for DriftCorrection.

Scientist path: angles from metadata via from_emd; bare arrays require
scan_direction_degrees. Crop FOV shared across maps; one static show check.
"""


import numpy as np
import pytest
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from quantem.core.datastructures.dataset2d import Dataset2d
from quantem.imaging.drift import DriftCorrection
from quantem.imaging.drift.io import scan_pairs
from tests.imaging.drift.simulation_fixture import make_synthetic_drift_data


def _fake_dataset(image, angle_deg, px_nm=0.01):
    """A stamped Dataset2d, as em.imaging.read_emd would return it."""
    ds = Dataset2d.from_array(np.asarray(image, dtype=np.float32))
    ds.sampling = [px_nm, px_nm]
    ds.metadata["scan_rotation_deg"] = float(angle_deg)
    return ds

def _patch_reader(monkeypatch, registry):
    """Route read_emd(path) -> registry[str(path)]."""
    monkeypatch.setattr(
        "quantem.imaging.drift.io.read_emd",
        lambda path: registry[str(path)],
    )


def test_scan_pairs_matches_orthogonal_scans_at_the_same_stage_position(
    tmp_path, monkeypatch
):
    files = [
        "001_5.20_Mx_0.emd",
        "002_5.20_Mx_90.emd",
        "003_5.20_Mx_90_far.emd",
    ]
    for name in files:
        (tmp_path / name).touch()

    metadata = {
        files[0]: (0.0, (0.0, 0.0), 1.0),
        files[1]: (90.0, (3e-9, 4e-9), 2.0),
        files[2]: (90.0, (100e-9, 0.0), 3.0),
    }

    def read_metadata(path):
        rotation, stage, acquired = metadata[path.name]
        return {
            "scan_shape": (128, 128),
            "pixel_size_nm": 0.1,
            "fov_m": 12.8e-9,
            "magnification": 5.2e6,
            "stage_xy_m": stage,
            "scan_rotation_deg": rotation,
            "acquisition_timestamp": acquired,
        }

    monkeypatch.setattr(
        "quantem.core.io.file_readers.read_emd_metadata", read_metadata
    )
    pairs = scan_pairs(tmp_path)

    reference = pairs[pairs.pair_order == 0].iloc[0]
    assert reference.file == files[0]
    assert reference.partner == files[1]
    assert reference.stage_distance_nm == pytest.approx(5.0)
    assert pairs.loc[pairs.file == files[2], "pair"].item() == ""

def _solve(dc):
    """Run the standard light pipeline and return the final cross-scan error."""
    dc.preprocess(padding_fraction=0.25, padding_value="median", smoothing_sigma=0.5,
                  num_knots=1, show_combined=False, show_scans=False)
    dc.correct_affine(max_drift_rate=0.04, num_rates=5, refine=False)
    dc.correct_nonrigid(num_refine_cycles=2, knot_smoothing_sigma=0.5, loss="mse",
                      show_combined=False, show_scans=False)
    return float(dc.error_track[-1, 1])

def _solved_pair():
    im0, im1, _ = make_synthetic_drift_data()
    dc = DriftCorrection.from_images(im0, im1, scan_direction_degrees=(0.0, -90.0))
    dc.preprocess(padding_fraction=0.25, padding_value="median", smoothing_sigma=0.5,
                  num_knots=1, show_combined=False, show_scans=False)
    dc.correct_affine(max_drift_rate=0.04, num_rates=5, refine=False)
    return dc, im0


def test_from_images_bare_arrays_require_angles():
    im0, im1, _ = make_synthetic_drift_data()
    with pytest.raises(TypeError, match="scan_direction_degrees is required"):
        DriftCorrection.from_images(im0, im1)


def test_from_emd_two_recovers_drift(monkeypatch):
    im0, im1, _ = make_synthetic_drift_data()
    _patch_reader(monkeypatch, {"a.emd": _fake_dataset(im0, 0.0),
                                "b.emd": _fake_dataset(im1, -90.0)})
    dc = DriftCorrection.from_emd("a.emd", "b.emd", verbose=False)
    assert _solve(dc) < 0.1


def test_from_emd_order_invariant(monkeypatch):
    im0, im1, _ = make_synthetic_drift_data()
    reg = {"a.emd": _fake_dataset(im0, 0.0), "b.emd": _fake_dataset(im1, -90.0)}
    _patch_reader(monkeypatch, reg)
    err_ab = _solve(DriftCorrection.from_emd("a.emd", "b.emd", verbose=False))
    err_ba = _solve(DriftCorrection.from_emd("b.emd", "a.emd", verbose=False))
    assert err_ab < 0.1 and err_ba < 0.1
    assert np.isclose(err_ab, err_ba, atol=0.02)


def test_crop_same_region_across_arrays():
    dc, im0 = _solved_pair()
    a = dc.crop(im0)
    b = dc.crop(np.zeros_like(im0))
    assert a.shape == b.shape


def test_from_images_picks_up_metadata_angles():
    im0, im1, _ = make_synthetic_drift_data()
    d0 = _fake_dataset(im0, 0.0)
    d1 = _fake_dataset(im1, -90.0)
    dc = DriftCorrection.from_images(d0, d1)
    assert list(np.asarray(dc.scan_direction_degrees)) == [0.0, -90.0]


def test_from_emd_uses_metadata_angles_not_default(monkeypatch):
    im0, im1, _ = make_synthetic_drift_data()
    _patch_reader(monkeypatch, {"a.emd": _fake_dataset(im0, 17.0),
                                "b.emd": _fake_dataset(im1, -73.0)})
    dc = DriftCorrection.from_emd("a.emd", "b.emd", verbose=False)
    assert list(np.asarray(dc.scan_direction_degrees)) == [17.0, -73.0]


def test_from_emd_carries_pixel_size(monkeypatch):
    im0, im1, _ = make_synthetic_drift_data()
    _patch_reader(monkeypatch, {"a.emd": _fake_dataset(im0, 0.0, px_nm=0.023),
                                "b.emd": _fake_dataset(im1, -90.0, px_nm=0.023)})
    dc = DriftCorrection.from_emd("a.emd", "b.emd", verbose=False)
    assert np.isclose(float(dc.imgs[0].sampling[0]), 0.023)


def test_show_can_return_static_matplotlib_figure():
    dc, _ = _solved_pair()

    figure = dc.show(mode="static", zoom=2, cmap="gray")

    assert isinstance(figure, Figure)
    assert len(figure.axes) == 6
    assert [axis.get_title() for axis in figure.axes] == [
        "0deg scan",
        "-90deg scan -> 0deg frame",
        "combined scan",
        "corrected 0deg",
        "corrected -90deg",
        "corrected combined scan",
    ]
    assert figure.axes[0].get_xlim()[1] - figure.axes[0].get_xlim()[0] == pytest.approx(64)
    plt.close(figure)
