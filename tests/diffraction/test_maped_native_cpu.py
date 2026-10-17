"""Load, align, merge and save experimental-format inputs with QuantEM alone."""

import builtins

import h5py
import numpy as np
import torch

import quantem.diffraction.maped as maped_module
from quantem.core.datastructures import Dataset4dstem
from quantem.core.io import load
from quantem.diffraction import MAPED


def test_native_cpu_load_align_merge_save(tmp_path, monkeypatch):
    """Native Arina inputs recover known shifts without optional GPU packages."""
    original_import = builtins.__import__

    def core_only(name, *args, **kwargs):
        if name.startswith(("quantem.gpu", "quantem.widget")):
            raise AssertionError(f"CPU MAPED imported an optional package: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", core_only)
    scan_row, scan_col = np.indices((32, 32))
    detector_row, detector_col = np.indices((16, 16))
    displacements = [((0, 0), (0, 0)), ((2, -1), (3, 1)), ((-1, 2), (-2, -3))]
    tilts = []
    for index, (beam, scan) in enumerate(displacements):
        row, col = scan_row - scan[0], scan_col - scan[1]
        sample = (
            1
            + np.exp(-((row - 16) ** 2 + (col - 16) ** 2) / 18)
            + 0.5 * np.exp(-((row - 8) ** 2 + (col - 22) ** 2) / 8)
        )
        disk = np.exp(-((detector_row - 8 - beam[0]) ** 2 + (detector_col - 8 - beam[1]) ** 2) / 8)
        counts = np.rint(200 * sample[..., None, None] * disk).astype(np.uint16)
        counts[..., 0, 0] = 65535
        mask = np.zeros((16, 16), dtype=np.uint32)
        mask[0, 0] = 1
        path = tmp_path / f"tilt_{index}_-17.0x_0.0y_master.h5"
        with h5py.File(path, "w") as file:
            file["entry/data/data_000001"] = counts.reshape(-1, 16, 16)
            file["entry/instrument/detector/detectorSpecific/pixel_mask"] = mask
        tilt = Dataset4dstem.from_file(path, "arina", apply_mask=True)
        expected = counts.copy()
        expected[..., 0, 0] = 0
        np.testing.assert_array_equal(tilt.array, expected)
        assert tilt.metadata["hot_pixel_correction"]["pixel_count"] == 1
        tilts.append(tilt)

    maped = MAPED.from_datasets(tilts)
    merged = maped.run(show=False, verbose=False)
    torch.testing.assert_close(
        maped.diffraction_shifts - maped.diffraction_shifts[0],
        -torch.tensor([beam for beam, _ in displacements], dtype=torch.float32),
        rtol=0,
        atol=0.07,
    )
    torch.testing.assert_close(
        maped.real_space_shifts - maped.real_space_shifts[0],
        -torch.tensor([scan for _, scan in displacements], dtype=torch.float32),
        rtol=0,
        atol=0.07,
    )
    assert maped.tilt_names == ["-17.0x 0.0y"] * 3
    assert isinstance(merged, Dataset4dstem)
    assert merged.array.dtype == np.float32
    assert np.isfinite(merged.array).all()
    row0, _, col0, _ = merged.metadata["maped_merge"]["scan_region"]
    np.testing.assert_array_equal(merged.origin, [row0, col0, 0, 0])
    output = tmp_path / "merged_cpu.zip"
    merged.save(output)
    reopened = load(output)
    np.testing.assert_array_equal(reopened.array, merged.array)
    assert reopened.metadata["maped_merge"] == merged.metadata["maped_merge"]

    # Inspect actual diffraction patterns at two matching specimen positions.
    plotted = []
    show_2d = maped_module.show_2d

    def capture_panels(arrays, **kwargs):
        plotted.extend(arrays)
        return show_2d(arrays, **kwargs)

    monkeypatch.setattr(maped_module, "show_2d", capture_panels)
    figure = maped.show(positions=[(8, 9), (16, 17)], tilt_indices=[0])
    np.testing.assert_array_equal(plotted[-1][1], merged.array[8, 9])
    np.testing.assert_array_equal(plotted[-1][2], merged.array[16, 17])
    assert not np.array_equal(plotted[-1][1], plotted[-1][2])
    assert len(figure.axes) == 6
    np.testing.assert_allclose(
        [figure.axes[0].lines[0].get_ydata()[0], figure.axes[0].lines[0].get_xdata()[0]],
        np.array([8 + row0, 9 + col0]) - maped.real_space_shifts[0].numpy(),
    )


def test_native_arina_counts_and_optional_calibration(tmp_path):
    """Rectangular scans retain uint32 counts with or without microscope metadata."""
    path = tmp_path / "scan_master.h5"
    counts = np.arange(4 * 6 * 8 * 8, dtype=np.uint32).reshape(24, 8, 8) + 70000
    with h5py.File(path, "w") as file:
        file.attrs["scan_shape"] = (4, 6)
        file["entry/data/data_000001"] = counts[:12]
        file["entry/data/data_000002"] = counts[12:]
    pixels = Dataset4dstem.from_file(
        path, "arina", navigation_shape=(4, 6), dtype=np.uint32
    )
    np.testing.assert_array_equal(pixels.array, counts.reshape(4, 6, 8, 8))
    assert pixels.array.dtype == np.uint32
    np.testing.assert_array_equal(pixels.sampling, [1, 1, 1, 1])

    with h5py.File(tmp_path / "scan.h5", "w") as file:
        file.create_group("STEM Metadata").attrs["Pixel Size"] = [0.25]
    calibrated = Dataset4dstem.from_file(
        path, "arina", navigation_shape=(4, 6), dtype=np.uint32
    )
    np.testing.assert_array_equal(calibrated.array, pixels.array)
    np.testing.assert_array_equal(calibrated.sampling, [2.5, 2.5, 1, 1])
    assert calibrated.units[:2] == ["A", "A"]


def test_native_arina_binning_and_flatfield(tmp_path):
    """Binning retains fractional intensities; flatfield zeros remain unscaled."""
    path = tmp_path / "scan_master.h5"
    counts = np.broadcast_to(np.arange(16, dtype=np.uint16).reshape(4, 4), (4, 4, 4))
    with h5py.File(path, "w") as file:
        file["entry/data/data_000001"] = counts
    binned = Dataset4dstem.from_file(path, "arina", rebin_diffraction=2, dtype=np.float32)
    np.testing.assert_array_equal(binned.array[0, 0], [[2.5, 4.5], [10.5, 12.5]])

    with h5py.File(path, "r+") as file:
        file["entry/data/data_000001"][:] = 16
    flatfield = np.full((4, 4), 2.0)
    flatfield[:, 0] = 4.0
    flatfield[0, 0] = 0
    corrected = Dataset4dstem.from_file(
        path, "arina", rebin_diffraction=2, flatfield=flatfield, dtype=np.uint16
    )
    assert corrected.array.dtype == np.float32
    np.testing.assert_array_equal(corrected.array[0, 0], [[14, 16], [12, 16]])
