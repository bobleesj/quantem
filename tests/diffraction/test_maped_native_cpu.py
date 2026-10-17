"""Load, align, merge and save experimental-format inputs with QuantEM alone."""

import builtins

import h5py
import numpy as np
import torch

import quantem.diffraction.maped._maped as maped_module
from quantem.core.datastructures import Dataset4dstem
from quantem.core.io import load
from quantem.diffraction import MAPED
from quantem.diffraction.maped import io


def test_native_cpu_load_align_merge_save(tmp_path, monkeypatch):
    """Native Arina inputs recover and report shifts with QuantEM's dependencies."""
    original_import = builtins.__import__

    def core_only(name, *args, **kwargs):
        if name.startswith(("quantem.gpu", "quantem.widget", "pandas")):
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
        original = Dataset4dstem.from_file(path, "arina")
        np.testing.assert_array_equal(original.array, counts)
        unmasked = io.load(path, "arina")
        np.testing.assert_array_equal(unmasked.array, original.array)
        tilt = io.load(path, "arina", apply_mask=True)
        expected = counts.copy()
        expected[..., 0, 0] = 0
        np.testing.assert_array_equal(tilt.array, expected)
        assert tilt.metadata["hot_pixel_correction"]["pixel_count"] == 1
        with h5py.File(path) as file:
            np.testing.assert_array_equal(
                file["entry/data/data_000001"][()], counts.reshape(-1, 16, 16)
            )
        tilts.append(tilt)

    maped = MAPED.from_datasets(tilts)
    merged = maped.run(show=False, verbose=False)
    report = maped.report()
    assert report["Tilt"] == maped.tilt_names
    for label, shifts in (
        ("Detector", maped.diffraction_shifts),
        ("Scan", maped.real_space_shifts),
    ):
        assert report[f"{label} row (px)"] == shifts[:, 0].tolist()
        assert report[f"{label} col (px)"] == shifts[:, 1].tolist()
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
    np.testing.assert_allclose(
        [figure.axes[0].lines[0].get_ydata()[0], figure.axes[0].lines[0].get_xdata()[0]],
        np.array([8 + row0, 9 + col0]) - maped.real_space_shifts[0].numpy(),
    )
