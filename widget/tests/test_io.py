"""Smoke tests for `load()` covering real scientist usage:

1. Load a folder of EMD files → ready-to-display Dataset3d
2. Mixed-resolution folder + size filter → only-the-bucket-I-want
3. Numpy array passthrough → wrap existing in-memory stack
4. Velox metadata auto-extracted onto Dataset3d.sampling + units

Disk fixtures write minimal Velox-EMD-shaped HDF5 so tests don't depend
on the user's local data dirs."""

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from quantem.core.datastructures import Dataset3d
from quantem.widget import Show3D, load


def _write_velox_emd(path: Path, frame: np.ndarray, pixel_size_m: float = 1.86e-11) -> None:
    """Write a minimal Velox EMD with PixelSize + Optics so the loader can
    surface Å/px sampling + voltage from real-shaped metadata."""
    with h5py.File(path, "w") as fp:
        grp = fp.create_group("Data/Image/abcdef")
        grp.create_dataset("Data", data=frame[..., None])  # Velox trailing 1-axis
        meta = {
            "BinaryResult": {"PixelSize": {"width": f"{pixel_size_m:.6e}"}, "DetectorName": "HAADF"},
            "Optics": {"AccelerationVoltage": 300000},
        }
        meta_bytes = json.dumps(meta).encode("utf-8")
        meta_arr = np.zeros((60000, 1), dtype=np.uint8)
        meta_arr[: len(meta_bytes), 0] = np.frombuffer(meta_bytes, dtype=np.uint8)
        grp.create_dataset("Metadata", data=meta_arr)


@pytest.fixture
def emd_folder(tmp_path):
    """Three 256² Velox EMD frames in one folder — the common single-mag case."""
    d = tmp_path / "scan"
    d.mkdir()
    rng = np.random.default_rng(0)
    for i in range(3):
        _write_velox_emd(d / f"frame_{i:03d}.emd", rng.integers(0, 65535, (256, 256), dtype=np.uint16))
    return d


@pytest.fixture
def mixed_folder(tmp_path):
    """Mixed-magnification folder: 2 frames at 256², 1 at 128² — exercises size filter."""
    d = tmp_path / "mixed"
    d.mkdir()
    rng = np.random.default_rng(1)
    _write_velox_emd(d / "lowmag.emd", rng.integers(0, 65535, (128, 128), dtype=np.uint16))
    _write_velox_emd(d / "hi_0.emd", rng.integers(0, 65535, (256, 256), dtype=np.uint16))
    _write_velox_emd(d / "hi_1.emd", rng.integers(0, 65535, (256, 256), dtype=np.uint16))
    return d


def test_load_folder_for_show3d(emd_folder):
    """The common workflow: point load() at a folder, hand the result to Show3D."""
    images = load(emd_folder, verbose=False)
    assert isinstance(images, Dataset3d)
    assert images.array.shape == (3, 256, 256)
    # Calibrated sampling (Å/px) auto-extracted from Velox PixelSize metadata.
    assert images.sampling[1] == pytest.approx(0.186, abs=1e-3)
    assert list(images.units) == ["index", "Å", "Å"]
    # Per-frame metadata (voltage, detector, etc) is accessible without parsing
    # raw Velox JSON.
    assert images.frame_metadata[0]["voltage_kV"] == pytest.approx(300.0)
    assert images.frame_metadata[0]["detector"] == "HAADF"
    # Show3D auto-extracts everything from the Dataset3d.
    Show3D(images)


def test_load_mixed_folder_with_size_filter(mixed_folder):
    """Filter mixed-magnification folder to one resolution."""
    images = load(mixed_folder, size=256, verbose=False)
    assert isinstance(images, Dataset3d)
    assert images.array.shape == (2, 256, 256)


def test_load_numpy_array_for_show3d():
    """When the scientist already has a stack in memory (denoise output,
    synthetic phantom, etc.) - wrap it for the widget without touching disk."""
    arr = np.random.default_rng(2).random((5, 64, 64), dtype=np.float32)
    images = load(arr, verbose=False)
    assert isinstance(images, Dataset3d)
    assert images.array.shape == (5, 64, 64)
    Show3D(images)
