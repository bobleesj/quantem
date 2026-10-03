"""Calibrated tensor analysis and bounded acquisition storage workflows."""

import numpy as np
import pytest
import torch

from quantem.core.datastructures import Dataset2d, Dataset4dstem
from quantem.core.io import load


def test_calibrated_tensor_selection_and_independent_copy():
    values_t = torch.arange(3 * 4 * 6 * 8).reshape(3, 4, 6, 8).to(torch.uint16)
    validity = np.ones((6, 8), dtype=bool)
    validity[3, 4] = False
    data = Dataset4dstem.from_tensor(
        values_t,
        name="calibrated counts",
        origin=(2, 3, -0.3, -0.4),
        sampling=(0.5, 0.6, 0.1, 0.2),
        units=["nm", "nm", "1/nm", "1/nm"],
        signal_units="electrons",
        metadata={"valid_pixels": validity, "acquisition": {"voltage_kV": 200}},
    )
    region = data[1:, 1:4:2, 1:6:2, 2:8:2]
    np.testing.assert_array_equal(region.numpy(), values_t[1:, 1:4:2, 1:6:2, 2:8:2])
    np.testing.assert_allclose(region.origin, (2.5, 3.6, -0.2, 0))
    np.testing.assert_allclose(region.sampling, (0.5, 1.2, 0.2, 0.4))
    np.testing.assert_array_equal(region.metadata["valid_pixels"], validity[1:6:2, 2:8:2])
    assert region.metadata["detector_shape"] == (3, 3)
    assert region.metadata["representation"] == "dense"
    assert region.residency == "host"
    assert repr(region).startswith("Dataset4dstem(")

    pattern = region[0, 1]
    assert isinstance(pattern, Dataset2d)
    np.testing.assert_allclose(pattern.origin, region.origin[2:])
    np.testing.assert_allclose(pattern.sampling, region.sampling[2:])
    clone = region.copy(copy_custom_attributes=False)
    clone.tensor[0, 0, 0, 0] = 60000
    clone.metadata["acquisition"]["voltage_kV"] = 300
    assert region.tensor[0, 0, 0, 0] != 60000
    assert region.metadata["acquisition"]["voltage_kV"] == 200
    np.testing.assert_array_equal(clone.sampling, region.sampling)
    assert clone.signal_units == "electrons"


@pytest.mark.parametrize("dtype", [torch.uint16, torch.float32])
def test_tensor_diffraction_statistics_and_virtual_images(dtype):
    values_t = torch.arange(6 * 5 * 7).reshape(2, 3, 5, 7).to(dtype)
    if dtype == torch.float32:
        values_t = values_t / 8 - 3
    data = Dataset4dstem.from_tensor(
        values_t,
        origin=(2, 3, -0.2, -0.3),
        sampling=(0.5, 0.6, 0.1, 0.1),
        units=["nm", "nm", "1/nm", "1/nm"],
        signal_units="electrons",
    )
    expected = values_t.numpy()
    np.testing.assert_allclose(data.dp_mean.numpy(), expected.mean(axis=(0, 1)))
    np.testing.assert_array_equal(data.dp_max.numpy(), expected.max(axis=(0, 1)))
    np.testing.assert_array_equal(data.min((0, 1)), expected.min(axis=(0, 1)))
    np.testing.assert_allclose(data.dp_median.numpy(), np.median(expected, axis=(0, 1)))

    rows, columns = np.ogrid[:5, :7]
    distance = np.sqrt((rows - 2) ** 2 + (columns - 3) ** 2)
    for mode, geometry, mask in (
        ("circle", ((2, 3), 2), distance <= 2),
        ("annular", ((2, 3), (1, 2)), (distance >= 1) & (distance <= 2)),
    ):
        image = data.get_virtual_image(mode=mode, geometry=geometry, name=mode)
        assert image.tensor.device == values_t.device
        np.testing.assert_allclose(image.numpy(), (expected * mask).sum(axis=(-1, -2)))
        np.testing.assert_allclose(image.sampling, data.sampling[:2])
        np.testing.assert_allclose(image.origin, data.origin[:2])
    weights = np.linspace(0.25, 1.5, 35).reshape(5, 7)
    weighted = data.get_virtual_image(mask=weights, attach=False)
    np.testing.assert_allclose(weighted.numpy(), (expected * weights).sum(axis=(-1, -2)))


class _BoundedStorage:
    """Keep test measurements accessible only through an explicit selection."""

    representation = "encoded"
    residency = "device"
    device = "cuda:1"

    def __init__(self):
        self.values_t = torch.arange(2 * 3 * 5 * 7).reshape(2, 3, 5, 7)
        self.shape = tuple(self.values_t.shape)
        self.dtype = np.dtype("int64")
        self.selections = []
        self.closed = False

    @property
    def data(self):
        raise AssertionError("This workflow must not materialize the acquisition")

    def __getitem__(self, key):
        self.selections.append(key)
        return self.values_t[key].clone()

    def close(self):
        self.closed = True

    def __exit__(self, *args):
        self.close()


def test_encoded_selection_lifetime_and_explicit_save(tmp_path):
    storage = _BoundedStorage()
    with Dataset4dstem(
        storage=storage,
        sampling=(0.5, 0.6, 0.1, 0.1),
        units=["nm", "nm", "1/nm", "1/nm"],
    ) as data:
        assert repr(data).startswith("Dataset4dstem(")
        assert data.device == "cuda:1"
        pattern = data[1, 2]
        assert len(storage.selections) == 1
        with pytest.raises(TypeError, match="quantem.gpu.io.save"):
            data.save(tmp_path / "encoded.zip")
        assert not (tmp_path / "encoded.zip").exists()
        with pytest.raises(NotImplementedError, match="Select a region first"):
            data.get_dp_median()
        with pytest.raises(NotImplementedError, match="NumPy-backed"):
            data.bin(bin_factors=2)
        with pytest.raises(TypeError, match="bounded region"):
            np.asarray(data)
    assert storage.closed
    np.testing.assert_array_equal(pattern.numpy(), storage.values_t[1, 2])
    np.testing.assert_allclose(pattern.sampling, (0.1, 0.1))
    path = tmp_path / "pattern.zip"
    pattern.save(path)
    restored = load(path)
    np.testing.assert_array_equal(restored.numpy(), pattern.numpy())
    np.testing.assert_array_equal(restored.sampling, pattern.sampling)


def test_encoded_virtual_image_uses_public_detector_reduction(monkeypatch):
    detector = pytest.importorskip("quantem.gpu.detector")
    storage = _BoundedStorage()
    data = Dataset4dstem(storage=storage, sampling=(0.5, 0.6, 0.1, 0.1))
    calls = []

    def masked_sum(source, mask):
        calls.append((source, mask))
        return (storage.values_t.numpy() * mask).sum(axis=(-1, -2)).astype(np.float32)

    monkeypatch.setattr(detector, "masked_sum", masked_sum)
    image = data.get_virtual_image(mode="annular", geometry=((2, 3), (1, 2)))
    assert calls[0][0] is data
    assert storage.selections == []
    np.testing.assert_allclose(image.sampling, (0.5, 0.6))
    np.testing.assert_array_equal(
        image.numpy(), (storage.values_t.numpy() * calls[0][1]).sum(axis=(-1, -2))
    )
