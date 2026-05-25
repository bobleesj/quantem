"""Tests for Dataset5dstem (quantem.core.datastructures.dataset5dstem)."""

import numpy as np
import pytest
import torch

from quantem.core.datastructures.dataset4dstem import Dataset4dstem
from quantem.core.datastructures.dataset5dstem import Dataset5dstem

_TWO_GPUS = torch.cuda.is_available() and torch.cuda.device_count() >= 2
_needs_2gpu = pytest.mark.skipif(not _TWO_GPUS, reason="needs >= 2 CUDA devices")


def _frame_on(device: str, n: int, *, scan=4, det=6) -> Dataset4dstem:
    """A small uint16 frame on a given device, content = arange + n (distinct per frame)."""
    t = ((torch.arange(scan * scan * det * det) + n)
         .reshape(scan, scan, det, det).to(torch.uint16)).to(device)
    return Dataset4dstem.from_tensor(t, sampling=(1, 1, 1, 1), name=f"f{n}")


def test_from_tensor():
    ds = Dataset5dstem.from_tensor(
        tensor=torch.rand(3, 5, 5, 8, 8),
        name="t",
        sampling=(0.5, 0.5, 0.1, 0.1),
        units=["nm", "nm", "1/nm", "1/nm"],
        series_type="time",
        series=[0.0, 2.0, 4.0],
    )
    assert ds.shape == (3, 5, 5, 8, 8)
    assert np.array_equal(ds.sampling, np.array([0.5, 0.5, 0.1, 0.1]))
    assert ds.units == ["nm", "nm", "1/nm", "1/nm"]
    assert ds.series_type == "time"
    assert np.array_equal(ds.series, np.array([0.0, 2.0, 4.0]))
    assert isinstance(ds[0], Dataset4dstem)


def test_slice():
    """A scientist slices a sub-stack - gets a smaller Dataset5dstem with series sliced."""
    ds = Dataset5dstem.from_tensor(
        tensor=torch.rand(5, 5, 5, 8, 8),
        series_type="time", series=[0.0, 1.0, 2.0, 3.0, 4.0],
    )
    sub = ds[1:4]
    assert isinstance(sub, Dataset5dstem)
    assert sub.shape == (3, 5, 5, 8, 8)
    assert np.array_equal(sub.series, np.array([1.0, 2.0, 3.0]))


def test_for_loop():
    """A scientist loops frame-by-frame - each yield is a Dataset4dstem."""
    ds = Dataset5dstem.from_tensor(tensor=torch.rand(3, 5, 5, 8, 8))
    seen = [f for f in ds]
    assert len(seen) == 3
    assert all(isinstance(f, Dataset4dstem) and f.shape == (5, 5, 8, 8) for f in seen)


def test_from_4dstem():
    d4_list = [
        Dataset4dstem.from_tensor(
            torch.rand(5, 5, 8, 8),
            sampling=(0.5, 0.5, 0.1, 0.1),
            units=("nm", "nm", "1/nm", "1/nm"),
            name=f"f{i}",
        )
        for i in range(3)
    ]
    ds = Dataset5dstem.from_4dstem(d4_list, series_type="tilt", series=[-30, 0, 30])
    assert ds.shape == (3, 5, 5, 8, 8)
    assert np.array_equal(ds.sampling, np.array([0.5, 0.5, 0.1, 0.1]))
    assert ds.units == ["nm", "nm", "1/nm", "1/nm"]
    assert ds.series_type == "tilt"
    assert np.array_equal(ds.series, np.array([-30.0, 0.0, 30.0]))
    assert ds.is_sharded is False  # all on one (cpu) device -> stacked single tensor


# --- multi-device (series of frames) ---


@_needs_2gpu
def test_from_4dstem_multidevice_routes_and_parity():
    """Frames on different cards -> series of frames; dataset[i] equals its source."""
    frames = [_frame_on(f"cuda:{i % 2}", i) for i in range(4)]
    ds = Dataset5dstem.from_4dstem(frames, series_type="tilt", series=[0, 1, 2, 3])
    assert ds.is_sharded is True
    assert len(ds) == 4
    assert ds.shape == (4, 4, 4, 6, 6)
    assert ds.devices == ["cuda:0", "cuda:1", "cuda:0", "cuda:1"]
    for i in range(4):
        got, src = ds[i].tensor, frames[i].tensor
        assert str(got.device) == f"cuda:{i % 2}"
        assert torch.equal(got.cpu(), src.cpu())


@_needs_2gpu
def test_multidevice_slice_keeps_frames():
    frames = [_frame_on(f"cuda:{i % 2}", i) for i in range(4)]
    ds = Dataset5dstem.from_4dstem(frames, series_type="tilt", series=[0, 1, 2, 3])
    sub = ds[1:3]
    assert isinstance(sub, Dataset5dstem)
    assert sub.shape == (2, 4, 4, 6, 6)
    assert sub.devices == ["cuda:1", "cuda:0"]
    assert np.array_equal(sub.series, np.array([1.0, 2.0]))


@_needs_2gpu
def test_single_device_4dstem_stacks_not_sharded():
    """All frames on one card -> compact single tensor, not a frame list."""
    frames = [_frame_on("cuda:0", i) for i in range(3)]
    ds = Dataset5dstem.from_4dstem(frames, series_type="time")
    assert ds.is_sharded is False
    assert ds.shape == (3, 4, 4, 6, 6)


@_needs_2gpu
def test_summary_per_device_totals():
    frames = [_frame_on(f"cuda:{i % 2}", i) for i in range(4)]
    ds = Dataset5dstem.from_4dstem(frames, series_type="tilt")
    per_device = ds.summary()
    assert set(per_device) == {"cuda:0", "cuda:1"}
    # 2 frames per card, equal size
    assert per_device["cuda:0"] == pytest.approx(per_device["cuda:1"])


@_needs_2gpu
def test_one_device_framelist_restacks_not_sharded():
    """A frame list that lands all on one device collapses to a single tensor
    (invariant: _frames backing ⟺ genuinely multi-device)."""
    frames = [_frame_on("cuda:0", i) for i in range(3)]
    ds = Dataset5dstem.from_4dstem(frames, series_type="tilt")
    assert ds.is_sharded is False
    # slicing a sharded dataset down to one device also collapses
    shd = Dataset5dstem.from_4dstem([_frame_on(f"cuda:{i % 2}", i) for i in range(4)],
                                    series_type="tilt")
    one_card = shd[0:4:2]  # frames 0,2 -> both cuda:0
    assert one_card.is_sharded is False
    assert one_card.shape == (2, 4, 4, 6, 6)


def test_empty_and_dtype_mismatch_raise():
    with pytest.raises(ValueError, match="at least one"):
        Dataset5dstem.from_4dstem([])
    a = Dataset4dstem.from_tensor(torch.zeros(4, 4, 6, 6, dtype=torch.uint16))
    b = Dataset4dstem.from_tensor(torch.zeros(4, 4, 6, 6, dtype=torch.uint32))
    with pytest.raises(ValueError, match="share dtype"):
        Dataset5dstem._from_frames([a.tensor.to("cpu"), b.tensor.to("cpu")],
                                   name="x", sampling=None, units=None, origin=None)


@_needs_2gpu
def test_numpy_gathers_full_series():
    """numpy() returns the whole 5D series (all frames, all cards), not just frame 0."""
    frames = [_frame_on(f"cuda:{i % 2}", i) for i in range(4)]
    ds = Dataset5dstem.from_4dstem(frames, series_type="tilt")
    arr = ds.numpy()
    assert arr.shape == (4, 4, 4, 6, 6)
    for i in range(4):
        assert np.array_equal(arr[i], frames[i].tensor.cpu().numpy())


def test_ndim_is_five():
    ds = Dataset5dstem.from_tensor(torch.zeros(3, 4, 4, 6, 6))
    assert ds.ndim == 5


def test_freed_dataset_errors_cleanly():
    ds = Dataset5dstem.from_tensor(torch.zeros(3, 4, 4, 6, 6))
    ds.free()
    with pytest.raises(RuntimeError, match="freed"):
        len(ds)
    with pytest.raises(RuntimeError, match="freed"):
        _ = ds.shape


@_needs_2gpu
def test_free_returns_vram():
    """free() drops frames and the CUDA allocator returns the memory."""
    # ~32 MiB per frame so the drop is clearly measurable.
    frames = [_frame_on(f"cuda:{i % 2}", i, scan=64, det=64) for i in range(4)]
    ds = Dataset5dstem.from_4dstem(frames, series_type="tilt")
    del frames  # drop external refs so free() is the only holder
    free_before, _ = torch.cuda.mem_get_info(0)
    ds.free()
    free_after, _ = torch.cuda.mem_get_info(0)
    assert ds._frames is None
    assert free_after >= free_before  # VRAM returned (>= guards against noise)
