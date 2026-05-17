"""Test widget.free() releases VRAM + RAM. Covers torch, cupy, timer paths."""
import gc
import sys

import numpy as np
import pytest
import torch

from quantem.widget import Show3D, Show3DVolume, Show4DSTEM


cuda_only = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)


@cuda_only
def test_show4dstem_free_releases_torch_vram():
    torch.cuda.empty_cache()
    baseline = torch.cuda.memory_allocated() / 1024**2
    arr = np.random.randint(0, 1000, size=(8, 8, 64, 64), dtype=np.uint16)
    w = Show4DSTEM(arr)
    assert torch.cuda.memory_allocated() / 1024**2 > baseline + 0.5
    w.free()
    gc.collect()
    torch.cuda.empty_cache()
    after_free = torch.cuda.memory_allocated() / 1024**2
    assert after_free < baseline + 16, f"leaked: {baseline:.1f} -> {after_free:.1f} MB"


@cuda_only
def test_show4dstem_free_clears_bytes_traits():
    arr = np.random.randint(0, 1000, size=(8, 8, 64, 64), dtype=np.uint16)
    w = Show4DSTEM(arr)
    w._compute_virtual_image_from_roi()
    w._update_frame()
    assert len(w.frame_bytes) > 0
    w.free()
    assert w.frame_bytes == b""
    assert w.virtual_image_bytes == b""
    assert w.vi_roi_dp_bytes == b""


@cuda_only
def test_del_alone_does_not_free():
    """sanity: del + gc + empty_cache leaks. traitlets observers pin the
    widget refcount, so del drops only one ref and data never collects."""
    torch.cuda.empty_cache()
    baseline = torch.cuda.memory_allocated() / 1024**2
    arr = np.random.randint(0, 1000, size=(8, 8, 64, 64), dtype=np.uint16)
    w = Show4DSTEM(arr)
    assert torch.cuda.memory_allocated() / 1024**2 > baseline + 0.5
    assert sys.getrefcount(w) - 1 > 1
    del w
    del arr
    gc.collect()
    torch.cuda.empty_cache()
    assert torch.cuda.memory_allocated() / 1024**2 > baseline + 0.5


@cuda_only
def test_show4dstem_back_to_back_no_leak():
    torch.cuda.empty_cache()
    baseline = torch.cuda.memory_allocated() / 1024**2
    arr = np.random.randint(0, 1000, size=(8, 8, 64, 64), dtype=np.uint16)
    w1 = Show4DSTEM(arr)
    w1._compute_virtual_image_from_roi()
    w1.free()
    torch.cuda.empty_cache()
    after1 = torch.cuda.memory_allocated() / 1024**2
    w2 = Show4DSTEM(arr)
    w2._compute_virtual_image_from_roi()
    w2.free()
    torch.cuda.empty_cache()
    after2 = torch.cuda.memory_allocated() / 1024**2
    assert abs(after2 - after1) < 1
    assert after1 < baseline + 16


@cuda_only
def test_show4dstem_free_cupy_pool():
    """When data is a torch view into cupy memory, free() must flush the
    cupy pool too. total_bytes() is what nvidia-smi sees, used_bytes() is
    what's still pinned."""
    cp = pytest.importorskip("cupy")
    pool = cp.get_default_memory_pool()
    pool.free_all_blocks()
    used_baseline = pool.used_bytes()
    total_baseline = pool.total_bytes()
    cp_arr = cp.zeros((8, 8, 64, 64), dtype=cp.uint16)
    assert pool.used_bytes() > used_baseline
    t = torch.as_tensor(cp_arr, device="cuda")
    w = Show4DSTEM(t)
    w._update_frame()
    del cp_arr, t
    w.free()
    gc.collect()
    torch.cuda.empty_cache()
    assert pool.used_bytes() <= used_baseline + 1024
    assert pool.total_bytes() <= total_baseline + 1024


@cuda_only
def test_show3d_free_releases_vram():
    torch.cuda.empty_cache()
    baseline = torch.cuda.memory_allocated() / 1024**2
    data = np.random.rand(32, 128, 128).astype(np.float32)
    w = Show3D(data, use_torch=True)
    assert torch.cuda.memory_allocated() / 1024**2 > baseline + 0.5
    w.free()
    gc.collect()
    torch.cuda.empty_cache()
    assert torch.cuda.memory_allocated() / 1024**2 < baseline + 16


def test_show3d_free_releases_ram_cpu_path():
    data = np.random.rand(32, 256, 256).astype(np.float32)
    w = Show3D(data, use_torch=False)
    assert w._data is not None
    w.free()
    assert w._data is None
    assert w._display_data is None
    assert w.frame_bytes == b""


def test_show3dvolume_free_releases_ram():
    data = np.random.rand(8, 64, 64).astype(np.float32)
    w = Show3DVolume(data)
    assert w._data is not None
    assert len(w.volume_bytes) > 0
    w.free()
    assert w._data is None
    assert w.volume_bytes == b""


def test_show3dvolume_free_releases_dual_mode_buffers():
    """Dual mode holds 2 volumes + 2 byte buffers; free() must clear all four
    plus the GIF/ZIP export buffers, otherwise tens of MB stay pinned."""
    a = np.random.rand(8, 64, 64).astype(np.float32)
    b = np.random.rand(8, 64, 64).astype(np.float32)
    w = Show3DVolume(a, data_b=b, dual_mode=True)
    assert w._data is not None and w._data_b is not None
    assert len(w.volume_bytes) > 0 and len(w.volume_bytes_b) > 0
    w.free()
    assert w._data is None
    assert w._data_b is None
    assert w.volume_bytes == b""
    assert w.volume_bytes_b == b""
    assert w._gif_data == b""
    assert w._zip_data == b""


def test_show3dvolume_slice_change_after_free_does_not_crash():
    """JS may send a slice trait update after the user clicks free() but before
    the comm message round-trips. Without a guard, _on_slice_change indexes a
    None _data and the kernel raises an unhelpful TypeError into the comm."""
    data = np.random.rand(8, 64, 64).astype(np.float32)
    w = Show3DVolume(data)
    w.free()
    w.slice_z = 3
    w.slice_y = 10
    w.show_stats = True


def test_show3d_roi_timer_canceled_on_free():
    """Pending ROI debounce timer must be canceled by free(), otherwise
    its callback fires 500ms later and crashes on nulled _display_data."""
    data = np.random.rand(16, 64, 64).astype(np.float32)
    w = Show3D(data, use_torch=False)
    w.roi_active = True
    w.roi_list = [{"shape": "square", "row": 5, "col": 5, "half_size": 2}]
    w.roi_selected_idx = 0
    w._on_roi_change()
    assert w._roi_plot_timer is not None
    w.free()
    assert w._roi_plot_timer is None
