"""Tests for the multi-GPU sharded 5D-STEM series + MAPED consuming it.

``Dataset5dstem`` keeps a tilt/time series distributed across GPUs so the whole
series stays resident (for viewing) and feeds MAPED one frame at a time. The
merge moves each frame to the compute device, so a series sharded across N GPUs
must produce a bit-for-bit identical merge to the same series on one GPU.

1. ``test_dataset5dstem_placement`` - the container's placement + introspection,
   on tiny synthetic tensors. Runs anywhere (1 GPU or CPU).
2. ``test_sharded_merge_bitexact_two_gpu`` (slow) - real 4D-STEM tilt series
   sharded across 2 GPUs, full MAPED pipeline, merge bit-exact vs single GPU.
"""

import os

import pytest
import torch

# Set MAPED_TEST_DIR (a 4D-STEM tilt-series dir of *_master.h5 files) to run the
# slow data tests; they skip when it is unset. MAPED_TEST_PREFIX is the master
# filename prefix within that dir.
MAPED_TEST_DIR = os.environ.get("MAPED_TEST_DIR", "")
MAPED_TEST_PREFIX = os.environ.get("MAPED_TEST_PREFIX", "")


def test_dataset5dstem_placement():
    """Frames distribute across the requested devices; introspection is correct."""
    from quantem.diffraction.dataset5dstem import Dataset5dstem

    frames = [torch.zeros((4, 4, 8, 8), dtype=torch.uint16) for _ in range(5)]
    n_gpu = torch.cuda.device_count()

    # single device: every frame lands there, not sharded
    one = Dataset5dstem.from_tensors(frames, devices=("cpu" if n_gpu == 0 else 0))
    assert len(one) == 5
    assert one.is_sharded is False
    assert one.dtype == torch.uint16
    assert one.frame_shape == (4, 4, 8, 8)
    assert len(one.bytes_per_device()) == 1

    if n_gpu >= 2:
        # round-robin across two GPUs -> genuinely sharded
        two = Dataset5dstem.from_tensors(frames, devices=[0, 1], series_type="tilt")
        assert two.is_sharded is True
        assert two.devices == ["cuda:0", "cuda:1", "cuda:0", "cuda:1", "cuda:0"]
        assert set(two.bytes_per_device()) == {"cuda:0", "cuda:1"}
        # series defaults to frame index
        assert two.series == [0, 1, 2, 3, 4]

    # shape mismatch is rejected
    with pytest.raises(ValueError):
        Dataset5dstem([torch.zeros((4, 4, 8, 8)), torch.zeros((4, 4, 8, 9))])


def _two_gpu_data_available():
    if torch.cuda.device_count() < 2:
        return False
    if not os.path.isdir(MAPED_TEST_DIR):
        return False
    try:
        import quantem.widget  # noqa: F401
    except ImportError:
        return False
    masters = [
        f
        for f in os.listdir(MAPED_TEST_DIR)
        if f.endswith("master.h5") and f.startswith(MAPED_TEST_PREFIX)
    ]
    return len(masters) >= 2


def _scan_bin(t, fac):
    """Device-agnostic scan-bin (sum) over fac x fac blocks, accumulating in int32."""
    r, c, h, w = t.shape
    v = t.reshape(r // fac, fac, c // fac, fac, h, w)
    out = v[:, 0, :, 0].to(torch.int32).clone()
    for i in range(fac):
        for j in range(fac):
            if i == 0 and j == 0:
                continue
            out += v[:, i, :, j].to(torch.int32)
    return out.to(torch.uint16)


def _freest_gpu():
    free = [torch.cuda.mem_get_info(i)[0] for i in range(torch.cuda.device_count())]
    return int(max(range(len(free)), key=lambda i: free[i]))


def _load_test_tilts(fac):
    """Load the tilt series (GPU LZ4) + scan-bin, returned on CPU.

    Loads on the GPU with the most free memory (the box may have other work on
    GPU0), bins on GPU, then parks on CPU so placement is up to the caller.
    """
    import cupy as cp

    from quantem.widget import load as wload

    cp.cuda.Device(_freest_gpu()).use()
    files = sorted(
        f
        for f in os.listdir(MAPED_TEST_DIR)
        if f.endswith("master.h5") and f.startswith(MAPED_TEST_PREFIX)
    )
    tilts = []
    for f in files:
        raw = wload(os.path.join(MAPED_TEST_DIR, f)).data  # cupy uint16
        tilts.append(_scan_bin(torch.from_dlpack(raw), fac).to("cpu"))
        del raw
        cp.get_default_memory_pool().free_all_blocks()
        torch.cuda.empty_cache()
    return tilts


def _run_maped(frames):
    from quantem.diffraction import MAPEDTorch

    m = MAPEDTorch.from_datasets(frames)
    m.preprocess(plot_summary=False)
    m.diffraction_origin(vmax=1000, sigma=1, plot_origins=False)
    m.diffraction_align(edge_blend=2, vmax=5000, plot_aligned=False)
    m.real_space_align(
        num_iter=20,
        hanning_filter=True,
        padding=2,
        edge_blend=5,
        pad_val="median",
        shift_method="bilinear",
        plot_aligned=False,
    )
    return m.merge_datasets(
        real_space_edge_blend=5, shift_method="fourier", batch_size=64, plot_result=False
    ).tensor


@pytest.mark.slow
@pytest.mark.skipif(
    not _two_gpu_data_available(), reason="needs 2 CUDA GPUs + quantem.widget + data"
)
def test_sharded_merge_bitexact_two_gpu():
    """Real tilt series sharded across 2 GPUs merges bit-exact vs 1 GPU.

    The 7 tilts stay resident, split across both cards (Dataset5dstem). MAPED
    moves each frame to the compute device on demand. The merged result must be
    bit-for-bit identical to running the same tilts all on one GPU - the sharding
    is a storage layout, not a numerical change.
    """
    from quantem.core.config import set_device
    from quantem.diffraction.dataset5dstem import Dataset5dstem

    tilts = _load_test_tilts(fac=4)  # uint16, on CPU
    assert len(tilts) >= 2

    set_device(torch.device("cuda:1"))
    sharded = Dataset5dstem.from_tensors(tilts, devices=[0, 1], series_type="tilt")
    assert sharded.is_sharded
    assert set(sharded.bytes_per_device()) == {"cuda:0", "cuda:1"}

    merged_sharded = _run_maped(sharded.to_list())
    merged_single = _run_maped([t.to("cuda:1") for t in tilts])

    assert merged_sharded.shape == merged_single.shape
    assert torch.equal(merged_sharded.cpu(), merged_single.cpu()), (
        "2-GPU sharded merge differs from single-GPU"
    )
