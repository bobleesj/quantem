"""Compare the multislice tutorial's EMPAD reader with quantem.gpu.

Set QUANTEM_MULTISLICE_RAW to the tutorial's scan_x128_y128.raw and run
``pytest tests/io/test_empad_gpu_parity.py --runslow -s``. This optional test
requires quantem.gpu and CUDA; it checks all measurements without rounding.
"""

import os
from pathlib import Path
from time import perf_counter

import numpy as np
import pytest


@pytest.mark.slow
def test_multislice_raw_matches_quantem_gpu(record_property):
    """Both readers preserve every float32 bit in tutorial detector orientation."""
    source = os.environ.get("QUANTEM_MULTISLICE_RAW")
    if not source:
        pytest.skip("Set QUANTEM_MULTISLICE_RAW to the multislice tutorial RAW file.")
    path = Path(source).expanduser()
    assert path.is_file(), f"RAW file not found: {path}. Set QUANTEM_MULTISLICE_RAW."
    io = pytest.importorskip("quantem.gpu.io")
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("This loader parity test requires CUDA.")

    # Match load_raw and the explicit detector transpose in tutorial cell 5.
    scan_shape = (128, 128)
    detector_shape = (128, 128)
    record_dtype = np.dtype([("data", np.float32, detector_shape), ("gap", np.uint8, 1024)])
    frame_count = int(np.prod(scan_shape))
    assert path.stat().st_size == frame_count * record_dtype.itemsize
    started = perf_counter()
    records = np.fromfile(path, dtype=record_dtype, count=frame_count)
    reference = records["data"].reshape(*scan_shape, *detector_shape)
    reference = reference.transpose(0, 1, 3, 2)
    reference_seconds = perf_counter() - started

    from quantem.core.datastructures import Dataset4dstem

    torch.cuda.synchronize()
    started = perf_counter()
    with io.load(
        path,
        scan_shape=scan_shape,
        backend="cuda",
        hot_pixel_correction="none",
        verbose=False,
    ) as loaded:
        torch.cuda.synchronize()
        resident_seconds = perf_counter() - started
        # Keep the acquisition compressed; decode at most 512 frames per read.
        decode_seconds = 0.0
        for first_row in range(0, scan_shape[0], 4):
            stop_row = first_row + 4
            started = perf_counter()
            block_t = loaded.read(scan_region=(first_row, stop_row, 0, scan_shape[1]))
            block_t = block_t.transpose(2, 3)  # Tutorial detector orientation.
            dset = Dataset4dstem.from_tensor(
                block_t,
                sampling=(0.4290, 0.4290, 1, 1),
                units=("A", "A", "pixels", "pixels"),
            )
            torch.cuda.synchronize()
            decode_seconds += perf_counter() - started
            assert dset.tensor is block_t
            assert dset.tensor.is_cuda
            assert dset.tensor.dtype == torch.float32
            # Only the parity assertion needs a host copy.
            restored = dset.tensor.cpu().numpy()
            np.testing.assert_array_equal(
                restored.view(np.uint32),
                reference[first_row:stop_row].view(np.uint32),
                err_msg=f"Detector sample bits differ at scan rows {first_row}:{stop_row}.",
            )

    for name, value in {
        "verified_float32_values": reference.size,
        "tutorial_raw_seconds": reference_seconds,
        "gpu_resident_load_seconds": resident_seconds,
        "gpu_decode_all_blocks_seconds": decode_seconds,
    }.items():
        record_property(name, value)
        print(f"{name}: {value}")


@pytest.mark.slow
def test_multislice_qem_entry_point_preserves_raw_bits():
    """The notebook's QEM entry point preserves the raw detector orientation and bits."""
    source = os.environ.get("QUANTEM_MULTISLICE_RAW")
    cache = os.environ.get("QUANTEM_MULTISLICE_QEM")
    if not source or not cache:
        pytest.skip("Set QUANTEM_MULTISLICE_RAW and QUANTEM_MULTISLICE_QEM for cache parity.")
    pytest.importorskip("quantem.gpu.io")
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("QEM entry-point parity requires CUDA.")
    from quantem.core import config
    from quantem.core.datastructures import Dataset4dstem

    config.set_device(0)
    dset = Dataset4dstem.from_qem(
        cache,
        sampling=(0.429, 0.429, 1, 1),
        units=("A", "A", "pixels", "pixels"),
        transpose_detector=True,
    )
    records = np.memmap(
        Path(source).expanduser(),
        mode="r",
        dtype=np.dtype([("data", np.float32, (128, 128)), ("gap", np.uint8, 1024)]),
        shape=(16384,),
    )
    assert dset.tensor.is_cuda
    # Frozen tutorial calibration; compare the original mask arithmetic directly.
    center = (63.97277653629759, 63.144974121061814)
    radius = 11.97339156710511
    detector_row, detector_col = np.ogrid[:128, :128]
    distance = np.sqrt((detector_row - center[0]) ** 2 + (detector_col - center[1]) ** 2)
    masks = (distance <= radius + 2, (distance >= radius + 10) & (distance <= radius * 5))
    images = (
        dset.get_virtual_image(mode="circle", geometry=(center, radius + 2), show=False),
        dset.get_virtual_image(
            mode="annular", geometry=(center, (radius + 10, radius * 5)), show=False
        ),
    )
    for row in range(0, 128, 4):
        expected = (
            records["data"][row * 128 : (row + 4) * 128]
            .reshape(4, 128, 128, 128)
            .transpose(0, 1, 3, 2)
        )
        actual = dset.tensor[row : row + 4].cpu().numpy()
        np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
        for image, mask in zip(images, masks):
            np.testing.assert_allclose(
                image.tensor[row : row + 4].cpu(),
                (expected * mask).sum(axis=(-2, -1)),
                rtol=1e-6,
                atol=1e-5,
            )
