"""Inspect exact merged regions before saving the full encoded acquisition."""

import os

import pytest
import torch
from quantem.gpu import io

from quantem.diffraction import MAPEDTorch


@pytest.mark.parametrize("device", ["cuda:0", "mps"])
def test_inspect_regions_then_save_complete_merge(tmp_path, device):
    """Selected float32 regions match the full merge and preserve owned inputs."""
    if device == "cuda:0":
        if os.environ.get("QUANTEM_CUDA_ANS_TEST") != "1":
            pytest.skip("Set QUANTEM_CUDA_ANS_TEST=1 in an owned GPU0 test window.")
    elif not torch.backends.mps.is_available():
        pytest.skip("Run this case on a physical MPS accelerator.")
    backend = torch.device(device).type
    shape = (19, 19, 8, 8)
    indices_t = torch.arange(19 * 19 * 8 * 8, device=device).reshape(shape)
    files = []
    for index in range(7):
        path = tmp_path / f"tilt_{index}_master.h5"
        values_t = ((indices_t * 13 + index * 37) % 65536).to(torch.uint16)
        io.save(path, values_t, dtype="uint16", backend=backend, verbose=False, wait=True)
        files.append(path)
    input_files = set(tmp_path.glob("*.h5"))
    maped = MAPEDTorch.from_files(files, device=device)
    try:
        sources = list(maped.datasets.sources)
        maped.preprocess(plot_summary=False)
        maped.real_space_shifts = torch.tensor(
            [[0, 0], [-1.25, 0.6], [0.75, -1.4], [2.3, 1.1], [-2.1, -0.2], [19, 0], [-18, 1]],
            device=device,
        )
        maped.diffraction_shifts = torch.tensor(
            [[0, 0], [0.4, -0.7], [-0.25, 0.5], [0.1, 0.2], [1.2, -0.1], [0, 0], [0.3, -0.1]],
            device=device,
        )
        complete = maped.merge_datasets(plot_result=False)
        for region in ((0, 3, 0, 3), (15, 19, 15, 19), (8, 11, 8, 11), (2, 7, 10, 15)):
            patch = maped.merge_datasets(scan_region=region, plot_result=False)
            row0, row1, column0, column1 = region
            assert torch.equal(patch.read(), complete.read()[row0:row1, column0:column1])
            assert patch.metadata["maped_merge"]["scan_region"] == list(region)
            assert all(not source.data.is_released for source in sources)
        assert set(tmp_path.glob("*.h5")) == input_files
        saved = maped.merge_datasets(save_to=tmp_path / "merged_master.h5", plot_result=False)
        torch.testing.assert_close(
            saved.read(), complete.read(), rtol=0, atol=saved.metadata["precision"]["scale"]
        )
        assert all(source.data.is_released for source in sources)
    finally:
        maped.close()
