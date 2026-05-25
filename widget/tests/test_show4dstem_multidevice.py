"""Show4DSTEM on a multi-device frame series (frames spread across GPUs).

Guards the device-mismatch class of bug: the detector mask + coordinate tensors
live on the widget's anchor device, but a frame may live on another card, so
every virtual-image reduction (BF/ADF/ROI/VI-ROI/auto-center) must follow the
frame's device. Needs >= 2 CUDA devices; skips otherwise.
"""

import numpy as np
import pytest
import torch

from quantem.widget import Show4DSTEM

_TWO_GPUS = torch.cuda.is_available() and torch.cuda.device_count() >= 2
pytestmark = pytest.mark.skipif(not _TWO_GPUS, reason="needs >= 2 CUDA devices")


def _frames(n=4, scan=6, det=12):
    """n frames, content distinct per frame, alternating cuda:0 / cuda:1."""
    return [
        ((torch.arange(scan * scan * det * det) + i * 50)
         .reshape(scan, scan, det, det).to(torch.uint16)).to(f"cuda:{i % 2}")
        for i in range(n)
    ]


def test_list_of_tensors_builds_sharded():
    w = Show4DSTEM(_frames(), frame_dim_label="tilt", verbose=False)
    assert w._sharded is True
    assert w.n_frames == 4


def test_every_vi_path_across_both_cards():
    """No device-mismatch on any virtual-image path, for frames on either card."""
    w = Show4DSTEM(_frames(), frame_dim_label="tilt", verbose=False)
    for fi in range(4):
        w.frame_idx = fi
        assert str(w._frame_data.device) == f"cuda:{fi % 2}"
        w._update_frame()  # CBED pull
        w._fast_masked_sum(w._create_circular_mask(w.center_col, w.center_row, w.bf_radius))  # BF
        w._fast_masked_sum(w._create_annular_mask(w.center_col, w.center_row,
                                                  w.bf_radius, w.bf_radius * 2))  # ADF
        w.roi_mode, w.roi_radius = "circle", 3.0
        w._compute_virtual_image_from_roi()
        w.vi_roi_mode, w.vi_roi_radius = "circle", 2.0
        w._compute_vi_roi_dp()
        w.auto_detect_center(update_roi=False)


def test_vi_values_match_single_device():
    """BF/ADF on the sharded series equal the same frames stacked on one card."""
    base = [(torch.arange(6 * 6 * 12 * 12) + i * 50).reshape(6, 6, 12, 12).to(torch.uint16)
            for i in range(4)]
    ref = Show4DSTEM(torch.stack([b.to("cuda:0") for b in base], 0), verbose=False)
    shd = Show4DSTEM([base[i].to(f"cuda:{i % 2}") for i in range(4)], verbose=False)

    def bf(w, fi):
        w.frame_idx = fi
        return w._fast_masked_sum(
            w._create_circular_mask(w.center_col, w.center_row, w.bf_radius)
        ).cpu().numpy()

    for fi in range(4):
        assert np.allclose(bf(ref, fi), bf(shd, fi))
