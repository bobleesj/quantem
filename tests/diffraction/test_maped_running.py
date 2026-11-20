"""The running merge: tilts added one at a time give the joint merge."""

import numpy as np
import torch

from quantem.core.datastructures import Dataset4dstem
from quantem.diffraction.maped import RunningMAPED
from quantem.diffraction.maped.core.merge import MergedRegions, RunningMerge

DISPLACEMENTS = [(0, 0), (2, -1), (-2, 3), (1, 2), (-1, -2)]


def tilts_of_one_sample(seed: int = 0):
    """Five native acquisitions of one sample, each rolled by a whole number of scan positions."""
    rng = np.random.default_rng(seed)
    row, col = np.indices((48, 48))
    sample = np.exp(-((row - 20) ** 2 + (col - 26) ** 2) / 30) + 0.5 * np.exp(-((row - 33) ** 2 + (col - 14) ** 2) / 12)
    disk = (np.hypot(*(np.indices((12, 12)) - 5.5)) < 4).astype(np.float32)
    tilts = []
    for shift in DISPLACEMENTS:
        image = np.roll(sample, shift, axis=(0, 1))
        counts = rng.poisson(20 * image[..., None, None] * disk).astype(np.float32)
        tilts.append(Dataset4dstem.from_array(counts))
    return tilts


def test_tilts_added_one_at_a_time_equal_the_joint_merge():
    tilts = tilts_of_one_sample()
    scan = -torch.tensor(DISPLACEMENTS, dtype=torch.float32) + torch.tensor([[0.0, 0.0], [0.3, 0.1], [0.0, -0.4], [0.25, 0.0], [0.0, 0.0]])
    detector = torch.tensor([[0.0, 0.0], [0.2, -0.1], [-0.3, 0.0], [0.0, 0.15], [0.1, 0.1]])
    for crop in (False, True):
        joint = MergedRegions(tilts, scan, detector, compile_merge=False, crop=crop)
        running = RunningMerge(tilts[0].shape, "cpu", compile_merge=False)
        for tilt, scan_shift, detector_shift in zip(tilts, scan.tolist(), detector.tolist()):
            running.add(tilt, scan_shift, detector_shift)
        running.finish(crop=crop)
        assert running.scan_box == joint.scan_box and running.shape == joint.shape
        joint.rows_per_region = running.rows_per_region
        merged = torch.cat(list(running.blocks()))
        torch.testing.assert_close(merged, torch.cat(list(joint.blocks())), atol=1e-5, rtol=1e-5)
        assert merged.shape[0] == running.shape[0] * running.shape[1]
        torch.testing.assert_close(running.scan_sum, joint.scan_sum, atol=1e-3, rtol=1e-5)


def test_each_tilt_is_placed_on_the_first_and_not_kept():
    tilts = tilts_of_one_sample()
    running = RunningMAPED(tilts[0], compile_merge=False)
    for tilt in tilts[1:]:
        running.add(tilt, iterations=40)
    assert len(running) == 5 and not hasattr(running, "tilts")
    found = torch.tensor(running.real_space_shifts)
    torch.testing.assert_close(found, -torch.tensor(DISPLACEMENTS, dtype=torch.float32), atol=0.15, rtol=0)
    assert running.real_space_shifts[0] == [0.0, 0.0] and running.diffraction_shifts[0] == [0.0, 0.0]
    merged = running.merge(crop=True, dtype="float32")
    assert isinstance(merged, Dataset4dstem) and merged.shape[2:] == (12, 12)
    assert merged.shape[0] <= 48 - 4 and merged.shape[1] <= 48 - 4          # the positions every tilt covers
    aligned = np.mean([np.roll(tilt.array, tuple(-np.array(shift)), axis=(0, 1)) for tilt, shift in zip(tilts, DISPLACEMENTS)], axis=0)
    row0, row1, col0, col1 = running.running.scan_box
    assert np.abs(merged.array - aligned[row0:row1, col0:col1]).mean() < 0.35  # the mean of the tilts moved back
