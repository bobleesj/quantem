import os
import runpy
import torch
from quantem.diffraction._maped_resident import ResidentMergeSource

original = ResidentMergeSource.blocks


def blocks(self, scan_region=None):
    if scan_region is not None:
        yield from original(self, scan_region)
        return
    rows, columns, height, width = self.shape
    calibration_frames = self.region_frames
    self.region_frames = max(columns, (calibration_frames // 2 // columns) * columns)
    count = 0
    first = 0
    output = None
    try:
        for values in original(self):
            if output is None:
                capacity = min(calibration_frames, rows * columns - first)
                output = torch.empty(
                    (capacity, height, width), device=self._torch_device, dtype=torch.float32
                )
            frames = values.shape[0]
            output[count : count + frames].copy_(values)
            count += frames
            del values
            if count == output.shape[0]:
                yield output
                first += count
                count = 0
                output = None
        assert count == 0
    finally:
        self.region_frames = calibration_frames
        self._update_merge_metadata()


ResidentMergeSource.blocks = blocks
runpy.run_path(os.environ["MAPED_BENCHMARK"], run_name="__main__")
