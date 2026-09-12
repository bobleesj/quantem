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
    outer_rows = max(1, self.region_frames // columns)
    inner_rows = max(1, outer_rows // 2)
    previous = len(self._pass_generation_seconds)
    for first in range(0, rows, outer_rows):
        stop = min(rows, first + outer_rows)
        output = torch.empty(
            (stop - first, columns, height, width), device=self._torch_device, dtype=torch.float32
        )
        for inner in range(first, stop, inner_rows):
            end = min(stop, inner + inner_rows)
            for values in original(self, (inner, end, 0, columns)):
                output[inner - first : end - first].copy_(
                    values.reshape(end - inner, columns, height, width)
                )
            del values
        yield output.flatten(0, 1)
        del output
    self._pass_generation_seconds[previous:] = [sum(self._pass_generation_seconds[previous:])]
    self._update_merge_metadata()


ResidentMergeSource.blocks = blocks
runpy.run_path(os.environ["MAPED_BENCHMARK"], run_name="__main__")
