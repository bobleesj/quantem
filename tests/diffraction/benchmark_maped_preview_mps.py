"""Measure exact no-save MAPED inspection and a bounded full-overview experiment.

Run on Phil with INPUT_DIRECTORY and a private OUTPUT_DIRECTORY. The public
patch workflow is measured separately from the benchmark-only complete overview.
Imports, independent parity checks, and browser rendering are outside timings.
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from pathlib import Path

import Metal
import torch
from quantem.gpu.detector import prepare
from quantem.widget import Show4DSTEM  # noqa: F401

from quantem.diffraction import MAPEDTorch
from quantem.diffraction._maped_resident import ResidentMergeSource


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
        raise RuntimeError("Disable CPU fallback for this GPU qualification.")
    torch.mps.set_per_process_memory_fraction(24 * 1024**3 / torch.mps.recommended_max_memory())
    files = sorted(args.inputs.glob("*_master.h5"))
    assert len(files) == 7
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"timings": {}, "peak_metal_bytes": 0}
    device = Metal.MTLCreateSystemDefaultDevice()
    stop = threading.Event()

    def monitor():
        while not stop.is_set():
            report["peak_metal_bytes"] = max(
                report["peak_metal_bytes"], int(device.currentAllocatedSize())
            )
            stop.wait(0.02)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()

    def timed(name, call):
        torch.mps.synchronize()
        start = time.perf_counter()
        value = call()
        torch.mps.synchronize()
        report["timings"][name] = time.perf_counter() - start
        print(name, report["timings"][name], flush=True)
        return value

    model = None
    try:
        start = time.perf_counter()
        model = timed("load", lambda: MAPEDTorch.from_files(files, device="mps"))
        timed("preprocess", lambda: model.preprocess(plot_summary=False))
        timed("diffraction_origin", lambda: model.diffraction_origin(sigma=1, plot_origins=False))
        timed("diffraction_align", lambda: model.diffraction_align(edge_blend=2, plot_aligned=False))
        timed("real_space_align", lambda: model.real_space_align(
            num_iter=20, edge_blend=5, padding=2, hanning_filter=True, plot_aligned=False
        ))
        sources = model.datasets.sources
        report["input_bytes"] = sum(source.resident_bytes for source in sources)
        assert all(source.representation.value == "encoded" for source in sources)
        assert all(source.metadata["source_read_passes"] == 1 for source in sources)
        regions = [(252, 260, 252, 260), (0, 8, 0, 8), (504, 512, 504, 512), (256, 257, 256, 257)]
        patches = []
        for index, region in enumerate(regions):
            patch = timed(f"patch_{index}", lambda: model.merge_datasets(
                scan_region=region, plot_result=False, verbose=False
            ))
            patches.append(patch)
            if index == 0:
                report["load_through_first_patch_seconds"] = time.perf_counter() - start
                viewer = timed("show4dstem_construction", lambda: model.show(verbose=False))
                report["viewer_type"] = type(viewer).__name__
                session = prepare(patch)
                timed("selected_dp", lambda: session.frame(4 * 8 + 4))
                viewer.close()
                model.viewer = None
        assert all(not source.data.is_released for source in sources)
        report["inputs_retained_after_inspection"] = True
        report["patch_bytes"] = [patch.logical_bytes for patch in patches]
        report["selected_regions"] = regions
        report["preview_peak_metal_bytes"] = report["peak_metal_bytes"]
        generated = ResidentMergeSource(
            sources, model.real_space_shifts, model.diffraction_shifts,
            close_sources_before_reopen=False,
        )
        rows, columns, detector_rows, detector_columns = generated.shape
        bf_t = torch.empty((rows, columns), device="mps")
        dp_sum_t = torch.zeros((detector_rows, detector_columns), device="mps")
        selected = [[] for _ in regions]
        torch.mps.synchronize()
        start = time.perf_counter()
        first = 0
        for block_t in generated.blocks():
            block_t = block_t.reshape(-1, columns, detector_rows, detector_columns)
            last = first + block_t.shape[0]
            bf_t[first:last] = block_t.mean(dim=(-2, -1))
            dp_sum_t += block_t.sum(dim=(0, 1))
            for index, (row0, row1, column0, column1) in enumerate(regions):
                overlap_first, overlap_last = max(row0, first), min(row1, last)
                if overlap_first < overlap_last:
                    selected[index].append(block_t[
                        overlap_first - first:overlap_last - first, column0:column1
                    ].clone())
            first = last
        dp_mean_t = dp_sum_t / (rows * columns)
        torch.mps.synchronize()
        report["timings"]["complete_overview_experiment"] = time.perf_counter() - start
        print("complete_overview_experiment", report["timings"]["complete_overview_experiment"], flush=True)
        for expected, patch in zip(selected, patches, strict=True):
            assert torch.equal(torch.cat(expected), patch.read())
        assert torch.isfinite(bf_t).all() and torch.isfinite(dp_mean_t).all()
        report["patch_float32_parity"] = {"exact": True, "values": sum(patch.data.numel() for patch in patches)}
        report["overview_pixels"] = bf_t.numel() + dp_mean_t.numel()
        report["passed"] = True
        generated.close()
    finally:
        if model is not None:
            model.close()
        stop.set()
        thread.join()
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
