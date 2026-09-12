"""Benchmark CUDA MAPED processing without export or reopening.

Run with INPUT_DIRECTORY OUTPUT_DIRECTORY on physical GPU0. Optionally pass
--reference with a frozen _maped_resident.py for an all-values paired audit.
The complete overview is a benchmark experiment, not a new public viewer API.
"""

import argparse
import importlib.util
import json
import os
import threading
import time
from pathlib import Path

import pynvml
import torch

from quantem.diffraction import MAPEDTorch
from quantem.diffraction._maped_resident import ResidentMergeSource


def main():
    """Measure synchronized processing and optional exact reference parity."""
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--frames", type=int, default=4096)
    args = parser.parse_args()
    if args.frames not in (512, 1024, 2048, 4096):
        parser.error("--frames must be 512, 1024, 2048, or 4096")
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "timings": {},
        "passes": [],
        "peak_process_bytes": 0,
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(),
    }
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    stop = threading.Event()

    def monitor():
        while not stop.wait(0.01):
            own = max(
                (
                    p.usedGpuMemory
                    for p in pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                    if p.pid == os.getpid()
                ),
                default=0,
            )
            report["peak_process_bytes"] = max(report["peak_process_bytes"], own)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    torch.cuda.set_per_process_memory_fraction(
        24 * 1024**3 / torch.cuda.get_device_properties(0).total_memory
    )

    def timed(name, fn):
        torch.cuda.synchronize()
        start = time.perf_counter()
        value = fn()
        torch.cuda.synchronize()
        report["timings"][name] = time.perf_counter() - start
        print(name, report["timings"][name], flush=True)
        return value

    model = None
    try:
        files = sorted(args.inputs.glob("*_master.h5"))
        assert len(files) == 7
        model = timed("load", lambda: MAPEDTorch.from_files(files, device="cuda:0"))
        timed("preprocess", lambda: model.preprocess(plot_summary=False))
        timed("diffraction_origin", lambda: model.diffraction_origin(sigma=1, plot_origins=False))
        timed(
            "diffraction_align", lambda: model.diffraction_align(edge_blend=2, plot_aligned=False)
        )
        timed(
            "real_space_align",
            lambda: model.real_space_align(
                num_iter=20, edge_blend=5, padding=2, hanning_filter=True, plot_aligned=False
            ),
        )
        sources = model.datasets.sources
        report["inputs"] = [
            {
                "representation": s.representation.value,
                "bytes": s.resident_bytes,
                "reads": s.metadata["source_read_passes"],
            }
            for s in sources
        ]
        assert all(s["representation"] == "encoded" and s["reads"] == 1 for s in report["inputs"])
        patch = timed(
            "patch",
            lambda: model.merge_datasets(
                scan_region=(252, 260, 252, 260), plot_result=False, verbose=False
            ),
        )
        report["patch_bytes"] = patch.logical_bytes
        viewer = timed("viewer_construction", lambda: model.show(verbose=False))
        viewer.close()
        model.viewer = None
        timed(
            "single_dp_merge",
            lambda: model.merge_datasets(
                scan_region=(256, 257, 256, 257), plot_result=False, verbose=False
            ),
        )
        generated = ResidentMergeSource(
            sources,
            model.real_space_shifts,
            model.diffraction_shifts,
            close_sources_before_reopen=False,
            compile_merge=False,
        )
        generated.region_frames = args.frames
        report["region_frames"] = args.frames
        rows, columns, height, width = generated.shape
        for repeat in range(3):

            def run():
                bf = torch.empty((rows, columns), device="cuda")
                dp = torch.zeros((height, width), device="cuda")
                low = torch.tensor(float("inf"), device="cuda")
                high = -low
                first = 0
                for block in generated.blocks():
                    region = block.reshape(-1, columns, height, width)
                    last = first + len(region)
                    bf[first:last] = region.mean((-2, -1))
                    dp += region.sum((0, 1))
                    lo, hi = torch.aminmax(region)
                    low = torch.minimum(low, lo)
                    high = torch.maximum(high, hi)
                    first = last
                return bf, dp / (rows * columns), low, high

            bf, dp, low, high = timed(f"overview_range_{repeat}", run)
            report["passes"].append(
                {
                    "min": low.item(),
                    "max": high.item(),
                    "bf_sum": bf.double().sum().item(),
                    "dp_sum": dp.double().sum().item(),
                }
            )
        if args.reference is not None:
            spec = importlib.util.spec_from_file_location("frozen_maped", args.reference)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            original = module.ResidentMergeSource(
                sources,
                model.real_space_shifts,
                model.diffraction_shifts,
                close_sources_before_reopen=False,
            )
            original.region_frames = generated.region_frames
            parity = {
                "original_seconds": 0.0,
                "candidate_seconds": 0.0,
                "values": 0,
                "exact": True,
            }
            try:
                original_iter, candidate_iter = iter(original.blocks()), iter(generated.blocks())
                for index in range(
                    (rows + (args.frames // columns) - 1) // (args.frames // columns)
                ):
                    values = [None, None]
                    for position in [0, 1] if index % 2 == 0 else [1, 0]:
                        torch.cuda.synchronize()
                        start = time.perf_counter()
                        values[position] = next((original_iter, candidate_iter)[position])
                        torch.cuda.synchronize()
                        parity[("original_seconds", "candidate_seconds")[position]] += (
                            time.perf_counter() - start
                        )
                    parity["exact"] &= torch.equal(*values)
                    parity["values"] += values[0].numel()
                    del values
                report["float32_parity"] = parity
                assert parity["exact"]
            finally:
                original.close()
        generated.close()
        assert report["peak_process_bytes"] < 24 * 1024**3
        report["passed"] = True
    finally:
        if model is not None:
            model.close()
        stop.set()
        thread.join()
        report["peak_torch_allocated"] = torch.cuda.max_memory_allocated()
        report["peak_torch_reserved"] = torch.cuda.max_memory_reserved()
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
