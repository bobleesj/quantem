"""Measure the public seven-tilt Torch workflow through resident viewing data."""
import argparse
import json
import os
from pathlib import Path
import threading
import time

import torch
from quantem.diffraction import MAPEDTorch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("directory", type=Path)
parser.add_argument("report", type=Path)
parser.add_argument("--device", required=True, choices=["cuda:0", "mps"])
args = parser.parse_args()
backend = args.device.split(":")[0]
peak = {"allocated_bytes": 0}
stop = threading.Event()
if backend == "cuda":
    import pynvml
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    torch.cuda.set_per_process_memory_fraction(24 * 2**30 / torch.cuda.get_device_properties(0).total_memory)
else:
    torch.mps.set_per_process_memory_fraction(24 * 2**30 / torch.mps.recommended_max_memory())


def sample():
    while not stop.wait(0.02):
        if backend == "cuda":
            used = max((p.usedGpuMemory for p in pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                        if p.pid == os.getpid()), default=0)
        else:
            used = torch.mps.driver_allocated_memory()
        peak["allocated_bytes"] = max(peak["allocated_bytes"], used)


thread = threading.Thread(target=sample, daemon=True)
thread.start()
timings = {}


def timed(name, operation):
    start = time.perf_counter()
    value = operation()
    (torch.cuda.synchronize if backend == "cuda" else torch.mps.synchronize)()
    timings[name] = time.perf_counter() - start
    print(name, timings[name], flush=True)
    return value


files = sorted(args.directory.glob("*_master.h5"))
assert len(files) == 7
model = timed("load", lambda: MAPEDTorch.from_files(files, device=args.device))
try:
    inputs = sum(source.resident_bytes for source in model.datasets.sources)
    timed("preprocess", lambda: model.preprocess(plot_summary=False))
    timed("origin", lambda: model.diffraction_origin(sigma=1, plot_origins=False))
    timed("diffraction_align", lambda: model.diffraction_align(edge_blend=2, plot_aligned=False))
    timed("real_space_align", lambda: model.real_space_align(
        num_iter=20, hanning_filter=True, padding=2, edge_blend=5,
        pad_val="median", shift_method="bilinear", plot_aligned=False))
    result = timed("merge_pack_summaries", lambda: model.merge_datasets(
        dtype="scaled_uint16", plot_result=False, verbose=False))
    timed("selected_dp", lambda: result.read(scan_region=(252, 253, 256, 257)))
    processing_total = sum(timings.values())
    viewer = timed("viewer_constructor", lambda: model.show())
    report = dict(backend=backend, timings=timings, input_resident_bytes=inputs,
                  output_resident_bytes=result.resident_bytes, peak=peak,
                  load_through_dp_seconds=processing_total,
                  precision=result.metadata["precision"],
                  merge=result.metadata["maped_merge"],
                  saved=False, browser_rendering_measured=False)
    args.report.write_text(json.dumps(report, indent=2)+"\n")
    print("REPORT", args.report, flush=True)
    from quantem.gpu.io.backends.mps.precision import MetalArray, _dispatch, _part_buffers
    import numpy as np
    source = result.data
    reference = MetalArray(source.det_shape, np.float32)
    actual = source.mean_dp()
    for index, part in enumerate(source.parts):
        parameters, calibration = source._params(part)
        parameters[0], parameters[8], parameters[9] = parameters[1], int(index > 0), source.n_frames
        with _part_buffers(part) as buffers:
            _dispatch("mean", [*buffers, reference], parameters, calibration)
    np.testing.assert_array_equal(actual.get(), reference.get())
    reference.release()
    actual.release()
    print("FULL_MEAN_EXACT_PARITY passed", flush=True)
finally:
    model.close()
    stop.set()
    thread.join()
