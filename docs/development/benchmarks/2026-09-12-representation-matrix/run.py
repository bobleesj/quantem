"""Measure the public seven-tilt Torch workflow through resident viewing data."""
import argparse
import json
import os
from pathlib import Path
import threading
import time

import torch
from quantem.diffraction import MAPEDTorch
from quantem.gpu import io

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("directory", type=Path)
parser.add_argument("report", type=Path)
parser.add_argument("--device", required=True, choices=["cuda:0", "mps"])
parser.add_argument("--representation", choices=["encoded", "packed"], required=True)
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
sources = timed("load", lambda: [io.load(
    path, backend=backend, representation=args.representation, dtype="native",
    apply_mask=False, verbose=False,
) for path in files])
assert all(source.representation.value == args.representation for source in sources)
model = MAPEDTorch.from_resident(sources, device=args.device)
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

    report = dict(backend=backend, representation=args.representation, timings=timings, input_resident_bytes=inputs,
                  output_resident_bytes=result.resident_bytes, peak=peak,
                  load_through_dp_seconds=processing_total,
                  precision=result.metadata["precision"],
                  merge=result.metadata["maped_merge"],
                  saved=False, browser_rendering_measured=False)
    samples = {
        "counts": torch.stack([source.read(scan_region=(252, 253, 256, 257)).to(torch.float32) for source in sources]),
        "dp_mean": torch.stack(model.dp_mean),
        "im_bf": torch.stack(model.im_bf),
        "real_space_shifts": model.real_space_shifts,
        "diffraction_shifts": model.diffraction_shifts,
        "display_dp": result.read(scan_region=(252, 253, 256, 257)),
    }
    reference_path = args.report.parent / f"{backend}-reference.pt"
    if args.representation == "encoded":
        torch.save(samples, reference_path)
    else:
        reference = torch.load(reference_path, map_location=args.device, weights_only=True)
        report["representation_parity"] = {
            key: bool(torch.equal(value, reference[key])) for key, value in samples.items()
        }
        report["representation_difference"] = {
            key: {"max_abs": float((value - reference[key]).abs().max()),
                  "rmse": float((value - reference[key]).square().mean().sqrt())}
            for key, value in samples.items()
        }
        torch.save(samples, args.report.parent / f"{backend}-packed-samples.pt")
    args.report.write_text(json.dumps(report, indent=2)+"\n")
    print("REPORT", args.report, flush=True)
    if args.representation == "packed":
        assert all(report["representation_parity"].values()), report["representation_difference"]
finally:
    model.close()
    for source in sources:
        source.close()
    stop.set()
    thread.join()
