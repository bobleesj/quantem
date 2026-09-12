"""Measure complete MPS MAPED runs and compare bounded float32 merge regions.

Run on the physical Apple GPU with a directory of seven tilt masters and a
private output directory. A reference module copied from the unchanged commit
provides an independent before/after execution oracle; it is never rewritten.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import os
import resource
import threading
import time
from pathlib import Path

import h5py
import Metal
import torch

from quantem.diffraction import MAPEDTorch, _maped_resident


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--compare-with", type=Path, help="A completed baseline output directory.")
    parser.add_argument(
        "--mode", choices=("baseline", "optimized", "parity", "profile"), required=True
    )
    args = parser.parse_args()
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
        raise RuntimeError("Disable CPU fallback for this GPU qualification.")
    if not torch.backends.mps.is_available():
        raise RuntimeError("Run this benchmark on a physical MPS accelerator.")
    torch.mps.set_per_process_memory_fraction(24 * 1024**3 / torch.mps.recommended_max_memory())
    spec = importlib.util.spec_from_file_location("reference_resident", args.reference)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    if args.mode == "baseline":
        _maped_resident.ResidentMergeSource = reference.ResidentMergeSource
        from quantem.gpu.io.backends.mps import precision

        precision_spec = importlib.util.spec_from_file_location(
            "quantem.gpu.io.backends.mps._reference_precision",
            args.reference.with_name("reference_precision.py"),
        )
        reference_precision = importlib.util.module_from_spec(precision_spec)
        precision_spec.loader.exec_module(reference_precision)
        precision.tensor_range = reference_precision.tensor_range
        precision.tensor_measure = reference_precision.tensor_measure
    files = sorted(args.inputs.glob("*_master.h5"))
    assert len(files) == 7, "Select exactly seven tilt masters."
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "mode": args.mode,
        "torch": torch.__version__,
        "reference_sha256": hashlib.sha256(args.reference.read_bytes()).hexdigest(),
        "timings": {},
        "peaks": {},
    }
    device = Metal.MTLCreateSystemDefaultDevice()
    report["device"] = str(device.name())
    stop = threading.Event()

    def monitor():
        while not stop.is_set():
            for name, value in (
                ("metal_allocated_bytes", int(device.currentAllocatedSize())),
                ("mps_driver_bytes", torch.mps.driver_allocated_memory()),
                ("mps_tensor_bytes", torch.mps.current_allocated_memory()),
            ):
                report["peaks"][name] = max(report["peaks"].get(name, 0), value)
            stop.wait(0.02)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()

    def timed(name, call):
        torch.mps.synchronize()
        started = time.perf_counter()
        value = call()
        torch.mps.synchronize()
        report["timings"][name] = time.perf_counter() - started
        print(name, report["timings"][name], flush=True)
        return value

    maped = None
    started = time.perf_counter()
    try:
        maped = timed("load", lambda: MAPEDTorch.from_files(files, device="mps", backend="mps"))
        sources = maped.datasets.sources
        report["inputs"] = [
            {
                "representation": source.representation.value,
                "resident_bytes": source.resident_bytes,
                "read_passes": source.metadata["source_read_passes"],
                "hot_pixel_correction": source.metadata["hot_pixel_correction"]["method"],
            }
            for source in sources
        ]
        assert all(
            s["representation"] == "encoded" and s["read_passes"] == 1 for s in report["inputs"]
        )
        timed("preprocess", lambda: maped.preprocess(plot_summary=False))
        timed("diffraction_origin", lambda: maped.diffraction_origin(sigma=1, plot_origins=False))
        timed(
            "diffraction_align", lambda: maped.diffraction_align(edge_blend=2, plot_aligned=False)
        )
        timed(
            "real_space_align",
            lambda: maped.real_space_align(
                num_iter=20, edge_blend=5, padding=2, hanning_filter=True, plot_aligned=False
            ),
        )
        report["shape"] = list(sources[0].shape)
        for name in ("diffraction_origins", "diffraction_shifts", "real_space_shifts"):
            report[name] = getattr(maped, name).cpu().tolist()
        report["summary_sha256"] = {
            name: [
                hashlib.sha256(value.cpu().numpy().tobytes()).hexdigest()
                for value in getattr(maped, name)
            ]
            for name in ("im_bf", "dp_mean")
        }
        if args.mode == "profile":
            from quantem.gpu import io

            def instrument(owner, name, label):
                function = getattr(owner, name)

                def measured(*a, **kw):
                    torch.mps.synchronize()
                    start = time.perf_counter()
                    value = function(*a, **kw)
                    torch.mps.synchronize()
                    report.setdefault("isolated_phases", {}).setdefault(label, []).append(
                        time.perf_counter() - start
                    )
                    return value

                setattr(owner, name, measured)

            instrument(io.FourDSTEMData, "read", "read")
            instrument(_maped_resident, "_sample_scan_rows", "scan_interpolation")
            instrument(_maped_resident, "_shift_detector", "detector_interpolation")
            instrument(torch, "einsum", "denominator")
            generated = _maped_resident.ResidentMergeSource(
                sources,
                maped.real_space_shifts,
                maped.diffraction_shifts,
                close_sources_before_reopen=False,
            )
            for values in itertools.islice(generated.blocks(), 4):
                torch.mps.synchronize()
                del values
            generated.close()
        elif args.mode == "parity":
            before = reference.ResidentMergeSource(
                sources,
                maped.real_space_shifts,
                maped.diffraction_shifts,
                close_sources_before_reopen=False,
            )
            after = _maped_resident.ResidentMergeSource(
                sources,
                maped.real_space_shifts,
                maped.diffraction_shifts,
                close_sources_before_reopen=False,
            )
            report["float32_parity"] = {"values": 0, "regions": 0, "exact": True}
            iterators = {"reference": iter(before.blocks()), "optimized": iter(after.blocks())}
            durations = {name: [] for name in iterators}
            assert before.region_frames == after.region_frames
            rows_per_region = max(1, before.region_frames // report["shape"][1])
            regions = (report["shape"][0] + rows_per_region - 1) // rows_per_region
            for index in range(regions):
                pair = {}
                # Alternate execution order on the same already-resident inputs.
                # These synchronized timings isolate merging from loading/writes.
                order = (
                    ("reference", "optimized") if index % 2 == 0 else ("optimized", "reference")
                )
                for name in order:
                    torch.mps.synchronize()
                    phase = time.perf_counter()
                    pair[name] = next(iterators[name])
                    torch.mps.synchronize()
                    durations[name].append(time.perf_counter() - phase)
                expected, actual = pair["reference"], pair["optimized"]
                assert expected.dtype == actual.dtype == torch.float32
                assert torch.equal(actual, expected), (
                    "Float32 output changed; investigate without loosening the gate."
                )
                report["float32_parity"]["values"] += actual.numel()
                report["float32_parity"]["regions"] += 1
                del actual, expected, pair
                if report["float32_parity"]["regions"] % 8 == 0:
                    print("exact regions", report["float32_parity"]["regions"], flush=True)
            for iterator in iterators.values():
                assert next(iterator, None) is None
            before.close()
            after.close()
            report["paired_merge_region_seconds"] = durations
        else:
            from quantem.gpu import io
            from quantem.gpu.io import _precision

            original_range = _precision._range
            original_save = io.save
            original_load = io.load
            _precision._range = lambda *a, **kw: timed(
                "range_pass", lambda: original_range(*a, **kw)
            )
            io.save = lambda *a, **kw: timed(
                "save_including_range", lambda: original_save(*a, **kw)
            )
            io.load = lambda *a, **kw: timed("reopen", lambda: original_load(*a, **kw))
            profile = {}
            result = timed(
                "merge_save_reopen",
                lambda: maped.merge_datasets(
                    save_to=args.output / "merged_master.h5",
                    plot_result=False,
                    profile_timings=profile,
                ),
            )
            report["precision"] = result.metadata["precision"]
            report["output_resident_bytes"] = result.resident_bytes
            report["merge"] = result.metadata["maped_merge"]
            report["profile"] = profile
            report["owned_inputs_released"] = all(source.data.is_released for source in sources)
            assert report["owned_inputs_released"]
        report["workflow_seconds"] = time.perf_counter() - started
        assert report["peaks"]["metal_allocated_bytes"] < 24 * 1024**3
        if args.compare_with:
            baseline = json.loads((args.compare_with / "report.json").read_text())
            for name in (
                "summary_sha256",
                "diffraction_origins",
                "diffraction_shifts",
                "real_space_shifts",
                "precision",
            ):
                assert report[name] == baseline[name], f"Changed {name}; investigate."
            # HDF5 orchestration compares compressed bytes only, without decoding
            # or doing scientific array arithmetic on the CPU.
            chunks = 0
            with (
                h5py.File(args.compare_with / "merged_master.h5", "r") as previous,
                h5py.File(args.output / "merged_master.h5", "r") as current,
            ):
                for name in previous["entry/data"]:
                    expected = previous["entry/data"][name]
                    actual = current["entry/data"][name]
                    assert expected.shape == actual.shape and expected.dtype == actual.dtype
                    for frame in range(expected.shape[0]):
                        assert expected.id.read_direct_chunk(
                            (frame, 0, 0)
                        ) == actual.id.read_direct_chunk((frame, 0, 0))
                        chunks += 1
            report["saved_chunk_parity"] = {"chunks": chunks, "exact": True}
        report["passed"] = True
    finally:
        if maped is not None:
            maped.close()
        stop.set()
        thread.join()
        report["peaks"]["maximum_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
