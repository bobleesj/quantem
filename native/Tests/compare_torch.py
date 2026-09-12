"""Compare a physical Metal benchmark with the current Torch MPS workflow.

Run with INPUT_DIRECTORY and the native benchmark's REPORT_JSON. Source data
and result paths stay outside the repository. All comparisons run on MPS;
NumPy provides an independent oracle for small saved diffraction patterns.
"""

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from quantem.diffraction import MAPEDTorch
from quantem.diffraction._maped_resident import (
    _sample_scan_rows,
    _shift_detector,
    _weights,
)


def compare(actual, expected):
    difference = actual - expected
    return {
        "max_absolute": float(difference.abs().max()),
        "rmse": float(difference.square().mean().sqrt()),
        "reference_max": float(expected.abs().max()),
        "exact": bool(torch.equal(actual, expected)),
    }


def merge_region(maped, rows, real_shifts, diffraction_shifts):
    shape = maped.datasets.sources[0].shape
    real_weights, detector_weights, grids = _weights(shape, real_shifts, diffraction_shifts)
    numerator = None
    for index, source in enumerate(maped.datasets.sources):
        offset = int(torch.floor(-real_shifts[index, 0]).item())
        first = max(0, rows[0] + offset)
        stop = min(shape[0], rows[1] + offset + 1)
        if first < stop:
            decoded = source.read(scan_region=(first, stop, 0, shape[1]))
            sampled = _sample_scan_rows(
                decoded, decoded_first_row=first, output_first_row=rows[0],
                output_stop_row=rows[1], shift=real_shifts[index],
            )
            del decoded
        else:
            sampled = torch.zeros(
                (rows[1] - rows[0], *shape[1:]),
                dtype=torch.float32, device=real_shifts.device,
            )
        shifted = _shift_detector(sampled.reshape(-1, *shape[2:]), grids[index]).reshape_as(sampled)
        del sampled
        weight = real_weights[index, rows[0]:rows[1], :, None, None]
        if numerator is None:
            numerator = shifted.mul_(weight)
        else:
            numerator.addcmul_(weight, shifted)
        del shifted
    denominator = torch.einsum("nrc,nhw->rchw", real_weights[:, rows[0]:rows[1]], detector_weights)
    denominator += (1 - detector_weights.sum(0).clamp(0, 1))[None, None]
    numerator.div_(denominator).masked_fill_(denominator == 0, 0)
    return numerator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path)
    parser.add_argument("native_report", type=Path)
    parser.add_argument("--saved", type=Path, help="Check a native HDF5 export through the existing Python loader.")
    args = parser.parse_args()
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"):
        raise RuntimeError("Disable MPS CPU fallback before this parity run.")
    native = json.loads(args.native_report.read_text())
    if native["shape"] != [512, 512, 192, 192]:
        raise ValueError("This real-data parity runner expects a 512×512 scan and 192×192 detector.")
    files = sorted(args.inputs.glob("*_master.h5"))
    if len(files) != 7:
        raise ValueError("Choose the directory containing exactly seven tilt masters.")
    result = {"comparisons": {}, "timings": {}}
    checks = result["comparisons"]

    def timed(name, call):
        torch.mps.synchronize()
        started = time.perf_counter()
        value = call()
        torch.mps.synchronize()
        result["timings"][name] = time.perf_counter() - started
        return value

    def exported(name, shape):
        path = args.native_report.with_suffix("." + name + ".f32")
        return torch.from_numpy(np.fromfile(path, dtype=np.float32).reshape(shape)).to("mps")

    maped = timed("load", lambda: MAPEDTorch.from_files(files, device="mps", backend="mps"))
    timed("preprocess", lambda: maped.preprocess(plot_summary=False))
    for name in ("dp_mean", "im_bf"):
        for index, expected in enumerate(getattr(maped, name)):
            checks[f"{name}_{index}"] = compare(exported(f"{name}-{index}", expected.shape), expected)
            assert checks[f"{name}_{index}"]["exact"], (name, index, checks[f"{name}_{index}"])
    timed("diffraction_origin", lambda: maped.diffraction_origin(sigma=1, plot_origins=False))
    timed("diffraction_align", lambda: maped.diffraction_align(edge_blend=2, plot_aligned=False))
    timed("real_space_align", lambda: maped.real_space_align(num_iter=20, edge_blend=5, padding=2, hanning_filter=True, plot_aligned=False))
    for name in ("diffraction_shifts", "real_space_shifts"):
        expected = getattr(maped, name)
        actual = torch.tensor(native[name], device="mps").reshape_as(expected)
        checks[name] = compare(actual, expected)
        result[name] = expected.cpu().tolist()
    result["diffraction_origins"] = maped.diffraction_origins.cpu().tolist()
    assert result["diffraction_origins"] == native["diffraction_origins"]
    native_region = exported("region", (8, 512, 192, 192))
    shared_real = torch.tensor(native["real_space_shifts"], device="mps").reshape(7, 2)
    shared_diff = torch.tensor(native["diffraction_shifts"], device="mps").reshape(7, 2)
    reference = timed("merge_with_native_shifts", lambda: merge_region(maped, (248, 256), shared_real, shared_diff))
    checks["merge_same_shifts"] = compare(native_region, reference)
    torch.testing.assert_close(native_region, reference, rtol=3e-6, atol=2e-5)
    del reference
    reference = timed("merge_with_torch_shifts", lambda: merge_region(maped, (248, 256), maped.real_space_shifts, maped.diffraction_shifts))
    checks["merge_independent_alignment"] = compare(native_region, reference)
    for name in ("diffraction_shifts", "real_space_shifts"):
        assert checks[name]["max_absolute"] < 0.01, (name, checks[name])
    assert checks["merge_independent_alignment"]["rmse"] < 0.001
    maped.close()
    del reference
    if args.saved:
        from quantem.gpu import io
        from quantem.gpu.detector import prepare

        # These small selected DPs are an independent NumPy parity oracle, not
        # a scientific runtime fallback. File decoding and packed reads use MPS.
        original_samples = {(row, column): native_region[row, column].cpu().numpy()
                            for row, column in [(0, 0), (0, 256), (7, 511)]}
        del native_region
        torch.mps.empty_cache()
        loaded = timed("python_reopen_native_export", lambda: io.load(args.saved, backend="mps", verbose=False))
        report = loaded.metadata["precision"]
        session = prepare(loaded)
        for (row, column), original in original_samples.items():
            codes = np.rint((original.astype(np.float64) - report["offset"]) / report["scale"]).clip(0, 65535)
            expected = (codes * report["scale"] + report["offset"]).astype(np.float32)
            actual = session.frame((row + 248) * 512 + column)
            np.testing.assert_array_equal(actual, expected)
        result["native_export_python_parity"] = "three selected DPs exact against float64 NumPy scaling oracle"
        result["saved_precision"] = report
        assert loaded.metadata["maped_summary"]["mean_bright_field"]["divisor"] == 192 * 192
        assert loaded.metadata["maped_merge"]["source_representation"] == "encoded"
        loaded.close()
    text = json.dumps(result, indent=2)
    args.native_report.with_suffix(".torch-parity.json").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
