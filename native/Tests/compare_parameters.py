"""Compare native parameter responses with the unchanged Torch MPS workflow.

Run INPUT_DIRECTORY CASES_JSON NATIVE_OUTPUT_DIRECTORY on the physical Mac.
Scientific operations and comparisons use MPS. File parsing and small exported
observations are test infrastructure, with no CPU scientific fallback.
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from compare_torch import compare, merge_region

from quantem.diffraction import MAPEDTorch


def sample_patterns(maped, positions, real_t, diffraction_t):
    """Read selected output DPs through the existing Torch merge operations."""
    patterns_t = []
    for row in sorted({point[0] for point in positions}):
        merged_t = merge_region(maped, (row, row + 1), real_t, diffraction_t)
        columns = [point[1] for point in positions if point[0] == row]
        patterns_t.append(merged_t[0, columns].clone())
        del merged_t
    return torch.cat(patterns_t)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path)
    parser.add_argument("cases", type=Path)
    parser.add_argument("native_output", type=Path)
    args = parser.parse_args()
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") != "0":
        raise RuntimeError("Unset PYTORCH_ENABLE_MPS_FALLBACK for GPU parity.")
    manifest = json.loads(args.cases.read_text())
    native = json.loads((args.native_output / "native.json").read_text())
    native_cases = {item["id"]: item for item in native["cases"]}
    limits = manifest["comparison"]
    positions = manifest["sample_positions"]
    files = sorted(args.inputs.glob("*_master.h5"))
    if len(files) != 7:
        raise ValueError("Choose the directory containing exactly seven masters.")
    maped = MAPEDTorch.from_files(files, device="mps", backend="mps")
    report = {"cases": [], "limits": limits, "sensitivity": []}
    observations = {}
    previous_preprocessing = None
    for case in manifest["cases"]:
        name = case["id"]
        started = time.perf_counter()
        result = {"id": name, "failures": []}
        native_result = native_cases[name]
        try:
            preprocessing = case.get("preprocess", {})
            if preprocessing != previous_preprocessing:
                maped.preprocess(**preprocessing, plot_summary=False)
                previous_preprocessing = preprocessing
            origin = dict(case.get("diffraction_origin", {}))
            if isinstance(origin.get("origins", [None])[0], int):
                origin["origins"] = tuple(origin["origins"])
            maped.diffraction_origin(**origin, plot_origins=False)
            maped.diffraction_align(**case.get("diffraction_align", {}), plot_aligned=False)
            maped.real_space_align(**case.get("real_space_align", {}), plot_aligned=False)
            if native_result["status"] != "passed":
                raise AssertionError(native_result.get("error", "Native case failed"))
            if maped.diffraction_origins.cpu().tolist() != native_result["origins"]:
                result["failures"].append("origins")
            if not native_result["source_bytes_unchanged"]:
                result["failures"].append("resident_lifetime")
            if maped.scales.cpu().tolist() != native_result["parameters"]["preprocess"]["scale"]:
                result["failures"].append("scale")
            native_diffraction_t = torch.tensor(
                native_result["diffraction_shifts"], device="mps"
            ).reshape(7, 2)
            native_real_t = torch.tensor(native_result["real_space_shifts"], device="mps").reshape(
                7, 2
            )
            for key, actual_t, expected_t in [
                ("diffraction_shifts", native_diffraction_t, maped.diffraction_shifts),
                ("real_space_shifts", native_real_t, maped.real_space_shifts),
            ]:
                result[key] = compare(actual_t, expected_t)
                if result[key]["max_absolute"] > limits["shift_max_absolute"]:
                    result["failures"].append(key)
            native_patterns_t = torch.from_numpy(
                np.fromfile(args.native_output / (name + ".f32"), dtype=np.float32).reshape(
                    len(positions), *native["shape"][2:]
                )
            ).to("mps")
            shared_t = sample_patterns(maped, positions, native_real_t, native_diffraction_t)
            result["same_shift_patterns"] = compare(native_patterns_t, shared_t)
            if not torch.allclose(
                native_patterns_t,
                shared_t,
                rtol=limits["same_shift_rtol"],
                atol=limits["same_shift_atol"],
            ):
                result["failures"].append("same_shift_patterns")
            del shared_t
            reference_t = sample_patterns(
                maped, positions, maped.real_space_shifts, maped.diffraction_shifts
            )
            result["aligned_patterns"] = compare(native_patterns_t, reference_t)
            if result["aligned_patterns"]["rmse"] > limits["aligned_dp_rmse"]:
                result["failures"].append("aligned_patterns")
            observations[name] = {
                "native_shifts": torch.cat((native_diffraction_t, native_real_t)).clone(),
                "torch_shifts": torch.cat(
                    (maped.diffraction_shifts, maped.real_space_shifts)
                ).clone(),
                "native_patterns": native_patterns_t,
                "torch_patterns": reference_t,
            }
            baseline = observations["defaults"]
            current = observations[name]
            shift_response_t = current["native_shifts"] - baseline["native_shifts"]
            reference_response_t = current["torch_shifts"] - baseline["torch_shifts"]
            result["shift_response"] = compare(shift_response_t, reference_response_t)
            if result["shift_response"]["max_absolute"] > 2 * limits["shift_max_absolute"]:
                result["failures"].append("shift_response")
            pattern_response_t = current["native_patterns"] - baseline["native_patterns"]
            reference_response_t = current["torch_patterns"] - baseline["torch_patterns"]
            result["pattern_response"] = compare(pattern_response_t, reference_response_t)
            if result["pattern_response"]["rmse"] > 2 * limits["aligned_dp_rmse"]:
                result["failures"].append("pattern_response")
            equivalent = case.get("equivalent_to")
            if case["effect"] in ("inactive", "origin_only"):
                equivalent = "defaults"
            if equivalent:
                for key in current:
                    if not torch.equal(current[key], observations[equivalent][key]):
                        result["failures"].append("invariance_" + key)
        except Exception as error:
            result["failures"].append(str(error))
        result["seconds"] = time.perf_counter() - started
        report["cases"].append(result)
        report["passed"] = all(not item["failures"] for item in report["cases"])
        (args.native_output / "torch-parameters.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        print(name, result["failures"] or "passed", flush=True)

    # Secant responses compare sensitivity without assuming monotonic alignment.
    groups = {}
    for case in manifest["cases"]:
        for stage in ("diffraction_origin", "diffraction_align", "real_space_align"):
            options = case.get(stage, {})
            if len(options) == 1 and case["id"] in observations:
                parameter, value = next(iter(options.items()))
                if type(value) in (int, float):
                    groups.setdefault((stage, parameter), []).append((value, case["id"]))
    for (stage, parameter), settings in groups.items():
        settings.sort()
        for (low, low_name), (high, high_name) in zip(settings, settings[1:]):
            if high == low:
                continue
            left, right = observations[low_name], observations[high_name]
            native_slope_t = (right["native_shifts"] - left["native_shifts"]) / (high - low)
            torch_slope_t = (right["torch_shifts"] - left["torch_shifts"]) / (high - low)
            metrics = compare(native_slope_t, torch_slope_t)
            bound = 2 * limits["shift_max_absolute"] / (high - low)
            pattern_metrics = compare(
                (right["native_patterns"] - left["native_patterns"]) / (high - low),
                (right["torch_patterns"] - left["torch_patterns"]) / (high - low),
            )
            pattern_bound = 2 * limits["aligned_dp_rmse"] / (high - low)
            report["sensitivity"].append(
                {
                    "stage": stage,
                    "parameter": parameter,
                    "interval": [low, high],
                    "shift_secant": metrics,
                    "max_absolute_limit": bound,
                    "pattern_secant": pattern_metrics,
                    "pattern_rmse_limit": pattern_bound,
                    "passed": metrics["max_absolute"] <= bound
                    and pattern_metrics["rmse"] <= pattern_bound,
                }
            )
    native_rejections = {
        item["id"]: item
        for item in json.loads((args.native_output / "validation.json").read_text())
    }
    report["validation"] = []
    for case in manifest["rejected_merge_cases"]:
        rejected = False
        try:
            maped.merge_datasets(
                **case["options"],
                plot_result=False,
                save_to=args.native_output / (case["id"] + "_torch_master.h5"),
            )
        except ValueError:
            rejected = True
        report["validation"].append(
            {
                "id": case["id"],
                "torch_rejected": rejected,
                "passed": rejected and native_rejections[case["id"]]["rejected"],
            }
        )
    report["passed"] = (
        len(report["cases"]) == len(manifest["cases"])
        and all(not item["failures"] for item in report["cases"])
        and all(item["passed"] for item in report["sensitivity"])
        and all(item["passed"] for item in report["validation"])
    )
    report["native_peak_metal_bytes"] = native["peak_metal_bytes"]
    report["source_read_passes"] = native["source_read_passes"]
    (args.native_output / "torch-parameters.json").write_text(json.dumps(report, indent=2) + "\n")
    maped.close()
    if not report["passed"]:
        raise SystemExit("Parameter parity failed; inspect the retained case failures.")


if __name__ == "__main__":
    main()
