# Native MAPED

`QuantEMMAPED` runs the MAPED scientific sequence in Swift using QuantEM.GPU's
native Metal infrastructure. It has no Python, Torch, or application UI runtime.
The Python `MAPEDTorch` API remains unchanged.

```swift
import Foundation
import QuantEMMAPED

let input = URL(fileURLWithPath: "/path/to/seven/tilts")
let files = try FileManager.default.contentsOfDirectory(
  at: input, includingPropertiesForKeys: nil
).filter { $0.lastPathComponent.hasSuffix("_master.h5") }
 .sorted { $0.path < $1.path }

let maped = try MAPEDNative.from_files(files)
defer { maped.close() }
try maped.preprocess()
try maped.diffraction_origin(sigma: 1)
try maped.diffraction_align(edge_blend: 2)
try maped.real_space_align(
  num_iter: 20, edge_blend: 5, padding: 2, hanning_filter: true
)
let merged = try maped.merge_datasets(dtype: "scaled_uint16")
let pattern = try merged.read((256 * 512 + 256)..<(256 * 512 + 257))
```

All seven inputs remain encoded on the GPU. Median correction and
complete-detector bright-field means follow the Python workflow. Bounded
float32 regions are merged once, converted to scaled uint16, and losslessly
ANS-encoded for viewing. Conversion is approximate; ANS preserves the resulting
codes exactly. Automatic region scales and error statistics are retained in
`merged.metadata`; callers do not choose a scaling region.

Saving is optional:

```swift
// Save existing display codes and calibration, without another merge.
try merged.save(to: URL(fileURLWithPath: "/path/to/new/scaled_master.h5"))

// Export original float32 merge values, recomputed from retained inputs.
// Upcasting the display codes cannot recover their discarded precision.
try maped.merge_datasets(
  dtype: "float32", save_to: URL(fileURLWithPath: "/path/to/new/float_master.h5")
)
```

The float32 export also returns a scaled display result. It does not allocate a
complete dense float32 dataset. Use a new path for each export. Saved attributes
record calibration/error statistics for scaled storage, MAPED parameters,
shifts, median correction, and bright-field summary conventions.

For the established call that supplies `save_to` but omits `dtype`, the earlier
global-scale, two-pass save/reopen path remains available. Specify
`dtype: "scaled_uint16"` for the resident single-pass workflow above.

Resident merging currently supports bilinear shifts, zero padding, scan edge
blend 1 and detector edge blend 0. Unsupported merge options raise an error;
shared parameter names do not imply every Torch option is implemented. Native
HDF5 export requires detector pixel counts divisible by 4096, including the
tested 192×192 detector. Float16 export and viewer integration are outside this
implementation. Native scaled-file reopening is tested; float32-file reopening
was qualified through the Python MPS loader, not a native application viewer.

Some established parameters affect plots only; their scientific meaning is
preserved. See the [numerical contract](../docs/development/native-maped-contract.md)
for the normalized-grid interpolation convention and currently inactive options.
The [parameter contract and sensitivity tests](../docs/development/native-maped-parameters.md)
explain every scientific control, shared case sweeps, and saved provenance.

## Build and test

Use the QuantEM repository as the working directory on a Mac. SwiftPM resolves
QuantEM.GPU from its `main` branch by default. The checked-in dependency lock pins the qualified
QuantEM.GPU revision; keep that lock when reproducing this run.
To work with both repositories locally, point the package at that checkout:

```bash
export QUANTEM_GPU_PACKAGE=/path/to/quantem.gpu
swift test -c release --filter MAPEDNativeTests
swift run -c release maped-native-benchmark \
  /path/to/seven/tilts /path/to/run/native.json /path/to/run/merged_master.h5
PYTHONPATH=src:/path/to/quantem.gpu/src python native/Tests/compare_torch.py \
  /path/to/seven/tilts /path/to/run/native.json \
  --saved /path/to/run/merged_master.h5
```

The benchmark retains bounded parity observations separately from its timed
workflow. Use a new output path for each run. Native tests include exact
integer reconstruction and median correction against NumPy, FFT and Gaussian
checks, exact uint16 codes/restoration, persisted coefficient equality, packed
HDF5 reopening, and borrowed-source lifetime. The real-data runner additionally
compares every summary pixel, both shift arrays, an eight-row merged region,
and selected saved diffraction patterns through the Python loader.

Fixtures under `Tests/QuantEMMAPEDTests/Fixtures` are independent expectations.
Do not regenerate them to silence a failure.

## Current seven-tilt qualification

the Apple M5 Max test host, Apple M5 Max (40 GPU cores, 128 GB), 512×512 scan, 192×192 detector,
2026-09-12. Native Swift/Metal; no Python or Torch runtime in processing.

| Stage | Seconds |
| --- | ---: |
| Load, median correction, ANS and summaries | 7.93 |
| Alignment and preparation remainder | 0.73 |
| Float32 merge | 6.97 |
| Display conversion and error measurement | 1.03 |
| ANS encode display | 1.30 |
| **Display ready, including loading** | **18.00** |
| One selected DP read | 0.0078 |

A final repeat completed in **24.85 s**: loading 9.06 s, alignment/preparation
0.89 s, merging 11.27 s, precision conversion 1.67 s and encoding 1.95 s.
The memory peak and whole-output display RMSE were unchanged. These two runs
establish an observed **18–25 s** range, not a fixed latency guarantee.

The input resident size is 7.01 GiB; display output is 6.13 GiB. Sampled peak
Metal allocation is **14.76 GiB** with all inputs retained. This is not a
16 GB Mac qualification: process overhead and the OS also need memory.

Optional scaled export is **6.05 GiB**, measured at **22.47 s**. Optional exact
float32 export is **32.98 GiB**, measured at **40.32 s** including recomputation
and display conversion. Both write timings use a shared network filesystem;
they are excluded from display-ready time and are not local SSD benchmarks.
The float32 logical array is 36 GiB. There is no claim of lossless 4× compression.

All 75 parameter cases and 45 sensitivity intervals passed the existing Torch
MPS gates. Five full scan rows (94,371,840 values) from saved files passed GPU
checks: native float32 against Torch at `atol=2e-5, rtol=3e-6`, and scaled output
against the saved float32 within its rounding bound. This is numerical float32
parity, not bit-identical cross-backend merging. Whole-output display RMSE was
approximately 0.00545. Twelve native tests passed, including ANS scaled-save
reopening across calibration boundaries.

See [the integration handoff](Benchmarks/results/2026-09-12-metal-resident/README.md)
for retained evidence and remaining qualifications. Older save-first results
remain in [the historical report](Benchmarks/results/2026-09-12-metal/README.md).

## Parameter qualification

The [expanded parameter run](Benchmarks/results/2026-09-12-metal-parameters/README.md)
qualifies 75 settings, 45 sensitivity intervals, seven rejected merge options,
and repeated changes on the same encoded inputs. It includes scan-boundary DPs.
Both shift arrays were bit-identical to Torch MPS for every case; the worst
sampled float32 DP RMSE was 1.47e-6. This finite matrix is not a proof for every
possible dataset or parameter value. The original benchmark above remains a
historical measurement; use the expanded run for current code measurements.

## Input representations

ANS (`encoded`) remains the default. The existing `from_resident` also accepts
bit-packed counts. See the [six-combination audit](../docs/development/benchmarks/2026-09-12-representation-matrix/README.md)
for the CUDA and native mask fixes, measured timings, and the remaining Torch
MPS ordinary-HDF5 packed-loader gap. Do not treat all six combinations as fully
qualified yet. The display dtype remains independent of the input codec.
