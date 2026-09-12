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
let merged = try maped.merge_datasets(
  save_to: URL(fileURLWithPath: "/path/to/new/merged_master.h5")
)
let pattern = try merged.read((256 * 512 + 256)..<(256 * 512 + 257))
```

All inputs remain encoded on the GPU. Median correction and complete-detector
bright-field means match the Python workflow. Merging uses float32 workspaces
of at most 4096 frames: first measure the global range, then recompute, convert,
measure the restored-value error, and write compressed HDF5. Owned inputs are
released before the complete output is reopened in packed GPU memory. Borrowed
`from_resident` inputs remain owned by their caller.

The output uses the existing `quantem_precision_v1`, `quantem_maped_summary_v1`,
and `quantem_maped_merge_v1` attributes. Python `quantem.gpu.io.load` can reopen
it directly. The result keeps the global scale, offset, RMSE, maximum error,
positive-to-zero count, and both shift arrays. The normal completion message
is one line.

Current resident merging supports bilinear shifts, zero padding, scan edge
blend 1, detector edge blend 0, and scaled uint16 output. Other choices raise
an explicit error. Native HDF5 currently requires a detector pixel count
divisible by 4096, including the tested 192×192 detector. Float16 native export
and native viewer integration are outside this implementation.

Some established parameters affect plots only; their scientific meaning is
preserved. See the [numerical contract](../docs/development/native-maped-contract.md)
for the normalized-grid interpolation convention and currently inactive options.
The [parameter contract and sensitivity tests](../docs/development/native-maped-parameters.md)
explain every scientific control, shared case sweeps, and saved provenance.

## Build and test

Use the QuantEM repository as the working directory on a Mac. SwiftPM resolves
QuantEM.GPU from its `main` branch by default (tested revision `6c171565`).
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

## Measured seven-tilt workflow

Physical Apple M5 Max, 40 GPU cores, 128 GB unified memory; seven acquisitions
with a 512×512 scan and 192×192 detector, measured on 2026-09-12:

| Stage | Seconds |
| --- | ---: |
| Load, median-correct, encode, and calculate summaries | 8.28 |
| Diffraction and real-space alignment | 0.53 |
| Merge to measure the global range | 13.34 |
| Merge again, convert, compress, and save | 22.70 |
| Reopen the full packed output | 3.44 |
| **Complete workflow** | **48.34** |

The 22.70-second write stage includes 14.23 seconds generating merged values,
**0.83 seconds converting and measuring precision**, 3.13 seconds compressing,
and 4.50 seconds writing HDF5. These components are included in the total;
they are not additional stages. Each input HDF5 is read once. Both merge
passes operate on the resident encoded inputs.

Encoded inputs occupy **7.01 GiB**; the fully reopened output occupies
**5.68 GiB**. Peak Metal allocation was **8.78 GiB** and process footprint
**10.09 GiB**. These are overlapping memory measurements, not additive.
The measured allocation fits a 24 GB budget, but this was a 128 GB Mac;
performance on a physical 24 GB machine remains to be measured.

All summary pixels match Torch MPS exactly. The largest shift difference is
0.00346 pixel; floating-point alignment is numerically equivalent within the
declared tolerance, not bit-identical. With common shifts, the 150,994,944-value
merged region passes `rtol=3e-6, atol=2e-5` with RMSE `6.17e-7`. Three saved DPs
reopened through the Python GPU loader match an independent float64 NumPy
scaling oracle exactly. Scaled-storage RMSE over the entire output is
`0.00695231`, with no clipping or overflow.

See the [retained measurements and qualification limits](Benchmarks/results/2026-09-12-metal/README.md).


## Parameter qualification

The [expanded parameter run](Benchmarks/results/2026-09-12-metal-parameters/README.md)
qualifies 75 settings, 45 sensitivity intervals, seven rejected merge options,
and repeated changes on the same encoded inputs. It includes scan-boundary DPs.
Both shift arrays were bit-identical to Torch MPS for every case; the worst
sampled float32 DP RMSE was 1.47e-6. This finite matrix is not a proof for every
possible dataset or parameter value. The original benchmark above remains a
historical measurement; use the expanded run for current code measurements.
