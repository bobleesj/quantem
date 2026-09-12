# MAPED scaled uint16 storage

Implemented for the Python CUDA and Torch MPS workflow. This supersedes the
initial proposal: there is no `scaling` argument. The existing `dtype` expresses
the scientist's storage choice; calibration and scheduling remain automatic.

```python
merged = maped.merge_datasets(dtype="scaled_uint16", plot_result=False)
viewer = maped.show()
```

All seven inputs remain encoded while MAPED performs one bounded float32 merge.
QuantEM.GPU measures each produced region, converts it to calibrated uint16,
and retains its packed codes. MAPED-owned inputs are then released. The complete
merged output stays resident for viewing, without a file or a second merge.
Borrowed inputs remain caller-owned. Float32 alignment, interpolation weights,
and accumulation order are unchanged. Ordinary small float32 `scan_region`
inspection remains available through the existing method.

## Meaning of `dtype` in MAPED

For resident CUDA/MPS inputs, `dtype` controls the merged result's storage;
alignment, interpolation and accumulation still use float32.

| Call | Result |
|---|---|
| `merge_datasets(dtype="scaled_uint16")` | Complete packed, calibrated output; float32 reconstructed reads; measured storage rounding |
| `merge_datasets(dtype="float32", scan_region=...)` | Float32 region without additional storage rounding; at most 4096 scan positions |
| `merge_datasets()` | Existing float32 inspection behavior; a large scan needs an explicit small region |
| `merge_datasets(save_to=...)` | Scaled uint16 by default; streams to disk, releases owned inputs, then reopens the complete packed result |

QuantEM.GPU IO additionally supports **float16** storage. That does not make
`merge_datasets(dtype="float16")` a supported resident MAPED call. The two
reduced-precision IO choices have different meanings: float16 stores rounded
floating-point intensities, while scaled uint16 stores calibrated integer
codes. Both reconstruct float32 values when read. Reconstruction does not undo
storage rounding. Plain uint16 is not a substitute for scaled uint16.

Use only `scaled_uint16`; `uint16_scaled` is not an alias. `dtype` does not choose
ANS versus bit packing, a GPU backend, or MAPED's scientific parameters.
Packing itself is exact relative to the chosen stored values. The result's
precision report describes before/after storage error, not physical accuracy.

## Lower-memory saved workflow

```python
merged = maped.merge_datasets(
    save_to="merged_master.h5", dtype="scaled_uint16", plot_result=False
)
maped.show()
```

For `from_files` inputs, saving avoids holding all input and output codes at
once. Torch computation uses smaller internal batches while retaining the same
storage-calibration boundaries. QuantEM.GPU streams the converted regions to
HDF5, then loads the complete result after MAPED releases its owned inputs.
Sources supplied through `from_resident` remain borrowed and cannot receive
this input-release memory saving automatically. The no-file call still retains
both inputs and output during merging and needs more memory.

The first measured saved workflow peaked at 11.89 GiB on a larger MPS machine
under a 12 GiB Torch cap. This is a candidate for 16 GB Macs, not physical-device
qualification; browser rendering and system memory pressure require testing.
See [measurements](maped-16gb-memory.md).

## Saving and loading

Save during merging with the existing `save_to` argument, or save the returned
resident later without repeating MAPED or applying another precision conversion:

```python
from quantem.gpu import io

io.save("merged_master.h5", merged)
reopened = io.load("merged_master.h5")
patch = reopened.read(scan_region=(252, 260, 252, 260))
```

The existing viewer consumes the loaded source. DP reads, means, detector sums,
and center-of-mass calculations operate in calibrated intensity units. Raw
uint16 codes are storage values, not detector counts. No caller selects regions,
codecs, or scales for reading. A bounded region read returns independently owned
Torch storage on the source accelerator.

Generic `io.save(path, source, dtype="scaled_uint16")` streams conversion and
writing without retaining the complete packed output. Generic
`io.load(source, dtype="scaled_uint16")` retains the complete packed result.
Sources can be files, GPU arrays, or existing generated sources with declared
`shape`, `dtype`, and ordered `blocks()`. Scientific generation stays in QuantEM;
precision conversion, packing, calibrated queries and file IO belong to
QuantEM.GPU. No MAPED kernels or algorithm executor were added to QuantEM.GPU.

## Precision and persisted calibration

Each stored region records its frame bounds, scale, offset, original range,
source dtype, and GPU-measured error report. The version-2 precision record also
contains full geometry, total RMSE, maximum error, changed values and overflow.
Conversion rounds to the nearest code with ties to even; reading reconstructs
float32 intensities from that region's calibration. The realized schedule is
saved, so reloading on another GPU does not recalibrate existing codes.

The error is relative to the corresponding float32 source, not ground-truth
scientific accuracy. Smaller local ranges can lower storage error; they do not
improve the underlying float32 merge. Changing the generation schedule can
change storage rounding. The saved schedule and scales preserve reproducibility.
The brief status line reports stored size, RMSE, maximum error and overflow;
detailed calibration stays in `merged.metadata["precision"]`.

Legacy globally scaled files still load with their original calibration.
Version-2 files retain the existing precision-attribute discovery path so older
Python readers reject the unsupported version instead of showing raw codes.
The existing native Swift global file reader does not yet accept this new
regional file format. Native Metal regional benchmarking remains separate;
Live4DSTEM integration is not part of this change.

Cropped reloads retain the original region audit metrics and label them as saved,
not newly measured error statistics for the crop. Saving such a selection
preserves its selected geometry and calibration. The HDF5 codec's existing
multiple-of-eight detector-element restriction still applies. MPS rejects
float32 subnormal inputs rather than silently flushing their intensities.

## Qualification

The shared NumPy oracle checks the same input values on CUDA and MPS: exact
codes and restored float32 values, signed/constant regions, rounding ties and
wide dynamic ranges. Workflow tests cover cross-region and detector selections,
BF/mean-DP and center-of-mass products, independent read ownership, full
save/reload equality, one-pass generated saving, and legacy global archives.
Existing MAPED dense-versus-resident and median hot-pixel tests also pass.
These checks distinguish storage parity from backend-specific float32 merge
rounding; they do not claim identical full MAPED arrays across GPU architectures.

See the [public-workflow measurements](maped-scaled-storage-performance.md)
and [matched DP review](native-maped-regional-storage.md#matched-diffraction-pattern-review).
