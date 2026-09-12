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
