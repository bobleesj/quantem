# Proposed MAPED scaled-storage API

Status: proposal. Native Metal regional storage has a full-data benchmark;
CUDA and Torch MPS adoption and generic file/viewer support are not implemented.
The existing default stays unchanged. The [measured native experiment](native-maped-regional-storage.md)
explains its precision, timing, memory, and limitations.

## Scientist-facing decision

Keep float32 scientific computation and choose how the merged result is stored.
Extend `merge_datasets`; do not introduce a regional MAPED class or a new merge
function. Reuse `dtype` for storage precision and add one explicit `scaling`
choice. The proposed calls below are not runnable in the current public API:

```python
# Proposed: full output resident on the GPU, no file required.
merged = maped.merge_datasets(
    dtype="scaled_uint16", scaling="regional", plot_result=False,
)
maped.show()

# Proposed: same scientific result and precision policy, additionally saved.
merged = maped.merge_datasets(
    dtype="scaled_uint16", scaling="regional",
    save_to="merged_master.h5", plot_result=False,
)
```

| Choice | Meaning | Consequence |
|---|---|---|
| `dtype="scaled_uint16", scaling="global"` | One scale and offset for the entire output | Existing complete-source range policy; normally two bounded merge passes |
| `dtype="scaled_uint16", scaling="regional"` | Each bounded scan region has its own scale and offset | One merge pass; local error bounds; usually finer precision in dimmer regions |
| `dtype="float32"` | Preserve float32 output values | No scaled-integer storage error; full residency may exceed the available memory |

For scaled uint16, `scaling` defaults to `"global"` to preserve the established
contract. Explicit `scaling` with another dtype is an error with a corrective
message. Existing `dtype=None` and unrelated `scale_output` behavior must be
preserved; `scale_output` must not be silently repurposed as storage scaling.
Support for the full resident no-file path must be implemented explicitly; the
current resident Python API otherwise asks for a file for a complete merge.

The source backend remains the backend already selected on the MAPED object:
CUDA, Torch MPS, or native Metal. All input residents remain encoded by default;
merged uint16 output remains packed by default. Scientists should not select
codecs, call decoding functions, provide region sizes, or pass backend flags at
each stage. The automatic region schedule is recorded for reproducibility.

Keep existing result types. Logical reads return calibrated intensities, with
float32 reconstruction where needed. Stored uint16 codes are not detector counts
and must never be presented as calibrated intensities without their scale.
The small status message should report the storage policy, resident size, RMSE,
maximum error, and overflow count; detailed region reports belong in metadata.
For example: `regional scaled_uint16 | 6.20 GiB | RMSE 0.00545 | max 0.0131 | overflow 0`.
The numerical values here describe the qualified acquisition, not universal defaults.

## QuantEM.GPU owns the reusable support

MAPED owns alignment, interpolation weights, and scientific stage order.
QuantEM.GPU owns precision conversion, packed residency, calibrated reads and
reductions, metadata validation, and save/load. Regional storage must be reusable
by other scientific algorithms. Do not add `quantem.gpu.maped_merge_*` functions
or put backend kernels in the Python MAPED algorithm.

Extend the existing generic save boundary consistently:

```python
# Proposed generic IO extension; not implemented yet.
gpu_io.save(path, source, dtype="scaled_uint16", scaling="regional")

# Existing load shape; future regional files detect their calibration metadata.
loaded = gpu_io.load(path, representation="packed")
```

No `scaling` argument is needed to read a saved file: the persisted calibration
is authoritative. Loading cannot invent a global scale for regional codes.
The native implementation should extend its existing packed source to hold
per-region calibration and support ordinary whole-array coordinates. The Python
boundary must provide the same logical reads without requiring MAPED or viewers
to reconstruct region tables. Reuse the current precision conversion and packing
kernels; the experiment already composes those operations successfully.

## Calibration and file contract

For each region, preserve consecutive frame boundaries, scale and offset with
sufficient precision, intensity units, finite range, source float32 dtype,
stored uint16 dtype, count, error metrics, and correction/merge provenance.
Preserve the full logical 4D shape and axes. Persist the realized region schedule;
reloading must reproduce the same values even if a different GPU would choose
a different batch size for new work.

Use a versioned, generic calibration schema, distinct from the existing global
precision record. New readers detect it. Readers without support must reject the
file explicitly. Do not export regionally scaled codes with only the old global
scale attribute. A successful save must preserve calibration and data together,
with incomplete output never admitted as complete.

Cross-region slices, detector reductions, BF/ADF/DPC summaries, ROI means and
exports must use calibrated intensities. Where calibration can be applied after
a reduction algebraically, include the correct offset contribution and selected
pixel count. Otherwise restore bounded float32 values on the GPU before reducing.
Do not compare or sum raw code values from regions with different scales.

For Show4DSTEM and Live4DSTEM, keep the existing viewer entry point. The source
reader handles regional calibration; the renderer must receive calibrated values
or an explicitly supported scale-aware GPU source. Independent per-image display
normalization must not hide precision differences in a validation comparison.
UI integration remains a separate task from this API proposal.

## Required qualification before adoption

- Same float32 fixture on CUDA, Torch MPS, and native Metal: independently verify
  scale, offset, round-to-even uint16 codes and restored values against NumPy.
  Include zeros, constant regions, signed finite values, ties, high dynamic range,
  partial final regions, nonfinite/subnormal handling, and explicit unsupported errors.
- Full MAPED results: retain frozen alignment and parameter-sensitivity tests;
  compare storage error to each backend's float32 result separately from existing
  cross-backend float32 execution tolerances. Threshold crossings can change codes
  when the input float32 values differ; do not conflate that with codec failure.
- Read tests: exact local restoration, slices crossing region boundaries,
  reordered/selected frames, detector subsets, scalar and image reductions, and
  fresh-versus-saved equivalence with matching metadata and ownership behavior.
- Viewing: paired float32/global/regional DPs with shared intensity limits and
  signed residual limits; exercise moves across region boundaries, BF maps and
  numerical readouts. Native DP read latency alone is not viewer qualification.
- Full-scale performance and memory on each physical target, including a 24 GiB
  Mac. Report loading, processing, optional saving and rendering separately.

The native experiment achieved 13.26–13.76 s without saving/rendering and lower
storage RMSE. It establishes feasibility, not completed CUDA/MPS/file/UI parity.
