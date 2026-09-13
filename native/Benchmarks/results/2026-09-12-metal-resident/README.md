# Native Metal resident MAPED handoff

Status: native processing reproduced on the Apple M5 Max test host; application UI integration is a
separate task. The canonical API and timing table are in [native/README.md](../../../README.md).

## Implementation ownership

- QuantEM `native/Sources/QuantEMMAPED/MAPEDNative.swift`: scientific sequence,
  parameters, float32 merging, summary provenance and optional export policy.
- QuantEM.GPU `Metal4DSTEMStreamingIO`: generic ANS storage, calibrated reads,
  precision reports and typed HDF5 writing. No MAPED algorithm was added there.
- `MetalPackedSource` is the existing public result type. Calibrated appends
  use ANS despite the historical class name; global packed sources still work.
- Native Swift runs without Torch. Torch MPS is the independent numerical
  oracle used by the qualification scripts.

## Reproduction

Use matching local QuantEM and QuantEM.GPU checkouts on the Apple M5 Max test host. Use the QuantEM commit containing this report and QuantEM.GPU
`34d2e6dcf69f05890ca3947ee39ca290d956d6a8`, recorded in `Package.resolved`.
The tests used the local dependency override with that implementation.

```bash
export QUANTEM_GPU_PACKAGE=/path/to/quantem.gpu
swift test -c release --filter MAPEDNativeTests
MAPED_NATIVE_DISPLAY=1 swift run -c release maped-native-benchmark \
  /path/to/seven/tilts /path/to/new/display.json
# Add MAPED_NATIVE_EXPORTS=/path/to/new/exports for both optional saves.
swift run -c release maped-parameter-parity \
  /path/to/seven/tilts native/Tests/parameter_cases.json /path/to/new/parameters
PYTORCH_ENABLE_MPS_FALLBACK=0 PYTHONPATH=src:/path/to/quantem.gpu/src \
  python native/Tests/compare_parameters.py \
  /path/to/seven/tilts native/Tests/parameter_cases.json /path/to/new/parameters
```

`display.json` contains timings and actual file sizes. `display-repeat.json`
records the final no-save repeat: 24.85 s versus 18.00 s initially, with the
same sampled memory peak and whole-output display RMSE. `parameter-summary.json`
records 75 cases and 45 sensitivity intervals against the unchanged gates.
`export-parity.json` records five complete scan-row comparisons performed on
MPS, including near both scan boundaries. Twelve native tests passed, including
an exact GPU comparison after scaled ANS save/reopen across region boundaries.
Full private observations and exports remain outside the repository. To rerun
saved-file checks against the full private benchmark report:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=0 PYTHONPATH=src:/path/to/quantem.gpu/src \
  python native/Tests/compare_resident_exports.py /path/to/seven/tilts /path/to/run
```

The compact committed `display.json` intentionally omits alignment arrays and
calibration details; use the full report emitted by the benchmark for this check.

## Integration constraints

Display reads are bounded GPU float32 buffers restored from scaled uint16 ANS.
The UI should use those reads, not allocate the entire 36 GiB float32 output.
Saving scaled data preserves existing codes and scales; requesting original
float32 remerges retained inputs. HDF5 files use GPU bitshuffle/LZ4, while the
resident display uses ANS. Disk bytes and resident bytes are therefore different.

Retain the MAPED object and seven sources until parameter changes or exact
exports are no longer needed. Replacing a result invalidates the old resident
result, and recomputing with an existing result can temporarily retain both;
the 14.76 GiB peak qualifies the initial display run, not every later operation.
Wire cancellation, progress and GPU buffer ownership in the integration task.
Do not present unimplemented merge options as working sliders. Use the existing
parameter contract to distinguish scientific, inactive and rejected options.

the Apple M5 Max test host has 128 GB unified memory. A sampled Metal peak is not process footprint,
and neither a physical 16 GB nor 24 GB Mac was qualified in this run. UI upload,
rendering and cold-start shader compilation are not part of display-ready time.
Native float32 file reopening still needs a native app reader qualification;
these saved float32 comparisons used the public Python MPS loader.

The earlier `quantem.gpu.remote.maped_api` service remains a separate ownership
exception to resolve with its integration owner; this change does not extend it.
