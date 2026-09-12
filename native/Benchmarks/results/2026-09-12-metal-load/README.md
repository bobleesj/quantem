# Seven-tilt native Metal measurements

Physical Apple M5 Max, 128 GiB unified memory. Acquisition identity and paths
are omitted. Counts have shape `(512, 512, 192, 192)` for each of seven tilts.

- `phil-load.json`: earlier per-tilt preparation diagnostic, 13.69–14.73 s total.
- `phil-end-to-end-diagnostic.json`: 85.71 s before optimization.
- `phil-optimized-1.json`, `phil-optimized-2.json`: final 32-thread prepared
  sampling, reused bounded storage and overlapped compressed-byte writing;
  22.56 and 22.87 s through packed GPU reopening.

`phil-full-float32-parity.json` records the final full GPU comparison;
`saved-chunk-parity.json` records the exact compressed-output comparison.

The `alignment_wall_seconds` field includes loading. `total_wall_seconds`
excludes validation-file exports. File-write time overlaps GPU work in the
optimized reports. Timing includes an existing index and uncontrolled OS cache;
these are not cold-storage or application-rendering measurements.

See [implementation, parity, and limitations](../../../../docs/development/native-maped-processing-performance.md).
