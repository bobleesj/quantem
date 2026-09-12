# Native MAPED IO qualification

Physical Apple M5 Max, 128 GiB. Seven `(512,512,192,192)` uint16 inputs.
Acquisition identity and paths are omitted.

`paired-reference.json`, `paired-candidate.json`, `paired-reference2.json` are
an ordered same-executable A/B/A comparison. All use input read-ahead; B retains
packed output rather than reopening it. `retained-full-1.json` is the initial
candidate run. `load-profiles.json` contains sequential/read-ahead phase reports.
`single-probe-2.json` records the rejected exact-byte float cache experiment.
`read-ahead-count-parity.json` was run with complete GPU count verification;
its slower loading time includes the audit. `saved-chunk-parity.json` records
exact serialized output and metadata parity.

[Interpretation and limits](../../../../docs/development/native-maped-io-performance.md).
