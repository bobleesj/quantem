# Scaled uint16 ANS residency

Use `../2026-09-12-scaled-storage/run.py` for no-file runs and
`../2026-09-12-maped-16gb/run.py` for the saved/reopened MPS run. Both accept
an input directory, output report and `--device`. The latter uses a 12 GiB
Torch cap; native driver memory must still be sampled.

The first CUDA run used 512-frame query reduction batches; the final uses
4096-frame query batches. Codec intervals, uint16 codes and calibration are
unchanged. Measurements are individual full runs, not controlled speedups.
Compact measurements preserve aggregate precision errors; MPS full regional
reports matched the previously committed baseline. See `../../maped-ans-output.md`.
