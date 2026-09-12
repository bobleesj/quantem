# ANS output kernel profiling

Seven full experimental inputs, each 512 x 512 scan and 192 x 192 detector,
retained in ANS on an Apple M5 Max with 128 GB. No saving/reopening. The public
Torch MAPED workflow is unchanged. Paths and source identities are excluded.

Trials in `runs.json` retain exact timer samples and precision-report digests:

- `before`: committed implementation.
- `after`, `refined`: fused native range/finite/subnormal reduction.
- `reciprocal`: additionally replace ANS integer division by exact reciprocal
  multiply/correction. Rejected: output encoding 1.131 s versus 1.108 s before.
- `fused`: range reduction plus direct calibrated ANS mean, separate submissions.
- `batched`: submit all direct mean regions in one ordered command buffer.
- `final_uninstrumented`: final implementation using the original public benchmark
  without profiling or the extra mean audit.

All seven complete precision reports have the same digest. The batched run also
compared the entire merged mean DP against the previous decoded GPU reduction
with exact array equality, after all reported workflow timings. This is not a
full-volume float32 bitwise comparison. Scaling error and algorithm parity
remain separate. Tolerances were not relaxed.

Reproduce from the QuantEM repository with both repositories on PYTHONPATH:

```sh
PROFILE_OUTPUT=profile.json \
MAPED_BENCHMARK=docs/development/benchmarks/2026-09-12-ans-kernel-profile/audit.py \
python docs/development/benchmarks/2026-09-12-ans-kernel-profile/profile.py \
  INPUT_DIRECTORY report.json --device mps
```

The audit includes a full-data exact mean comparison outside timed workflow
stages. Its profiling totals include that extra query and the reference decode;
therefore `batched-profile.json` has two output means, and ANS decode totals
include the audit. Other trials used the existing scaled-storage `run.py`.

`before`, `after` and `reciprocal` profiles include pending Torch producer work
in `conversion`. `refined`, `fused` and `batched` separate that work under
`producer_wait`. Native GPU durations include input and output encoding unless
explicitly named `output_ans`; nested counters must not be added together. No
hardware occupancy counters were taken.

Filesystem caches and background services were uncontrolled. These are
individual full runs, not a statistically qualified end-to-end speedup. Native
allocation was sampled through Torch every 20 ms; this is not physical 16 GB
qualification. Viewer construction was timed, not browser presentation.
