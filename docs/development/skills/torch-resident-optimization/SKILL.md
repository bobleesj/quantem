---
name: torch-resident-optimization
description: Optimize scientific Torch processing over encoded GPU-resident data while preserving algorithm ownership, bounded memory, and numerical parity. Use for resident read, interpolation, reduction, and processing-to-view performance on CUDA or MPS.
---

# Torch resident optimization

Keep scientific algorithms in their scientific package. QuantEM.GPU provides
reusable IO, residency, bounded reads, codecs, reductions, and precision
conversion. Do not move MAPED or another scientific workflow into a GPU IO
helper. Python scientific code remains Torch; backend-specific code belongs
in the infrastructure package when it provides a reusable capability.

## Start with the actual data path

Check the current implementation before assuming a copy exists. CUDA's existing
DLPack path shares decoded native storage with Torch without a copy. MPS's
qualified direct read writes into Torch-owned storage. These are different
ownership arrangements with the same goal: avoid copying decoded arrays.
Returned data must survive later reads and source closure. Respect stream/queue
dependencies before writing recycled memory. Never manually release storage
owned by Torch or expose an output buffer that the next read overwrites.

Retain encoded inputs and bounded working regions. Measure process-level GPU
memory as well as Torch allocation: the Torch allocator omits other backend
allocations. A 24 GiB allocator cap on a larger machine is not a physical
24 GB laptop qualification. Scalar control metadata and file IO may use the
host; scientific array computation stays on the selected GPU.

## Optimize memory traffic before inventing another API

- Reuse invariant indices, weights, and grids where their inputs are unchanged.
- Pass changing alignment parameters as tensors to compiled functions. Measure
  cold compilation separately; avoid specialization on each scalar setting.
- Fixed region shapes can outperform dynamic-shape compilation. Benchmark both
  when relevant; do not assume compilation helps every backend.
- For ordered accumulation into zeros, the first valid term may initialize the
  output directly. Zero only uncovered borders, then retain the original order
  for later terms. Validate empty overlaps, boundary regions, fractional and
  integer shifts, and full-range counts against the original arithmetic.
- Combine compatible range/overview reductions during a traversal when useful,
  without silently changing reduction order or reported precision.
- Tune region sizes only with speed and parity evidence. Smaller regions may
  reduce memory while adding launches, reads, and different reduction rounding.

## Qualification

Freeze the original implementation before edits. Compare on the same resident
inputs, alignment parameters, device, and precision. Alternate original and
candidate order, synchronize timing boundaries, repeat complete passes, and
retain negative trials. Do not recapture baselines or widen tolerances to make
an optimization pass. Synthetic edge tests complement full-volume comparisons.

Separate loading, preprocessing/alignment, selected-DP compute, complete merge,
overview reductions, global-range measurement, scaled conversion, packing,
saving, reopening, and viewer construction. Constructor or backend frame-read
timing is not browser rendering or interaction latency.

Exact global scaled uint16 requires the final range. If a full float32 result
cannot remain resident, two bounded passes may be necessary. Do not quote a
one-pass float32 overview as the latency of the complete scaled resident view.
Approximate storage error is separate from float32 algorithm parity.

## Retained examples

- [CUDA MAPED](../../cuda-maped-processing-performance.md): direct first-term
  initialization reduced paired complete merges from 6.27 s to 5.79–5.83 s.
  Two 9.66-billion-value audits were exact. The change is entirely Torch.
  The MPS compiled form failed exact CUDA parity and was rejected; 2048-frame
  regions were slower than 4096. QuantEM.GPU was unchanged in this experiment.
- [MPS MAPED](../../mps-maped-inspection-performance.md): direct reads into
  Torch-owned storage plus fixed-shape interpolation compilation reduced a
  paired complete merge from 19.93 s to 10.02 s with all float32 values exact.
  This combined result includes infrastructure and Torch changes; do not
  attribute it solely to either one.

These are measured examples, not universal latency or cross-device parity
promises. Keep benchmark scripts, evidence, notebook examples, and this skill
in the same repository; local skill discovery may link to this directory.
