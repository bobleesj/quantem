# Resident memory experiments

Use the public benchmark in `../2026-09-12-scaled-storage/run.py` for baseline
and accepted production changes. Set `MAPED_BENCHMARK` to that script when
running a prototype wrapper, with the same input directory, output JSON and
`--device` arguments. All scientific work remains on the selected GPU.

The wrappers are rejected experiments, not supported APIs. The cache trial flushed native and Torch allocator caches once before merging. `try_subregions.py` disables the compiled MPS interior
path by requesting explicit regions. `try_subregions_compiled.py` keeps that
path but temporarily changes internal frame sizing; it is specialized to the
512-column qualification geometry, not arbitrary shapes.

`results.json` retains anonymous measurements. Before/after refer to the
float32-copy and generator-lifetime edits. Timing variation between separate
runs must not be interpreted as a repeatable speedup. The full qualification
and limitations are in `../../maped-scaled-storage-performance.md`.
