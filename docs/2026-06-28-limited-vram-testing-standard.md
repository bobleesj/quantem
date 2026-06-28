# Standard: testing that a workload fits a limited-VRAM card

2026-06-28

## Why this exists

We kept claiming "fits a 24 GB card" off proxy signals and got burned. The honest
rule: **never claim a workload fits a card you do not own without running the real,
full-size workload with only that much VRAM free and confirming it does not OOM.**

A memory estimate is a guess. The real peak includes transients an estimate misses:
the grid_sample cast slab (9.6 GB at no-bin), FFT pairs, the per-batch weighted
product. And an auto-sizing workload (batch size, chunk size) picks a BIGGER config
when it sees a big card, so a 96 GB run never exercises the 24/48 GB path. The only
test that means anything is: cap the VRAM, run the real thing, look at whether it
survives.

## Proxies that do NOT prove a fit

Each of these was green while the real 24 GB run OOM'd in seconds:

| Proxy | Why it lies |
|---|---|
| Auto-trigger fired (chose the out-of-core path) | Choosing the path is not fitting it |
| Bit-exact parity test passed | That tests correctness, not memory |
| Binned (`det_bin=4`) run passed | Small data dodges the real peak |
| Ran fine on the 96 GB card | The cap is the whole point; a big card hides the OOM |

## The procedure

1. **Pick the target card size** in GB (the smallest card you want to support).
2. **Load the real, full-size workload** - all data, no binning, the exact thing
   that runs in production. NOT a 2-item subset, NOT `det_bin`.
3. **Cap the VRAM to the target** with `vram_capped(target_gb)` AFTER the resident
   data is loaded (see gotcha below).
4. **Run the operation under test inside the cap.** If it raises
   `torch.cuda.OutOfMemoryError`, it does not fit. If it completes, it fits.
5. **Read the real peak** with `torch.cuda.max_memory_allocated`. That number is
   the definitive "this workload needs N GB."
6. **Make it an automated test** so the claim is machine-checked forever, not
   re-asserted by hand.

## The tooling

`tests/diffraction/_vram_cap.py`:

```python
from _vram_cap import vram_capped, fits_in_vram

# context-manager form - you control the assertion
maped.real_space_align(...)            # resident data loaded FIRST
with vram_capped(48) as cap_bytes:     # only 48 GB free now
    merged = maped.merge_datasets()    # must not OOM
peak = (torch.cuda.max_memory_allocated(0) - cap_bytes) / 1e9   # cap subtracted

# one-shot form - returns (fits, peak_gb)
fits, peak = fits_in_vram(48, lambda: maped.merge_datasets())
assert fits and peak < 48
```

`vram_capped(target_gb)` allocates a uint8 block of `free - target_gb` so only
`target_gb` stays free, resets peak stats, and frees the block on exit.

## Gotchas (every one of these cost real time)

- **Cap AFTER the resident data is loaded.** `vram_capped` leaves `target_gb` free
  *at the moment it is entered*. If you cap before loading, the load eats the
  budget. For the merge: align first, then cap, then merge.
- **A workload that releases memory will see more than `target_gb`.** If the op
  frees resident data at the start (the merge releases its held tilt), free grows
  past the cap. Run the test from a FRESH state (not after another full run) so the
  cap is representative, and trust the measured peak over the cap arithmetic.
- **Measure the peak; do not trust the formula.** The peak from
  `max_memory_allocated` is the truth. An estimate that says "fits" can still OOM.
- **Verify with the workload's OWN completion signal,** not a downstream check that
  itself blows up. A full-res bit-exact diff `(a - b).abs().max()` allocates three
  38 GB host arrays (~114 GB RAM) and gets OOM-killed AFTER the merge already
  finished - that is a test-harness failure, not a merge failure. Use the merge's
  `tqdm 100%` / returned tensor as proof it ran; check bit-exactness separately on
  small (`det_bin`) data.
- **Never truncate the log you are reading the result from.** Piping a run through
  `| tail -N` can drop the result line out of the saved file before it lands. Write
  the full stream to a file, then grep the file.

## Worked example: MAPED no-bin merge

7-tilt, full-res 512x512x192x192. The resident tilt is uint16 = 19.3 GB = 79% of a
24 GB card. Measured with this procedure:

| Target card | Result | Note |
|---|---|---|
| 96 GB | in-VRAM, 12.4 s | accumulator stays on GPU |
| 48 GB | out-of-core, 34 s, fits | accumulator streams to CPU RAM, one tilt on GPU |
| 24 GB | OOM | tilt 19 GB + 9.6 GB cast slab = ~29 GB, over 24 |

So the honest support statement is: **no-bin out-of-core wants a ~48 GB card; a true
24 GB card needs `det_bin=2`** (halves the tilt). We learned that by capping and
running, not by estimating.

## The one-line rule

If you are about to say "this fits X GB," and you have not run the real workload
under `vram_capped(X)` to no-OOM, you do not know it fits. Cap, run, look.
