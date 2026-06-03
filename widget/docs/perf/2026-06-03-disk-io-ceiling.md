# Disk-IO ceiling for cold 4D-STEM loads (2026-06-03)

Motivation: with 40-50 datasets and limited VRAM/RAM, **cold disk read is the load cost
that matters** (warm page cache is the lucky case). Goal: drive cold read toward zero and
toward using disk as an out-of-core memory tier for reconstruction. Measured on mjgoat,
gold master (27 files, 3.15 GB compressed) on the WD SN850X. No loader changes — pure
measurement. Cold = `posix_fadvise(DONTNEED)` self-evict (no root needed).

## Hardware

| NVMe | model | mount | negotiated link | ~seq BW |
|---|---|---|---|---|
| nvme0 | Samsung 9100 PRO 1TB | **unmounted** (only /boot/efi) | **32 GT/s Gen5 x4** | ~14 GB/s |
| nvme1 | Samsung 9100 PRO 1TB | `/`, `/home/owner`, `/tmp` | 16 GT/s Gen4 x4 | ~7 GB/s |
| nvme2 | WD_BLACK SN850X 8TB | `/home/owner/ssd` (DATA) | 16 GT/s Gen4 x4 | ~7 GB/s |

GPU0 PCIe Gen5 x8 (~32 GB/s H2D) — the bus is NOT the floor; the disk is.

## Cold vs warm read scaling (WD nvme2, gold 3.15 GB)

| threads | cold GB/s | warm GB/s |
|---|---|---|
| 1 | 2.36 | 9.46 |
| 8 | **6.02** | 30.6 |
| 12 (loader default) | 5.83 | 31.8 |
| 16 | 6.08 | 31.3 |
| 32-48 | ~5.5 | ~29 |

- **Cold ceiling ≈ 6.0 GB/s** = 82% of the WD's 7.3 GB/s rating. The loader's 12-thread pool
  is already in the optimal 8-16 band. The READ path is near-optimal for this drive.
- 3.15 GB cold ≈ **525 ms**; warm ≈ 100 ms. The 376 ms full-load number reported earlier was warm.

## Things that did NOT help (measured, rejected)

- **kvikio 26.02 (GPUDirect Storage lib) in compat mode**: 0.22 cold / 2.77 warm GB/s on a
  single file — *slower* than POSIX. Without the `nvidia-fs` kernel driver kvikio can't do
  real NVMe→VRAM DMA; compat falls back to synchronous unaligned host reads. Useless until
  the GDS driver is installed.
- **Striping across nvme1 + nvme2 concurrently**: 4.58 GB/s aggregate — LESS than nvme2
  alone (6.02). nvme1 is the busy root drive (3.48 GB/s under OS/dashboard contention) and
  drags the aggregate. A real striping win needs dedicated, idle drives, not the root disk.
- More threads (>16): no gain, slight loss (host queue depth).

## The real levers to push cold toward zero (all need operator/root action)

1. **Mount + use the idle Gen5 nvme0 (~14 GB/s) for hot datasets.** It negotiates 32 GT/s
   (true Gen5) but holds no data today. Moving working datasets there is ~2.3× the WD's cold
   BB for the cost of an `fstab` mount. Biggest near-free win. NEEDS ROOT (mount).
2. **GPUDirect Storage** — install `nvidia-fs` + `modprobe`, then kvikio (already installed)
   does NVMe→VRAM DMA, bypassing the CPU and the host-pinned bounce. This is the enabler for
   disk-as-memory out-of-core recon (read batches straight to GPU, CPU free). NEEDS ROOT.
3. **Stream-overlap read with GPU compute (software, parity-gated).** Today prepare (disk)
   and decompress (GPU) are serial. Pipelining them hides the GPU phase under the read so
   *felt* latency approaches the raw read time; for recon, stream batches NVMe→GPU on demand
   and never hold the full dataset — disk BW only matters if it can't keep the GPU fed
   (6 GB/s/drive vs the kernel consume rate). This is the out-of-core recon architecture.
4. **Dedicated RAID0 across multiple idle NVMe** (nvme0 + a second dedicated drive) →
   additive cold BW (~14+14 ≈ 28 GB/s). NEEDS ROOT + dedicating drives.

## Bottom line

The loader's read is already at 82% of this Gen4 drive's floor at the optimal thread count —
software thread-tuning is done. Cold read speed now is a **hardware + architecture** problem:
faster/striped drives (Gen5 nvme0 is idle), GPUDirect Storage for CPU-free NVMe→VRAM, and
stream-overlap so the read hides behind GPU compute. The last one is the path to "disk as a
memory tier for reconstruction."
