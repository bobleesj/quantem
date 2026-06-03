# How offline data is embedded

The interactive widgets in these docs run with **no Python kernel** - the data
lives in the browser and the compute runs in WebGPU. The hard part is getting a
multi-megabyte array into the page *fast* without losing precision. This page
documents how Show4DSTEM (the heaviest case) does it.

## The problem

A 4D-STEM stack is large (a full 512x512 scan x 24x24 detector is 288 MB as
uint16). The naive path - base64 the array into the widget-state JSON inside the
HTML - is slow and deploy-hostile:

- The browser must parse the whole multi-hundred-MB HTML text, then `JSON.parse`
  the embedded string, on the main thread, before the widget even mounts.
- A 200 MB+ HTML file cannot be committed to GitHub (>100 MB is rejected, >50 MB
  warns) and bloats the repo forever.

## The pipeline

```
detector counts (uint16)
  → uint8 quantize (global linear)        # 2x smaller, near-lossless for the VI
  → gzip (lossless)                        # ~2-3x smaller again
  → [inline base64]  or  [companion .gz]   # two delivery modes (below)
  → DecompressionStream('gzip')            # native, off the parse path, lossless
  → WebGPU storage buffer                  # masked_sum / reduce_frames in WGSL
```

**Why uint8 is fine for bright-field.** The virtual image is a *sum* over many
detector pixels. For a large, bright detector (BF) where each pixel carries many
quantization levels, the per-pixel rounding error is ~zero-mean and averages down
by ~1/sqrt(N) across the aperture, so the summed image is visually identical to
the kernel result; the colormap auto-scales. The raw counts stay uint16 in the
live kernel path - quantization is only the offline display pack.

```{warning}
This is **only safe for bright, large virtual detectors (BF)**. The quantization
is **global-linear** - one 8-bit scale set by the brightest pixel (the central
disk). For **HAADF / ADF, point detectors, or DPC/center-of-mass** the faint
high-angle counts fall below one uint8 level and truncate to **zero** before the
sum - a coherent bias that does *not* average out (the 1/sqrt(N) argument fails
when per-pixel signal < 1 level). For dark-field or few-pixel detectors, apply a
**log/gamma transform before quantizing**, or keep uint16. The three rows in the
table below are bit-identical *to each other* (same uint8 source); they are not
bit-identical to the uint16 kernel, only visually identical for bright detectors.
```

**Why gzip.** `DecompressionStream('gzip')` is native (Baseline since May 2023:
Safari 16.4+, Chrome 80+, Firefox 113+), runs in C++ off the HTML-parse path, and
is bit-exact (lossless). Detector data (lots of low counts) compresses ~2-3x.

## Measured (full 512x512 gold, real data)

| Pack | HTML size | Cold open (Linux / 8 GB Mac) |
|---|---|---|
| uint16 base64 inline | 404 MB | 32 s / ~100 s (or freeze) |
| uint8 base64 inline | 203 MB | 9 s / 51 s |
| **uint8 + gzip inline** | **92 MB** | **3 s / ~15 s** |

All three render a virtual image **bit-identical to each other** (same uint8
source) - the speedups are pure size/parse wins. gzip is lossless; the only
quantization is the shared uint16→uint8 step (see the bright-field warning above).

## Two delivery modes

| | Docs (GitHub Pages) | Share with a colleague |
|---|---|---|
| Opened over | HTTP | double-click (`file://`) |
| Layout | tiny HTML **+ companion `.gz`** fetched at runtime | **one self-contained `.html`** |
| Why | a 90 MB inline file bloats the repo + trips GitHub's 50 MiB warning; Pages serves a sibling and same-origin `fetch()` works over HTTP | people won't manage a folder, and `fetch()` of a sibling is CORS-blocked under `file://` |
| Trait | `data_url=` (relative) | `offline=True` (inline gzipped) |

Both share the same gzip + `DecompressionStream` + WebGPU code; only the *source
of bytes* differs (a `fetch()` vs an inline base64 trait).

## Going faster still (roadmap)

The goal: least internet (bytes) **and** least compute (decode). Ideas, lossless:

- **Instant first paint (recommended next step):** ship a tiny precomputed virtual
  image inline (KB) so the page renders immediately, and load the full stack in the
  background - detector interaction lights up a couple seconds later. This is the
  only lever that feels instant on **both** surfaces: on `file://` the cold-open
  floor is Blink's single-threaded HTML-text parse of the inline bytes (which no
  worker or GPU trick can beat), so the only way to feel instant there is to not
  block on the big payload at all. Adds ~zero bytes, zero precision loss.
- **OPFS / IndexedDB cache:** after the first fetch, cache the decoded bytes;
  re-opening the page is instant with zero network and zero decompress.
- **Better codec (measure first):** brotli (`DecompressionStream('brotli')`) is
  ~15-20% smaller than gzip, but Streams-API brotli is newer + unevenly shipped
  (Safari 18.4+; not broadly in Chromium yet) - check support before relying on it.
  zstd and bitshuffle+LZ4 (the upstream HDF5 detector codec) compress integer data
  even better but need a **wasm decoder** that itself costs bytes + compute to
  load - often erasing the ratio gain. For a widget whose problem is bytes+compute,
  native gzip is usually the sweet spot; reserve wasm codecs for a measured spike.
- **Streaming decode:** overlap fetch + decompress + GPU upload so first frame
  appears before the whole stack lands.
- **Off-main-thread:** decompress in a Web Worker (transferable buffer) so the UI
  never blocks during load.
