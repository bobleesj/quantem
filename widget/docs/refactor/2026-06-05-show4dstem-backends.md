# Show4DSTEM — Backend vs Backendless refactor

Architecture ledger for the phased refactor that landed on
`widget-show3d-show4dstem-kernels` 2026-06-05.

## Why

Show4DSTEM had grown two divergent classes (~3.6k LOC total) — `Show4DSTEM`
(torch on CUDA/MPS/CPU) and `Show4DSTEMMPS` (raw-Metal chunked for Phil's
19.3 GB Samsung-class stacks where torch.MPS hits the >2^31-element buffer
limit). The compute paths were already abstracted in
`kernels/compute/backends.py`, but the MPS subclass duplicated lifecycle
logic (fast_vi sidecar, radial cache, multi-dataset proxy) by reaching into
`self._data.vi.*` directly. Adding a third backend (WebGPU online / offline)
would have meant carving a third parallel widget file. This refactor
collapses the abstraction so backend selection is a runtime concern, not a
class-hierarchy concern.

Tied issues: #772 (single-source quantem.live widgets/), #775 (single-source
WebGPU frontend), #746/#745/#744/#743 (MetalRaw lifecycle + perf), #747 (5D
time-series binned), #754 (backendless HTML with sibling .h5),
#740/#737/#763 (Mac MPS expansion).

## Framing — backend vs backendless

| Mode | Data lives | Compute runs | Status |
|---|---|---|---|
| **Backend / Torch** | Python torch.Tensor on CUDA / MPS / CPU | Python via torch | shipped (`TorchBackend`) |
| **Backend / MetalRaw** | Python `MPSChunked4DSTEM` (Metal unified-memory chunks) | Python via raw Metal kernels (`MetalVirtualImage`) | shipped (`MetalRawBackend`) |
| **Backendless / Offline** | Browser, embedded in `<script>` block of standalone HTML | Browser WebGPU (`js/engine/compute.ts`) | shipped — `widget.export_html()` + `_pack_offline_bslz4()` |
| **Backendless / Online** | Browser WebGPU GPU buffer (kernel streams chunks) | Browser WebGPU | designed, not implemented |

## Phase 1 — Shipped 2026-06-05

Commits `831bb577` (refactor) + `e208cc8b` (snapshot label work).

### `kernels/compute/backend.py` (new, 109 LOC)

`ComputeBackend` `Protocol`, runtime-checkable. Required surface:

```python
backend.scan_shape    -> (int, int)
backend.det_shape     -> (int, int)
backend.n_frames      -> int
backend.device        -> str | torch.device
backend.capabilities  -> tuple[ComputeCapability, ...]

backend.frame(idx)                     -> np.ndarray (det_r, det_c)
backend.masked_sum(det_mask)           -> np.ndarray (scan_r, scan_c) float32
backend.mean_dp()                      -> np.ndarray (det_r, det_c) float32
backend.reduce_frames(idx, "mean"|"sum"|"max") -> np.ndarray (det_r, det_c) float32
backend.center_of_mass(det_mask=None)  -> (com_col, com_row) flat (N,) float32
```

Optional capability hooks are advertised via `capabilities` tuple and
called only after a `'<cap>' in backend.capabilities` check. They're NOT
part of the runtime Protocol so `isinstance(b, ComputeBackend)` passes
for any backend that implements just the required surface.

Capability strings:
- `'fast_sidecar'` — `ensure_fast_sidecar(verbose)`, `cache_fast_presets(masks)`, `fast_bin`, `has_fast`
- `'radial_cache'` — `ensure_radial_cache(row, col)`, `radial_cache_ready(row, col)`, `radial_masked_sum(...)`
- `'row_prefix_exact'` — marker that radial cache uses exact row-prefix sums (no bin)
- `'multi_dataset'` — `set_active_dataset(idx)`, `multi_n_ready`, `multi_names`, `multi_active_idx`, `multi_total()`, `set_multi_ready_callback(cb)`

### `kernels/compute/backends.py` — renames + lifecycle move

- `TorchCompute` → `TorchBackend` (alias kept for one release)
- `MetalCompute` → `MetalRawBackend` (alias kept for one release)
- `MetalRawBackend.capabilities` advertised dynamically based on what
  `ChunkedFrames` actually supports:
    - `'fast_sidecar'` always
    - `'radial_cache'` + `'row_prefix_exact'` when `data.vi.row_prefix_enabled`
    - `'multi_dataset'` when `data` has `set_active` + `on_ready`
- All MPS lifecycle methods now live ON the backend, not in the widget subclass:
  - `ensure_fast_sidecar(verbose)` (blocking; idempotent on already-binned data)
  - `cache_fast_presets({"bf": mask, ...})` → `dict[str, np.ndarray]`
  - `ensure_radial_cache(row, col, *, idle_delay_s=0.75)` (async; idempotent at the same center; cancels stale builds when center moves)
  - `radial_cache_ready(row, col)` / `radial_masked_sum(...)` / `radial_building` / `radial_error`
  - `set_active_dataset(idx)` / `multi_n_ready` / `multi_names` / `multi_active_idx` / `multi_total()` / `set_multi_ready_callback(cb)`

### `show4dstem_mps.py` — UI-only subclass (908 → 802 LOC)

`Show4DSTEMMPS` keeps:
- The four MPS traits (`fast_interaction`, `fast_interaction_ready`,
  `fast_interaction_building`, `radial_interaction_ready`,
  `radial_interaction_building`) and their observers
- The detector preset cache (BF/ABF/ADF/HAADF arrays cached in instance attrs)
- The numpy ROI-mask builder (`_detector_mask_np`)
- `set_fast_interaction(enabled, wait=)` / `wait_for_fast_interaction` /
  `wait_for_radial_interaction` operator-facing controls

Removed:
- Direct `self._data.vi.*` calls — every Metal interaction now flows through
  `self._compute.<lifecycle_method>`
- The radial-cache background thread + idle-delay polling — now owned by
  `MetalRawBackend.ensure_radial_cache`
- The multi-dataset proxy wiring details — now owned by
  `MetalRawBackend.set_multi_ready_callback`

### Verification (Phase 1)

- mjgoat CUDA path: `Show4DSTEM(load('.../logic_013_master.h5', det_bin=4))`
  → `_compute = TorchBackend`, `capabilities = ()`. masked_sum + frame +
  mean_dp all produce expected shapes/dtypes. `live notebook publish` hook
  baked real Show4DSTEM canvas PNG into ipynb in 3.8 s. Verified
  bit-identical against pre-refactor `publish_dict` output.
- Phil MPS path (`load(backend='mps', det_bin=4)`):
  - `_compute = MetalRawBackend`, `capabilities = ('fast_sidecar',)`
  - `masked_sum(full mask)`: 13 ms (real-time)
  - `mean_dp()`: 57 ms, `frame(0)`: 0 ms
  - `virtual_image_bytes` (1 MB) + `frame_bytes` (9 KB) populated
  - Zero Chrome processes touched on Phil

## Phase 3 — Backendless / Offline (already shipped, verified post-refactor)

`widget.export_html(path)` produces a self-contained `.html` that mounts
the live anywidget JS bundle with all current widget state embedded.

The existing implementation uses:
- `_clone_for_html_export()` → builds an export-only widget with
  `_offline_stack` (uint8 quantized) + `_offline_bslz4` (HDF5
  bitshuffle+LZ4 metadata for 8-plane GPU fast path) + `_offline_bad_px`
  populated
- `embed_minimal_html()` from `ipywidgets.embed` writes the standalone HTML
- Browser-side `Show4DSTEMCompute` (`js/engine/compute.ts`) does ALL
  subsequent reductions in WebGPU: `maskedSum`, `frameAt`, `reduceFrames`,
  `maskedCoM` — same shaders the live offline-mode interactive widget uses

Post-refactor verification (mjgoat → Linux Chrome on `:1`):
- `Show4DSTEM(load('.../det_bin=4'))` → `export_html('/tmp/x.html')` →
  2 MB self-contained file in <0.1 s
- Open in Linux headed Chrome → 8 canvases mounted (BF detector + CBED +
  scale bars + histograms + sliders), full interactivity
- Title, magma CBED, virtual image, ROI mode, presets all working

Conceptually `Show4DSTEM` in offline mode IS a "WebGPUOfflineBackend" — the
Python widget object has no `self._compute` after `_clone_for_html_export`
runs; all derived compute happens in the browser. The Protocol stays
unviolated because offline-export widgets are write-only (the operator
exports them; they're never queried via `self._compute.<method>` after).

Open question: #754 (backendless HTML that auto-loads a sibling .h5).
Today's `_pack_offline_bslz4` bakes the FULL data into the HTML. The
sibling-.h5 alternative would have the HTML fetch the .h5 via `fetch()` +
do bslz4 decode in the browser. That's a future optimization; not needed
to call Phase 3 "shipped".

## Phase 2 — Backendless / Online (designed, not implemented)

The future addition. Goal: Python kernel HOLDS the raw 4D stack
(any data type), but the BROWSER does all reductions via WebGPU. Useful
for cross-platform Mac users where torch.MPS isn't viable and the data
is too big to bake into an offline HTML.

### Protocol changes

Add a new backend that implements the required surface BUT delegates
each compute call to a JS Comm channel:

```python
class WebGPUOnlineBackend:
    capabilities = ()  # JS lifecycle has its own model

    def __init__(self, widget):
        self._widget = widget                       # weak ref
        self._pending: dict[int, asyncio.Future] = {}

    def masked_sum(self, det_mask):
        req_id = self._next_req()
        self._widget.send({"op": "masked_sum", "req_id": req_id,
                           "mask_bytes": det_mask.tobytes()})
        return self._await(req_id)  # blocks Python until JS replies
```

### JS side

Browser holds the stack as a single WebGPU storage buffer (chunked if >2
GB to dodge per-buffer caps). Mounts on first request via `widget.recv`:

```typescript
model.on("msg:custom", async (msg) => {
  if (msg.op === "masked_sum") {
    const vi = compute.maskedSum(stackBuffer, msg.mask_bytes, scanShape);
    model.send({req_id: msg.req_id, result_bytes: vi.buffer});
  }
});
```

### Streaming protocol

For large stacks, kernel ships chunks lazily — JS requests block N, kernel
ships, JS caches in GPU buffer with LRU eviction. Reuse the existing
`_offline_bslz4` chunk metadata format so the JS decode path is shared.

### Estimated effort

~1-2 weeks. Touches:
- `kernels/compute/webgpu_online.py` (new, ~200 LOC) — Python side
- `js/engine/online-channel.ts` (new, ~300 LOC) — JS Comm + buffer cache
- `show4dstem.py` — accept `backend='webgpu'` and route compute calls
  through the async backend
- Comm protocol + serialization tests

### When to build

Triggers:
- Mac user without working torch.MPS (or stack > torch.MPS limit) and the
  data is too big for offline HTML (>500 MB after uint8 quantization)
- Cross-platform browser-first deployment (no Jupyter kernel needed for
  most operations — wait, that's Phase 3 territory)

Phase 2 specifically helps when: Python kernel HOLDS the data (because it
came from disk via Python), but Python can't do reductions efficiently
(no CUDA, no Metal, torch.MPS overflows). Today's path either uses
TorchBackend('cpu') (slow) or fails. WebGPUOnlineBackend would solve this
by deferring to the browser GPU.

## Follow-ups (post-Phase-1)

- **#772 closure** — quantem.live `widgets/show4dstem_*.py` (~3.3 kLOC)
  becomes a `from quantem.widget.show4dstem import Show4DSTEM` re-export.
  Gated on this commit landing in widget upstream + live notebooks getting
  re-pointed. ~1 hour work, separate commit.
- Capability `'fft'` — `MetalRawBackend` could implement FFT via
  `MetalVirtualImage`'s row-prefix engine for exact mode. Today FFT is
  CPU-side numpy in Python (`np.fft.fft2`). Worth measuring.
- Capability `'com_cache'` — formalize `MetalRawBackend._com_cache` as a
  capability hook so the widget can `if 'com_cache' in caps: use_cached`.
