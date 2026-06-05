"""WebGPUOnlineBackend — Phase 2 of the backend-vs-backendless refactor.

Python kernel HOLDS the raw 4D-STEM stack. Browser does ALL reductions via
WebGPU. Bytes flow once from kernel to browser (full stack or chunks);
every subsequent compute call (drag ROI, scrub frame, change ADF radius)
runs entirely in the browser GPU via the existing ``Show4DSTEMCompute``
shaders in ``js/engine/compute.ts``.

Who benefits — anyone with a GPU + a browser:
* NVIDIA / AMD / Intel users without a CUDA-built torch
* Apple M-series Mac users (WebGPU via Metal) without working torch.MPS
* Laptop-first adopters (no Python GPU drivers, but a modern browser)
* Anyone whose data exceeds torch.MPS's >2^31-element ceiling

Compared to:
* TorchBackend — Python compute. Requires working torch + GPU.
* MetalRawBackend — Python compute via raw Metal. Mac-only, custom path.
* WebGPUOnlineBackend — browser compute. Universal GPU access via WebGPU.
* WebGPUOfflineBackend (export_html) — same browser shaders, but no
  kernel after export. Read-only artifact.

Protocol (Python ↔ JS over the anywidget custom Comm):

  Python → JS (request)         JS → Python (response)
  ─────────────────────────     ──────────────────────
  {op: "init_stack",            {req_id, ok, error?}
   dtype: "float32"|"uint16",
   scan_shape, det_shape,
   n_chunks, chunk_meta}        (then N follow-up `init_stack_chunk`
                                 messages each with a chunk's bytes)
  {op: "masked_sum",            {req_id, result_bytes: float32 (scan)}
   req_id, mask_bytes}
  {op: "frame",                 {req_id, result_bytes: float32 (det)}
   req_id, idx}
  {op: "mean_dp", req_id}       {req_id, result_bytes}
  {op: "reduce_frames",         {req_id, result_bytes}
   req_id, scan_indices,
   reduce: "mean"|"sum"|"max"}
  {op: "com",                   {req_id, com_col_bytes, com_row_bytes}
   req_id, mask_bytes?}

Implementation note (this file): the Python side is a SYNCHRONOUS
ComputeBackend (the widget calls ``backend.masked_sum(mask)`` and
expects a numpy array back). The Comm is asynchronous (anywidget posts
custom messages). We bridge via a per-request ``threading.Event`` keyed
by an integer ``req_id``; each compute method blocks on the event until
the JS side replies.

JS-side implementation (not in this commit) lives at
``js/engine/online-channel.ts``. The Python side here will work AGAINST
that JS module once it lands. Until then, ``WebGPUOnlineBackend`` raises
``RuntimeError`` on the first compute call to make accidental
selection-without-JS obvious.
"""
from __future__ import annotations

import itertools
import threading
from typing import Any, Iterable, Literal

import numpy as np


_RESPONSE_TIMEOUT_S = 30.0  # widget user-interaction budget; bail before they notice


class WebGPUOnlineBackend:
    """ComputeBackend that defers every reduction to the browser via WebGPU.

    Constructor takes ``widget`` (a live anywidget instance — usually the
    Show4DSTEM the backend belongs to) so we can send custom messages and
    observe the responses. The 4D stack is shipped to the browser ONCE in
    ``__init__`` via the ``init_stack`` message; every subsequent compute
    call is a per-request RPC.

    Capability set: ``()`` — JS owns the lifecycle (sidecar, radial cache)
    inside its WebGPU pipeline. No Python-side capability hooks.
    """

    capabilities: tuple[str, ...] = ()

    def __init__(self, data: Any, widget: Any | None = None, *,
                 scan_shape: tuple[int, int] | None = None):
        """Bind to ``data`` (numpy / torch / MPSChunked4DSTEM / etc.) and the
        ``widget`` whose Comm channel we'll use. Sends the stack to JS.

        ``widget`` may be ``None`` during static analysis / tests; in that
        case compute calls raise immediately. The widget normally wires it
        in via ``Show4DSTEM.__init__`` when ``backend='webgpu'``.
        """
        # Resolve shape from input
        arr = _as_ndarray(data)  # may be a torch CPU tensor view, no GPU work
        self.scan_shape, self.det_shape = _resolve_shapes(arr, scan_shape)
        self.n_frames = int(self.scan_shape[0] * self.scan_shape[1])
        self.device = "webgpu"  # surfaced to Python-side debug only
        self._widget = widget
        self._req_counter = itertools.count(1)
        self._pending: dict[int, dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._stack_initialized = False
        if widget is not None:
            self._wire_widget(widget)
            self._init_stack(arr)

    # ------------------------------------------------------------------ Comm wiring
    def _wire_widget(self, widget: Any) -> None:
        """Subscribe to custom messages on the widget so JS replies route here."""
        widget.on_msg(self._on_msg)

    def _on_msg(self, _widget: Any, content: dict, buffers: list) -> None:
        """Dispatch a JS reply back to the waiting compute call."""
        req_id = content.get("req_id")
        if req_id is None or req_id not in self._pending:
            return
        slot = self._pending[req_id]
        slot["content"] = content
        slot["buffers"] = list(buffers or [])
        slot["event"].set()

    def _rpc(self, op: str, *, expect_array: bool = True, **payload) -> np.ndarray | None:
        """Send {op, req_id, ...payload} → block until JS replies → return bytes."""
        if self._widget is None:
            raise RuntimeError(
                "WebGPUOnlineBackend has no widget attached — cannot send Comm. "
                "Construct via Show4DSTEM(data, backend='webgpu')."
            )
        req_id = next(self._req_counter)
        event = threading.Event()
        with self._lock:
            self._pending[req_id] = {"event": event, "content": None, "buffers": []}
        self._widget.send({"op": op, "req_id": req_id, **payload})
        ok = event.wait(_RESPONSE_TIMEOUT_S)
        with self._lock:
            slot = self._pending.pop(req_id, None)
        if not ok:
            raise TimeoutError(
                f"WebGPU JS response timeout after {_RESPONSE_TIMEOUT_S}s "
                f"(op={op}, req_id={req_id}). JS side may not be wired yet."
            )
        if slot is None:
            raise RuntimeError(f"WebGPU response for req_id {req_id} dropped before delivery.")
        content = slot["content"] or {}
        if not content.get("ok", True):
            raise RuntimeError(f"WebGPU JS error on op={op}: {content.get('error', '?')}")
        if not expect_array:
            return None
        buffers = slot["buffers"]
        if not buffers:
            raise RuntimeError(f"WebGPU op={op} returned no buffer.")
        return np.frombuffer(buffers[0], dtype=np.float32)

    def _init_stack(self, arr: Any) -> None:
        """Ship the full 4D stack to JS. For now: single-shot upload via Bytes
        buffer. Phase 2.1 will add chunked streaming for stacks > 2 GB."""
        # TODO(Phase 2): chunk if arr.nbytes > 2 GB (per-buffer WebGPU cap)
        flat = np.ascontiguousarray(arr).reshape(self.n_frames, *self.det_shape)
        payload = {
            "op": "init_stack",
            "scan_shape": list(self.scan_shape),
            "det_shape": list(self.det_shape),
            "n_frames": self.n_frames,
            "dtype": str(flat.dtype),
        }
        # one-shot upload
        self._rpc("init_stack", expect_array=False, **payload)
        self._stack_initialized = True

    # ------------------------------------------------------------------ ComputeBackend
    def frame(self, idx: int) -> np.ndarray:
        buf = self._rpc("frame", idx=int(idx))
        return buf.reshape(self.det_shape)

    def masked_sum(self, det_mask: np.ndarray) -> np.ndarray:
        mask = np.ascontiguousarray(det_mask, dtype=np.uint8)
        buf = self._rpc("masked_sum", mask_shape=list(mask.shape),
                        mask_bytes=mask.tobytes())
        return buf.reshape(self.scan_shape).astype(np.float32, copy=False)

    def mean_dp(self) -> np.ndarray:
        buf = self._rpc("mean_dp")
        return buf.reshape(self.det_shape)

    def reduce_frames(
        self, scan_indices: Iterable[int],
        reduce: Literal["mean", "sum", "max"] = "mean",
    ) -> np.ndarray:
        idx = np.ascontiguousarray(np.asarray(scan_indices, dtype=np.uint32))
        buf = self._rpc("reduce_frames", reduce=reduce,
                        scan_indices_bytes=idx.tobytes(),
                        n_indices=int(idx.size))
        return buf.reshape(self.det_shape)

    def center_of_mass(
        self, det_mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        payload: dict = {}
        if det_mask is not None:
            mask = np.ascontiguousarray(det_mask, dtype=np.uint8)
            payload["mask_bytes"] = mask.tobytes()
            payload["mask_shape"] = list(mask.shape)
        # CoM returns TWO buffers, so use _rpc_multi
        if self._widget is None:
            raise RuntimeError("WebGPUOnlineBackend has no widget attached.")
        req_id = next(self._req_counter)
        event = threading.Event()
        with self._lock:
            self._pending[req_id] = {"event": event, "content": None, "buffers": []}
        self._widget.send({"op": "com", "req_id": req_id, **payload})
        if not event.wait(_RESPONSE_TIMEOUT_S):
            with self._lock:
                self._pending.pop(req_id, None)
            raise TimeoutError(f"WebGPU CoM timeout (req_id={req_id})")
        with self._lock:
            slot = self._pending.pop(req_id)
        if not slot["content"].get("ok", True):
            raise RuntimeError(f"CoM JS error: {slot['content'].get('error')}")
        bufs = slot["buffers"]
        if len(bufs) < 2:
            raise RuntimeError("CoM expects two buffers (com_col, com_row).")
        com_col = np.frombuffer(bufs[0], dtype=np.float32)
        com_row = np.frombuffer(bufs[1], dtype=np.float32)
        return com_col, com_row


# ----------------------------------------------------------------------- helpers
def _as_ndarray(data: Any) -> np.ndarray:
    """Coerce torch tensors / cupy arrays / numpy / dataset-likes to numpy."""
    if hasattr(data, "detach") and hasattr(data, "cpu"):  # torch
        return data.detach().cpu().numpy()
    if type(data).__module__.startswith("cupy"):
        return data.get()  # cupy → numpy host copy
    if hasattr(data, "array"):  # quantem Dataset
        return _as_ndarray(data.array)
    return np.asarray(data)


def _resolve_shapes(arr: np.ndarray, scan_shape: tuple[int, int] | None):
    if arr.ndim == 4:
        sr, sc, dr, dc = arr.shape
        return (int(sr), int(sc)), (int(dr), int(dc))
    if arr.ndim == 3:
        n, dr, dc = arr.shape
        if scan_shape is not None:
            sr, sc = scan_shape
        else:
            sr = int(round(n ** 0.5))
            sc = n // sr
        return (int(sr), int(sc)), (int(dr), int(dc))
    raise ValueError(f"expected 3D/4D array, got {arr.shape}")
