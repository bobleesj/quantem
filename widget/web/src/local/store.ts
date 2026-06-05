// Local data layer for the standalone WebGPU 4D-STEM browser. Replaces the
// quantem.live server: a locally-picked folder of Arina .h5 files is scanned into
// the same Session/MasterFile tree the Browse GUI expects, and every image
// (virtual image, CBED frame, summed DP) is produced on the GPU by the shared
// js/engine WGSL engine. No Python, no network.

import { readH5Volume } from "../engine/h5reader";
import { Show4DSTEMCompute } from "../engine/compute";
import { decodeBslz4Batch, type Bslz4Spec } from "../engine/bslz4";
import type { Session, MasterFile, RawData, DetectorMode, DetShape, ShapeParams } from "../pages/browse/types";

// A picked file, uniform over File System Access handles and <input webkitdirectory>.
// `source` (the File or directory handle) lets a worker read the bytes off the main thread -
// reading in parallel across worker threads is ~4x the main-thread File.arrayBuffer rate.
export interface LocalFile { name: string; relPath: string; bytes(): Promise<ArrayBuffer>; source?: File | FileSystemFileHandle; }

interface ParsedSpec { id: number; nFrames: number; nBlocksPerFrame: number; blockElems: number; detSize: number; srcDtype: "uint8" | "uint16" | "uint32"; blockMeta: Uint32Array; buffer: ArrayBuffer; }
let READERS: Worker[] | null = null;
function readerPool(): Worker[] {
  if (!READERS) {
    const n = Math.min(8, navigator.hardwareConcurrency || 4);
    READERS = Array.from({ length: n }, () => new Worker(new URL("./readWorker.ts", import.meta.url), { type: "module" }));
  }
  return READERS;
}
// Read + parse every slab across the worker pool (parallel disk reads ~10 GB/s vs ~2.3 GB/s
// on the main thread). Calls onReady(id, spec) as each file completes, in arrival order.
function readParseInWorkers(slabs: LocalFile[], onReady: (s: ParsedSpec) => void): void {
  const pool = readerPool();
  let next = 0;
  const pump = (w: Worker) => {
    if (next >= slabs.length) return;
    const id = next++;
    w.onmessage = (e: MessageEvent<ParsedSpec>) => { onReady(e.data); pump(w); };
    const src = slabs[id].source!;
    w.postMessage({ id, name: slabs[id].name, file: src instanceof File ? src : undefined, handle: src instanceof File ? undefined : src });
  };
  pool.forEach(pump);
}

const MASTER_RE = /_master\.h5$/i;
const DATA_RE = /_data_\d+\.h5$/i;

interface Handles { master: LocalFile | null; dataFiles: LocalFile[]; }
interface LoadedDS {
  compute: Show4DSTEMCompute;
  meanDP: Float32Array;
  scanRows: number; scanCols: number; detRows: number; detCols: number; detSize: number; scanCount: number;
  bf: { cy: number; cx: number; r_bf: number };
  badPx: Uint32Array;
}

let SESSIONS: Session[] = [];
const HANDLES = new Map<string, Handles>();          // fileKey -> file handles
// scanCols = the master's true raster WIDTH (from ntrigger). Needed so a PARTIAL dataset (fewer
// data files than the full scan) reshapes into the correct-width grid instead of a sqrt() square
// (which wraps the partial frames at the wrong stride -> diagonal garbage).
const GEOM = new Map<string, { detRows: number; detCols: number; badPx: Uint32Array; scanCols: number }>();
const LOADED = new Map<string, Promise<LoadedDS>>();  // fileKey -> decoded dataset (LRU)
const LRU: string[] = [];
const MAX_RESIDENT = 2;
// Keys of the ACTIVE 5D set are PINNED: never evicted, so scrubbing the time/tilt axis stays
// live (every binned frame resident on the GPU). The set is auto-binned to fit VRAM, so pinning
// the whole series is safe; a non-5D single dataset just keeps the 2-deep LRU.
const PINNED = new Set<string>();
export function setPinned5DKeys(keys: string[]): void { PINNED.clear(); for (const k of keys) PINNED.add(k); }
// # of files the last scan skipped (unreadable / corrupt / truncated / junk) - surfaced to the user.
let LAST_SCAN_SKIPPED = 0;
export function lastScanSkipped(): number { return LAST_SCAN_SKIPPED; }

function key(source: string, date: string, name: string): string { return `${source}/${date}/${name}`; }
function humanSize(bytes: number): string {
  if (bytes >= 1e9) return `${(bytes / 1e9).toFixed(1)} GB`;
  if (bytes >= 1e6) return `${(bytes / 1e6).toFixed(0)} MB`;
  return `${(bytes / 1e3).toFixed(0)} KB`;
}

// Scan a picked folder into Session[]. A *_master.h5 + its *_data_*.h5 siblings is one
// dataset; the master gives scan size (ntrigger) + hot pixels (pixel_mask). A bare .h5 is
// a standalone single-slab volume. Called once before the Browse GUI mounts.
export async function scanFolder(files: LocalFile[]): Promise<void> {
  const jsfive = await import("jsfive/esm/high-level.js") as { File: new (ab: ArrayBuffer, name: string) => unknown };
  SESSIONS = []; HANDLES.clear(); GEOM.clear();
  const byKey = new Map<string, Session>();
  const sessionFor = (relPath: string): { source: string; date: string } => {
    const parts = relPath.split("/").filter(Boolean);
    if (parts.length >= 3) return { source: parts[0], date: parts.slice(1, -1).join("/") };
    if (parts.length === 2) return { source: parts[0], date: "local" };
    return { source: "datasets", date: "local" };
  };
  const pushFile = (source: string, date: string, mf: MasterFile, h: Handles) => {
    const sk = `${source}/${date}`;
    let sess = byKey.get(sk);
    if (!sess) { sess = { source, date, files: [] }; byKey.set(sk, sess); SESSIONS.push(sess); }
    sess.files.push(mf);
    HANDLES.set(key(source, date, mf.name), h);
  };
  // Skip macOS AppleDouble sidecars (._foo.h5): they match *master*.h5 but are 4 KB resource-fork
  // junk that jsfive can't parse - and one throw used to kill the whole folder scan.
  const realFiles = files.filter((f) => !f.name.startsWith("._"));
  let skipped = 0;
  // Claim each master's data-file siblings up front (filename-only, no IO).
  const masters = realFiles.filter((f) => MASTER_RE.test(f.name));
  const claimed = new Set<string>();
  const masterPlan = masters.map((m) => {
    const prefix = m.name.replace(MASTER_RE, "");
    const dataFiles = realFiles.filter((f) => DATA_RE.test(f.name) && f.name.startsWith(prefix))
      .sort((a, b) => a.name.localeCompare(b.name));
    dataFiles.forEach((f) => claimed.add(f.relPath)); claimed.add(m.relPath);
    return { m, dataFiles };
  });
  // Read ALL master headers in PARALLEL (the scan bottleneck was serial reads - tens of seconds
  // over a slow link). A corrupt / truncated / non-HDF5 master must not abort the rest - skip it.
  const masterBytes = await Promise.all(masterPlan.map((p) => p.m.bytes().catch(() => null)));
  masterPlan.forEach((p, i) => {
    const bytes = masterBytes[i];
    if (!bytes) { skipped++; return; }
    try {
      const f = new jsfive.File(bytes, p.m.name) as { get(path: string): { value: ArrayLike<number>; shape: number[] } };
      const ntrigger = Number(f.get("entry/instrument/detector/detectorSpecific/ntrigger").value[0]);
      const pm = f.get("entry/instrument/detector/detectorSpecific/pixel_mask");
      const detRows = pm.shape[0], detCols = pm.shape[1];
      // jsfive's .value is a lazy getter that re-decodes the whole dataset on EVERY access. Read it
      // ONCE - the old loop touched pm.value in both the test and the body per pixel = 262k decodes
      // per master = 21s each (the entire folder-scan bottleneck). Hoisting drops it to ~1ms.
      const maskVals = pm.value;
      const bad: number[] = [];
      for (let k = 0; k < maskVals.length; k++) if (maskVals[k] !== 0) bad.push(k);
      const side = Math.round(Math.sqrt(ntrigger));
      const scanRows = side, scanCols = Math.ceil(ntrigger / side);
      const { source, date } = sessionFor(p.m.relPath);
      const mf: MasterFile = {
        name: p.m.name, shape: [scanRows, scanCols, detRows, detCols], cal: "un",
        size: humanSize(ntrigger * detRows * detCols), loadable: p.dataFiles.length > 0,
      };
      GEOM.set(key(source, date, p.m.name), { detRows, detCols, badPx: new Uint32Array(bad), scanCols });
      pushFile(source, date, mf, { master: p.m, dataFiles: p.dataFiles });
    } catch { skipped++; }   // unreadable master (corrupt/truncated/not HDF5)
  });
  // Bare .h5 volumes not owned by a master, also read in parallel. Exclude orphan _data_NNNNNN.h5
  // files: a data shard whose master is missing/corrupt can't decode standalone, and full-reading
  // gigabytes of frame shards during a folder scan is the scan-bottleneck bug (was 110s on big trees).
  const bare = realFiles.filter((f) => !claimed.has(f.relPath) && /\.h5$/i.test(f.name) && !DATA_RE.test(f.name));
  const bareBytes = await Promise.all(bare.map((f) => f.bytes().catch(() => null)));
  bare.forEach((file, i) => {
    const bytes = bareBytes[i];
    if (!bytes) { skipped++; return; }
    let head: ReturnType<typeof readH5Volume>;
    try { head = readH5Volume(bytes, file.name); }
    catch { return; }   // not a 4D bslz4 stack (e.g. a Velox virtual-image export) - silently ignore
    const side = Math.round(Math.sqrt(head.nFrames));
    const { source, date } = sessionFor(file.relPath);
    const mf: MasterFile = {
      name: file.name, shape: [side, Math.ceil(head.nFrames / side), head.detRows, head.detCols], cal: "un",
      size: humanSize(head.nFrames * head.detSize), loadable: true,
    };
    GEOM.set(key(source, date, file.name), { detRows: head.detRows, detCols: head.detCols, badPx: new Uint32Array(0), scanCols: Math.ceil(head.nFrames / side) });
    pushFile(source, date, mf, { master: null, dataFiles: [file] });
  });
  SESSIONS.sort((a, b) => `${a.source}/${a.date}`.localeCompare(`${b.source}/${b.date}`));
  LAST_SCAN_SKIPPED = skipped;
}

export function getSessions(): Session[] { return SESSIONS; }

// Background-warm a 5D series: decode every frame sequentially (one at a time so the transient
// decode buffer never OOMs) so the time/tilt scrub is LIVE from the start instead of decoding
// each frame on its first view. Pinned, so all frames stay resident once warm. Fire-and-forget;
// errors per frame are swallowed (a bad frame just stays cold). A new warm cancels the old via gen.
let warmGen = 0;
export async function warmSet5D(frames: { source: string; date: string; name: string }[]): Promise<void> {
  const gen = ++warmGen;
  for (const f of frames) {
    if (gen !== warmGen) return;   // a newer set superseded this warm-up
    try { await ensureLoaded(f.source, f.date, f.name); } catch { /* leave this frame cold */ }
  }
}

// Decode a dataset's slabs into one chunked GPU compute (LRU, dispose on evict). Computes
// the mean DP once and auto-fits the bright-field disk to seed the aperture.
async function ensureLoaded(source: string, date: string, name: string): Promise<LoadedDS> {
  const k = key(source, date, name);
  const hit = LOADED.get(k);
  if (hit) { const i = LRU.indexOf(k); if (i >= 0) { LRU.splice(i, 1); LRU.push(k); } return hit; }
  const handles = HANDLES.get(k);
  const geom = GEOM.get(k);
  if (!handles || !geom) throw new Error(`unknown dataset ${k}`);
  const p = (async (): Promise<LoadedDS> => {
    const PERF = ((window as unknown as { __perf: unknown[] }).__perf ||= []) as Record<string, unknown>[];
    const tA = performance.now();
    const slabs = handles.master ? handles.dataFiles : handles.dataFiles;
    const detSize = geom.detRows * geom.detCols;
    // Pipeline parse and decode in groups: parse a group on the main thread (disk read +
    // jsfive B-tree walk) while the GPU decodes the previous group, so the wall is ~max(parse,
    // decode) not their sum. (A worker-pool parse was tried and lost to transfer overhead.)
    const GROUP = 7;
    const chunks: { buffer: GPUBuffer; startScan: number; nScan: number }[] = [];
    let startScan = 0, device: GPUDevice | null = null, mode = 1;
    let pending: Promise<{ device: GPUDevice; buffers: GPUBuffer[]; mode: number } | null> | null = null;
    let pendingSpecs: { startScan: number; nScan: number }[] = [];
    const drain = async () => {
      if (!pending) return;
      const r = await pending; if (!r) throw new Error("WebGPU unavailable");
      device = r.device; mode = r.mode;
      r.buffers.forEach((buffer, i) => chunks.push({ buffer, startScan: pendingSpecs[i].startScan, nScan: pendingSpecs[i].nScan }));
      pending = null;
    };
    let srcDtype: "uint8" | "uint16" | "uint32" = "uint16";   // detected from the data
    // Picker path -> parallel worker reads. But file:// (the double-click single-HTML artifact)
    // blocks `new Worker()` (origin "null"), so fall back to the main-thread read there.
    const useWorkers = slabs.every((s) => s.source) && location.protocol !== "file:";
    // Read + parse all slabs (in workers when we have File/handle sources, else on the main
    // thread). Collect parsed specs keyed by file index so we can decode in scan order.
    const parsed: (ParsedSpec | null)[] = new Array(slabs.length).fill(null);
    const ready: (() => void)[] = []; const readyP = slabs.map((_, i) => new Promise<void>((r) => (ready[i] = r)));
    if (useWorkers) {
      readParseInWorkers(slabs, (ps) => { parsed[ps.id] = ps; ready[ps.id](); });
    } else {
      (async () => { for (let i = 0; i < slabs.length; i++) {
        const buf = await slabs[i].bytes(); const vol = readH5Volume(buf, slabs[i].name);
        const c = vol.chunks[0]; parsed[i] = { id: i, nFrames: c.nFrames, nBlocksPerFrame: c.nBlocksPerFrame, blockElems: c.blockElems, detSize: c.detSize, srcDtype: vol.srcDtype, blockMeta: c.blockMeta, buffer: buf }; ready[i]();
      } })();
    }
    // Decode groups in scan order as their files become available; the worker reads (fast)
    // overlap the GPU decode of earlier groups.
    for (let g = 0; g < slabs.length; g += GROUP) {
      const specs: (Bslz4Spec & { startScan: number; nScan: number })[] = [];
      for (let i = g; i < Math.min(g + GROUP, slabs.length); i++) {
        await readyP[i]; const ps = parsed[i]!;
        srcDtype = ps.srcDtype;
        specs.push({ compressed: new Uint8Array(ps.buffer), blockMeta: ps.blockMeta, nFrames: ps.nFrames,
          nBlocksPerFrame: ps.nBlocksPerFrame, blockElems: ps.blockElems, detSize: ps.detSize, startScan, nScan: ps.nFrames });
        startScan += ps.nFrames;
      }
      await drain();
      pending = decodeBslz4Batch(specs, "uint8", srcDtype, GROUP);
      pendingSpecs = specs;
    }
    await drain();
    if (!device) throw new Error("WebGPU unavailable");
    const tC = performance.now();
    const compute = Show4DSTEMCompute.fromGpuChunks(device, chunks, startScan, detSize, mode);
    compute.badPx = geom.badPx;
    const meanDP = await compute.reduceFrames(new Uint32Array(startScan).fill(1), true);
    PERF.push({ key: k, loadDecodeMs: Math.round(tC - tA), reduceMs: Math.round(performance.now() - tC), totalMs: Math.round(performance.now() - tA) });
    const bf = fitBfDisk(meanDP, geom.detRows, geom.detCols);
    // Reshape with the master's TRUE raster width; rows = actual decoded frames / width (so a
    // partial dataset shows the rows it has, correctly, instead of a wrong-stride square).
    const scanCols = geom.scanCols || Math.round(Math.sqrt(startScan));
    const scanRows = Math.ceil(startScan / scanCols);
    return { compute, meanDP, scanRows, scanCols,
      detRows: geom.detRows, detCols: geom.detCols, detSize, scanCount: startScan, bf, badPx: geom.badPx };
  })();
  LOADED.set(k, p); LRU.push(k);
  // Evict the oldest NON-pinned datasets beyond the resident budget. Pinned = the active 5D
  // series, which stays fully resident so its scrub is live. The budget counts only unpinned.
  while (LRU.filter((x) => !PINNED.has(x)).length > MAX_RESIDENT) {
    const evIdx = LRU.findIndex((x) => !PINNED.has(x));
    if (evIdx < 0) break;
    const ev = LRU.splice(evIdx, 1)[0];
    const old = LOADED.get(ev); LOADED.delete(ev);
    old?.then((d) => d.compute.dispose()).catch(() => {});
  }
  return p;
}

// Auto-fit the bright-field disk from the mean DP: centroid of the bright region (the disk)
// gives the center; the bright-pixel area gives the radius (area = pi r^2).
function fitBfDisk(meanDP: Float32Array, detRows: number, detCols: number): { cy: number; cx: number; r_bf: number } {
  let max = 0; for (const v of meanDP) if (v > max) max = v;
  const thr = max * 0.5;
  let sum = 0, sr = 0, sc = 0, n = 0;
  for (let r = 0; r < detRows; r++) for (let c = 0; c < detCols; c++) {
    if (meanDP[r * detCols + c] >= thr) { sr += r; sc += c; n++; }
    sum += 0;
  }
  void sum;
  if (n === 0) return { cy: detRows / 2, cx: detCols / 2, r_bf: detRows / 8 };
  return { cy: sr / n, cx: sc / n, r_bf: Math.sqrt(n / Math.PI) };
}

// --- masks ---------------------------------------------------------------
function diskMask(detRows: number, detCols: number, cy: number, cx: number, rad: number): Uint32Array {
  const m = new Uint32Array(detRows * detCols), r2 = rad * rad;
  for (let r = 0; r < detRows; r++) for (let c = 0; c < detCols; c++) {
    const dr = r - cy, dc = c - cx; if (dr * dr + dc * dc <= r2) m[r * detCols + c] = 1;
  }
  return m;
}
function annulusMask(detRows: number, detCols: number, cy: number, cx: number, inner: number, outer: number): Uint32Array {
  const m = new Uint32Array(detRows * detCols), i2 = inner * inner, o2 = outer * outer;
  for (let r = 0; r < detRows; r++) for (let c = 0; c < detCols; c++) {
    const dr = r - cy, dc = c - cx, d2 = dr * dr + dc * dc; if (d2 >= i2 && d2 <= o2) m[r * detCols + c] = 1;
  }
  return m;
}

function mean(a: Float32Array): number { let s = 0; for (const v of a) s += v; return s / (a.length || 1); }

// In-place radix-2 Cooley-Tukey FFT (re/im length must be a power of two). sign=-1 forward.
function fft1d(re: Float32Array, im: Float32Array, sign: number): void {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) { [re[i], re[j]] = [re[j], re[i]]; [im[i], im[j]] = [im[j], im[i]]; }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = sign * 2 * Math.PI / len, wr = Math.cos(ang), wi = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let cr = 1, ci = 0;
      for (let k = 0; k < len / 2; k++) {
        const ar = re[i + k], ai = im[i + k];
        const br = re[i + k + len / 2], bi = im[i + k + len / 2];
        const tr = br * cr - bi * ci, ti = br * ci + bi * cr;
        re[i + k] = ar + tr; im[i + k] = ai + ti;
        re[i + k + len / 2] = ar - tr; im[i + k + len / 2] = ai - ti;
        const ncr = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = ncr;
      }
    }
  }
}
function fft2d(re: Float32Array, im: Float32Array, rows: number, cols: number, sign: number): void {
  const rr = new Float32Array(cols), ri = new Float32Array(cols);
  for (let r = 0; r < rows; r++) {
    rr.set(re.subarray(r * cols, r * cols + cols)); ri.set(im.subarray(r * cols, r * cols + cols));
    fft1d(rr, ri, sign);
    re.set(rr, r * cols); im.set(ri, r * cols);
  }
  const cr = new Float32Array(rows), ci = new Float32Array(rows);
  for (let c = 0; c < cols; c++) {
    for (let r = 0; r < rows; r++) { cr[r] = re[r * cols + c]; ci[r] = im[r * cols + c]; }
    fft1d(cr, ci, sign);
    for (let r = 0; r < rows; r++) { re[r * cols + c] = cr[r]; im[r * cols + c] = ci[r]; }
  }
}

// Integrated CoM (DPC phase): solve grad(phi) = (dx, dy) in Fourier space,
// phi_hat = (-i kx Gx - i ky Gy)/(kx^2 + ky^2), DC = 0. Real part is the phase image.
// Falls back to CoM magnitude when the scan isn't a power-of-two square (FFT needs it).
function integrateICoM(dx: Float32Array, dy: Float32Array, rows: number, cols: number): Float32Array {
  const pow2 = (x: number) => (x & (x - 1)) === 0;
  if (!pow2(rows) || !pow2(cols)) { const m = new Float32Array(dx.length); for (let i = 0; i < m.length; i++) m[i] = Math.hypot(dx[i], dy[i]); return m; }
  const gxr = Float32Array.from(dx), gxi = new Float32Array(dx.length);
  const gyr = Float32Array.from(dy), gyi = new Float32Array(dy.length);
  fft2d(gxr, gxi, rows, cols, -1);
  fft2d(gyr, gyi, rows, cols, -1);
  const pr = new Float32Array(dx.length), pi = new Float32Array(dx.length);
  for (let r = 0; r < rows; r++) {
    const ky = 2 * Math.PI * (r <= rows / 2 ? r : r - rows) / rows;
    for (let c = 0; c < cols; c++) {
      const kx = 2 * Math.PI * (c <= cols / 2 ? c : c - cols) / cols;
      const k2 = kx * kx + ky * ky;
      const i = r * cols + c;
      if (k2 === 0) continue;
      // numerator = -i*kx*Gx - i*ky*Gy ; dividing complex G by treating -i*k as multiplier
      const nr = kx * gxi[i] + ky * gyi[i];        // real part of (-i k)(Gr+iGi) = k*Gi
      const ni = -(kx * gxr[i] + ky * gyr[i]);     // imag part = -k*Gr
      pr[i] = nr / k2; pi[i] = ni / k2;
    }
  }
  fft2d(pr, pi, rows, cols, +1);
  const out = new Float32Array(dx.length), norm = rows * cols;
  for (let i = 0; i < out.length; i++) out[i] = pr[i] / norm;   // inverse FFT normalization
  return out;
}

function reshapeVI(vi: Float32Array, ds: LoadedDS): RawData { return { data: vi, width: ds.scanCols, height: ds.scanRows }; }
function reshapeDP(dp: Float32Array, ds: LoadedDS): RawData { return { data: dp, width: ds.detCols, height: ds.detRows }; }

// --- public image ops (called by the rewritten fetch* in types.ts) -------
export async function bfGeometry(source: string, date: string, name: string): Promise<{ cy: number; cx: number; r_bf: number }> {
  const ds = await ensureLoaded(source, date, name); return ds.bf;
}

// The cached mean diffraction pattern (uint8-clipped integer sum / nFrames, bad px zeroed) -
// used for parity checks against an h5py reference.
export async function datasetMeanDp(source: string, date: string, name: string): Promise<Float32Array> {
  const ds = await ensureLoaded(source, date, name); return ds.meanDP;
}

// GPU-resident virtual image for the 60fps aperture-drag fast path: returns the maskedSum result
// as a GPU buffer (NO readback) so the caller colormaps it straight to the canvas. Only the
// mask modes (BF/ADF/DF) - CoM/iCoM need a CPU post-process (descan, integrate) so they stay on
// the readback path. Returns null for those (caller falls back to the normal path).
export async function virtualImageBufferGpu(
  source: string, date: string, name: string, mode: DetectorMode,
  inner: number, outer: number, cx: number | null, cy: number | null,
): Promise<{ buffer: GPUBuffer; width: number; height: number } | null> {
  const ds = await ensureLoaded(source, date, name);
  const ccx = cx ?? ds.bf.cx, ccy = cy ?? ds.bf.cy, r = ds.bf.r_bf;
  let mask: Uint32Array;
  if (mode === "BF") mask = diskMask(ds.detRows, ds.detCols, ccy, ccx, (outer || 1) * r);
  else if (mode === "ADF" || mode === "DF") mask = annulusMask(ds.detRows, ds.detCols, ccy, ccx, (inner || 1.2) * r, (outer || 4) * r);
  else return null;
  const { buffer } = ds.compute.maskedSumBuffer(mask);
  return { buffer, width: ds.scanCols, height: ds.scanRows };
}

// Virtual image for a detector MODE with alpha-unit ring radii (1 = BF disk edge), like the
// server's /realspace. BF = disk; ADF/DF = annulus. CoM/iCoM/SSB not yet on the GPU path.
export async function virtualImage(
  source: string, date: string, name: string, mode: DetectorMode,
  inner: number, outer: number, cx: number | null, cy: number | null,
): Promise<RawData | null> {
  const ds = await ensureLoaded(source, date, name);
  const ccx = cx ?? ds.bf.cx, ccy = cy ?? ds.bf.cy, r = ds.bf.r_bf;
  let mask: Uint32Array;
  if (mode === "BF") { mask = diskMask(ds.detRows, ds.detCols, ccy, ccx, (outer || 1) * r);
    return reshapeVI(await ds.compute.maskedSum(mask), ds); }
  if (mode === "ADF" || mode === "DF") { mask = annulusMask(ds.detRows, ds.detCols, ccy, ccx, (inner || 1.2) * r, (outer || 4) * r);
    return reshapeVI(await ds.compute.maskedSum(mask), ds); }
  // CoM / iCoM (DPC): intensity-weighted centroid over the BF disk per scan position.
  const comMask = diskMask(ds.detRows, ds.detCols, ccy, ccx, 1.5 * r);
  const { comY, comX } = await ds.compute.maskedCoM(comMask, ds.detCols);
  const my = mean(comY), mx = mean(comX);
  const dy = new Float32Array(comY.length), dx = new Float32Array(comX.length);
  for (let i = 0; i < dy.length; i++) { dy[i] = comY[i] - my; dx[i] = comX[i] - mx; }  // remove descan offset
  let field: Float32Array;
  if (mode === "CoMx") field = dx;
  else if (mode === "CoMy") field = dy;
  else if (mode === "CoMmag") { field = new Float32Array(dx.length); for (let i = 0; i < dx.length; i++) field[i] = Math.hypot(dx[i], dy[i]); }
  else if (mode === "iCoM") field = integrateICoM(dx, dy, ds.scanRows, ds.scanCols);
  else return null;
  return reshapeVI(field, ds);
}

// CoM parity probe: raw per-scan intensity-weighted centroid (comY, comX) over the BF-disk
// aperture (1.5 x r_bf), BEFORE descan subtraction - the direct maskedCoM output. Returns
// reduction sums + geometry so a numpy reference can be compared bit-close on real data.
export async function datasetComStats(source: string, date: string, name: string): Promise<{
  detRows: number; detCols: number; scanCount: number; cx: number; cy: number; r: number; rad: number;
  comYsum: number; comXsum: number; comY0: number; comX0: number; nbad: number;
}> {
  const ds = await ensureLoaded(source, date, name);
  const rad = 1.5 * ds.bf.r_bf;
  const mask = diskMask(ds.detRows, ds.detCols, ds.bf.cy, ds.bf.cx, rad);
  const { comY, comX } = await ds.compute.maskedCoM(mask, ds.detCols);
  let comYsum = 0, comXsum = 0;
  for (let i = 0; i < comY.length; i++) { comYsum += comY[i]; comXsum += comX[i]; }
  return { detRows: ds.detRows, detCols: ds.detCols, scanCount: ds.scanCount,
    cx: ds.bf.cx, cy: ds.bf.cy, r: ds.bf.r_bf, rad,
    comYsum, comXsum, comY0: comY[0], comX0: comX[0], nbad: ds.badPx.length };
}

// Virtual image for a free-form detector SHAPE (params already in detector px), like the
// server's /realspace-shape. Drives the live detector drag in the Viewer.
export async function virtualImageShape(
  source: string, date: string, name: string, shape: DetShape, p: ShapeParams,
): Promise<RawData | null> {
  const ds = await ensureLoaded(source, date, name);
  const { detRows, detCols } = ds;
  let mask: Uint32Array;
  if (shape === "circle") mask = diskMask(detRows, detCols, p.cy, p.cx, p.r);
  else if (shape === "annulus") mask = annulusMask(detRows, detCols, p.cy, p.cx, p.inner, p.outer);
  else if (shape === "point") mask = diskMask(detRows, detCols, p.py, p.px, 0.75);
  else if (shape === "square") {
    mask = new Uint32Array(detRows * detCols);
    for (let r = 0; r < detRows; r++) for (let c = 0; c < detCols; c++)
      if (Math.abs(r - p.cy) <= p.half && Math.abs(c - p.cx) <= p.half) mask[r * detCols + c] = 1;
  } else {  // rect
    const r0 = Math.min(p.row0, p.row1), r1 = Math.max(p.row0, p.row1);
    const c0 = Math.min(p.col0, p.col1), c1 = Math.max(p.col0, p.col1);
    mask = new Uint32Array(detRows * detCols);
    for (let r = Math.max(0, r0); r <= Math.min(detRows - 1, r1); r++)
      for (let c = Math.max(0, c0); c <= Math.min(detCols - 1, c1); c++) mask[r * detCols + c] = 1;
  }
  const vi = await ds.compute.maskedSum(mask);
  return reshapeVI(vi, ds);
}

// One CBED frame at scan position (sx=col, sy=row).
export async function cbedFrame(source: string, date: string, name: string, sx: number, sy: number): Promise<RawData | null> {
  const ds = await ensureLoaded(source, date, name);
  const frame = await ds.compute.frameAt(sy * ds.scanCols + sx);
  return reshapeDP(frame, ds);
}

// Summed CBED over a rectangular scan ROI.
export async function cbedRoi(
  source: string, date: string, name: string, row0: number, col0: number, row1: number, col1: number,
): Promise<RawData | null> {
  const ds = await ensureLoaded(source, date, name);
  const mask = new Uint32Array(ds.scanCount);
  const r0 = Math.max(0, Math.min(row0, row1)), r1 = Math.min(ds.scanRows - 1, Math.max(row0, row1));
  const c0 = Math.max(0, Math.min(col0, col1)), c1 = Math.min(ds.scanCols - 1, Math.max(col0, col1));
  for (let r = r0; r <= r1; r++) for (let c = c0; c <= c1; c++) mask[r * ds.scanCols + c] = 1;
  const dp = await ds.compute.reduceFrames(mask, true);
  return reshapeDP(dp, ds);
}

// Free VRAM proxy from the WebGPU adapter (the app has no server GPU monitor).
export function freeVramBytes(): number { return 8 * 1024 * 1024 * 1024; }
