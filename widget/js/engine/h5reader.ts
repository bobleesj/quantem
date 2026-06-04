// In-browser native Arina/HDF5 reader. Pulls the raw bitshuffle+LZ4 (bslz4) chunks
// straight out of an .h5 file with jsfive (pure-JS HDF5, no wasm) and packs them into
// the Bslz4Spec the WGSL engine decodes. This is the ONLY path that turns a user's
// .h5 file into GPU-decodable bytes with no Python and no server - the colleague opens
// the HTML, points at a folder, and every dataset decodes on the GPU.
//
// The raw chunk jsfive hands back is byte-identical to h5py read_direct_chunk (verified
// 2026-06-03 on full gold 512 Arina, frame0 = 12586 bytes, bit-exact), so the bytes the
// engine eats here are the same bytes CUDA eats. Parity is by composition: jsfive chunk
// == h5py chunk, and engine decode == CUDA decode.

import * as jsfive from "jsfive/esm/high-level.js";
import { BTreeV1RawDataChunks } from "jsfive/esm/btree.js";
import type { Bslz4Spec } from "./bslz4";

export interface H5Volume {
  name: string;
  detRows: number;
  detCols: number;
  detSize: number;
  blockElems: number;          // elements per bitshuffle block (read from the chunk header)
  nBlocksPerFrame: number;
  srcDtype: "uint8" | "uint16";
  nFrames: number;             // frames in THIS file (one Arina data file is a scan slab)
  chunks: Bslz4Spec[];         // scan-frame chunked so each decoded buffer <= the GPU cap
  chunkScanCounts: number[];   // frame count per chunk (== spec.nFrames)
}

// jsfive dataset path candidates, in priority order. Arina data files use entry/data/data;
// fall back to a search for the first 3D unsigned-int dataset for other layouts.
const PATH_CANDIDATES = ["entry/data/data", "entry/data", "data"];

function readBE32(b: Uint8Array, off: number): number {
  return ((b[off] << 24) | (b[off + 1] << 16) | (b[off + 2] << 8) | b[off + 3]) >>> 0;
}

// Walk a chunked dataset's B-tree and return per-frame raw bslz4 chunk bytes, indexed by
// the scan-frame index (chunk_offset[0]). Each chunk is the exact filter-input bytes -
// jsfive reads the B-tree but does NOT apply the bitshuffle filter (filter 32008 is
// unknown to it), so the raw slab is the native codec stream the WGSL decoder wants.
function rawFrameChunks(ds: any): Uint8Array[] {
  const dobj = ds._dataobjects;
  dobj._get_chunk_params();
  const bt = new BTreeV1RawDataChunks(dobj.fh, dobj._chunk_address, dobj._chunk_dims);
  const nFrames = ds.shape[0];
  const out: Uint8Array[] = new Array(nFrames);
  for (const node of bt.all_nodes.get(0)) {
    const keys = node.get("keys"), addrs = node.get("addresses");
    for (let i = 0; i < keys.length; i++) {
      const frame = keys[i].get("chunk_offset")[0];
      const size = keys[i].get("chunk_size");
      out[frame] = new Uint8Array(dobj.fh.slice(addrs[i], addrs[i] + size));
    }
  }
  return out;
}

// Find the 3D detector-stack dataset inside the file (scan frames x detRows x detCols).
function findStack(file: any): any {
  for (const path of PATH_CANDIDATES) {
    try { const d = file.get(path); if (d && d.shape && d.shape.length === 3) return d; } catch { /* not present */ }
  }
  // Fallback: first 3D dataset anywhere in the tree.
  const walk = (grp: any): any => {
    for (const key of grp.keys) {
      const child = grp.get(key);
      if (child?.shape?.length === 3) return child;
      if (child?.keys) { const found = walk(child); if (found) return found; }
    }
    return null;
  };
  const found = walk(file);
  if (!found) throw new Error("no 3D detector-stack dataset found in HDF5 file");
  return found;
}

// Parse one Arina/HDF5 file's raw chunks into chunked Bslz4Spec(s). framesPerChunk bounds
// the decoded GPU buffer (uint8 stack <= ~0.95 GB): detSize bytes/frame, so the default
// keeps each chunk under the 1 GB per-buffer cap.
export function readH5Volume(buffer: ArrayBuffer, name: string, framesPerChunk?: number): H5Volume {
  const file = new jsfive.File(buffer, name);
  const ds = findStack(file);
  const [nFrames, detRows, detCols] = ds.shape;
  const detSize = detRows * detCols;
  const srcDtype: "uint8" | "uint16" = ds.dtype && /u?int8|\|u1|<u1|u1/.test(String(ds.dtype)) ? "uint8" : "uint16";
  const srcBytes = srcDtype === "uint8" ? 1 : 2;
  const frames = rawFrameChunks(ds);
  // Block geometry comes from the first chunk's 12-byte header: bytes 8-11 (BE) are the
  // per-block uncompressed byte count; blockElems = that / element bytes.
  const blockBytes = readBE32(frames[0], 8);
  const blockElems = blockBytes / srcBytes;
  const nBlocksPerFrame = Math.ceil(detSize / blockElems);
  const cap = framesPerChunk ?? Math.max(1, Math.floor((950 * 1024 * 1024) / detSize));
  const specs: Bslz4Spec[] = [];
  const scanCounts: number[] = [];
  for (let f0 = 0; f0 < nFrames; f0 += cap) {
    const f1 = Math.min(nFrames, f0 + cap);
    const parts: Uint8Array[] = [];
    const meta: number[] = [];
    let base = 0;
    for (let f = f0; f < f1; f++) {
      const chunk = frames[f];
      // 12B header (8B total uncompressed + 4B block size), then per block [4B BE clen][lz4].
      let pos = 12;
      for (let b = 0; b < nBlocksPerFrame; b++) {
        const clen = readBE32(chunk, pos);
        meta.push(base + pos + 4, clen);
        pos += 4 + clen;
      }
      parts.push(chunk);
      base += chunk.byteLength;
      // pad the concatenated blob to a 4-byte boundary so the next frame's offsets and the
      // GPU writeBuffer stay 4-aligned (matches the Python packer exactly).
      const padLen = (-base) & 3;
      if (padLen) { parts.push(new Uint8Array(padLen)); base += padLen; }
    }
    const compressed = new Uint8Array(base);
    let w = 0;
    for (const p of parts) { compressed.set(p, w); w += p.byteLength; }
    specs.push({ compressed, blockMeta: new Uint32Array(meta), nFrames: f1 - f0, nBlocksPerFrame, blockElems, detSize });
    scanCounts.push(f1 - f0);
  }
  return { name, detRows, detCols, detSize, blockElems, nBlocksPerFrame, srcDtype, nFrames, chunks: specs, chunkScanCounts: scanCounts };
}
