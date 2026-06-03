"""bslz4 offline pack round-trip: the forward encoder in Show4DSTEM._pack_offline_bslz4
must produce a companion the browser WebGPU decoder can invert bit-exactly.

We decode the companion in Python with the SAME algorithm the WGSL decoder uses
(LZ4 block decode + plane-major LSB-first inverse bitshuffle, uint8 clip) and assert
it matches clip(data, 0, 255). This locks the on-disk format + the JS decoder contract.
Skips cleanly without lz4.
"""
import json
import struct
import numpy as np
import pytest

pytest.importorskip("lz4.block")
import lz4.block as _lz4


def _decode_bslz4_uint8(raw, blockMeta, n_frames, n_blocks, block_elems, det_size):
    """Mirror of the WGSL two-pass decode: LZ4 per block -> inverse bitshuffle -> clip uint8."""
    plane_bytes = block_elems // 8
    out = np.zeros((n_frames, det_size), dtype=np.uint8)
    for f in range(n_frames):
        for b in range(n_blocks):
            mi = (f * n_blocks + b) * 2
            coff, clen = blockMeta[mi], blockMeta[mi + 1]
            planes = np.frombuffer(_lz4.decompress(raw[coff:coff + clen], uncompressed_size=block_elems * 2), dtype=np.uint8)
            for e in range(block_elems):
                v = 0
                for bit in range(16):
                    byte = planes[bit * plane_bytes + (e >> 3)]
                    v |= ((int(byte) >> (e & 7)) & 1) << bit
                out[f, b * block_elems + e] = min(v, 255)
    return out


def test_bslz4_offline_roundtrip(tmp_path):
    from quantem.widget import Show4DSTEM
    rng = np.random.default_rng(0)
    # small sparse integer detector data (det 64x64 = 4096 -> block_elems 1024, 4 blocks)
    data = rng.integers(0, 40, size=(4, 4, 64, 64), dtype=np.uint16)
    url = tmp_path / "stack.bin"
    w = Show4DSTEM(data, scan_shape=(4, 4), offline=True, offline_codec="bslz4", data_url=str(url))
    meta = json.loads(w._offline_bslz4)
    raw = np.fromfile(url, dtype=np.uint8)
    decoded = _decode_bslz4_uint8(raw, meta["blockMeta"], meta["nFrames"],
                                  meta["nBlocksPerFrame"], meta["blockElems"], 64 * 64)
    expected = np.clip(data.reshape(16, -1), 0, 255).astype(np.uint8)
    np.testing.assert_array_equal(decoded, expected)   # GPU-decoder contract: bit-exact uint8
