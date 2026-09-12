"""Check exact serialized output after native writer changes.

Usage: python compare_saved_chunks.py REFERENCE_H5 CANDIDATE_H5

This checks file bytes and precision metadata only. HDF5 filters are bypassed;
scientific arrays are never decompressed or processed on the host.
"""

import argparse
import json
import math
from pathlib import Path

import h5py


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()
    with h5py.File(args.reference, "r") as reference, h5py.File(args.candidate, "r") as candidate:
        expected = reference["/entry/data/data"]
        actual = candidate["/entry/data/data"]
        assert expected.shape == actual.shape, "Saved detector/scan shapes differ."
        assert expected.dtype == actual.dtype, "Saved count types differ."
        assert expected.chunks == actual.chunks == (1, *expected.shape[1:])
        assert json.loads(reference.attrs["quantem_precision_v1"]) == json.loads(
            candidate.attrs["quantem_precision_v1"]
        ), "Saved precision coefficients or error measurements differ."
        for frame in range(expected.shape[0]):
            position = (frame, 0, 0)
            assert expected.id.read_direct_chunk(position) == actual.id.read_direct_chunk(position), (
                f"Compressed output differs at scan frame {frame}."
            )
        print(json.dumps({
            "compressed_chunks": expected.shape[0],
            "stored_values": math.prod(expected.shape),
            "exact": True,
            "host_array_decoding": False,
        }))


if __name__ == "__main__":
    _main()
