"""Fresh-install end-to-end check for quantem.widget Show4DSTEM.

Run inside a CLEAN conda env that has ONLY `pip install quantem_widget-*.whl` (no
editable source on the path). Proves a brand-new user can install the wheel and run
the documented API on real 4D-STEM data:

    from quantem.widget import load, Show4DSTEM
    Show4DSTEM(load(master, det_bin=4))            # single
    Show4DSTEM(load([m0, m1, m2], det_bin=4))      # many

Pass the data dir as argv[1] (default mjgoat path). Prints ALL PASS on success;
any failure raises and exits non-zero.
"""
import glob
import os
import sys

DATA = sys.argv[1] if len(sys.argv) > 1 else "/home/owner/data/samsung/20260512_dram"


def main():
    # the install must NOT resolve to an editable source tree
    import quantem.widget as w
    src = os.path.dirname(w.__file__)
    print(f"quantem.widget {w.__version__} from {src}")
    assert "site-packages" in src, f"not a clean install: {src}"

    from quantem.widget import load, Show4DSTEM
    from quantem.widget.io import detect_backend
    backend = detect_backend()
    print(f"backend: {backend}")

    masters = sorted(glob.glob(f"{DATA}/*master.h5"))
    assert masters, f"no masters under {DATA}"
    print(f"{len(masters)} masters")

    # single
    v1 = Show4DSTEM(load(masters[0], det_bin=4, verbose=False), verbose=False)
    print(f"single: {type(v1).__name__} scan={v1._scan_shape} det={v1._det_shape}")
    assert v1._scan_shape[0] > 0 and v1._det_shape[0] > 0

    # multi (>=2 datasets)
    sub = masters[:3]
    v2 = Show4DSTEM(load(sub, det_bin=4, verbose=False), verbose=False)
    print(f"multi: {type(v2).__name__} n_frames={v2.n_frames} frame_dim={v2.frame_dim_label}")
    assert v2.n_frames >= 1

    # the public surface is exactly the unified API (no legacy name)
    assert not hasattr(w, "load_4dstem_macbook"), "legacy load_4dstem_macbook still exported"

    print("ALL PASS")


if __name__ == "__main__":
    main()
