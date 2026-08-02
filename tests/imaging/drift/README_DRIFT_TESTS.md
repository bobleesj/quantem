# Drift tests, under `tests/imaging/drift/`

Grouped by role, not split by module, so other imaging submodules can carry
their own test folders alongside this one:

```text
tests/imaging/
  drift/
    test_drift*.py          # all drift tests
    simulation_fixture.py   # shared forward-model helpers, not a test file
    README_DRIFT_TESTS.md   # this inventory
```

Run:

```bash
pytest tests/imaging/drift/
```

## Inventory

| File | Role |
|------|------|
| `test_drift2d.py` | 2D drift correction reproduces the paper result, on real data *(data-gated)* |
| `test_drift3d.py` | 3D (XEDS spectrum image) drift correction reproduces the paper result, on real data *(data-gated)* |
| `test_drift_simulations.py` | Synthetic 2D/3D/4D workflows: forward model, preprocess/affine/strip/nonrigid, from_reference, from_4dstem, corrected(), save/load, and frozen affine/nonrigid baselines |
| `test_drift_utils.py` | Core building blocks vs numpy/scipy, or vs a hand-computable analytic answer where no reference exists |
| `test_drift_io.py` | EMD load, metadata angles, `scan_pairs`, crop, show |

Naming rule: **`test_drift<topic>.py`** or **`test_drift_<topic>.py`**, one file per role.

## Policy

Keep scientist workflows + frozen baselines + kernel/forward-model parity.
Do not add pure `raises` / "runs without crash" smoke unless a silent wrong answer was proven.
([ophusgroup/dev](https://github.com/ophusgroup/dev) D2)
