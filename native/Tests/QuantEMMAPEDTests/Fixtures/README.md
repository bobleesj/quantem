Independent native parity fixtures and expected scientific values belong here.
Never replace expectations to hide differences between implementations.

# Normalized-grid regression

`normalized_grid.json` is generated independently by
`../../generate_sampling_fixture.py`. It freezes the float32 zero-padding
boundary for a translated constant image. Preserve separately rounded grid
operations; compiler contraction changes these values.
