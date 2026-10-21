# Scientific backend boundaries

- MAPED and other algorithms own scientific sequencing, alignment, merging,
  plots, and reports. Keep them usable with native CPU datasets and QuantEM alone.
- ANS, encoding/decoding, device residency, and accelerator storage belong to
  `quantem.gpu`, not the algorithm package. Use dataset operations without
  introducing codec-specific branches, imports, assertions, or report labels.
- Core tests check scientific values, calibration, shapes, and lifecycle;
  codec conformance belongs in backend tests. Reuse small synthetic fixtures.
- Keep rounding error and clipping visible when precision changes. Do not
  call scaled uint16 lossless or remove verification to simplify the API.
- Optional table output must not make pandas necessary for computation;
  retain the dictionary report path for installations without pandas.
