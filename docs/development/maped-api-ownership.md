# MAPED API and package ownership

Use the existing `MAPEDTorch` workflow on both CUDA and MPS. The recent recipe and precision experiments do not introduce a second notebook API, a public recipe codec, or public batch/ANS tuning controls.

```python
from quantem.diffraction import MAPEDTorch

model = MAPEDTorch.from_files(files, device="mps")  # or "cuda:0"
model.preprocess(plot_summary=False)
model.diffraction_origin(sigma=1, plot_origins=False)
model.diffraction_align(edge_blend=2, plot_aligned=False)
model.real_space_align(
    num_iter=20,
    hanning_filter=True,
    padding=2,
    edge_blend=5,
    pad_val="median",
    shift_method="bilinear",
    plot_aligned=False,
)
merged = model.merge_datasets(dtype="scaled_uint16", plot_result=False)
model.show()
```

These are the existing scientific parameters, not new defaults. Omit the custom `read` callback to use the accelerator's encoded-resident loading path. The same loading policy performs the existing median hot-pixel correction. Explicit custom readers retain their existing behavior and are not silently rewritten.

`dtype="scaled_uint16"` describes approximate calibrated storage for viewing. MAPED interpolation and accumulation remain float32. QuantEM.GPU automatically converts and ANS-encodes the output; users do not call decode kernels, set partial-reduction counts, or construct private source adapters. Conversion reports distinguish storage error from algorithm parity. Float16 remains a separate supported precision choice at the generic IO layer; it is not silently substituted for this MAPED workflow.

For the existing file-backed workflow, add `save_to="merged_master.h5"` to `merge_datasets`. This saves a merged result; it does not save the experimental input-plus-recipe archive. Saving/reopening costs must be reported separately from no-file processing.

## Ownership

| Responsibility | Owner |
|---|---|
| MAPED scientific parameters, preprocessing, registration and merge order | QuantEM |
| Torch interpolation, weighting and accumulation | QuantEM |
| Generic encoded reads, ANS/bit packing, native storage ownership | QuantEM.GPU |
| Generic dtype conversion, calibration and error measurement | QuantEM.GPU |
| Native CUDA/Metal kernels supporting those generic operations | QuantEM.GPU |
| Widget presentation and native-app integration | Their respective frontend packages |

`ResidentMergeSource` is an internal QuantEM implementation consumed through existing `quantem.gpu.io.load/save` contracts. It is not a user-facing replacement for `merge_datasets`. Hardware experiments may use internal codec probes, but those are not examples for ordinary notebooks.

## Optimization status

The local MPS changes affect only generic precision conversion: GPU aggregation of error statistics and conversion occupancy. They automatically apply through the existing load/merge path. Public method signatures, scientific defaults, calibration policy and returned types were not changed. See [MPS measurements](benchmarks/2026-09-12-mps-recipe-pipeline/README.md) for exact test boundaries and the unachieved few-second target.

The approximately 7 GiB input-plus-recipe archive remains experimental. It reconstructs output by recomputing MAPED and is not a general lossless float32 compression format. Its benchmark reader must not become a second public MAPED API.

## Existing remote-service exception

The repository audit found an older `quantem.gpu.remote.maped_api` integration service. It contains MAPED-specific request parameters, workflow sequencing and a custom dense reader. This was not added by the precision changes, and it is not unified with the encoded-resident notebook path. Therefore the whole QuantEM.GPU repository cannot yet be described as free of MAPED-specific code.

The integration owner should relocate MAPED-specific policy/sequencing to the scientific or integration package, leave only generic transport/resource infrastructure in QuantEM.GPU, preserve the existing remote wire contract, and qualify the existing `MAPEDTorch` workflow. Do not silently switch a remote implementation pinned to an earlier revision or change its precision semantics while doing that migration.

The AST boundary regression test covers both `maped.py` and `_maped_resident.py`: native accelerator imports and private QuantEM.GPU/backend imports are prohibited in production MAPED code.
