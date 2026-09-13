"""Compare MPS recipe reconstruction batches on seven local acquisitions."""
import argparse
import json
from pathlib import Path
import time
from types import SimpleNamespace
import torch
from quantem.gpu import io
from quantem.diffraction._maped_resident import ResidentMergeSource
from quantem.gpu.io._precision import _convert_region, _regional_report
from quantem.gpu.io.backends.mps.precision import encode_ans, PrecisionSource

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('directory', type=Path)
parser.add_argument('recipe', type=Path)
parser.add_argument('report', type=Path)
parser.add_argument('--rows', nargs='+', type=int, default=[8, 16, 4, 8])
args = parser.parse_args()
assert torch.backends.mps.is_available()
files = sorted(args.directory.glob('*_master.h5'))
assert len(files) == 7
recipe = json.loads(args.recipe.read_text())
started = time.perf_counter()
sources = [io.load(path, backend='mps', representation='encoded', verbose=False) for path in files]
torch.mps.synchronize()
report = {'load_seconds': time.perf_counter()-started, 'trials': []}
print('LOAD', report['load_seconds'], flush=True)
shifts = [torch.tensor(recipe['alignment'][name], device='mps', dtype=torch.float32)
          for name in ('real_space_shifts_row_column', 'diffraction_shifts_row_column')]

def make_source(rows):
    source = ResidentMergeSource(sources, *shifts, close_sources_before_reopen=False)
    source.region_frames = rows * 512
    source._compute_region_frames = None
    return source

for rows in args.rows:
    source = make_source(rows)
    context = SimpleNamespace(backend='mps', dtype='float32', shape=source.shape, saved=None)
    parts, reports = [], []
    first = 0
    merge_seconds = conversion_seconds = 0.
    peak = torch.mps.driver_allocated_memory()
    started = time.perf_counter()
    blocks = iter(source.blocks())
    while True:
        t = time.perf_counter()
        try:
            values_t = next(blocks)
        except StopIteration:
            break
        torch.mps.synchronize()
        merge_seconds += time.perf_counter()-t
        t = time.perf_counter()
        # Keep calibration identical while varying only computation batches.
        for cursor in range(0, len(values_t), 1024):
            values = values_t[cursor:cursor+1024]
            encoded, precision = _convert_region(context, values)
            precision.update(first_frame=first, stop_frame=first+len(values))
            reports.append(precision)
            parts.append(encode_ans(encoded, (1, len(values), *source.shape[2:])))
            first += len(values)
            del values, encoded
        torch.mps.synchronize()
        conversion_seconds += time.perf_counter()-t
        peak = max(peak, torch.mps.driver_allocated_memory())
        del values_t
    precision = _regional_report(source.shape, reports)
    view = PrecisionSource(parts, source.shape, precision)
    t = time.perf_counter()
    dp = view.frame_native(131328)
    torch.mps.synchronize()
    query = time.perf_counter()-t
    trial = dict(rows=rows, merge_seconds=merge_seconds, conversion_seconds=conversion_seconds,
                 processing_seconds=time.perf_counter()-started, query_seconds=query,
                 output_bytes=view.nbytes, sampled_driver_peak_bytes=peak,
                 precision=precision)
    report['trials'].append(trial)
    args.report.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in trial.items() if k!='precision'}), flush=True)
    del dp
    view.release()
    source.close()
    del view, parts
    torch.mps.empty_cache()
# Full same-backend float32 qualification. Do not compare MPS to CUDA as exact.
reference = make_source(8)
candidate = make_source(16)
a = iter(reference.blocks())
values = mismatches = 0
for block_t in candidate.blocks():
    for offset in range(0, len(block_t), 4096):
        reference_t = next(a)
        candidate_t = block_t[offset:offset+len(reference_t)]
        mismatches += int(torch.count_nonzero(reference_t.view(torch.int32) != candidate_t.view(torch.int32)).item())
        values += reference_t.numel()
        del reference_t, candidate_t
report['float32_parity'] = dict(values=values, bit_mismatches=mismatches)
args.report.write_text(json.dumps(report, indent=2)+'\n')
print('PARITY', report['float32_parity'], flush=True)
for source in sources:
    source.close()
