"""Tune conversion occupancy on a real MPS merged region, with GPU parity."""
from pathlib import Path
exec(Path(__file__).with_name('run.py').read_text().split('for rows in args.rows:')[0])
import quantem.gpu.io.backends.mps.precision as module
source = make_source(8)
blocks = iter(source.blocks())
values_t = next(blocks)[:1024].clone()
context = SimpleNamespace(backend='mps', dtype='float32', shape=source.shape, saved=None)
reference, reference_report = _convert_region(context, values_t)
reference_t = reference.to_torch().view(torch.int16)
results = []
for partials in (8192, 32768, 65536, 131072, 8192):
    module._MEASUREMENT_PARTIALS = partials
    timings = []
    for repeat in range(4):
        torch.mps.synchronize()
        started = time.perf_counter()
        encoded, precision = _convert_region(context, values_t)
        torch.mps.synchronize()
        timings.append(time.perf_counter()-started)
        observed_t = encoded.to_torch().view(torch.int16)
        assert torch.equal(observed_t, reference_t)
        encoded.release()
    results.append(dict(partials=partials, seconds=timings, rmse=precision['rmse'],
                        reference_rmse=reference_report['rmse'], codes_exact=True))
args.report.write_text(json.dumps(results, indent=2)+'\n')
print(results, flush=True)
for loaded in sources:
    loaded.close()
