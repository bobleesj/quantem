"""Verify all scaled viewing codes against the original reduction topology."""
from pathlib import Path
exec(Path(__file__).with_name('run.py').read_text().split('for rows in args.rows:')[0])
import quantem.gpu.io.backends.mps.precision as module
source = make_source(8)
context = SimpleNamespace(backend='mps', dtype='float32', shape=source.shape, saved=None)
values = 0
maximum_rmse_difference = 0.
for block_t in source.blocks():
    for first in range(0, len(block_t), 1024):
        original_t = block_t[first:first+1024]
        module._MEASUREMENT_PARTIALS = 8192
        before, before_report = _convert_region(context, original_t)
        module._MEASUREMENT_PARTIALS = 65536
        after, after_report = _convert_region(context, original_t)
        assert torch.equal(before.to_torch().view(torch.int16), after.to_torch().view(torch.int16))
        for key in ('scale', 'offset', 'max_abs_error', 'changed', 'positive_to_zero', 'overflow', 'clipped'):
            assert before_report[key] == after_report[key], key
        difference = abs(before_report['rmse'] - after_report['rmse'])
        maximum_rmse_difference = max(maximum_rmse_difference, difference)
        assert difference <= 1e-7 * max(before_report['rmse'], 1e-30)
        values += original_t.numel()
        before.release()
        after.release()
report['conversion_parity'] = dict(values=values, code_mismatches=0,
    maximum_regional_rmse_absolute_difference=maximum_rmse_difference,
    counters_and_calibration_exact=True)
args.report.write_text(json.dumps(report, indent=2)+'\n')
print(report, flush=True)
for loaded in sources:
    loaded.close()
