"""Attribute MPS time with synchronized diagnostic stage boundaries."""
from pathlib import Path
# Reuse the frozen benchmark setup, keeping paths supplied by its CLI.
exec(Path(__file__).with_name('run.py').read_text().split('for rows in args.rows:')[0])
from collections import defaultdict
import quantem.diffraction._maped_resident as merge_module
import quantem.gpu.io.backends.mps.precision as precision_module
stats = defaultdict(float)

def timed_function(name, operation):
    def measured(*positional, **keywords):
        torch.mps.synchronize()
        started = time.perf_counter()
        result = operation(*positional, **keywords)
        torch.mps.synchronize()
        stats[name] += time.perf_counter() - started
        return result
    return measured

sources = [SimpleNamespace(shape=loaded.shape, metadata=loaded.metadata,
                           read=timed_function('decode', loaded.read), close=loaded.close)
           for loaded in sources]
for name in ('_sample_scan_rows', '_sample_scan_interior', '_shift_detector'):
    setattr(merge_module, name, timed_function(name, getattr(merge_module, name)))
for name in ('source_range', 'encode_measure', '_accumulate_measurement'):
    setattr(precision_module, name, timed_function(name, getattr(precision_module, name)))
source = make_source(8)
context = SimpleNamespace(backend='mps', dtype='float32', shape=source.shape, saved=None)
started = time.perf_counter()
for block_t in source.blocks():
    for first in range(0, len(block_t), 1024):
        values = block_t[first:first+1024]
        encoded, measured = _convert_region(context, values)
        part = timed_function('ans', encode_ans)(encoded, (1, len(values), *source.shape[2:]))
        part.release()
        del values, encoded
report['profile'] = dict(stats)
report['profile_wall_seconds'] = time.perf_counter()-started
args.report.write_text(json.dumps(report, indent=2)+'\n')
print(report, flush=True)
for loaded in sources:
    loaded.close()
