"""Measure native command execution separately from host orchestration."""
import json
from pathlib import Path
import runpy
import sys
import time
from collections import defaultdict
import quantem.gpu.io.backends.mps.precision as precision
import quantem.gpu.io.backends.mps._streamed as streamed

stats = defaultdict(lambda: dict(calls=0, complete_wall_seconds=0., gpu_seconds=0.))
processing = False
def instrument(operation):
    def complete(command, label):
        global processing
        if label == "ANS Torch range read":
            processing = True
        started = time.perf_counter()
        result = operation(command, label)
        item = stats[('processing: ' if processing else 'loading: ') + label]
        item['calls'] += 1
        item['complete_wall_seconds'] += time.perf_counter()-started
        item['gpu_seconds'] += max(0., command.GPUEndTime()-command.GPUStartTime())
        return result
    return complete
precision._complete = instrument(precision._complete)
streamed._complete = instrument(streamed._complete)
runpy.run_path(str(Path(__file__).with_name('profile_stages.py')), run_name='__main__')
path = Path(sys.argv[3])
report = json.loads(path.read_text())
report['command_timings'] = dict(stats)
path.write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(dict(stats)), flush=True)
