import atexit
import json
import os
import runpy
import time
from collections import defaultdict
from quantem.gpu.io.backends.mps import precision, _streamed
from quantem.gpu.io import _precision
metrics = defaultdict(lambda: [0, 0.0])
def wrap(owner, name, key):
    original = getattr(owner, name)
    def call(*args, **kwargs):
        started = time.perf_counter()
        if key == "conversion":
            import torch
            torch.mps.synchronize()
            metrics["producer_wait"][0] += 1
            metrics["producer_wait"][1] += time.perf_counter() - started
            started = time.perf_counter()
        try:
            return original(*args, **kwargs)
        finally:
            metrics[key][0] += 1
            metrics[key][1] += time.perf_counter() - started
    setattr(owner, name, call)
for owner, name, key in [
    (_precision, '_convert_region', 'conversion'),
    (precision, 'tensor_range', 'range_wall'),
    (precision, 'encode_measure', 'encode_measure_wall'),
    (precision, '_accumulate_measurement', 'report_scalars'),
    (precision, 'encode_ans', 'output_ans'),
    (precision.PrecisionSource, 'mean_dp', 'output_mean_dp'),
    (precision.PrecisionSource, 'masked_sum_native', 'output_detector'),
    (_streamed.MPSStreamedCounts, 'decode_scan_range_device', 'ans_decode'),
]:
    wrap(owner, name, key)
for module in (precision, _streamed):
    original = module._complete
    def complete(command, label, original=original):
        value = original(command, label)
        elapsed = command.GPUEndTime() - command.GPUStartTime()
        metrics['gpu:' + label][0] += 1
        metrics['gpu:' + label][1] += elapsed
        return value
    module._complete = complete
@atexit.register
def report():
    with open(os.environ['PROFILE_OUTPUT'], 'w') as stream:
        json.dump(dict(metrics), stream, indent=2)
runpy.run_path(os.environ['MAPED_BENCHMARK'], run_name='__main__')
