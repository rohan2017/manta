"""Paired simulator storage benchmark against an explicit baseline source file.

Export the prior _sim.py from git, then pass its path as the only argument.
Both runtimes share the same module, native compilation and input workload.
"""

import dataclasses
import importlib.util
import json
import statistics
import sys
import time
import tracemalloc

from benchmarks.simulator_step import make_sim

name = "manta.codegen.numpy._d14_baseline"
spec = importlib.util.spec_from_file_location(name, sys.argv[1])
module = importlib.util.module_from_spec(spec)
sys.modules[name] = module
spec.loader.exec_module(module)


def run(sim, count):
    start = time.perf_counter_ns()
    for _ in range(count):
        sim.step(0.002)
    return (time.perf_counter_ns() - start) / count / 1000


results = {}
for sensors in (1, 8):
    candidate = make_sim(sensors, compile_kernels=True)
    baseline = module.NumpySim(candidate.module)
    baseline._enable_compile(optimization="O1")
    for sim in (baseline, candidate):
        for _ in range(200):
            sim.step(0.002)
    pairs = []
    for repeat in range(7):
        ordered = (baseline, candidate) if repeat % 2 == 0 else (candidate, baseline)
        values = {id(sim): run(sim, 5000) for sim in ordered}
        pairs.append(
            {"baseline_us": values[id(baseline)], "packed_us": values[id(candidate)]}
        )
    assert dataclasses.asdict(baseline.checkpoint()) == dataclasses.asdict(
        candidate.checkpoint()
    )
    allocations = {}
    for label, sim in (("baseline", baseline), ("packed", candidate)):
        tracemalloc.start()
        for _ in range(100):
            sim.step(0.002)
        allocations[label] = dict(
            zip(("retained_bytes", "peak_bytes"), tracemalloc.get_traced_memory())
        )
        tracemalloc.stop()
    b = statistics.median(p["baseline_us"] for p in pairs)
    c = statistics.median(p["packed_us"] for p in pairs)
    results[str(sensors)] = {
        "pairs": pairs,
        "median_baseline_us": b,
        "median_packed_us": c,
        "reduction_percent": 100 * (b - c) / b,
        "traced_memory": allocations,
        "exact_checkpoint_parity": True,
    }
print(json.dumps(results, indent=2))
