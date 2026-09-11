# Packed simulator state — 2026-09-10

The packed-state prototype reduces median step time by **40.4%** for four sensor
ports (46.06 → 27.44 µs) and **22.9%** for 32 ports (81.66 → 62.97 µs) on this
WSL development host. Both use the existing simulator benchmark's zero-gravity
20 kg craft, fixed 2 ms steps, zero noise, and O1 native kernels. Construction
and compilation are outside the timed region. These are local simulator results,
not a fleet-wide or Jetson performance claim.

Seven paired rounds alternate execution order, with 5,000 steps per runtime
per round after 200 warmup steps. Both runtimes finish with exactly equal complete
checkpoints. Tracemalloc's 100-step peak falls from 6,650 to 5,666 bytes for four
ports and 23,785 to 20,386 bytes for 32 ports. This is peak traced allocation,
not a total allocation count. Raw timings are in the adjacent JSON file.

Reproduce from the Manta checkout:

```bash
git show 4b5dd82:manta/codegen/numpy/_sim.py > /tmp/manta-baseline-sim.py
XDG_CACHE_HOME=/tmp/manta-packed-benchmark PYTHONPATH=. python benchmarks/packed_state.py /tmp/manta-baseline-sim.py
```

The implementation retains live nested owner dictionaries while manifold arrays
view the current packed state. Each successful step replaces those views;
previously held slot arrays remain stale as documented. Ordinary dictionary edits
invalidate the prepared layout and undergo full validation before the next step.
In-place array edits directly affect packed state and receive a finite-value check
before execution. Inputs/noise retain their existing typed validation.

Regressions cover owner aliases, array edits, expired arrays, update/pop/setdefault/
union mutations, whole-state assignment, deepcopy, failed-kernel retry, stochastic
checkpoint replay, folded steps, coupled models and owned sensor outputs.
The retained-caller dictionary from whole-state assignment is copied into the
runtime's tracked dictionaries; read live state through `sim.state` thereafter.
