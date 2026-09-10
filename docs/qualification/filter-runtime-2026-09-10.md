# Filter runtime validation cost — 2026-09-10

The public raw INS runtime used **29.5% less CPU** on the local synthetic
benchmark after specializing covariance symmetry validation and eliminating a
discarded covariance copy. This is a Python runtime improvement; the generated
mathematics, covariance constraints and compiler policy are unchanged.

## Change and preserved contracts

The profile over 61,000 predictions and 6,100 updates (including warmup) recorded
73,204 calls to NumPy's general `allclose` machinery. Covariance callers have
already established square float arrays and rejected nonfinite entries.
The specialized comparison uses the same elementwise condition:
`abs(A - A.T) <= 1e-12 + 1e-10 * abs(A.T)`. It does not substitute a looser
matrix-norm tolerance. Both directions of each asymmetric pair are checked.

Internal staged-state validation previously copied the validated covariance
and discarded that copy. It now checks the already owned state directly.
Public covariance boundaries still copy caller inputs, and native output
ownership is unchanged. Finiteness, symmetry, eigenvalue-based PSD validation,
measurement Cholesky validation, Schmidt joint-covariance checks and atomic
state commit remain in place. The float epsilon used in the existing roundoff
bound is resolved once at import.

## Measured workload

Baseline: Manta `564d0bad18df14a3b2efd1fa540738784ac78ce3`.
Host: Intel Core Ultra 9 386H, Linux/WSL x86-64, Python 3.12.3,
NumPy 2.5.1, CasADi 3.7.2. Each run used single-threaded BLAS/OpenMP and the
same expanded, O1-compiled synthetic 15-state linearized Earth INS fixture.

[`benchmarks/ins_runtime.py`](../../benchmarks/ins_runtime.py) measures seven
batches, each containing 20,000 predictions at a modeled 100 Hz and 2,000 DVL
updates at 10 Hz, with an explicit per-sample covariance. Construction,
compilation/cache loading, 1,000 warmup cycles, resets and RSS sampling are
outside the timing region. No profiler ran during these comparisons.

| Metric per 200 seconds of modeled input | Before | After |
| --- | ---: | ---: |
| Median process CPU seconds | 1.4742 | 1.0388 |
| Median wall seconds | 1.4067 | 0.9881 |
| Equivalent percentage of one core at declared rates | 0.737% | 0.519% |
| Sampled RSS range, KiB | 74,248–74,908 | 74,236–74,888 |

The seven CPU samples were 1.370–1.552 s before and 0.979–1.146 s after.
Final nominal state and covariance arrays were **exactly equal** between the
two runs and across the seven resets/repetitions within each run. Complete
samples and final arrays are recorded in
[`before.json`](data/filter-runtime-2026-09-10/before.json) and
[`after.json`](data/filter-runtime-2026-09-10/after.json).

The absolute saving here is about **0.218 percentage points of one core** at
the specified rates. This is not a 29.5% whole-nav or whole-vehicle saving.
The workload excludes packet preintegration, Shiver observation handling,
Fathom, logging and every other vehicle process. It does not use the saved
Mako commissioning estimator artifact, whose schema predates current Shiver.
RSS includes retained construction objects; this short accelerated benchmark
does not establish long-duration memory bounds or count allocation churn.
ARM timings and deadline tails remain unmeasured while the Jetson is repaired.

Run from the Manta checkout, with a writable task-specific cache directory:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=. \
  XDG_CACHE_HOME=/tmp/manta-filter-runtime-cache \
  .venv/bin/python benchmarks/ins_runtime.py --output /tmp/ins-runtime.json
```

## Validation

The filter and replay suite passed all 87 tests, including generated C++
parity. Added public-boundary cases cover elementwise symmetry tolerances
across scales, absolute tolerance near zero, nonfinite rejection, failure
atomicity and reset covariance ownership. The changed files pass Ruff.
Wheel and source archive builds pass using the already installed Hatchling
backend in Shiver's environment; an import and EKF predict/update smoke test
against the built wheel passes.

The full Manta suite, including `server/mako/test_smoke.py` outside the default
test path, finished with **1,355 passed, 5 skipped, 6 failed**. All six failures
reproduced on the clean baseline checkout (identical Git tree to `564d0ba`),
with identical error output after normalizing Python function addresses. They
cover existing consistency expectations, stale simulator reset fields,
UKF attitude tolerance, and missing gravity in a server smoke fixture.
Failure node IDs are retained in
[`validation.json`](data/filter-runtime-2026-09-10/validation.json).

Shiver's navigation filter, preintegration runtime, replay and delayed-fix
integration checks passed **41 tests**. A separate seeded comparison over
4,000 finite matrices, with scales from approximately 1e-200 to 1e200, found
no symmetry-decision differences from the previous `allclose` check. The full
Manta Ruff scan still reports nine existing import/export ordering findings
in files untouched by this change; the changed Python files pass.
