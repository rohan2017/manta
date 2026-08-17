"""Shared pytest configuration: auto-marking of slow / cpp tests.

Rather than scatter ``@pytest.mark`` decorators across the suite, the policy
for which tests are expensive lives here in one auditable place. Two markers
are applied at collection time:

* ``cpp``  — the test shells out to an external toolchain (C/C++ compiler,
  emcc, node). These already self-skip when no toolchain is on PATH; the
  marker lets you *also* exclude them when a compiler *is* present.
* ``slow`` — long-running fits, Monte-Carlo consistency runs, or fine-dt
  convergence sweeps.

Fast inner-loop run:  ``pytest -m "not slow and not cpp"``
Full suite (CI):      ``pytest``           (markers are not filtered)

Name-substring matching is silently brittle — a renamed test would drop its
marker with no error — so on a full-suite collection every pattern is
checked to still match something, and a dead pattern fails collection.
"""

# Tests whose *name* contains any of these substrings invoke a compiler.
_CPP_NAME_MARKERS = (
    "cpp_roundtrip",        # *_python_cpp_roundtrip across pid/madgwick/etc.
    "multicraft_roundtrip",
    "compiles_with_cc",
    "emits_scalar_ref",     # test_codegen_emit_cpp _syntax_check tests
)

# Whole modules dominated by toolchain work (cc / emcc / node shells).
_CPP_MODULES = frozenset({
    "test_codegen_wasm",
    "test_filter_replay",
})

# Whole modules dominated by long fits / Monte-Carlo work.
_SLOW_MODULES = frozenset({
    "test_noise_fit",
    "test_consistency",
    "test_fit",
})

# Individual slow tests living in otherwise-fast modules (long sims / sweeps).
_SLOW_NAME_MARKERS = (
    "frictionless_energy_converges_with_dt",
    "double_pendulum_conserves_energy_undamped",
    "double_pendulum_converges_to_textbook_rk4",
    "gimbal_pendulum_precesses_at_minus_omega",
    "no_precession_without_hub_spin",
    "gimbal_conserves_momentum_and_energy",
    "eskf_nees_consistency_over_seeds",
    "random_walk_bias_walks_with_driver",
    "grad_through_rollout_is_finite",
)

# Below this many collected items the run is a subset (one file, -k) and
# unmatched patterns are expected; at or above it we treat the collection
# as the full suite and enforce that every pattern still matches.
_FULL_SUITE_THRESHOLD = 500


def pytest_collection_modifyitems(config, items):
    import pytest

    hit_names: set[str] = set()
    hit_modules: set[str] = set()
    for item in items:
        module = item.module.__name__.rsplit(".", 1)[-1]
        name = item.name
        cpp_hits = [s for s in _CPP_NAME_MARKERS if s in name]
        if cpp_hits or module in _CPP_MODULES:
            item.add_marker(pytest.mark.cpp)
        slow_hits = [s for s in _SLOW_NAME_MARKERS if s in name]
        if module in _SLOW_MODULES or slow_hits:
            item.add_marker(pytest.mark.slow)
        hit_names.update(cpp_hits)
        hit_names.update(slow_hits)
        if module in _CPP_MODULES or module in _SLOW_MODULES:
            hit_modules.add(module)

    if len(items) >= _FULL_SUITE_THRESHOLD:
        dead = (set(_CPP_NAME_MARKERS + _SLOW_NAME_MARKERS) - hit_names) | (
            (_CPP_MODULES | _SLOW_MODULES) - hit_modules)
        if dead:
            raise pytest.UsageError(
                f"conftest marker patterns matched no test: {sorted(dead)} "
                f"— a rename dropped its cpp/slow marker; update the "
                f"pattern lists in tests/conftest.py.")
