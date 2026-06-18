"""Shared pytest configuration: auto-marking of slow / cpp tests.

Rather than scatter ``@pytest.mark`` decorators across the suite, the policy
for which tests are expensive lives here in one auditable place. Two markers
are applied at collection time:

* ``cpp``  — the test shells out to a C/C++ compiler (codegen roundtrips,
  syntax checks). These already self-skip when no toolchain is on PATH; the
  marker lets you *also* exclude them when a compiler *is* present.
* ``slow`` — long-running fits, Monte-Carlo consistency runs, or fine-dt
  convergence sweeps.

Fast inner-loop run:  ``pytest -m "not slow and not cpp"``
Full suite (CI):      ``pytest``           (markers are not filtered)
"""

# Tests whose *name* contains any of these substrings invoke a compiler.
_CPP_NAME_MARKERS = (
    "cpp_roundtrip",        # *_python_cpp_roundtrip across pid/madgwick/etc.
    "multicraft_roundtrip",
    "compiles_with_cc",
    "emits_scalar_ref",     # test_codegen_emit_cpp _syntax_check tests
)

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


def pytest_collection_modifyitems(config, items):
    import pytest

    for item in items:
        module = item.module.__name__.rsplit(".", 1)[-1]
        name = item.name
        if any(s in name for s in _CPP_NAME_MARKERS):
            item.add_marker(pytest.mark.cpp)
        if module in _SLOW_MODULES or any(s in name for s in _SLOW_NAME_MARKERS):
            item.add_marker(pytest.mark.slow)
