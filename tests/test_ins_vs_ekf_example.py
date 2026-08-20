"""Executable smoke coverage for the EKF/INS A/B example."""

import numpy as np

from examples.vehicles.ins_vs_ekf import build_world, run
from manta.parts import DragSurface, Mass, PointBuoy


def test_truth_and_estimator_models_are_deliberately_different():
    truth = build_world(truth=True)
    estimate = build_world(truth=False)
    truth_parts = {part.name: part for part in truth.crafts[0].parts}
    estimate_parts = {part.name: part for part in estimate.crafts[0].parts}

    truth_masses = [part for part in truth_parts.values()
                    if isinstance(part, Mass)]
    estimate_masses = [part for part in estimate_parts.values()
                       if isinstance(part, Mass)]
    truth_buoys = [part for part in truth_parts.values()
                   if isinstance(part, PointBuoy)]
    estimate_buoys = [part for part in estimate_parts.values()
                      if isinstance(part, PointBuoy)]
    truth_drag = [part for part in truth_parts.values()
                  if isinstance(part, DragSurface)]
    estimate_drag = [part for part in estimate_parts.values()
                     if isinstance(part, DragSurface)]

    assert (len(truth_masses), len(estimate_masses)) == (3, 1)
    assert (len(truth_buoys), len(estimate_buoys)) == (5, 1)
    assert (len(truth_drag), len(estimate_drag)) == (5, 1)
    truth_mass = sum(float(part.mass) for part in truth_masses)
    estimate_mass = sum(float(part.mass) for part in estimate_masses)
    assert abs(truth_mass / estimate_mass - 1.0) < 0.02
    assert np.allclose(truth_parts["surge"].force,
                       estimate_parts["surge"].force, rtol=0.04)
    assert "wave_fx" in truth_parts and "wave_fx" not in estimate_parts
    assert "model_force" not in truth_parts and "model_force" in estimate_parts


def test_ab_example_runs_both_estimators_headless():
    result = run(duration=0.4, dt=0.02, warmup=0.1, seed=3,
                 progress=False)
    values = (*vars(result.ekf).values(), *vars(result.ins).values(),
              result.wave_force_rms_n, result.wave_torque_rms_nm)
    assert all(np.isfinite(value) and value >= 0.0 for value in values)
    assert result.rho_by_sensor == {
        "auv.model_force.specific_force": 0.012,
    }
