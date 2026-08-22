"""Executable smoke coverage for the EKF/INS A/B example."""

import numpy as np

from examples.vehicles.ins_vs_ekf import (
    build_world,
    calibrate_model_force,
    run,
)
from manta.estimation.ins import MODEL_FORCE_RHO_CEILING
from manta.fit import FitEvidence
from manta.parts import DragSurface, Mass, ModelForce, PointBuoy


def test_truth_and_estimator_models_are_deliberately_different():
    truth = build_world(truth=True)
    # The reduction carries no ModelForce until its model error has been
    # identified on held-out data; the evidence-less form is the
    # calibration model.
    assert "model_force" not in {
        part.name for part in build_world(truth=False).crafts[0].parts}
    evidence = calibrate_model_force(seed=3, dt=0.02, windows=4)
    assert isinstance(evidence, FitEvidence)
    estimate = build_world(truth=False, evidence=evidence)
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
    assert "turbulence" in truth_parts and "turbulence" not in estimate_parts
    assert "model_force" not in truth_parts and "model_force" in estimate_parts
    assert isinstance(estimate_parts["model_force"], ModelForce)
    assert estimate_parts["model_force"].evidence == evidence


def test_ab_example_runs_both_estimators_headless():
    result = run(duration=0.4, dt=0.02, warmup=0.1, seed=3,
                 progress=False)
    values = (*vars(result.ekf).values(), *vars(result.ins).values(),
              result.wave_force_rms_n, result.wave_torque_rms_nm)
    assert all(np.isfinite(value) and value >= 0.0 for value in values)
    # The model error is identified, not declared: rho follows the
    # evidence's white floor and must sit inside the INS's valid regime.
    assert result.evidence.accepted, result.evidence.summary()
    assert result.evidence.channel == "auv.imu.accel"
    rho = result.rho_by_sensor["auv.model_force.specific_force"]
    assert 0.0 < rho <= MODEL_FORCE_RHO_CEILING
    assert {ax.noise_model.kind for ax in result.evidence.axes} \
        == {"gauss_markov"}
