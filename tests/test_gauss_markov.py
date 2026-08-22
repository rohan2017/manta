"""First-order Gauss–Markov noise channels — `GaussMarkovNoise` end to end.

The correlated error is filter state with the exact discrete transition
`φ = exp(-dt/τ)` and process noise `(1-φ²)·σ²`, synthesized once in the IR
so every backend shares it. Checked here: declaration plumbing (slot,
driver, `<name>_tau` attribute and override, validation), the simulated
recursion, the auto-assembled Q, EKF/UKF tracking, and JAX parity. The
C++ compile-and-run parity lives in `test_filter_runtime.py`.
"""

import numpy as np
import pytest

from manta import EKF, UKF, Craft, Sim, TargetNumpy, World, state_spec_from_craft
from manta.fields import GravityField
from manta.ir.frames import PartFrame, WorldFrame
from manta.ir.types import Vec3
from manta.ir.wrench import Wrench
from manta.parts import (
    GaussMarkovNoise,
    Mass,
    Output,
    Part,
    PartUpdate,
    WhiteNoise,
)

TAU, SIGMA, DT = 1.5, 0.05, 0.01


class CorrelatedGyro(Part):
    """Gyro whose error is white noise plus a Gauss–Markov drift."""
    gyro_noise = WhiteNoise("R3", frame=PartFrame, sigma=0.002)
    gyro_drift = GaussMarkovNoise("R3", frame=PartFrame, sigma=SIGMA, tau=TAU)
    gyro = Output()

    def update(self, ctx):
        zero = Vec3[PartFrame].constant((0.0, 0.0, 0.0))
        omega = ctx.orientation.conjugate().apply(
            ctx.angular_velocity[WorldFrame])
        return PartUpdate(
            wrench=Wrench(force=zero, torque=zero),
            outputs={"gyro": omega + self.gyro_drift + self.gyro_noise})


def _world(**gyro_overrides):
    w = World().add_field(GravityField(g=(0.0, 0.0, -9.81)))
    c = Craft("drone")
    c.add(Mass("body", mass=1.0, moi=(0.1, 0.1, 0.1)))
    c.add(CorrelatedGyro("g", **gyro_overrides))
    w.add_craft(c, position=(0.0, 0.0, 5.0))
    return w, c


# ---------------------------------------------------------------------------
# Declaration plumbing
# ---------------------------------------------------------------------------

def test_declaration_metadata_slot_and_runtime_attributes():
    decl = GaussMarkovNoise("R3", frame=PartFrame, sigma=0.1, tau=2.0)
    assert decl.kind == "gauss_markov"
    assert decl.contributes_state
    assert decl.driver_input_name("e") == "e_driver"
    assert decl.state_manifold().kind == "vec"
    assert decl.runtime_attributes("e") == {"e_sigma": (0.1, True),
                                            "e_tau": (2.0, False)}
    _, c = _world()
    g = c.parts[-1]
    assert g.gyro_drift_sigma == SIGMA and g.gyro_drift_tau == TAU
    spec = state_spec_from_craft(c)
    assert "g.gyro_drift" in [s.name for s in spec.slots]
    init = c.initial_state()
    assert "g.gyro_drift" in init and "g.gyro_drift_driver" in init


def test_tau_is_validated_and_overridable_per_instance():
    with pytest.raises(TypeError):
        GaussMarkovNoise("R3", sigma=0.1)               # tau is required
    for bad in (0.0, -1.0, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="tau"):
            GaussMarkovNoise("R3", sigma=0.1, tau=bad)
    _, c = _world(gyro_drift_tau=0.25)
    assert c.parts[-1].gyro_drift_tau == 0.25
    with pytest.raises(ValueError, match="gyro_drift_tau"):
        _world(gyro_drift_tau=0.0)
    with pytest.raises(TypeError, match="unknown parameter"):
        _world(gyro_drift_bogus=1.0)


def test_inert_channel_contributes_no_state():
    _, c = _world(gyro_drift_sigma=0.0)
    spec = state_spec_from_craft(c)
    assert "g.gyro_drift" not in [s.name for s in spec.slots]
    assert "g.gyro_drift" not in c.initial_state()


# ---------------------------------------------------------------------------
# Kernel: exact discrete recursion and its process noise
# ---------------------------------------------------------------------------

def test_sim_applies_exact_exponential_transition():
    w, _ = _world()
    sim = TargetNumpy(Sim(w))
    x0 = np.array([0.4, -0.2, 0.1])
    sim.state["drone"]["g.gyro_drift"] = x0.copy()
    drv = np.array([0.5, -0.25, 0.1])
    expected = x0.copy()
    phi = np.exp(-DT / TAU)
    for _ in range(40):
        sim.state["drone"]["g.gyro_drift_driver"] = drv.copy()
        sim.step(DT)
        expected = phi * expected + np.sqrt(1.0 - phi * phi) * drv
    np.testing.assert_allclose(sim.state["drone"]["g.gyro_drift"], expected,
                               atol=1e-12)


def test_stationary_variance_is_sigma_squared_under_driver_samples():
    """Driving the exact recursion with N(0, σ²) samples keeps the slot's
    variance at σ² — the stationary Gauss–Markov process, for any dt."""
    rng = np.random.default_rng(3)
    phi = np.exp(-0.05 / TAU)
    e = np.zeros(20000)
    for k in range(1, e.size):
        e[k] = phi * e[k - 1] + np.sqrt(1 - phi * phi) * rng.normal(0, SIGMA)
    assert abs(np.std(e[2000:]) / SIGMA - 1.0) < 0.1


def test_auto_Q_is_one_minus_phi_squared_times_sigma_squared():
    w, _ = _world()
    ekf_t = EKF(w)
    ekf = TargetNumpy(ekf_t)
    slot = ekf_t.spec.slot("drone.g.gyro_drift")
    for dt in (0.01, 0.2, 2.0):
        L = np.asarray(ekf_t.sys.L_fn(ekf.x, np.zeros(0), dt, 0.0))
        Q = L @ ekf_t.sys.Sigma @ L.T
        block = Q[slot.tangent_offset:slot.tangent_offset + 3,
                  slot.tangent_offset:slot.tangent_offset + 3]
        phi = np.exp(-dt / TAU)
        np.testing.assert_allclose(
            block, (1.0 - phi * phi) * SIGMA ** 2 * np.eye(3), atol=1e-14)
        F = np.asarray(ekf_t.sys.F_fn(ekf.x, np.zeros(0), dt, 0.0))
        np.testing.assert_allclose(
            F[slot.tangent_offset:slot.tangent_offset + 3,
              slot.tangent_offset:slot.tangent_offset + 3],
            phi * np.eye(3), atol=1e-14)


@pytest.mark.parametrize("filter_cls", [EKF, UKF])
def test_filters_track_a_gauss_markov_drift(filter_cls):
    rng = np.random.default_rng(11)
    sim_w, sim_c = _world()
    est_w, _ = _world()
    sim = TargetNumpy(Sim(sim_w))
    transform = filter_cls(est_w)
    filt = TargetNumpy(transform)
    slot = transform.spec.slot("drone.g.gyro_drift")
    sim.state["drone"]["g.gyro_drift"] = np.array([0.08, -0.05, 0.03])
    P = np.eye(transform.spec.tangent_dim) * 1e-4
    P[slot.tangent_offset:slot.tangent_offset + 3,
      slot.tangent_offset:slot.tangent_offset + 3] = np.eye(3) * 0.1
    filt.reset(P=P)
    errors = []
    for k in range(400):
        sim.state["drone"].update(sim_c.sample_noise(rng))
        sim.step(DT)
        z = np.array(sim.outputs()["drone"]["g.gyro"]).ravel()
        filt.predict(DT)
        filt.update("g.gyro", z)
        if k >= 200:
            errors.append(np.array(filt.state_dict()["drone"]["g.gyro_drift"])
                          - np.array(sim.state["drone"]["g.gyro_drift"]))
    rms = np.sqrt(np.mean(np.square(errors)))
    assert rms < 0.02, rms


def test_jax_lowering_shares_the_kernel():
    pytest.importorskip("jax")
    from manta import TargetJax
    w, _ = _world()
    sim = Sim(w)
    mod, jm = sim.module(), TargetJax(sim)
    rng = np.random.default_rng(0)
    x = np.array(jm.initial_state(), dtype=float)
    spec = mod.spec
    slot = spec.slot("drone.g.gyro_drift")
    x[slot.ambient_offset:slot.ambient_offset + 3] = (0.3, -0.1, 0.2)
    noise = rng.standard_normal(mod.port("noise").size) * 0.5
    ref = mod.functions["step"](x, np.zeros(0), noise, 0.3, 0.0)
    out = jm.kernel("step")(x, np.zeros(0), noise, 0.3, 0.0)
    for a, b in zip(ref, out):
        np.testing.assert_allclose(np.array(a).ravel(), np.array(b).ravel(),
                                   atol=1e-12)
    phi = np.exp(-0.3 / TAU)
    drift_next = np.array(out[0]).ravel()[
        slot.ambient_offset:slot.ambient_offset + 3]
    driver = noise[[i for i, f in enumerate(_noise_fields(mod))
                    if f == "drone.g.gyro_drift_driver"]]
    np.testing.assert_allclose(
        drift_next, phi * np.array((0.3, -0.1, 0.2))
        + np.sqrt(1 - phi * phi) * driver, atol=1e-12)


def _noise_fields(mod):
    names = []
    for f in mod.port("noise").fields:
        names.extend([f.name] * f.dim)
    return names
