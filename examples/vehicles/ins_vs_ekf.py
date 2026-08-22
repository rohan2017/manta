"""A/B test a dynamics-driven EKF against strapdown INS under reduction.

One seeded truth simulation drives two estimators from the same sensor samples
and commanded actuator inputs:

* ``EKF`` propagates the deliberately imperfect vehicle dynamics and folds
  GPS, DVL, heading, and gyro measurements. Its acceleration comes from the
  vehicle model.
* ``INS`` propagates the IMU strapdown recurrence, folds the same GPS/DVL
  samples, and uses ``ModelForce`` with the raw accelerometer sample as a
  specific-force disturbance observation.

The truth vehicle is a higher-fidelity assembly with three distributed masses,
five buoyancy samples, and five offset drag panels. The estimator model is its
lumped reduction: one equivalent mass/inertia, one buoyancy point, one
translational drag tensor, and one rotational-damping tensor. Aggregate mass,
inertia, thrust, buoyancy, and drag remain within a few percent, so its nominal
dynamics are reasonably accurate without duplicating the simulation topology.

Truth is also subjected to a seeded, band-limited random force/torque process
representing wave buffet, plus white sub-resolution forcing (thruster ripple,
flow noise). Those disturbances exist only on the truth model, so neither
filter is handed the answer. A ``CraftWindBubble`` supplies a
drifting current state that both filters are allowed to estimate through the
model.

The INS's ``ModelForce`` does not guess its model error. A calibration log
(truth seed distinct from the A/B run) is split into training and held-out
windows, and ``held_out_evidence`` identifies the reduced model's
specific-force residual on the held-out set: per-axis bias, a white floor,
and the time-correlated (Gauss–Markov) component the wave buffet leaves
behind. ``ModelForce(evidence=...)`` consumes that typed artifact — the bias
as a deterministic correction, the correlated part as filter state — and
``INS`` refuses a part without accepted evidence.

This is an architectural comparison, not a universal claim that one transform
wins. Change the reduction fidelity, sensor rates, acceptance criteria, and
disturbance spectrum to match a real vehicle; ``rho`` is reported as a
diagnostic and the rho ceiling is the only generic threshold applied.

Run::

    .venv/bin/python -m examples.vehicles.ins_vs_ekf
    .venv/bin/python -m examples.vehicles.ins_vs_ekf --duration 60 --seed 12
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

from manta import EKF, INS, Craft, NoiseDriver, Sim, TargetNumpy, Window, World
from manta.fields import CraftWindBubble, FluidField, GravityField
from manta.fit import FitAcceptanceCriteria, FitEvidence, held_out_evidence, hold_out
from manta.parts import (
    IMU,
    DragSurface,
    HeadingSensor,
    Mass,
    ModelForce,
    PointBuoy,
    PositionSensor,
    ProcessNoise,
    RotationalDrag,
    Thruster,
    VelocitySensor,
)

RHO_WATER = 1025.0
G = 9.81
WAVE_INPUTS = ("wave_fx.throttle", "wave_fy.throttle",
               "wave_fz.throttle", "wave_tx.throttle",
               "wave_ty.throttle", "wave_tz.throttle")


@dataclass(frozen=True)
class Metrics:
    """Time-aggregated navigation error for one estimator."""

    position_rms_m: float
    position_p95_m: float
    velocity_rms_mps: float
    attitude_rms_deg: float
    current_rms_mps: float


@dataclass(frozen=True)
class Comparison:
    """Reproducible A/B result returned by :func:`run`."""

    ekf: Metrics
    ins: Metrics
    evidence: FitEvidence
    rho_by_sensor: dict[str, float]
    wave_force_rms_n: float
    wave_torque_rms_nm: float
    model_reduction: dict[str, str]


class WaveBuffet:
    """Seeded Ornstein-Uhlenbeck wrench plus a small regular swell.

    The state is generated outside Manta because it represents an unmodeled
    environment input. Only the truth simulator receives the resulting six
    force/torque commands.
    """

    def __init__(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)
        self.force = np.zeros(3)
        self.torque = np.zeros(3)

    def sample(self, t: float, dt: float) -> tuple[np.ndarray, np.ndarray]:
        tau = 1.4
        decay = np.exp(-dt / tau)
        innovation = np.sqrt(1.0 - decay * decay)
        self.force = (decay * self.force
                      + innovation * np.array((20.0, 15.0, 11.0))
                      * self.rng.standard_normal(3))
        self.torque = (decay * self.torque
                       + innovation * np.array((0.05, 0.07, 0.09))
                       * self.rng.standard_normal(3))
        swell_force = np.array((8.0 * np.sin(0.75 * t),
                                5.0 * np.sin(1.10 * t + 0.8),
                                4.0 * np.sin(0.55 * t + 1.7)))
        swell_torque = np.array((0.02 * np.sin(0.80 * t + 0.2),
                                 0.025 * np.sin(0.60 * t + 1.1),
                                 0.03 * np.sin(0.95 * t + 2.0)))
        return self.force + swell_force, self.torque + swell_torque


def build_world(*, truth: bool, evidence: FitEvidence | None = None) -> World:
    """Build detailed truth or its reasonably accurate lumped reduction.

    The reduced model mounts a ``ModelForce`` only once ``evidence`` (its
    identified specific-force error) exists; the evidence-less reduction is
    the calibration model the evidence is computed against.
    """
    auv = Craft("auv")
    if truth:
        # Detailed mass layout. The symmetric equipment offsets keep the
        # aggregate COM at the craft origin while contributing parallel-axis
        # inertia that the reduced model captures in one tensor.
        auv.add(Mass("hull", mass=40.0, moi=(6.5, 8.5, 10.0)))
        auv.add(Mass("battery", mass=8.0, moi=(0.25, 0.35, 0.40),
                     mount_offset=(0.45, 0.0, -0.05)))
        auv.add(Mass("payload", mass=8.0, moi=(0.25, 0.35, 0.40),
                     mount_offset=(-0.45, 0.0, 0.05)))

        # Distributed displacement and drag create full lever-arm moments in
        # the simulation model.
        for index, x in enumerate((-0.8, -0.4, 0.0, 0.4, 0.8)):
            auv.add(PointBuoy(
                f"buoy_{index}", volume=(56.0 / RHO_WATER) / 5.0,
                mount_offset=(x, 0.0, 0.12)))
        total_drag = np.array((-0.032, -0.050, -0.062))
        panels = {
            "drag_center": ((0.0, 0.0, 0.0), 0.20),
            "drag_fore": ((0.8, 0.0, 0.0), 0.20),
            "drag_aft": ((-0.8, 0.0, 0.0), 0.20),
            "drag_top": ((0.0, 0.0, 0.22), 0.20),
            "drag_bottom": ((0.0, 0.0, -0.22), 0.20),
        }
        for name, (offset, fraction) in panels.items():
            auv.add(DragSurface(
                name, force=tuple(total_drag * fraction),
                mount_offset=offset))
        # Sub-resolution forcing (thruster ripple, flow noise): a white
        # wrench the reduction does not carry. It is what gives the model
        # error a genuine white floor — the regime the INS's separated-noise
        # pseudo-measurement is valid in (`rho` well below its ceiling).
        auv.add(ProcessNoise("turbulence", force_noise_sigma=11.0,
                             torque_noise_sigma=0.03))
        thrust = 80.0
        yaw_torque = 12.0
    else:
        # Reduced estimator model: aggregate properties reproduce the
        # detailed model to within a few percent, but its topology is much
        # smaller and its rotational drag is a fitted lumped approximation.
        auv.add(Mass("body", mass=55.0, moi=(7.0, 12.3, 13.8)))
        auv.add(PointBuoy("buoy", volume=55.0 / RHO_WATER,
                          mount_offset=(0.0, 0.0, 0.12)))
        auv.add(DragSurface("drag", force=(-0.031, -0.048, -0.060)))
        auv.add(RotationalDrag(
            "angular_drag", torque=(-0.0011, -0.0150, -0.0125)))
        thrust = 82.5
        yaw_torque = 12.3

    auv.add(Thruster("surge", force=(thrust, 0.0, 0.0)))
    auv.add(Thruster("sway", force=(0.0, thrust * 0.65, 0.0)))
    auv.add(Thruster("heave", force=(0.0, 0.0, thrust * 0.70)))
    auv.add(Thruster("yaw", torque=(0.0, 0.0, yaw_torque)))

    imu = IMU(
        "imu", rate=50.0,
        accel_noise_sigma=0.015, gyro_noise_sigma=0.003,
        accel_bias_sigma=8e-4, gyro_bias_sigma=2e-4,
    )
    auv.add(imu)
    auv.add(PositionSensor("gps", rate=1.0, position_noise_sigma=0.35))
    auv.add(VelocitySensor("dvl", rate=5.0, velocity_noise_sigma=0.06))
    auv.add(HeadingSensor("heading", rate=2.0,
                          heading_vector_noise_sigma=0.02))

    if truth:
        # Unit wrench channels: their throttle values are directly N / N m.
        # They are absent from the estimator model and therefore genuinely
        # unknown to both filters.
        auv.add(Thruster("wave_fx", force=(1.0, 0.0, 0.0)))
        auv.add(Thruster("wave_fy", force=(0.0, 1.0, 0.0)))
        auv.add(Thruster("wave_fz", force=(0.0, 0.0, 1.0)))
        auv.add(Thruster("wave_tx", torque=(1.0, 0.0, 0.0)))
        auv.add(Thruster("wave_ty", torque=(0.0, 1.0, 0.0)))
        auv.add(Thruster("wave_tz", torque=(0.0, 0.0, 1.0)))
    elif evidence is not None:
        # Ordinary part declaration: INS does not contain special force-aid
        # equations. The selected accelerometer sample is supplied as z; the
        # error model (bias, white floor, Gauss–Markov state) is the
        # identified held-out evidence, never a hand-picked sigma.
        auv.add(ModelForce("model_force", imu=imu, evidence=evidence))

    fluid = FluidField().add_uniform(
        density=RHO_WATER, viscosity=1.35e-3)
    fluid.add(CraftWindBubble(
        auv, radius=500.0, sigma=0.035, name="auv_current"))
    world = (World("ins_ekf_ab")
             .add_field(GravityField(g=(0.0, 0.0, -G)))
             .add_field(fluid))
    world.add_craft(auv, position=(0.0, 0.0, -5.0))
    return world


def calibrate_model_force(*, seed: int, dt: float, windows: int = 24,
                          window_s: float = 2.0,
                          criteria: FitAcceptanceCriteria | None = None
                          ) -> FitEvidence:
    """Identify the reduced model's specific-force error on held-out data.

    A calibration log is recorded from the truth vehicle under the known
    maneuver commands and its own wave buffet (seeded apart from the A/B
    run). Windows are split training / held-out; the evidence is computed
    on the held-out tail only, against the reduced model's mean prediction
    from each window's true initial state over the recorded commands.
    """
    truth = TargetNumpy(Sim(build_world(truth=True)))
    truth.attach_driver(NoiseDriver(seed=seed + 2000))
    reduced = build_world(truth=False)
    template = TargetNumpy(Sim(reduced)).state
    wave = WaveBuffet(seed + 1)
    K = round(window_s / dt)
    log: list[Window] = []
    t = 0.0
    for _ in range(windows):
        x0 = {owner: {key: np.array(truth.state[owner][key], copy=True)
                      for key in slots if key in truth.state[owner]}
              for owner, slots in template.items()}
        commands: dict[str, list[float]] = {}
        accel: list[np.ndarray] = []
        for _ in range(K):
            command = _command(t)
            force, torque = wave.sample(t, dt)
            truth.step(dt, u={**command, **_wave_controls(force, torque)}, t=t)
            for name, value in command.items():
                commands.setdefault(name, []).append(value)
            accel.append(np.array(truth.reading("auv.imu.accel"), copy=True))
            t += dt
        log.append(Window(
            x0=x0, u={k: np.array(v) for k, v in commands.items()},
            z={"imu.accel": np.array(accel)}, dt=dt, t0=t - K * dt))
    _training, held_out = hold_out(log, fraction=0.5)
    return held_out_evidence(reduced, held_out, sensor="auv.imu.accel",
                             criteria=criteria)


def _command(t: float) -> dict[str, float]:
    """Known maneuver inputs given identically to truth and both filters."""
    return {
        "surge.throttle": 0.34 + 0.13 * np.sin(0.27 * t),
        "sway.throttle": 0.15 * np.sin(0.41 * t + 0.6),
        "heave.throttle": 0.08 * np.sin(0.33 * t + 1.4),
        "yaw.throttle": 0.05 * np.sin(0.22 * t),
    }


def _wave_controls(force: np.ndarray, torque: np.ndarray) -> dict[str, float]:
    values = np.concatenate((force, torque))
    return {name: float(value) for name, value in zip(WAVE_INPUTS, values)}


def _attitude_error_deg(q_true, q_est) -> float:
    a = np.asarray(q_true, dtype=float).reshape(4)
    b = np.asarray(q_est, dtype=float).reshape(4)
    cosine = abs(float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))))
    return float(np.degrees(2.0 * np.arccos(np.clip(cosine, -1.0, 1.0))))


def _estimate_errors(sim, runtime) -> tuple[float, float, float, float]:
    truth = sim.state["auv"]
    estimate = runtime.state_dict()["auv"]
    position = np.linalg.norm(
        np.asarray(estimate["position"]) - np.asarray(truth["position"]))
    velocity = np.linalg.norm(
        np.asarray(estimate["velocity"]) - np.asarray(truth["velocity"]))
    attitude = _attitude_error_deg(
        truth["orientation"], estimate["orientation"])
    current_true = np.asarray(sim.state["auv_current"]["wind"])
    current_est = np.asarray(runtime.state_dict()["auv_current"]["wind"])
    current = np.linalg.norm(current_est - current_true)
    return float(position), float(velocity), attitude, float(current)


def _metrics(samples: np.ndarray) -> Metrics:
    return Metrics(
        position_rms_m=float(np.sqrt(np.mean(samples[:, 0] ** 2))),
        position_p95_m=float(np.percentile(samples[:, 0], 95.0)),
        velocity_rms_mps=float(np.sqrt(np.mean(samples[:, 1] ** 2))),
        attitude_rms_deg=float(np.sqrt(np.mean(samples[:, 2] ** 2))),
        current_rms_mps=float(np.sqrt(np.mean(samples[:, 3] ** 2))),
    )


def run(*, duration: float = 40.0, dt: float = 0.02, seed: int = 7,
        warmup: float = 5.0, progress: bool = True) -> Comparison:
    """Run the common truth log through EKF and INS online.

    ``warmup`` is excluded from aggregate metrics so the result emphasizes
    propagation under model reduction rather than identical initial transients.
    """
    if duration <= 0.0 or dt <= 0.0:
        raise ValueError("duration and dt must be positive")
    if not 0.0 <= warmup < duration:
        raise ValueError("warmup must satisfy 0 <= warmup < duration")

    evidence = calibrate_model_force(seed=seed, dt=dt)
    if progress:
        print(evidence.summary())
    truth_world = build_world(truth=True)
    estimator_world = build_world(truth=False, evidence=evidence)
    sim = TargetNumpy(Sim(truth_world))
    sim.attach_driver(NoiseDriver(seed=seed + 1000))

    ekf_ir = EKF(estimator_world, sensors=[
        "auv.gps.position", "auv.dvl.velocity",
        "auv.heading.heading_vector", "auv.imu.gyro",
    ])
    ins_ir = INS(estimator_world, imu="auv.imu", sensors=[
        "auv.gps.position", "auv.dvl.velocity",
        "auv.heading.heading_vector",
        "auv.model_force.specific_force",
    ])
    ekf = TargetNumpy(ekf_ir)
    ins = TargetNumpy(ins_ir)
    ekf.reset(P=np.eye(ekf_ir.spec.tangent_dim) * 0.05)
    ins.reset(P=np.eye(ins_ir.spec.tangent_dim) * 0.05)

    wave = WaveBuffet(seed)
    ekf_errors: list[tuple[float, float, float, float]] = []
    ins_errors: list[tuple[float, float, float, float]] = []
    wave_forces: list[np.ndarray] = []
    wave_torques: list[np.ndarray] = []
    steps = round(duration / dt)
    gps_every = max(1, round(1.0 / (1.0 * dt)))
    dvl_every = max(1, round(1.0 / (5.0 * dt)))
    heading_every = max(1, round(1.0 / (2.0 * dt)))

    if progress:
        print("estimator reduction: 3 masses / 5 buoys / 5 drag panels -> "
              "1 equivalent mass / buoy / drag + rotational damping")
        print("aggregate mass, inertia, thrust, and drag differ by only a "
              "few percent; wave wrench remains hidden from both filters")
        print(f"INS rho: {dict(ins.rho_by_sensor)}")
        print(f"{'t':>6}  {'wave |F|':>9}  {'EKF pos':>9}  {'INS pos':>9}  "
              f"{'EKF att':>9}  {'INS att':>9}")

    for i in range(steps):
        t = i * dt
        command = _command(t)
        force, torque = wave.sample(t, dt)
        truth_u = {**command, **_wave_controls(force, torque)}
        sim.step(dt, u=truth_u, t=t)

        accel = sim.reading("auv.imu.accel")
        gyro = sim.reading("auv.imu.gyro")
        ins_u = {**command, "auv.imu.accel": accel, "auv.imu.gyro": gyro}

        # GPS/DVL/heading are common aiding. IMU placement is the A/B:
        # EKF folds gyro against its dynamics-propagated omega state; INS
        # uses gyro+accel for strapdown and accel again for ModelForce.
        ekf.update("auv.imu.gyro", gyro, u=command, t=t)
        ins.update("auv.model_force.specific_force", accel, u=ins_u, t=t)
        if i % dvl_every == 0:
            dvl = sim.reading("auv.dvl.velocity")
            ekf.update("auv.dvl.velocity", dvl, u=command, t=t)
            ins.update("auv.dvl.velocity", dvl, u=ins_u, t=t)
        if i % gps_every == 0:
            gps = sim.reading("auv.gps.position")
            ekf.update("auv.gps.position", gps, u=command, t=t)
            ins.update("auv.gps.position", gps, u=ins_u, t=t)
        if i % heading_every == 0:
            heading = sim.reading("auv.heading.heading_vector")
            ekf.update("auv.heading.heading_vector", heading, u=command, t=t)
            ins.update("auv.heading.heading_vector", heading, u=ins_u, t=t)
        ekf.predict(dt, u=command, t=t)
        ins.predict(dt, u=ins_u, t=t)

        ekf_error = _estimate_errors(sim, ekf)
        ins_error = _estimate_errors(sim, ins)
        if t >= warmup:
            ekf_errors.append(ekf_error)
            ins_errors.append(ins_error)
            wave_forces.append(force.copy())
            wave_torques.append(torque.copy())
        if progress and (i == 0 or (i + 1) % max(1, int(5.0 / dt)) == 0):
            print(f"{t + dt:6.1f}  {np.linalg.norm(force):9.2f}  "
                  f"{ekf_error[0]:9.3f}  {ins_error[0]:9.3f}  "
                  f"{ekf_error[2]:9.2f}  {ins_error[2]:9.2f}")

    ekf_samples = np.asarray(ekf_errors, dtype=float)
    ins_samples = np.asarray(ins_errors, dtype=float)
    force_samples = np.asarray(wave_forces, dtype=float)
    torque_samples = np.asarray(wave_torques, dtype=float)
    result = Comparison(
        ekf=_metrics(ekf_samples),
        ins=_metrics(ins_samples),
        evidence=evidence,
        rho_by_sensor=dict(ins.rho_by_sensor),
        wave_force_rms_n=float(np.sqrt(np.mean(np.sum(force_samples ** 2,
                                                       axis=1)))),
        wave_torque_rms_nm=float(np.sqrt(np.mean(np.sum(torque_samples ** 2,
                                                        axis=1)))),
        model_reduction={
            "mass": "three truth masses totaling 56 kg / lumped 55 kg",
            "inertia": "parallel-axis truth rollup / fitted lumped tensor",
            "surge thrust": "truth 80 N / estimator 82.5 N",
            "linear drag": "five offset panels / one near-equivalent tensor",
            "wave wrench": "truth-only colored force and torque inputs",
        },
    )
    if progress:
        _print_summary(result, duration=duration, warmup=warmup)
    return result


def _print_summary(result: Comparison, *, duration: float, warmup: float) -> None:
    print(f"\nRMS over t={warmup:g}..{duration:g} s "
          f"(wave RMS {result.wave_force_rms_n:.1f} N, "
          f"{result.wave_torque_rms_nm:.1f} N m)")
    print(f"{'metric':<24} {'EKF':>12} {'INS':>12} {'INS/EKF':>10}")
    rows = (
        ("position RMS [m]", result.ekf.position_rms_m,
         result.ins.position_rms_m),
        ("position p95 [m]", result.ekf.position_p95_m,
         result.ins.position_p95_m),
        ("velocity RMS [m/s]", result.ekf.velocity_rms_mps,
         result.ins.velocity_rms_mps),
        ("attitude RMS [deg]", result.ekf.attitude_rms_deg,
         result.ins.attitude_rms_deg),
        ("current RMS [m/s]", result.ekf.current_rms_mps,
         result.ins.current_rms_mps),
    )
    for name, ekf_value, ins_value in rows:
        ratio = ins_value / ekf_value if ekf_value > 0.0 else float("nan")
        print(f"{name:<24} {ekf_value:12.4f} {ins_value:12.4f} {ratio:10.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=40.0)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--warmup", type=float, default=5.0)
    args = parser.parse_args()
    run(duration=args.duration, dt=args.dt, seed=args.seed,
        warmup=args.warmup)


if __name__ == "__main__":
    main()
