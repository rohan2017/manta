"""System identification — fit a quadcopter model to flight data.

The first manta tuning demo. A "real" quad (the truth model — which the
fitter never sees) flies a noisy, aggressively excited trajectory while
we record exactly what a flight log would hold: per-motor throttle
commands and IMU readings (gyro + accel). The model we hand the fitter
starts from datasheet-grade guesses: thrust coefficients ~20% off, one
motor mounted 3 cm from where the CAD says, the IMU lever arm unknown,
and the mass off by 200 g.

`Fit` promotes those Parameters to a live parameter vector
(`Sim(world, parameters=[...])`), rolls the oracle kernel over short
windows of the log (CasADi `mapaccum`, exact gradients), and solves the
MAP problem

    Σ windows ‖predicted − measured‖²_weighted  +  ‖(θ − θ₀)/σ₀‖²

with IPOPT. Priors make the jointly-unobservable directions well-posed
(thrust and mass trade off; the prior pins what the scale said), and the
Gauss-Newton posterior says exactly which numbers the flight actually
nailed down — read the `post/prior` column: ≪ 1 means the data spoke,
≈ 1 means the prior did.

Run::

    .venv/bin/python -m examples.vehicles.sysid_drone
"""

from __future__ import annotations

import copy

import numpy as np

from manta import (
    Craft, Fit, NoiseDriver, NoiseFit, Prior, Sim, TargetNumpy, Window,
    hold_out,
    World,
)
from manta.fields import GravityField
from manta.parts import IMU, Mass, Thruster

DT = 0.004          # 250 Hz log
WINDOW = 100        # 0.4 s windows
N_WINDOWS = 8

# What the truth quad actually is vs. what the model guesses.
TRUTH = dict(mass=1.32, kf=11.0, t1_mount=(0.12, 0.15, 0.0),
             imu_mount=(0.05, -0.02, 0.0))
GUESS = dict(mass=1.52, kf=9.0, t1_mount=(0.15, 0.15, 0.0),
             imu_mount=(0.0, 0.0, 0.0))

GYRO_SIGMA, ACCEL_SIGMA = 0.002, 0.02      # per-tick sensor noise


def build(mass, kf, t1_mount, imu_mount) -> World:
    """An X-quad: four quadratic thrusters on 15 cm arms + IMU."""
    arm = 0.15
    mounts = {"t1": t1_mount, "t2": (-arm, arm, 0.0),
              "t3": (-arm, -arm, 0.0), "t4": (arm, -arm, 0.0)}
    spin = {"t1": 1, "t2": -1, "t3": 1, "t4": -1}
    quad = Craft("quad")
    quad.add(Mass("body", mass=mass, moi=(0.021, 0.022, 0.038)))
    for nm, mount in mounts.items():
        quad.add(Thruster(nm, force_quad=(0.0, 0.0, kf),
                          torque_quad=(0.0, 0.0, 0.02 * spin[nm]),
                          mount_offset=mount))
    quad.add(IMU("imu", mount_offset=imu_mount,
                 gyro_noise_sigma=GYRO_SIGMA, accel_noise_sigma=ACCEL_SIGMA))
    world = World()
    world.add_field(GravityField(g=(0.0, 0.0, -9.81)))
    world.add_craft(quad, position=(0.0, 0.0, 20.0))
    return world


def record_flight(world: World, *, seed=11) -> list[Window]:
    """Fly the truth and keep what a flight log keeps: throttles + IMU.

    Identifiability needs excitation: hover data pins almost nothing,
    so each motor gets its own jittered throttle (differential thrust →
    torque arms show up in the gyro; thrust steps → kf/m in the accel).
    Window initial states are taken from truth — with a real log you'd
    seed them from your estimator instead.
    """
    rng = np.random.default_rng(seed)
    truth = TargetNumpy(Sim(world))
    truth.attach_driver(NoiseDriver(seed=seed))
    windows: list[Window] = []
    for _ in range(N_WINDOWS):
        x0 = copy.deepcopy(truth.state)
        base = np.clip(0.45 + 0.10 * rng.standard_normal(), 0.2, 0.8)
        throttle = {f"t{i}": np.clip(
            base + 0.10 * rng.standard_normal(WINDOW), 0.05, 1.0)
            for i in range(1, 5)}
        gyro, accel = [], []
        for k in range(WINDOW):
            for nm, tr in throttle.items():
                truth.state["quad"][f"{nm}.throttle"] = tr[k]
            truth.step(DT)
            out = truth.outputs()["quad"]
            gyro.append(out["imu.gyro"].copy())
            accel.append(out["imu.accel"].copy())
        windows.append(Window(
            x0=x0,
            u={f"{nm}.throttle": tr for nm, tr in throttle.items()},
            z={"imu.gyro": np.array(gyro), "imu.accel": np.array(accel)},
            dt=DT))
    return windows


def main() -> None:
    # The acceptance set is split off before anything is fitted and never
    # enters either fit: the derived artifact's evidence (held-out residual
    # bias, white/Gauss–Markov model error, acceptance decision) is computed
    # on it alone.
    windows, held_out = hold_out(record_flight(build(**TRUTH)), fraction=0.3)
    model = build(**GUESS)

    fit = Fit(model, parameters={
        # Weighed on a kitchen scale: ±5%, relative (log-space).
        "body.mass": Prior(sigma=0.05, log=True),
        # Datasheet thrust coefficients: loose.
        "t1.force_quad": Prior(sigma=4.0),
        "t2.force_quad": Prior(sigma=4.0),
        "t3.force_quad": Prior(sigma=4.0),
        "t4.force_quad": Prior(sigma=4.0),
        # CAD said 15 cm; trust it to ±5 cm.
        "t1.mount_offset": Prior(sigma=0.05),
        # IMU somewhere near the center: ±10 cm.
        "imu.mount_offset": Prior(sigma=0.10),
    })
    result = fit.solve(
        windows,
        # Whiten: weight each sensor by 1/σ² so rad/s and m/s² mix sanely.
        weights={"imu.gyro": 1.0 / GYRO_SIGMA**2,
                 "imu.accel": 1.0 / ACCEL_SIGMA**2},
    )

    print(result.summary())
    print()
    truth_vals = {
        "quad.body.mass": TRUTH["mass"],
        **{f"quad.t{i}.force_quad": (0.0, 0.0, TRUTH["kf"])
           for i in range(1, 5)},
        "quad.t1.mount_offset": TRUTH["t1_mount"],
        "quad.imu.mount_offset": TRUTH["imu_mount"],
    }
    print(f"{'parameter':24s}  {'started':>22s}  {'fitted':>22s}  "
          f"{'truth':>22s}")
    for name, tv in truth_vals.items():
        fv = result.values[name]
        fmt = lambda v: (f"{v:.4g}" if np.isscalar(v) else
                         "(" + ", ".join(f"{x:.4g}" for x in np.atleast_1d(v))
                         + ")")
        started = {"quad.body.mass": GUESS["mass"],
                   "quad.t1.mount_offset": GUESS["t1_mount"],
                   "quad.imu.mount_offset": GUESS["imu_mount"]}.get(
            name, (0.0, 0.0, GUESS["kf"]))
        print(f"{name:24s}  {fmt(started):>22s}  {fmt(fv):>22s}  "
              f"{fmt(tv):>22s}")

    # The weakest-identified direction of the data alone — with mass AND
    # thrust both free, expect the thrust/mass scale combo near the
    # bottom (the prior is what pinned it).
    ev, comp = result.weak_directions(1)[0]
    top = sorted(comp.items(), key=lambda kv: -abs(kv[1]))[:4]
    print(f"\nleast-identified data direction (eig {ev:.3g}): "
          + ", ".join(f"{k} ({v:+.2f})" for k, v in top))

    # Bake the fitted values into the model: from here, Sim(model) /
    # EKF(model) / a C++ deploy all use the identified physics.
    result.apply()
    print("\nfitted values applied — `Sim(model)` now bakes them in.")

    # ---- stage 2: noise σ, by innovation likelihood --------------------
    # An L2 loss can't see σ (the mean prediction doesn't depend on it).
    # With the dynamics now right, `NoiseFit` runs a Kalman filter over
    # the same windows and finds the σ whose predicted innovation
    # covariance matches the data's actual scatter. Start the model's σ
    # guesses 5× off in both directions.
    imu = next(p for p in model.crafts[0].parts if p.name == "imu")
    imu.gyro_noise_sigma = 5 * GYRO_SIGMA
    imu.accel_noise_sigma = ACCEL_SIGMA / 5
    nresult = NoiseFit(model, noise={
        "imu.gyro_noise": Prior(sigma=2.0),    # ±e² relative — loose
        "imu.accel_noise": Prior(sigma=2.0),
    }).solve(windows)
    print("\n" + nresult.summary())
    print(f"true: gyro σ={GYRO_SIGMA}  accel σ={ACCEL_SIGMA}  "
          f"(started {5 * GYRO_SIGMA} / {ACCEL_SIGMA / 5})")
    nresult.apply()
    print("noise σ applied — `EKF(model)` now auto-builds Q/R from the "
          "identified noise model.")

    # ---- stage 3: held-out evidence and the derived artifact ----------
    # `derive()` freezes the fitted model as an immutable ModelArtifact.
    # Its provenance carries typed evidence computed on the untouched
    # held-out windows: per-axis residual bias, the fitted process-noise
    # model (white, or Gauss–Markov tau/sigma when the residual is time
    # correlated — the fallback is recorded, never silent), and the
    # acceptance decision with the thresholds that produced it. A
    # `ModelForce(evidence=...)` consumes exactly this artifact.
    evidence = nresult.evidence(held_out, sensor="imu.accel")
    print("\n" + evidence.summary())
    artifact = nresult.derive(evidence=evidence)
    print(f"derived {artifact!r}: accepted="
          f"{artifact.derivation['noise_fit'].accepted}")


if __name__ == "__main__":
    main()
