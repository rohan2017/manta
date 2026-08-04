"""Structured system identification — one arm length, one thrust curve.

The lesson this demo exists to teach: a parameter fit that matches the
data is not the same as a parameter fit you can ship. Hand a fitter
twelve free mount coordinates and twelve free thrust coefficients and it
will happily land on an airframe whose four motors sit at four different
radii, each pushing a slightly different direction with a slightly
different gain. That model predicts THIS log better than the honest one
does — and describes no quadcopter that was ever built. Fly the next
airframe off the line with it and it is wrong.

So don't fit an airframe, fit a DESIGN. The truth quad is a symmetric
X-frame: four identical rotors on equal arms. Those words are
constraints, and `Free`/`Tied` state them as such —

    "arm": Free(...)                                   one number...
    "fl.transform":  Tied("arm", scale=[[c], [c], [0.0]])   ...four mounts
    "fl.force_quad": Tied("kf",  scale=[[0.0], [0.0], [1.0]])

— collapsing 40 decision variables to 7. Every motor is forced to the
same radius in the propeller plane, thrust is forced along body z, and
every window that excites ANY rotor now informs the one shared number.
`Prior(lower=, upper=)` adds the sanity rails on top: a negative arm or
a rotor outpulling its own motor rating is not a worse fit, it is a
different vehicle.

The demo fits the same log twice — structured, then with all 40
parameters free — and reads both back as vehicle geometry. Both match
the data. Only one is a quadcopter.

Two things worth stealing beyond the tying:

* **Excitation is designed, not sprinkled.** Each window drives one
  mixer axis with a doublet (roll, pitch, yaw, collective in turn). The
  first version of this demo used broadband jitter on every rotor
  instead; the yaw coefficient came back 4x too large, because a quad's
  yaw authority is weak and reaction torque drowned in the thrust
  ripple. A dedicated yaw doublet fixed it with no change to the fitter.
  Identifiability is a property of the maneuver, not just the solver.

* **Inertia is NOT fitted here**, it is trusted from CAD. The gyro
  observes angular acceleration `kf·arm·Δt²/I`, so inertia and arm
  length enter as a product: free them both and the pair is a flat
  direction the data cannot split (`weak_directions()` would say so).
  Fix the one you actually know.

Run::

    .venv/bin/python -m examples.vehicles.sysid_quad_tied
"""

from __future__ import annotations

import copy
import math

import numpy as np

from manta import (
    Craft, Fit, Free, NoiseDriver, Prior, Sim, TargetNumpy, Tied, Window,
    World,
)
from manta.fields import GravityField
from manta.parts import IMU, Mass, Thruster

DT = 0.004          # 250 Hz log
WINDOW = 50         # 0.2 s excitation bursts
N_WINDOWS = 8

ROTORS = ["fl", "fr", "br", "bl"]
# X-frame corners in FLU body coords, as unit diagonals: a motor at arm
# length L sits at (sx·L/√2, sy·L/√2, 0).
CORNER = {"fl": (+1, +1), "fr": (+1, -1), "br": (-1, -1), "bl": (-1, +1)}
# Prop rotation about +z. A CCW (+z) prop drags the body the other way,
# so its reaction torque on the airframe is −spin·kappa.
SPIN = {"fl": +1, "fr": -1, "br": +1, "bl": -1}
DIAG = 1.0 / math.sqrt(2.0)

# One doublet per mixer axis — the excitation that makes each parameter
# visible: roll/pitch show the arm length, yaw shows the drag
# coefficient, collective shows the thrust curve.
MIX = {"roll":  {"fl": +1, "fr": -1, "br": -1, "bl": +1},
       "pitch": {"fl": +1, "fr": +1, "br": -1, "bl": -1},
       "yaw":   {"fl": +1, "fr": -1, "br": +1, "bl": -1},
       "coll":  {"fl": +1, "fr": +1, "br": +1, "bl": +1}}
AMPLITUDE = {"roll": 0.12, "pitch": 0.12, "yaw": 0.18, "coll": 0.07}

# The vehicle that flew (the fitter never sees these) vs. the numbers we
# start from. Mass is measurable, so the guess is good (a scale, ±1%).
# The rest are not: the booms clamp anywhere along a slot and nobody
# measured the build (arm 3 cm short), and the thrust curve and yaw
# coefficient are datasheet values for the prop, 17% and 24% low.
TRUTH = dict(mass=1.35, arm=0.22, kf=11.5, kappa=0.021,
             imu_mount=(0.02, -0.015, 0.01))
GUESS = dict(mass=1.36, arm=0.19, kf=9.50, kappa=0.016,
             imu_mount=(0.0, 0.0, 0.0))

MOI = (0.021, 0.021, 0.038)                # from CAD — trusted, not fitted
GYRO_SIGMA, ACCEL_SIGMA = 0.002, 0.02      # per-tick IMU noise
FORCE_SIGMA = 0.05                         # per-rotor thrust ripple (N)


def build(mass, arm, kf, kappa, imu_mount) -> World:
    """A symmetric X-quad: four identical rotors at radius `arm`."""
    quad = Craft("quad")
    quad.add(Mass("body", mass=mass, moi=MOI))
    for nm in ROTORS:
        sx, sy = CORNER[nm]
        quad.add(Thruster(
            nm,
            force_quad=(0.0, 0.0, kf),
            torque_quad=(0.0, 0.0, -SPIN[nm] * kappa),
            transform=(sx * arm * DIAG, sy * arm * DIAG, 0.0),
            force_noise_sigma=FORCE_SIGMA))
    quad.add(IMU("imu", transform=imu_mount,
                 gyro_noise_sigma=GYRO_SIGMA, accel_noise_sigma=ACCEL_SIGMA))
    world = World()
    world.add_field(GravityField(g=(0.0, 0.0, -9.81)))
    world.add_craft(quad, position=(0.0, 0.0, 20.0))
    return world


def record_flight(world: World, *, seed=5) -> list[Window]:
    """Fly the excitation schedule and keep what a flight log keeps:
    per-rotor throttle commands and the IMU.

    Each window is one open-loop doublet on one mixer axis — the rotors
    step one way for half the window, the other way for the rest — plus
    a little jitter so the excitation isn't perfectly rank-deficient.
    Between bursts the pilot re-trims, modeled here by re-seeding near
    hover; that also keeps the windows independent and open-loop.
    Closed-loop data would be far weaker: a stabilized quad's
    differential commands are correlated with its own motion, and the
    fit cannot tell the airframe from the autopilot.

    The truth sim runs with a `NoiseDriver`, so the log carries real IMU
    noise AND real per-rotor thrust ripple. The ripple is process noise
    the deterministic-rollout loss has no term for — which is exactly
    the regime where structure earns its keep.
    """
    rng = np.random.default_rng(seed)
    truth = TargetNumpy(Sim(world))
    truth.attach_driver(NoiseDriver(seed=seed))
    hover = math.sqrt(TRUTH["mass"] * 9.81 / (4.0 * TRUTH["kf"]))
    schedule = list(MIX)
    doublet = np.where(np.arange(WINDOW) < WINDOW // 2, 1.0, -1.0)

    windows: list[Window] = []
    for w in range(N_WINDOWS):
        # Re-trim, with a little attitude/rate diversity so the windows
        # don't all sample the same corner of the dynamics.
        st = truth.state["quad"]
        st["position"] = np.array([0.0, 0.0, 20.0])
        st["velocity"] = 0.2 * rng.standard_normal(3)
        st["angular_velocity"] = 0.15 * rng.standard_normal(3)
        axis = rng.standard_normal(3)
        axis /= np.linalg.norm(axis)
        half = 0.5 * rng.uniform(-0.2, 0.2)
        st["orientation"] = np.concatenate(
            [[math.cos(half)], axis * math.sin(half)])

        x0 = copy.deepcopy(truth.state)
        mode = schedule[w % len(schedule)]
        throttle = {
            nm: np.clip(hover + AMPLITUDE[mode] * MIX[mode][nm] * doublet
                        + 0.05 * rng.standard_normal(WINDOW), 0.05, 1.0)
            for nm in ROTORS}
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


WEIGHTS = {"imu.gyro": 1.0 / GYRO_SIGMA**2,     # whiten: rad/s vs m/s²
           "imu.accel": 1.0 / ACCEL_SIGMA**2}


def fit_structured(model: World, windows: list[Window]):
    """The design is the model: 7 decision variables, four rotors."""
    params = {
        # The three numbers that ARE the design, with hard sanity rails.
        "arm":   Free(GUESS["arm"],
                      prior=Prior(sigma=0.04, lower=0.05, upper=0.60)),
        "kf":    Free(GUESS["kf"],
                      prior=Prior(sigma=3.0, lower=1.0, upper=30.0)),
        "kappa": Free(GUESS["kappa"],
                      prior=Prior(sigma=0.01, lower=0.0, upper=0.10)),
        # Weighed on a kitchen scale: ±3%, relative (log-space).
        "body.mass": Prior(sigma=0.03, log=True),
        # The one thing that is genuinely per-vehicle: where the flight
        # controller ended up on its foam pad. Untied, ±10 cm.
        "imu.transform": Prior(sigma=0.10),
    }
    for nm in ROTORS:
        sx, sy = CORNER[nm]
        # One arm length → four mounts. The zero row is a constraint
        # too: the props are in the body's xy plane, not above or below.
        params[f"{nm}.transform"] = Tied(
            "arm", scale=[[sx * DIAG], [sy * DIAG], [0.0]])
        # One thrust curve → four rotors, thrust along body z.
        params[f"{nm}.force_quad"] = Tied("kf", scale=[[0.0], [0.0], [1.0]])
        # One yaw coefficient → four reaction torques, signed by spin.
        params[f"{nm}.torque_quad"] = Tied(
            "kappa", scale=[[0.0], [0.0], [-SPIN[nm]]])
    return Fit(model, parameters=params).solve(windows, weights=WEIGHTS)


def fit_unstructured(model: World, windows: list[Window]):
    """Every coordinate free: 40 decision variables, no design. Same
    data, same priors, same solver — the only thing removed is the
    knowledge that this is a quadcopter."""
    params = {
        "body.mass": Prior(sigma=0.03, log=True),
        "imu.transform": Prior(sigma=0.10),
    }
    for nm in ROTORS:
        params[f"{nm}.transform"] = Prior(sigma=0.04)
        params[f"{nm}.force_quad"] = Prior(sigma=3.0)
        params[f"{nm}.torque_quad"] = Prior(sigma=0.01)
    return Fit(model, parameters=params).solve(windows, weights=WEIGHTS)


def airframe_report(values: dict) -> dict:
    """Read a fitted parameter set back as vehicle geometry: each
    rotor's radius in the prop plane, how far it sits out of that plane,
    its thrust gain, and how far its thrust axis leans off body z."""
    radii, gains, tilts, out_of_plane = [], [], [], []
    for nm in ROTORS:
        mount = np.atleast_1d(values[f"quad.{nm}.transform"])
        force = np.atleast_1d(values[f"quad.{nm}.force_quad"])
        radii.append(float(np.linalg.norm(mount[:2])))
        out_of_plane.append(abs(float(mount[2])))
        gains.append(float(np.linalg.norm(force)))
        tilts.append(math.degrees(
            math.atan2(float(np.linalg.norm(force[:2])), float(force[2]))))
    return {"radii": np.array(radii), "gains": np.array(gains),
            "tilts": np.array(tilts),
            "out_of_plane": np.array(out_of_plane)}


def print_airframe(label: str, rep: dict) -> None:
    fmt = lambda a: "[" + " ".join(f"{v:6.3f}" for v in a) + "]"
    print(f"  {label}")
    print(f"    rotor radius (m)   {fmt(rep['radii'])}   spread "
          f"{np.ptp(rep['radii']) * 100:6.2f} cm")
    print(f"    out of prop plane  {fmt(rep['out_of_plane'])}   max    "
          f"{rep['out_of_plane'].max() * 100:5.2f} cm")
    print(f"    thrust gain (N)    {fmt(rep['gains'])}   spread "
          f"{np.ptp(rep['gains']):6.3f} N")
    print(f"    thrust axis tilt   {fmt(rep['tilts'])}   max    "
          f"{rep['tilts'].max():5.2f} deg")


def predict_rmse(model: World, windows: list[Window]) -> tuple[float, float]:
    """Roll `model` (noise-free, the mean prediction) through each
    window's recorded commands from its recorded initial state, and
    score the gyro/accel RMSE against what the vehicle actually did.

    This is the question a fit is really answering: not "how well does
    this explain the log I fitted on" but "how well does it predict a
    vehicle it has never seen".
    """
    sim = TargetNumpy(Sim(model))
    sq_gyro, sq_accel, n = 0.0, 0.0, 0
    for w in windows:
        sim.state = copy.deepcopy(w.x0)
        for k in range(len(w.z["imu.gyro"])):
            for nm in ROTORS:
                sim.state["quad"][f"{nm}.throttle"] = \
                    float(w.u[f"{nm}.throttle"][k])
            sim.step(w.dt)
            out = sim.outputs()["quad"]
            sq_gyro += float(np.sum((out["imu.gyro"]
                                     - w.z["imu.gyro"][k]) ** 2))
            sq_accel += float(np.sum((out["imu.accel"]
                                      - w.z["imu.accel"][k]) ** 2))
            n += 3
    return math.sqrt(sq_gyro / n), math.sqrt(sq_accel / n)


def truth_values() -> dict:
    """The truth airframe in the same shape `airframe_report` reads."""
    out = {}
    for nm in ROTORS:
        sx, sy = CORNER[nm]
        out[f"quad.{nm}.transform"] = (sx * TRUTH["arm"] * DIAG,
                                       sy * TRUTH["arm"] * DIAG, 0.0)
        out[f"quad.{nm}.force_quad"] = (0.0, 0.0, TRUTH["kf"])
    return out


def main() -> None:
    print(f"recording {N_WINDOWS} excitation bursts "
          f"({WINDOW * DT:.2f} s each at {1 / DT:.0f} Hz): "
          f"{', '.join(list(MIX)[i % len(MIX)] for i in range(N_WINDOWS))}")
    windows = record_flight(build(**TRUTH))

    # ---- the fit that knows it is fitting a quadcopter -----------------
    model = build(**GUESS)
    res = fit_structured(model, windows)
    print("\n" + "=" * 74)
    print("STRUCTURED FIT — one arm, one thrust curve, one yaw coeff "
          "(7 unknowns)")
    print("=" * 74)
    print(res.summary())

    print(f"\n  {'quantity':16s} {'guess':>10s} {'fitted':>10s} "
          f"{'truth':>10s} {'error':>10s}")
    checks = [("arm (m)", "arm", GUESS["arm"], TRUTH["arm"], 0.005),
              ("kf (N)", "kf", GUESS["kf"], TRUTH["kf"], 0.30),
              ("kappa (N·m)", "kappa", GUESS["kappa"], TRUTH["kappa"], 0.003),
              ("mass (kg)", "quad.body.mass", GUESS["mass"], TRUTH["mass"],
               0.05)]
    failures = []
    for label, key, guess, truth, tol in checks:
        got = float(np.atleast_1d(res.values[key])[0])
        err = got - truth
        ok = abs(err) <= tol
        failures += [] if ok else [f"{label}: |{err:+.4g}| > {tol}"]
        print(f"  {label:16s} {guess:10.4f} {got:10.4f} {truth:10.4f} "
              f"{err:+10.4f}   {'ok' if ok else 'FAIL'}")

    # ---- the same data, fit with no structure at all -------------------
    loose_model = build(**GUESS)
    loose = fit_unstructured(loose_model, windows)
    print("\n" + "=" * 74)
    print("UNSTRUCTURED FIT — every mount and coefficient free "
          "(40 unknowns)")
    print("=" * 74)
    print(f"\n  data fit:  structured {res.objective:.6g}"
          f"   unstructured {loose.objective:.6g}"
          f"   ({100 * (1 - loose.objective / res.objective):.1f}% better)")
    print("  More freedom always fits the log better. That is precisely why "
          "fitting\n  the log is not the property you want.\n")

    print_airframe("truth vehicle", airframe_report(truth_values()))
    print_airframe("structured fit", airframe_report(res.values))
    print_airframe("unstructured fit", airframe_report(loose.values))

    s_rep = airframe_report(res.values)
    u_rep = airframe_report(loose.values)
    print(f"\n  The structured fit's four rotors are identical by "
          f"construction, and its radius\n  lands "
          f"{abs(s_rep['radii'].mean() - TRUTH['arm']) * 1000:.1f} mm from "
          f"truth. The unstructured fit spreads the same four rotors\n  over "
          f"{np.ptp(u_rep['radii']) * 100:.2f} cm of radius and "
          f"{np.ptp(u_rep['gains']):.2f} N of thrust gain — asymmetries the "
          f"vehicle does not\n  have, invented to absorb this particular "
          f"log's thrust ripple.")

    # ---- the test that decides which model you would ship --------------
    # A second airframe off the same line: same design, its own noise
    # realization and its own excitation. Neither fit has seen it.
    res.apply()          # writes onto `model`
    loose.apply()        # writes onto `loose_model`
    fleet = record_flight(build(**TRUTH), seed=41)

    print("\n" + "=" * 74)
    print("FLEET TEST — predicting a second airframe neither fit has seen")
    print("=" * 74)
    print(f"\n  {'model':22s} {'gyro RMSE':>12s} {'accel RMSE':>12s}")
    rows = [("starting guess", build(**GUESS)),
            ("structured fit", model),
            ("unstructured fit", loose_model)]
    scores = {}
    for label, m in rows:
        g, a = predict_rmse(m, fleet)
        scores[label] = (g, a)
        print(f"  {label:22s} {g:12.5f} {a:12.5f}")
    print(f"\n  Sensor noise floor:    {GYRO_SIGMA:12.5f} {ACCEL_SIGMA:12.5f}")

    sg, ug = scores["structured fit"][0], scores["unstructured fit"][0]
    verdict = ("the structured model predicts the new airframe better, "
               "despite fitting\n  the training log worse"
               if sg < ug else
               "the two models predict the new airframe about equally well")
    print(f"\n  On the training log the unstructured fit won by "
          f"{100 * (1 - loose.objective / res.objective):.1f}%. On a vehicle "
          f"neither\n  has seen, {verdict} — 40 free coordinates bought "
          f"a better\n  fit to one airframe's noise, not a better model of "
          f"the design.")
    print("\nstructured values applied — `Sim(model)` now bakes them in.")

    if failures:
        raise SystemExit("FAILED to recover: " + "; ".join(failures))
    print("PASS — arm length and thrust curve recovered within tolerance.")


if __name__ == "__main__":
    main()
