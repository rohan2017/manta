"""`MPC(world)` — the receding-horizon transform, lowered and flown.

The controller thesis this transform graduates from the manta-mpc lab:
one MPC, no per-vehicle control code — `MPC(world)` consumes the model
like `Sim`/`EKF`/`LQR` do, emits a Module whose HELD state is the warm
plan, and `TargetNumpy` gives back a runtime that IS a receding-horizon
controller with no controller code of its own. Deeper cross-checks
(IPOPT oracle, basin behavior, plant mismatch, device benchmarks) live
in the lab; what belongs HERE is the transform contract: module shape,
view selection, the α=0 never-worse guarantee, terminal-cost modes,
and a closed loop against the world's own Sim.
"""

import math

import numpy as np
import pytest

from manta import MPC, Craft, Sim, TargetNumpy, World
from manta.codegen.numpy import NumpyMpc
from manta.fields import FluidField, GravityField
from manta.parts import (ControlSurface, DragSurface, Mass,
                         RotationalDrag, Thruster)

RHO = 1025.0


def _tug():
    """Fully-actuated test hull: 3 axis thrusters through the COM, no
    torque authority — attitude stays wherever it started, which is
    exactly the point-to-point cost family's free-attitude contract."""
    c = Craft("tug")
    c.add(Mass("hull", mass=8.0, moi=(0.2, 0.5, 0.5)))
    c.add(DragSurface.directional_quadratic(
        "drag", areas=(0.05, 0.12, 0.12), drag_coefficient=1.0))
    c.add(DragSurface("skin", force=(-3.0 / RHO, -6.0 / RHO,
                                     -6.0 / RHO)))
    c.add(RotationalDrag("spin", torque=(-1.0 / RHO, -2.0 / RHO,
                                         -2.0 / RHO)))
    for name, axis in (("fx", (20.0, 0.0, 0.0)), ("fy", (0.0, 15.0, 0.0)),
                       ("fz", (0.0, 0.0, 15.0))):
        c.add(Thruster(name, force=axis))
    w = (World(name="tug_world")
         .add_field(GravityField().add_uniform((0.0, 0.0, 0.0)))
         .add_field(FluidField().add_uniform(density=RHO)))
    w.add_craft(c)
    bounds = {f"{n}.throttle": (-1.0, 1.0) for n in ("fx", "fy", "fz")}
    return w, bounds


def _pack_plant_state(sim, spec, craft="tug"):
    flat = {f"{craft}.{k}": np.asarray(v, dtype=float).ravel()
            for k, v in sim.state[craft].items()}
    return spec.pack_any(flat)


def test_module_shape_selects_the_mpc_view():
    """The structural contract: HELD + returns CONTROL → NumpyMpc, the
    plan is a (nu × N) matrix state field, and the tick entry writes it
    while returning typed u and J."""
    w, bounds = _tug()
    mpc = MPC(w, u_bounds=bounds, horizon=15)
    m = mpc.module()
    assert m.hosting.value == "held"
    assert m.state.field("plan").shape == (3, 15)
    ep = m.entry("tick")
    assert ep.writes == ("plan",) and ep.returns == ("u", "J")

    rt = TargetNumpy(mpc)
    assert isinstance(rt, NumpyMpc)
    # input fields carry the craft-prefixed Part Input names
    assert [f.name for f in m.port("u").fields] == [
        "tug.fx.throttle", "tug.fy.throttle", "tug.fz.throttle"]


def test_tick_never_degrades_the_held_plan():
    """The α = 0 guarantee, tested as stated: one tick's J may not
    exceed the cost of flying the held plan UNCHANGED (α = 0 is always
    among the scored candidates). Deliberately bad plan — full
    sideways thrust — so there is real room in both directions."""
    w, bounds = _tug()
    N = 15
    mpc = MPC(w, u_bounds=bounds, horizon=N)   # default weights
    rt = TargetNumpy(mpc)
    goal = np.array([2.0, 0.0, 0.0])
    x0 = np.asarray(mpc.module().port("x").init, dtype=float)

    bad = np.zeros((3, N))
    bad[1, :] = 1.0
    rt.reset_plan(bad)
    rt.tick(x0, goal)
    j_tick = rt.J
    assert j_tick is not None

    # the α = 0 candidate's cost, computed independently: roll the bad
    # plan through the world's own step and price it with the same
    # default cost family (running pos 0.4 + effort 0.05, terminal
    # pos 120), substeps matching the transform's default
    sim_module = Sim(w).module()
    step = sim_module.functions["step"].expand()
    spec = sim_module.state.fields[0].spec
    po = spec.slot("tug.position").ambient_offset
    dt, sub = 0.25, 5
    x, J0 = x0.copy(), 0.0
    for i in range(N):
        p_err = x[po:po + 3] - goal
        J0 += dt * (0.05 * float(bad[:, i] @ bad[:, i])
                    + 0.4 * float(p_err @ p_err))
        for j in range(sub):
            x = np.asarray(step(x, bad[:, i], np.zeros(step.size_in(2)[0]),
                                dt / sub, j * dt / sub)).ravel()
    pT = x[po:po + 3] - goal
    J0 += 120.0 * float(pT @ pT)

    assert j_tick <= J0 + 1e-9, (
        f"tick returned {j_tick:.2f}, worse than the untouched plan "
        f"{J0:.2f} — the α=0 guarantee is broken")
    # and the fixed-work tick is deterministic: same state, same held
    # plan → the same J to the last bit
    rt.reset_plan(bad)
    rt.tick(x0, goal)
    assert rt.J == j_tick


def test_closed_loop_station_keeping_against_the_sim():
    """The graduation loop: TargetNumpy(MPC(w)) flying
    TargetNumpy(Sim(w)) — plan born from ZEROS by the tick itself (a
    fully-actuated hull needs no seed), goal reached and held."""
    w, bounds = _tug()
    mpc = MPC(w, u_bounds=bounds, horizon=15,
              w_pos_terminal=60.0, w_vel_terminal=40.0)
    rt = TargetNumpy(mpc)
    plant = TargetNumpy(Sim(_tug()[0]))
    spec = mpc.module().port("x").spec
    goal = np.array([2.0, 0.0, -1.0])

    miss = np.inf
    for _ in range(40):                    # 10 s at the 0.25 s period
        x = _pack_plant_state(plant, spec)
        u = rt.tick(x, goal)
        for _ in range(5):
            plant.step(0.05, u=u)
        p = np.asarray(plant.state["tug"]["position"], dtype=float)
        miss = min(miss, np.linalg.norm(p - goal))
    v = np.asarray(plant.state["tug"]["velocity"], dtype=float)
    assert miss < 0.3, f"closest approach {miss:.2f} m"
    assert np.linalg.norm(p - goal) < 0.3, "did not HOLD station"
    assert np.linalg.norm(v) < 0.2, f"still moving {np.linalg.norm(v):.2f}"


def test_lqr_terminal_mode_builds_and_flies():
    """terminal='lqr' — the Riccati tail as terminal cost, solved at
    construction from the same world. Shorter-horizon planning is its
    point; here we pin that it builds, flies, and actually differs
    from the weight terminal."""
    w, bounds = _tug()
    mpc = MPC(w, u_bounds=bounds, horizon=10, terminal="lqr")
    rt = TargetNumpy(mpc)
    x0 = np.asarray(mpc.module().port("x").init, dtype=float)
    rt.tick(x0, (1.5, 0.0, 0.0))
    j_lqr = rt.J

    mpc_w = MPC(w, u_bounds=bounds, horizon=10)
    rt_w = TargetNumpy(mpc_w)
    rt_w.tick(x0, (1.5, 0.0, 0.0))
    assert rt_w.J != pytest.approx(j_lqr)


def _fin():
    """Underactuated test hull: forward-only prop, one elevator whose
    lift is ½ρV²·A·CL — dead at rest. Depth is reachable ONLY through
    the pitch+drive coupling."""
    c = Craft("fin")
    c.add(Mass("hull", mass=10.0, moi=(0.2, 2.0, 2.0)))
    c.add(DragSurface.directional_quadratic(
        "drag", areas=(0.02, 0.15, 0.15), drag_coefficient=1.0))
    c.add(DragSurface("skin", force=(-2.0 / RHO, -5.0 / RHO,
                                     -5.0 / RHO)))
    c.add(RotationalDrag("spin", torque=(-1.0 / RHO, -3.0 / RHO,
                                         -3.0 / RHO)))
    c.add(Thruster("prop", force=(20.0, 0.0, 0.0)))
    tail = dict(area=0.01, chord=0.1, servo_gain=4.0,
                hinge_damping=0.4)
    c.add(ControlSurface("elevator", mount_offset=(-0.5, 0.0, 0.0),
                         **tail))
    half = math.pi / 4.0
    c.add(ControlSurface("rudder", mount_offset=(-0.5, 0.0, 0.0),
                         mount_orientation=(math.cos(half),
                                            math.sin(half), 0.0, 0.0),
                         **tail))
    w = (World(name="fin_world")
         .add_field(GravityField().add_uniform((0.0, 0.0, 0.0)))
         .add_field(FluidField().add_uniform(density=RHO)))
    w.add_craft(c)
    return w, {"prop.throttle": (0.0, 1.0),
               "elevator.deflection_cmd": (-0.4, 0.4),
               "rudder.deflection_cmd": (-0.4, 0.4)}


def test_underactuated_hull_refuses_the_lqr_terminal():
    """A forward-only prop with fins dead at rest has no stabilizable
    rest equilibrium: terminal='lqr' must refuse loudly at
    CONSTRUCTION, not hand back a meaningless P."""
    w, bounds = _fin()
    with pytest.raises(RuntimeError, match="not stabilizable"):
        MPC(w, u_bounds=bounds, horizon=10, terminal="lqr")


def test_plan_birth_discovers_pitch_to_dive_and_seeds_the_runtime():
    """The whole controller lifecycle, manta alone. `plan()` from
    ZEROS on the underactuated hull — no cruise hint, no attitude
    cost, fins dead at rest — must discover speed-up-then-pitch-to-
    dive (the finctrl acceptance, inherited from the lab), and its
    plan must seed `reset_plan` so the tick flies the dive closed-loop
    against the world's own Sim."""
    w, bounds = _fin()
    mpc = MPC(w, u_bounds=bounds, horizon=40, w_pos_terminal=120.0,
              w_vel_terminal=0.0)
    x0 = np.asarray(mpc.module().port("x").init, dtype=float)
    goal = np.array([8.0, 0.0, -2.5])

    p = mpc.plan(x0, goal)
    assert p.converged
    assert p.U.shape == (3, 40)           # reset_plan orientation

    rt = TargetNumpy(mpc)
    rt.reset_plan(p.U)
    plant = TargetNumpy(Sim(_fin()[0]))
    spec = mpc.module().port("x").spec
    miss = np.inf
    for _ in range(50):
        flat = {f"fin.{k}": np.asarray(v, dtype=float).ravel()
                for k, v in plant.state["fin"].items()}
        u = rt.tick(spec.pack_any(flat), goal)
        for _ in range(5):
            plant.step(0.05, u=u)
        pos = np.asarray(plant.state["fin"]["position"], dtype=float)
        miss = min(miss, np.linalg.norm(pos - goal))
    assert miss < 1.0, f"closest approach {miss:.2f} m"
    # it dived, and it did so by driving (authority is V²-bought)
    assert pos[2] < -1.5 or miss < 1.0


def test_goal_shorthand_equals_the_explicit_objective():
    """The legacy surface is the default objective, exactly: a tick
    driven by the `goal` shorthand must price and solve identically to
    one driven by `set_objective(position=goal)` — same kernel, same
    runtime data, bit-equal J."""
    w, bounds = _tug()
    mpc = MPC(w, u_bounds=bounds, horizon=15)
    x0 = np.asarray(mpc.module().port("x").init, dtype=float)
    goal = (2.0, 0.5, -0.5)

    a = TargetNumpy(mpc)
    ua = a.tick(x0, goal)
    b = TargetNumpy(mpc)
    b.set_objective(position=goal)
    ub = b.tick(x0)
    assert a.J == b.J
    for k in ua:
        assert ua[k] == ub[k]


def test_cruise_transect_heading_speed_depth():
    """The finctrl mission the stack exists for, with NO position
    waypoint at all: hold 2 m depth (w_p = (0,0,w) — x/y masked),
    make 1.1 m/s along a 40° course (velocity target), nose on the
    course (the heading term, Euler-free). The underactuated hull must
    discover throttle-for-authority and hold trim — and x/y must
    remain genuinely free (the vehicle sails away from the origin
    unpunished)."""
    w, bounds = _fin()
    mpc = MPC(w, u_bounds=bounds, horizon=40)
    d = np.array([np.cos(np.radians(40.0)), np.sin(np.radians(40.0)),
                  0.0])
    objective = dict(position=(0.0, 0.0, -2.0), w_position=(0.0, 0.0, 40.0),
                     velocity=1.1 * d, w_velocity=(6.0, 6.0, 6.0),
                     heading=d, w_heading=8.0,
                     P=np.zeros((mpc.nx, mpc.nx)))
    x0 = np.asarray(mpc.module().port("x").init, dtype=float)
    p = mpc.plan(x0, **objective)
    assert p.converged

    rt = TargetNumpy(mpc)
    rt.set_objective(position=(0.0, 0.0, -2.0),
                     w_position=(0.0, 0.0, 40.0),
                     velocity=1.1 * d, w_velocity=(6.0, 6.0, 6.0),
                     heading=d, w_heading=8.0,
                     P=np.zeros((mpc.nx, mpc.nx)))
    rt.reset_plan(p.U)
    plant = TargetNumpy(Sim(_fin()[0]))
    plant.state["fin"]["position"] = np.array([0.0, 0.0, -2.0])
    spec = mpc.module().port("x").spec
    for _ in range(60):
        st = plant.state["fin"]
        flat = {f"fin.{n}": np.asarray(v, dtype=float).ravel()
                for n, v in st.items()}
        u = rt.tick(spec.pack_any(flat))
        for _ in range(5):
            plant.step(0.05, u=u)

    st = plant.state["fin"]
    pos = np.asarray(st["position"], dtype=float).ravel()
    v = np.asarray(st["velocity"], dtype=float).ravel()
    q = np.asarray(st["orientation"], dtype=float).ravel()
    nose = np.array([1 - 2 * (q[2] ** 2 + q[3] ** 2),
                     2 * (q[1] * q[2] + q[0] * q[3]),
                     2 * (q[1] * q[3] - q[0] * q[2])])
    assert abs(pos[2] + 2.0) < 0.2, f"depth {pos[2]:.2f}"
    assert nose @ d > 0.95, f"nose off course: {nose @ d:.2f}"
    assert abs(np.linalg.norm(v) - 1.1) < 0.3, \
        f"speed {np.linalg.norm(v):.2f}"
    # x/y really are free: it sailed well away from the origin
    assert np.linalg.norm(pos[:2]) > 10.0


def test_plan_multi_start_and_second_order_run():
    """The plan-birth extras execute and do no harm on a hull without
    saddles (the deep saddle science is the lab's record): multi-start
    is at least as good as single, full DDP converges to the same
    neighborhood as Gauss-Newton."""
    w, bounds = _tug()
    mpc = MPC(w, u_bounds=bounds, horizon=15)
    x0 = np.asarray(mpc.module().port("x").init, dtype=float)
    goal = (2.0, 0.0, 0.0)
    single = mpc.plan(x0, goal)
    multi = mpc.plan(x0, goal, multi_start=True)
    fxx = mpc.plan(x0, goal, second_order=True)
    assert single.converged
    assert multi.cost <= single.cost + 1e-9
    assert fxx.cost <= 1.1 * single.cost


# ---------------------------------------------------------------------------
# MPC.terminal() — the on-demand terminal quadratic
# ---------------------------------------------------------------------------

def test_terminal_places_weights_on_the_position_and_velocity_blocks():
    """`terminal()` is the runtime-port shape of the constructor's
    `terminal="weights"` family. It shipped with no caller and no test,
    so nothing pinned that it lands the weights on the right slices of
    the (nx × nx) matrix — a silently mis-offset diagonal would just
    pull on the wrong states."""
    w, bounds = _tug()
    mpc = MPC(w, u_bounds=bounds, horizon=5)

    PT = mpc.terminal(w_position=7.0, w_velocity=3.0)
    assert PT.shape == (mpc.nx, mpc.nx)
    np.testing.assert_allclose(PT, np.diag(np.diag(PT)))     # diagonal
    d = np.diag(PT)
    po, pd, vo = mpc._po, mpc._pd, mpc._vo
    np.testing.assert_allclose(d[po:po + pd], 7.0)
    np.testing.assert_allclose(d[vo:vo + 3], 3.0)
    # Everything else — orientation, rates, any part state — untouched.
    rest = np.delete(d, list(range(po, po + pd)) + list(range(vo, vo + 3)))
    np.testing.assert_allclose(rest, 0.0)


def test_terminal_defaults_to_no_terminal_cost():
    w, bounds = _tug()
    mpc = MPC(w, u_bounds=bounds, horizon=5)
    np.testing.assert_allclose(mpc.terminal(), np.zeros((mpc.nx, mpc.nx)))


def test_terminal_broadcasts_per_axis_masks():
    """The documented reason it takes vectors: a guidance-law switch that
    MASKS an axis in the running cost must be able to mask the same axis
    in the terminal cost, or the masked axis gets an anchor instead of a
    hold."""
    w, bounds = _tug()
    mpc = MPC(w, u_bounds=bounds, horizon=5)
    PT = mpc.terminal(w_position=(0.0, 0.0, 25.0), w_velocity=(1.0, 2.0, 3.0))
    d = np.diag(PT)
    po, vo = mpc._po, mpc._vo
    np.testing.assert_allclose(d[po:po + 3], (0.0, 0.0, 25.0))
    np.testing.assert_allclose(d[vo:vo + 3], (1.0, 2.0, 3.0))


def test_terminal_feeds_the_PT_port_and_changes_the_cost():
    """End to end: the matrix it returns is accepted by
    `set_objective(P=...)` and actually prices the horizon tail."""
    w, bounds = _tug()
    mpc = MPC(w, u_bounds=bounds, horizon=10)
    x0 = np.asarray(mpc.module().port("x").init, dtype=float)

    loose = TargetNumpy(mpc)
    loose.set_objective(P=mpc.terminal(w_position=0.0, w_velocity=0.0))
    loose.tick(x0, (1.5, 0.0, 0.0))

    tight = TargetNumpy(mpc)
    tight.set_objective(P=mpc.terminal(w_position=200.0, w_velocity=50.0))
    tight.tick(x0, (1.5, 0.0, 0.0))

    # A heavy terminal pull prices the same opening state far higher and
    # commands a more aggressive first move toward the goal.
    assert tight.J > loose.J
