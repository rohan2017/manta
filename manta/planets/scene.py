"""`Scene` — a local reference frame for placement, readout, and rendering.

A `Scene` is a frame **rigidly fixed in a planet's body frame**: a local
patch of ground (a launch site, a dive site, a stretch of sea) with a
human-friendly East/North/Up orientation. It exists purely as I/O
convenience — the simulator integrates everything in the inertial
`WorldFrame`; a Scene only translates the *inputs* (where to place a
craft) and *outputs* (poses / state to report or draw) between WorldFrame
and the local frame. No manta math or dynamics run in a Scene frame, and
it is never symbolic.

It answers three needs at once:

  * **Placement** — `scene.at_rest(position=...)` returns the full initial
    state for a craft sitting (co-rotating) in the scene at a small,
    local scene-frame coordinate. The planet can stay at the world origin
    (its true scale); you never hand-derive `ω × r` orbital velocities or
    body spin rates.
  * **Reporting** — `scene.relative(state)` re-expresses a craft's
    WorldFrame state (position, attitude, velocity, body rate) in the
    scene frame, so printed numbers and logged poses are local and small
    (velocity / angular rate come out *relative to the co-rotating
    scene* — i.e. relative to the ground).
  * **Rendering** — `scene.world_pose(t)` gives the scene's own world
    transform to publish as a parent entity (e.g. `viz.pose("world/scene",
    *scene.world_pose(t))`), so child poses logged via `relative()` stay
    near the origin and the view can simply track the scene.

The scene is fixed in PlanetFrame, so its orientation relative to the
planet (`R_planet_from_scene`) and its anchor point are *constants*; only
the planet's own `R_world_from_planet(t)` carries the time dependence.
"""

from __future__ import annotations

import numpy as np

from ..ir._rotation import (
    quat_conj_np, quat_from_rotmat_np, quat_mul_np, quat_to_rotmat_np,
)


class Scene:
    """A local East/North/Up frame fixed in a planet's body frame.

    Construct via `planet.scene_at(position, heading=...)` rather than
    directly. `position` is the anchor point in PlanetFrame (typically a
    point on the surface); the scene's axes are the local tangent frame
    there — +z up (radial), +x north (the planet spin axis projected into
    the tangent plane), +y = up×north — optionally yawed by `heading`
    (radians) about up.
    """

    def __init__(self,
                 planet,
                 position: tuple[float, float, float],
                 *,
                 heading: float = 0.0) -> None:
        self.planet = planet
        self.anchor_planet = np.asarray(position, dtype=float)
        if self.anchor_planet.shape != (3,):
            raise ValueError(
                f"Scene: position must be length-3, got {position!r}")
        self.heading = float(heading)
        # The scene is fixed in PlanetFrame. At t=0 PlanetFrame and
        # WorldFrame share orientation (R_world_from_planet(0) = I), so the
        # tangent basis computed at the world point `anchor + centre` *is*
        # R_planet_from_scene — a constant we cache once.
        p_world0 = tuple(self.anchor_planet + planet.center)
        self.R_planet_from_scene = planet._local_tangent_rotmat(
            p_world0, self.heading)

    # ---- World transforms (numeric, time-varying via the planet) --------

    def R_world_from_scene(self, t: float = 0.0) -> np.ndarray:
        """3×3 rotation from the scene frame to WorldFrame at time `t`."""
        return self.planet.R_world_from_planet(t) @ self.R_planet_from_scene

    def origin_world(self, t: float = 0.0) -> np.ndarray:
        """The scene origin's WorldFrame position at time `t`."""
        return (self.planet.R_world_from_planet(t) @ self.anchor_planet
                + self.planet.center)

    def world_pose(self, t: float = 0.0) -> tuple[tuple, tuple]:
        """`(origin_world, quat_world_from_scene)` at time `t` — the scene's
        own pose, to publish as a parent/anchor entity for rendering."""
        q = quat_from_rotmat_np(self.R_world_from_scene(t))
        return (tuple(float(v) for v in self.origin_world(t)),
                tuple(float(v) for v in q))

    # ---- Placement (input side) -----------------------------------------

    def at_rest(self,
                position: tuple[float, float, float] = (0.0, 0.0, 0.0),
                *,
                heading: float = 0.0) -> dict:
        """Initial state for a craft **at rest in the scene** (co-rotating
        with the planet) at scene-frame coordinate `position`, yawed
        `heading` (radians) about local up from the scene's north.

        Returns a kwargs dict (`position`, `velocity`, `orientation`,
        `angular_velocity`, all WorldFrame) to splat into
        `World.add_craft`::

            w.add_craft(sub,  **scene.at_rest((0, 0, -0.2)))       # 0.2 m down
            w.add_craft(buoy, **scene.at_rest(heading=np.radians(35)))

        The orbital velocity `ω × r` and body spin rate `R_craftᵀ·ω` are
        filled in so the craft is genuinely fixed to the planet (a gyro
        reads the planet's spin; it does not drift through the co-rotating
        sea/air).
        """
        R_ps = self.R_planet_from_scene              # = R_world_from_scene(0)
        p_world = self.origin_world(0.0) + R_ps @ np.asarray(position, float)
        r_world = p_world - self.planet.center
        omega_w = self.planet.omega_vec_world()
        v_world = np.cross(omega_w, r_world)
        c, s = np.cos(heading), np.sin(heading)
        Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        R_wc = R_ps @ Rz
        orientation = quat_from_rotmat_np(R_wc)
        omega_body = R_wc.T @ omega_w
        return {
            "position":         tuple(float(v) for v in p_world),
            "velocity":         tuple(float(v) for v in v_world),
            "orientation":      tuple(float(v) for v in orientation),
            "angular_velocity": tuple(float(v) for v in omega_body),
        }

    # ---- Reporting (output side) ----------------------------------------

    def relative(self, state: dict, t: float = 0.0) -> dict:
        """Re-express a craft's WorldFrame `state` in the scene frame.

        `state` is a per-craft state dict (as from `sim.state[name]` or
        `ekf.state_dict()[name]`): `position`/`velocity` (WorldFrame),
        `orientation` (world-from-craft quaternion), `angular_velocity`
        (body rates). Returns a dict of the same shape with:

          * `position`    — in scene coordinates,
          * `orientation` — scene-from-craft quaternion,
          * `velocity`    — velocity **relative to the co-rotating scene**
                            (i.e. relative to the ground), in scene coords,
          * `angular_velocity` — body rate **relative to the scene's spin**
                            (zero for a craft sitting still on the ground).

        Any other keys (part states like joint angles) pass through
        unchanged. `t` is the sim time the state was sampled at (needed for
        a spinning planet; default 0).
        """
        R_ws = self.R_world_from_scene(t)
        R_sw = R_ws.T
        o = self.origin_world(t)
        omega_w = self.planet.omega_vec_world()

        out = dict(state)
        p = np.asarray(state["position"], dtype=float).ravel()
        out["position"] = tuple(float(v) for v in (R_sw @ (p - o)))

        q_wc = None
        if "orientation" in state:
            q_wc = np.asarray(state["orientation"], dtype=float).ravel()
            q_ws = quat_from_rotmat_np(R_ws)
            q_sc = quat_mul_np(quat_conj_np(q_ws), q_wc)
            if q_sc[0] < 0.0:                 # canonicalise to w ≥ 0 (q ≡ −q)
                q_sc = -q_sc
            out["orientation"] = tuple(float(v) for v in q_sc)
        if "velocity" in state:
            v = np.asarray(state["velocity"], dtype=float).ravel()
            v_rel = v - np.cross(omega_w, p - self.planet.center)
            out["velocity"] = tuple(float(v_) for v_ in (R_sw @ v_rel))
        if "angular_velocity" in state and q_wc is not None:
            wb = np.asarray(state["angular_velocity"], dtype=float).ravel()
            R_wc = quat_to_rotmat_np(q_wc)
            out["angular_velocity"] = tuple(
                float(v_) for v_ in (wb - R_wc.T @ omega_w))
        return out

    def __repr__(self) -> str:
        return (f"<Scene planet={self.planet.name!r} "
                f"anchor={tuple(self.anchor_planet.tolist())} "
                f"heading={self.heading:.4g}>")
