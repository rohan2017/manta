"""Planet — body-fixed rotating frame + field-disturbance source.

A `Planet` defines:

  1. A coordinate frame (`PlanetFrame`) whose origin is at the planet's
     center in `WorldFrame` and which rotates with constant angular
     velocity `omega` about a fixed axis.
  2. A set of field disturbances the planet contributes to the World's
     shared GravityField / FluidField / MagField when it's added via
     `World.add_planet(planet)`.

The framework integrates in `WorldFrame` (inertial). Pseudo-forces
(Coriolis, centrifugal) emerge automatically from the frame transform
and the co-rotating field disturbances — the integrator itself adds
none.

Subclass `Planet` to provide a concrete preset (e.g., `Earth`). The
base class itself adds no disturbances; subclasses override
`register_disturbances(world)` to install gravity / ocean / atmosphere
/ magnetic-dipole sources.

Multi-planet worlds are supported — register each via `add_planet`.
Each planet's disturbances are summed into the world's shared fields.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import casadi as ca
import numpy as np

if TYPE_CHECKING:
    from ..world import World
    from .scene import Scene
    from .state import PlanetState


class Planet:
    """Body-fixed rotating planet frame + field-disturbance source.

    Args:
        name           — identifier (used in repr + lookups).
        position       — planet center in WorldFrame (m). Default origin.
        rotation_axis  — unit rotation axis in WorldFrame. Default (0,0,1).
        omega          — angular rate, rad/s. Positive ⇒ right-hand-rule
                         rotation about `rotation_axis`. Earth sidereal
                         is ~7.272e-5 rad/s; default 0 (non-rotating).
    """

    def __init__(self,
                 name: str = "planet",
                 *,
                 position: tuple[float, float, float] = (0.0, 0.0, 0.0),
                 rotation_axis: tuple[float, float, float] = (0.0, 0.0, 1.0),
                 omega: float = 0.0) -> None:
        from ..ir.module import check_name
        self.name = check_name(str(name), who=type(self).__name__)
        pos = np.asarray(position, dtype=float)
        if pos.shape != (3,):
            raise ValueError(f"Planet: position must be length-3, got {position!r}")
        self.center = pos
        axis = np.asarray(rotation_axis, dtype=float)
        n = float(np.linalg.norm(axis))
        if n == 0.0:
            raise ValueError("Planet: rotation_axis must be nonzero.")
        self.axis = axis / n
        self.omega = float(omega)

    # ------------------------------------------------------------------
    # Numpy transforms (numeric, eager)
    # ------------------------------------------------------------------

    def R_world_from_planet(self, t: float) -> np.ndarray:
        """3×3 rotation matrix from PlanetFrame to WorldFrame at time t."""
        theta = self.omega * float(t)
        c = np.cos(theta)
        s = np.sin(theta)
        ux, uy, uz = self.axis
        K = np.array([[0.0, -uz,  uy],
                      [ uz, 0.0, -ux],
                      [-uy,  ux, 0.0]], dtype=float)
        return np.eye(3) + s * K + (1.0 - c) * (K @ K)

    def R_planet_from_world(self, t: float) -> np.ndarray:
        return self.R_world_from_planet(t).T

    def omega_vec_world(self) -> np.ndarray:
        """Constant angular-velocity 3-vector in WorldFrame coords."""
        return self.omega * self.axis

    def planet_to_world(self,
                        p_planet: tuple[float, float, float],
                        v_planet: tuple[float, float, float],
                        t: float
                        ) -> tuple[np.ndarray, np.ndarray]:
        """Position + velocity of a point that, in PlanetFrame at time
        `t`, has coords (p_planet, v_planet). Returns (p_world, v_world).

        Velocity transform:
            v_world = R · v_planet + ω × (p_world − planet.position)
        """
        R = self.R_world_from_planet(t)
        p_planet_arr = np.asarray(p_planet, dtype=float)
        p_world = R @ p_planet_arr + self.center
        omega_w = self.omega_vec_world()
        v_world = (R @ np.asarray(v_planet, dtype=float)
                   + np.cross(omega_w, p_world - self.center))
        return p_world, v_world

    def world_to_planet(self,
                        p_world: tuple[float, float, float],
                        v_world: tuple[float, float, float],
                        t: float
                        ) -> tuple[np.ndarray, np.ndarray]:
        R_inv = self.R_planet_from_world(t)
        p_world_arr = np.asarray(p_world, dtype=float)
        v_world_arr = np.asarray(v_world, dtype=float)
        omega_w = self.omega_vec_world()
        p_planet = R_inv @ (p_world_arr - self.center)
        v_planet = R_inv @ (v_world_arr
                            - np.cross(omega_w, p_world_arr - self.center))
        return p_planet, v_planet

    # ------------------------------------------------------------------
    # Symbolic transforms (used by Disturbance.contribute_at_sym(point, t))
    # ------------------------------------------------------------------

    def position_world_sym(self) -> ca.MX:
        """3×1 MX of the planet's center in WorldFrame (constant)."""
        return ca.DM(self.center.reshape(3, 1))

    def omega_world_sym(self) -> ca.MX:
        """3×1 MX of the angular-velocity vector in WorldFrame (constant)."""
        return ca.DM((self.omega * self.axis).reshape(3, 1))

    def R_world_from_planet_sym(self, t_sym) -> ca.MX:
        """3×3 MX rotation from PlanetFrame to WorldFrame at symbolic t.

        Rodrigues' formula with angle = omega·t. Branch-free.
        """
        theta = self.omega * t_sym
        c = ca.cos(theta)
        s = ca.sin(theta)
        ux, uy, uz = float(self.axis[0]), float(self.axis[1]), float(self.axis[2])
        K = ca.DM(np.array([[0.0, -uz,  uy],
                            [ uz, 0.0, -ux],
                            [-uy,  ux, 0.0]], dtype=float))
        I = ca.DM.eye(3)
        return I + s * K + (1.0 - c) * (K @ K)

    # ------------------------------------------------------------------
    # Initial-state factories — PlanetFrame position/velocity seeds for
    # World.add_craft(position= / velocity=).
    # ------------------------------------------------------------------

    def position(self,
                 x: float, y: float, z: float) -> "PlanetState":
        """Return a `PlanetState` wrapping a PlanetFrame position. Pass
        directly to `World.add_craft(..., position=...)` to seed the
        craft's initial WorldFrame position from PlanetFrame coords."""
        from .state import PlanetState
        return PlanetState(self, "position", (float(x), float(y), float(z)))

    def velocity(self,
                 vx: float, vy: float, vz: float) -> "PlanetState":
        """Return a `PlanetState` wrapping a PlanetFrame velocity."""
        from .state import PlanetState
        return PlanetState(self, "velocity",
                           (float(vx), float(vy), float(vz)))

    # ------------------------------------------------------------------
    # Rigid-attachment kinematics — full WorldFrame initial state for a
    # craft fixed to (co-rotating with) the planet. Cartesian throughout;
    # general for any rotating planet (no per-subclass code).
    # ------------------------------------------------------------------

    def local_tangent_basis(self,
                            position: tuple[float, float, float]
                            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Local East/North/Up unit vectors (WorldFrame) at a WorldFrame
        point — a purely Cartesian local-tangent frame, no lat/lon.

        `Up` is the local radial (from the planet centre out through the
        point). `North` is the planet's spin axis projected into the
        tangent plane and normalised — the same true-north direction a
        gyrocompass finds from the spin vector. `East = North × Up`.

        Where North is undefined — the planet isn't rotating, or the
        point sits on the spin axis — it falls back to a stable
        tangential reference (world +x, else +y), so the basis is always
        well-formed (only its azimuth is then arbitrary). Returns
        `(east, north, up)`.
        """
        r_world = np.asarray(position, dtype=float) - self.center
        n = float(np.linalg.norm(r_world))
        if n == 0.0:
            raise ValueError(
                f"{type(self).__name__}.local_tangent_basis: undefined at "
                f"the planet centre")
        up = r_world / n
        north = self.axis - float(np.dot(self.axis, up)) * up
        nn = float(np.linalg.norm(north))
        if nn < 1e-9:
            for ref in (np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])):
                north = ref - float(np.dot(ref, up)) * up
                nn = float(np.linalg.norm(north))
                if nn > 1e-9:
                    break
        north = north / nn
        east = np.cross(north, up)
        return east, north, up

    def _local_tangent_rotmat(self, position, heading) -> np.ndarray:
        """3×3 world-from-craft rotation for the local-tangent attitude:
        body +x = North, +z = Up, +y = Up×North, yawed by `heading` (rad)
        about Up (right-handed)."""
        _, north, up = self.local_tangent_basis(position)
        R_base = np.column_stack([north, np.cross(up, north), up])
        c, s = np.cos(heading), np.sin(heading)
        Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        return R_base @ Rz

    def local_tangent_orientation(self,
                                  position: tuple[float, float, float],
                                  heading: float = 0.0) -> tuple:
        """World-from-craft quaternion `(w, x, y, z)` placing the craft in
        the local-tangent frame at WorldFrame point `position`: body
        forward (+x) along North, up (+z) along the local radial, yawed by
        `heading` (radians, right-handed about Up — 0 faces North).

        Cartesian and general: 'North' is the spin-axis tangential
        projection (see `local_tangent_basis`)."""
        from ..ir._rotation import quat_from_rotmat_np
        R_wc = self._local_tangent_rotmat(position, float(heading))
        return tuple(float(v) for v in quat_from_rotmat_np(R_wc))

    def scene_at(self,
                 position: tuple[float, float, float],
                 *,
                 heading: float = 0.0) -> "Scene":
        """A local **`Scene`** anchored at PlanetFrame point `position` — a
        ground patch with a human-friendly East/North/Up frame, used to
        place craft and to translate poses/state for reporting + rendering.

        `position` is in the planet-fixed frame (origin at the planet
        centre), so a point on the surface is a planet-radius vector — with
        the planet left at the world origin you place a craft anywhere on
        it: north pole `(0, 0, R_EQ)`, equator `(R_EQ, 0, 0)`. The scene's
        axes are the local tangent frame there (+z up, +x north), optionally
        yawed by `heading` (radians) about up. See `Scene` for the full API
        (`at_rest`, `relative`, `world_pose`).
        """
        from .scene import Scene
        return Scene(self, position, heading=heading)

    # ------------------------------------------------------------------
    # Disturbance registration (subclass override hook)
    # ------------------------------------------------------------------

    def register_disturbances(self, world: "World") -> None:
        """Called by `Sim(world)` to attach this planet's standing
        contributions to the world's shared fields. Subclasses (Earth,
        Moon, ...) override to install gravity / ocean / atmosphere /
        magnetic-dipole disturbances. Base default: no-op.

        Subclasses should use `world.get_or_create_field(FieldClass)` to
        get the shared instance, then `.add(disturbance)`.
        """
        return None

    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (f"<{type(self).__name__} '{self.name}' "
                f"pos={tuple(self.center.tolist())} "
                f"axis={tuple(self.axis.tolist())} "
                f"omega={self.omega:.4g}>")
