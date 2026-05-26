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
        self.name = str(name)
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
        self._world = None   # set by World.add_planet

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

    def axis_world_sym(self) -> ca.MX:
        """3×1 MX of the unit rotation axis in WorldFrame (constant)."""
        return ca.DM(self.axis.reshape(3, 1))

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
    # Initial-state factories (Phase D — populated when frame-carrying
    # init lands; stubbed here for forward use).
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

    def at_rest(self) -> "PlanetState":
        """Shorthand for `planet.velocity(0, 0, 0)` — sets the WorldFrame
        velocity such that the craft sits at rest in PlanetFrame
        (i.e., co-rotates with the planet)."""
        from .state import PlanetState
        return PlanetState(self, "velocity", (0.0, 0.0, 0.0))

    # ------------------------------------------------------------------
    # Disturbance registration (subclass override hook)
    # ------------------------------------------------------------------

    def register_disturbances(self, world: "World") -> None:
        """Called by `World.compile()` to attach this planet's standing
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
