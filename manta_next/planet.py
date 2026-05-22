"""Planet — runtime parameters for a `PlanetFrame` body-fixed frame.

A `PlanetFrame` is the planet's body-fixed coordinate frame. It rotates
with constant angular velocity `omega` about a fixed axis in
`WorldFrame`. At t=0 the PlanetFrame axes coincide with WorldFrame; at
time t they are rotated by angle `omega·t` about the rotation axis
(right-hand rule).

The framework integrates in `WorldFrame` (inertial); `Planet` provides
the transform that lets the user author initial conditions or read
results in the planet's rotating frame. **Pseudo-forces (Coriolis,
centrifugal) emerge automatically** from this transform — the
integrator itself adds none.

Usage::

    earth = Planet(rotation_axis=(0, 0, 1), omega=7.27e-5)
    # Drone at PlanetFrame (R_earth, 0, 0), at rest in PlanetFrame:
    p_world, v_world = earth.planet_to_world(
        p_planet=(R_earth, 0, 0), v_planet=(0, 0, 0), t=0.0)
    # → p_world = (R_earth, 0, 0)        (axes coincide at t=0)
    # → v_world ≈ (0, ω·R_earth, 0)      (eastward tangential velocity)

    # After the sim has advanced t seconds, recover PlanetFrame state:
    p_planet, v_planet = earth.world_to_planet(p_world, v_world, t)
"""

from __future__ import annotations

import numpy as np


class Planet:
    """Body-fixed planet frame rotating about a fixed `WorldFrame` axis.

    Attributes:
        axis  — unit rotation axis in WorldFrame coords (3-vector).
        omega — angular rate, rad/s. Positive ⇒ right-hand-rule
                rotation about `axis`. Earth: ~7.272e-5 rad/s.
    """

    __slots__ = ("axis", "omega")

    def __init__(self,
                 rotation_axis: tuple[float, float, float] = (0.0, 0.0, 1.0),
                 omega: float = 0.0) -> None:
        axis = np.asarray(rotation_axis, dtype=float)
        n = float(np.linalg.norm(axis))
        if n == 0.0:
            raise ValueError(
                "Planet: rotation_axis must be nonzero.")
        self.axis = axis / n
        self.omega = float(omega)

    # ---- Rotations -----------------------------------------------------

    def R_world_from_planet(self, t: float) -> np.ndarray:
        """3×3 rotation matrix that takes a PlanetFrame vector at time t
        into WorldFrame coords. Rodrigues' formula with angle = ω·t."""
        theta = self.omega * t
        c = np.cos(theta)
        s = np.sin(theta)
        ux, uy, uz = self.axis
        K = np.array([[0.0, -uz,  uy],
                      [ uz, 0.0, -ux],
                      [-uy,  ux, 0.0]], dtype=float)
        return np.eye(3) + s * K + (1.0 - c) * (K @ K)

    def R_planet_from_world(self, t: float) -> np.ndarray:
        """Inverse of R_world_from_planet — same matrix transposed."""
        return self.R_world_from_planet(t).T

    def omega_vec_world(self) -> np.ndarray:
        """Angular-velocity 3-vector of the planet in WorldFrame coords.
        Constant in time (the planet spins about a fixed axis)."""
        return self.omega * self.axis

    # ---- State transforms ----------------------------------------------

    def planet_to_world(self,
                        p_planet: tuple[float, float, float],
                        v_planet: tuple[float, float, float],
                        t: float
                        ) -> tuple[np.ndarray, np.ndarray]:
        """Position and velocity of a point that, in PlanetFrame at time
        t, has coords (p_planet, v_planet). Returns (p_world, v_world).

        Velocity transform:
            v_world = R · v_planet + ω × p_world
        The `ω × p_world` term is the rotation contribution — a point
        at rest in PlanetFrame still has tangential velocity in
        WorldFrame because the planet is spinning under it.
        """
        R = self.R_world_from_planet(t)
        p_world = R @ np.asarray(p_planet, dtype=float)
        omega_w = self.omega_vec_world()
        v_world = R @ np.asarray(v_planet, dtype=float) + np.cross(omega_w, p_world)
        return p_world, v_world

    def world_to_planet(self,
                        p_world: tuple[float, float, float],
                        v_world: tuple[float, float, float],
                        t: float
                        ) -> tuple[np.ndarray, np.ndarray]:
        """Inverse of planet_to_world. Returns (p_planet, v_planet)."""
        R_inv = self.R_planet_from_world(t)
        p_world_arr = np.asarray(p_world, dtype=float)
        v_world_arr = np.asarray(v_world, dtype=float)
        omega_w = self.omega_vec_world()
        p_planet = R_inv @ p_world_arr
        v_planet = R_inv @ (v_world_arr - np.cross(omega_w, p_world_arr))
        return p_planet, v_planet

    def __repr__(self) -> str:
        return (f"<Planet axis=({self.axis[0]:.3f}, {self.axis[1]:.3f}, "
                f"{self.axis[2]:.3f}) omega={self.omega:.4g} rad/s>")
