"""Build consistent local INS gravity and rotation from one planet scene."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import casadi as ca
import numpy as np

from ..fields import GravityField
from ..fields.base import Disturbance
from ..ir.frames import WorldFrame
from ..ir.types import Vec3
from ..planets.scene import Scene
from ..world import World
from .navigation_frame import NavigationFrame


class _LocalGravity(Disturbance):
    field_value_shape = GravityField.value_shape

    def __init__(self, function):
        super().__init__()
        self.function = function

    def contribute_at_sym(self, point, t):
        return self.field_value_shape.from_mx(
            self.function(point.mx, getattr(t, "mx", t))
        )


@dataclass(frozen=True)
class NavigationEnvironment:
    """A matched ``frame`` and ``gravity`` for a local estimator World.

    The estimator's World coordinates are scene coordinates. This is a
    separate World from an inertial, planet-scale simulation. Add ``gravity``
    to that World and pass ``frame`` to INS. Other environmental fields and
    sensor reference frames must also be authored in these local coordinates.
    """

    frame: NavigationFrame
    gravity: GravityField

    @classmethod
    def from_scene(cls, scene: Scene, *, frame_id: str, epoch: str):
        """Snapshot a planet's rotation and spatially varying effective gravity.

        Uses the planet's registered GravityField, including custom sources.
        Gravity is evaluated at the transformed position/time, then projected
        into the scene axes, with centrifugal acceleration included once.
        No Earth constants or constant-gravity fit are used. The clock origin
        is the planet's t=0; ``epoch`` labels that origin, not a time offset.
        Rebuild after changing the planet or anchor. Only stateless gravity
        sources are supported here; stateful environmental fields belong in
        an explicitly authored estimator World.
        """
        if not isinstance(scene, Scene):
            raise TypeError("from_scene requires a planet Scene")
        scene = copy.deepcopy(scene)
        planet = scene.planet
        axes = np.asarray(scene.R_planet_from_scene)
        frame = NavigationFrame(
            frame_id=frame_id,
            epoch=epoch,
            angular_velocity=axes.T @ planet.omega_vec_world(),
            origin_from_rotation_center=axes.T @ scene.anchor_planet,
            gravity_convention="effective",
        )
        source = World("navigation_gravity").add_planet(planet).snapshot()
        gravity = source.get_field(GravityField)
        for disturbance in gravity.disturbances:
            if (
                disturbance.state_declarations()
                or disturbance.input_declarations()
                or disturbance.noise_declarations()
            ):
                raise ValueError(
                    "scene navigation gravity must be stateless and deterministic"
                )
        p, t = ca.MX.sym("position", 3), ca.MX.sym("time")
        rotation = planet.R_world_from_planet_sym(t) @ ca.DM(axes)
        origin = ca.DM(frame.origin_from_rotation_center)
        point = ca.DM(planet.center) + rotation @ (origin + p)
        g = rotation.T @ gravity.value_at_sym(Vec3[WorldFrame].from_mx(point), t).mx
        omega = ca.DM(frame.angular_velocity)
        effective = g - ca.cross(omega, ca.cross(omega, origin + p))
        function = ca.Function("navigation_gravity", [p, t], [effective])
        return cls(frame, GravityField().add(_LocalGravity(function)))
