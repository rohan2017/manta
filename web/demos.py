"""Vehicle models for the homepage example carousel.

Each `build_*` returns a `World` with one craft, kept short enough to show
verbatim as the carousel's code blurb — so the Python on the left of the
carousel is exactly the model flying on the right (lowered to WebAssembly by
`TargetWasm`). Regenerate the bundles with `web/build_demos.py`.
"""

from __future__ import annotations

from manta import Craft, World
from manta.fields import GravityField
from manta.parts import IMU, DragSurface, Mass, PositionSensor, Thruster

# --- quadcopter -------------------------------------------------------------
# Four rotors on an X-frame: each lifts along +z and spins its prop, so the
# pair-wise counter-rotation lets differential throttle yaw the airframe.
ARM = 0.25                       # rotor offset from the centre (m)
MAX_THRUST = 6.0                 # per-rotor max thrust (N)
YAW_COEFF = 0.05                 # reaction torque per unit thrust
ROTORS = {                       # mount point + spin sign (CCW = +1)
    "front": ((+ARM, 0, 0), +1), "right": ((0, +ARM, 0), -1),
    "back":  ((-ARM, 0, 0), +1), "left":  ((0, -ARM, 0), -1),
}


def build_quad() -> World:
    quad = Craft("quad")
    quad.add(Mass("body", mass=1.0, moi=(0.01, 0.01, 0.02)))
    for name, (mount, spin) in ROTORS.items():
        quad.add(Thruster(name, force=(0, 0, MAX_THRUST),
                          torque=(0, 0, spin * YAW_COEFF * MAX_THRUST),
                          transform=mount))
    quad.add(DragSurface("drag", force=(-0.2, -0.2, -0.3)))
    quad.add(IMU("imu", gyro_noise_sigma=0.01))
    quad.add(PositionSensor("gps", position_noise_sigma=0.04))

    world = World().add_field(GravityField(g=(0, 0, -9.81)))
    world.add_craft(quad, position=(0, 0, 2.0))
    return world
