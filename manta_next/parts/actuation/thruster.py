"""Thruster — scalar thrust command along a fixed body-frame axis.

The canonical "actuator" Part: a single commanded scalar drives a force
along a known direction in CraftFrame. Useful for rocket motors, fans,
ducted fans, EDFs — anything where the user commands magnitude and the
direction is fixed by mounting.
"""

from __future__ import annotations

from ...ir.frames import CraftFrame
from ...ir.types import Vec3
from ..base import Input, Parameter, Part
from ..wrench import Wrench


class Thruster(Part):
    """Single-axis thruster.

    Parameters:
        axis        : tuple   body-frame unit thrust direction (unit-norm
                              not enforced; magnitude scales with thrust).
                              Default (0, 0, 1) — upward in body frame.

    Inputs:
        thrust_cmd  : scalar  newtons of thrust. The output force is
                              `thrust_cmd · axis_unit_vec` in CraftFrame.

    The thruster contributes pure force. The torque a thruster generates
    about the craft origin is the standard force-at-offset term, handled
    by Part.transform → _wrench_to_craft in Craft.compile_tick().
    """

    axis:       tuple = Parameter((0.0, 0.0, 1.0))
    thrust_cmd: float = Input(default=0.0)

    def update(self, ctx):
        axis_v = Vec3[CraftFrame].constant(tuple(self.axis))
        f      = axis_v * self.thrust_cmd
        zero_t = Vec3[CraftFrame].constant((0.0, 0.0, 0.0))
        return Wrench(force=f, torque=zero_t)
