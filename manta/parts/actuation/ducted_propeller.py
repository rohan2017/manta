"""Advance-aware, force-commanded ducted propeller.

This is an intentionally low-order open-water model for propellers whose
available data is a static bollard-pull point rather than a full ``K_T(J)`` /
``K_Q(J)`` chart. The input remains requested *static thrust* in newtons, which
lets a vehicle controller and its actuator bus keep a physically meaningful
contract. Until an RPM-dependent coefficient chart exists, axial inflow uses
the simplest signed quadratic curve that honors the two known anchors:

``T = T_command - T_static_max * (V_axial / V_zero_full) *
                                  |V_axial / V_zero_full|``

The axial term is propeller windmilling drag: it opposes through-disk flow even
at zero command. Most importantly for estimation and optimal control,
``dT/dT_command`` is positive everywhere; a moving craft never acquires a
flat, locally uncontrollable command interval. Oblique inflow scales the
command-produced component, while shaft reaction torque unloads smoothly:

``Q = Q_static(command) /
     (1 + max(0, V_axial_along_thrust / V_zero_full)^exponent)``

``V_zero_full`` defaults to the ideal static far-wake speed
``sqrt(2 T_static_max / (rho_ref A_disk))``. That is a provisional scale, not
a substitute for RPM and open-water data. Supplying a measured zero-thrust
advance speed replaces it directly. Signed-speed and positive-advance norms
are regularized so ordinary MPC/EKF Jacobians remain finite.

Oblique inflow is represented by a cosine-like loss based on crossflow over
the same shaft-speed scale. Side force and disk moments are deliberately not
invented yet; this part models only axial thrust and shaft reaction torque.
"""

from __future__ import annotations

import math
from typing import ClassVar

import casadi as ca

from ...fields import FluidField
from ...ir.frames import PartFrame, WorldFrame
from ...ir.types import Scalar, Vec3
from ...ir.wrench import Wrench
from ...smoothing import smooth_max0
from .._declarations import Input, Output, Parameter, PartUpdate
from .._trace import scalar_mx
from ..base import Part, PartRole

_COMMAND_EPS_N = 1e-6
_FLOW_NORM_EPS_SQ = 1e-12


class DuctedPropeller(Part):
    """A static-calibrated propeller with quadratic axial-flow loss.

    Parameters:
        max_static_thrust — magnitude of the reference bollard thrust, N.
        max_static_torque — magnitude of shaft reaction torque at that point,
                            N·m.
        diameter — propeller diameter, m.
        reaction_sign — +1/-1 rotation handedness.
        reference_density — density at which the static point was measured.
        zero_thrust_advance_speed — full-command unloading speed, m/s. Zero
                            derives the ideal far-wake provisional value.
        torque_unload_exponent — torque load factor exponent. Values greater
                            than one make torque unload faster than thrust;
                            the provisional default is 1.5.
        oblique_inflow_scale — crossflow loss strength. One uses the same
                            velocity scale as axial advance; zero disables it.

    Input:
        thrust_command — signed thrust that the static calibration would
                         produce, N.

    Outputs:
        thrust — achieved axial force, N.
        reaction_torque — achieved axial shaft reaction, N·m.
        advance_fraction — signed axial advance divided by the provisional
                           full-command zero-load speed.
        oblique_factor — multiplicative crossflow efficiency in [0, 1].
    """

    role = PartRole.ACTUATOR

    requires_fields: ClassVar = [FluidField]

    max_static_thrust: float = Parameter(1.0)
    max_static_torque: float = Parameter(0.0)
    diameter: float = Parameter(0.1)
    reaction_sign: float = Parameter(1.0)
    reference_density: float = Parameter(1025.0)
    zero_thrust_advance_speed: float = Parameter(0.0)
    torque_unload_exponent: float = Parameter(1.5)
    oblique_inflow_scale: float = Parameter(1.0)

    thrust_command: float = Input(default=0.0)

    thrust = Output()
    reaction_torque = Output()
    advance_fraction = Output()
    oblique_factor = Output()

    def __init__(self, name: str, **overrides) -> None:
        super().__init__(name, **overrides)
        who = f"DuctedPropeller {name!r}"
        for parameter in ("max_static_thrust", "diameter",
                          "reference_density"):
            if float(getattr(self, parameter)) <= 0.0:
                raise ValueError(
                    f"{who}: {parameter} must be > 0, got "
                    f"{getattr(self, parameter)!r}")
        if float(self.max_static_torque) < 0.0:
            raise ValueError(f"{who}: max_static_torque must be >= 0")
        if float(self.reaction_sign) not in (-1.0, 1.0):
            raise ValueError(f"{who}: reaction_sign must be -1 or +1")
        if float(self.zero_thrust_advance_speed) < 0.0:
            raise ValueError(
                f"{who}: zero_thrust_advance_speed must be >= 0")
        if float(self.torque_unload_exponent) <= 0.0:
            raise ValueError(
                f"{who}: torque_unload_exponent must be > 0")
        if float(self.oblique_inflow_scale) < 0.0:
            raise ValueError(f"{who}: oblique_inflow_scale must be >= 0")
        if float(self.zero_thrust_advance_speed) == 0.0:
            disk_area = math.pi * (float(self.diameter) / 2.0) ** 2
            self.zero_thrust_advance_speed = math.sqrt(
                2.0 * float(self.max_static_thrust)
                / (float(self.reference_density) * disk_area))

    def update(self, ctx) -> PartUpdate:
        command = scalar_mx(self.thrust_command)
        command_abs = ca.sqrt(command * command + _COMMAND_EPS_N ** 2)
        direction = command / command_abs

        p_world = ctx.position[WorldFrame]
        fluid = ctx.field(FluidField).value_at_sym(p_world, ctx.t)
        v_rel_world = ctx.velocity[WorldFrame] - fluid.velocity
        v_part = ctx.orientation.conjugate().apply(v_rel_world)._mx
        axial = v_part[0]
        crossflow = ca.sqrt(v_part[1] ** 2 + v_part[2] ** 2
                            + _FLOW_NORM_EPS_SQ)

        velocity_scale = float(self.zero_thrust_advance_speed)
        advance = axial / velocity_scale
        helpful_advance = smooth_max0(
            direction * advance, _FLOW_NORM_EPS_SQ)

        oblique_ratio = (float(self.oblique_inflow_scale) * crossflow
                         / velocity_scale)
        oblique = 1.0 / ca.sqrt(1.0 + oblique_ratio * oblique_ratio)
        density_scale = fluid.density / float(self.reference_density)

        # Smooth x|x| at zero only to keep the symbolic derivative defined.
        # Unlike the previous clipped load curve, command remains an affine
        # term: its local control derivative never vanishes at positive
        # advance.  The signed quadratic also represents windmilling drag.
        advance_abs = ca.sqrt(advance * advance + _FLOW_NORM_EPS_SQ)
        axial_loss = float(self.max_static_thrust) * advance * advance_abs
        achieved_thrust = (
            command * oblique - axial_loss) * density_scale
        static_torque = (
            command * float(self.max_static_torque)
            / float(self.max_static_thrust) * float(self.reaction_sign))
        torque_load = 1.0 / (
            1.0 + helpful_advance ** float(self.torque_unload_exponent))
        achieved_torque = (
            static_torque
            * torque_load * oblique * density_scale)

        force = Vec3[PartFrame].from_mx(
            ca.vertcat(achieved_thrust, 0.0, 0.0))
        torque = Vec3[PartFrame].from_mx(
            ca.vertcat(achieved_torque, 0.0, 0.0))
        return PartUpdate(
            wrench=Wrench(force=force, torque=torque),
            outputs={
                "thrust": Scalar.from_mx(achieved_thrust),
                "reaction_torque": Scalar.from_mx(achieved_torque),
                "advance_fraction": Scalar.from_mx(advance),
                "oblique_factor": Scalar.from_mx(oblique),
            },
        )


__all__ = ["DuctedPropeller"]
