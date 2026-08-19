"""Opt-in electrical endpoints for mechanical actuators.

The stock mechanical parts retain their existing command interfaces.  The
classes in this module add the A1 radial-DC endpoint contract through a plain
mixin, so one physical part has one mechanical identity and one electrical
port.  A powered part is wired exactly like any other endpoint::

    source.connect(bus).connect(PoweredMotor("fin", ...))

No ESC protocol, PWM calibration, or mission policy lives here.  Those belong
at the Shiver device boundary.  The calibrated propulsor and servo adapters
also make an important modeling limit explicit: the underlying low-order
parts have no shaft-speed/current state.  Their rated mechanical power and
efficiency are therefore required calibration parameters, not values inferred
from thrust or hinge torque.
"""

from __future__ import annotations

from typing import Any

import casadi as ca

from ...ir.types import Scalar
from .._declarations import Input, Output, Parameter, PartUpdate
from .._trace import scalar_mx as _mx
from ..actuation.ducted_propeller import DuctedPropeller
from ..actuation.thruster import Thruster
from ..aero.control_surface import ControlSurface
from ..articulation.motor import Motor
from .core import ConstantPowerLoad, ElectricalPort
from ._invariants import (
    c1_gate as _c1_gate,
    finite_scalar as _finite_scalar,
    positive as _positive,
    unit_interval as _unit_interval,
)


def _merge_update(update, outputs: dict[str, Scalar]) -> PartUpdate:
    """Append electrical observables without altering mechanical results."""
    if isinstance(update, PartUpdate):
        merged = dict(update.outputs)
        merged.update(outputs)
        return PartUpdate(
            wrench=update.wrench,
            new_state=update.new_state,
            outputs=merged,
            rates=update.rates,
        )
    return PartUpdate(wrench=update, outputs=outputs)


class PoweredLoadMixin(ElectricalPort):
    """A non-supplying electrical endpoint mixed into a mechanical part."""

    # ``supply_voltage`` is intentionally distinct from Motor's established
    # signed ``voltage`` command input.
    supply_voltage = Output()
    input_current = Output()
    output_current = Output()
    input_power = Output()
    output_power = Output()
    loss_power = Output()
    brownout = Output()
    open = Output()
    tripped = Output()
    kcl_residual = Output()
    energy_residual = Output()
    drive_fraction = Output()
    mechanical_power = Output()

    enabled: float = Input(default=1.0)
    brownout_voltage: float = Parameter(1.0)
    recovery_voltage: float = Parameter(2.0)
    voltage_floor: float = Parameter(0.1)

    _can_supply = False

    def __init__(self, name: str, **overrides: Any) -> None:
        super().__init__(name, **overrides)
        who = f"{type(self).__name__}({name!r})"
        low = _positive(self.declared_value("brownout_voltage"),
                        name=f"{who}.brownout_voltage")
        high = _positive(self.declared_value("recovery_voltage"),
                         name=f"{who}.recovery_voltage")
        if high <= low:
            raise ValueError(
                f"{who}.recovery_voltage must be > brownout_voltage")
        floor = _positive(self.declared_value("voltage_floor"),
                          name=f"{who}.voltage_floor")
        if floor > low:
            raise ValueError(
                f"{who}.voltage_floor must be <= brownout_voltage so the "
                f"load fades before its denominator floor")
        _unit_interval(self.declared_value("enabled"),
                       name=f"{who}.enabled")

    def _local_voltage(self) -> ca.MX:
        if self._upstream is None:
            raise RuntimeError(f"{type(self).__name__}({self.name!r}) has no upstream")
        return self._upstream._local_voltage()

    def _brownout_gate(self, voltage: ca.MX) -> ca.MX:
        return _c1_gate(
            voltage,
            float(self.declared_value("brownout_voltage")),
            float(self.declared_value("recovery_voltage")),
        )

    def _power_terms(self, voltage: ca.MX) -> tuple[ca.MX, ca.MX, ca.MX, ca.MX]:
        """Return input, mechanical, heat, and drive fraction powers."""
        raise NotImplementedError

    def _supply_current(self, upstream_voltage: ca.MX) -> ca.MX:
        input_power, _, _, _ = self._power_terms(upstream_voltage)
        return input_power / ca.fmax(
            upstream_voltage, float(self.declared_value("voltage_floor")))

    def dissipated_heat(self) -> ca.MX:
        _, _, loss, _ = self._power_terms(self._local_voltage())
        return loss

    def update(self, ctx) -> PartUpdate:
        mechanical_update = super().update(ctx)
        voltage = self._local_voltage()
        _, mechanical_power, loss, drive = self._power_terms(voltage)
        current = self._supply_current(voltage)
        # input_power is constructed from voltage*current in the active
        # region. The explicit expression below protects the diagnostic at
        # the voltage floor without hiding a physically invalid rail state.
        actual_input_power = voltage * current
        residual = actual_input_power - mechanical_power - loss
        zero = ca.MX(0.0)
        values = {
            "supply_voltage": voltage,
            "input_current": current,
            "output_current": current,
            "input_power": actual_input_power,
            "output_power": mechanical_power,
            "loss_power": loss,
            "brownout": 1.0 - self._brownout_gate(voltage),
            "open": 1.0 - _mx(self.enabled),
            "tripped": zero,
            "kcl_residual": zero,
            "energy_residual": residual,
            "drive_fraction": drive,
            "mechanical_power": mechanical_power,
        }
        return _merge_update(
            mechanical_update,
            {name: Scalar.from_mx(value) for name, value in values.items()},
        )


class PoweredMotor(PoweredLoadMixin, Motor):
    """A direct-voltage :class:`Motor` supplied by an A1 electrical rail.

    The existing signed ``voltage`` input remains the requested winding
    voltage.  The drive clips it to available rail magnitude and fades it
    through brownout.  Reverse power is not returned to A1's unidirectional
    network; regenerative/dynamic braking energy is reported as heat.
    """

    controller_efficiency: float = Parameter(1.0)
    idle_power: float = Parameter(0.0)
    armature_current = Output()

    def __init__(self, name: str, **overrides: Any) -> None:
        super().__init__(name, **overrides)
        who = f"PoweredMotor({name!r})"
        efficiency = _finite_scalar(
            self.declared_value("controller_efficiency"),
            name=f"{who}.controller_efficiency")
        if not 0.0 < efficiency <= 1.0:
            raise ValueError(f"{who}.controller_efficiency must be in (0, 1]")
        _positive(self.declared_value("idle_power"),
                  name=f"{who}.idle_power", allow_zero=True)

    def _terminal_voltage_mx(self) -> ca.MX:
        rail = self._local_voltage()
        command = _mx(self.voltage)
        available = ca.fmax(rail, 0.0)
        clipped = ca.fmin(ca.fmax(command, -available), available)
        return (_mx(self.enabled) * self._brownout_gate(rail) * clipped)

    def _power_terms(self, voltage: ca.MX):
        terminal_voltage = self._terminal_voltage_mx()
        current = self._current_mx()
        _, rate_name = self.dof_state_names()
        output_rate = _mx(getattr(self, rate_name))
        motor_torque = (float(self.declared_value("gear_ratio"))
                        * _mx(self.torque_constant) * current)
        mechanical = motor_torque * output_rate
        terminal_power = terminal_voltage * current
        gate = _mx(self.enabled) * self._brownout_gate(voltage)
        input_power = (
            ca.fmax(terminal_power, 0.0)
            / float(self.declared_value("controller_efficiency"))
            + gate * float(self.declared_value("idle_power"))
        )
        # Includes winding copper loss, controller loss, and any shaft energy
        # absorbed while A1's deliberately non-regenerative bus is braking.
        loss = input_power - mechanical
        requested = ca.fabs(_mx(self.voltage))
        drive = ca.if_else(
            requested > 1e-9,
            ca.fabs(terminal_voltage) / requested,
            gate,
        )
        return input_power, mechanical, loss, drive

    def update(self, ctx) -> PartUpdate:
        update = super().update(ctx)
        update.outputs["armature_current"] = Scalar.from_mx(self._current_mx())
        return update


class _CalibratedPoweredActuator(PoweredLoadMixin):
    """Power-map seam for primitives without a resolved motor shaft."""

    rated_voltage: float = Parameter(12.0)
    rated_mechanical_power: float = Parameter(1.0)
    conversion_efficiency: float = Parameter(0.8)
    idle_power: float = Parameter(0.0)
    power_exponent: float = Parameter(2.0)

    def __init__(self, name: str, **overrides: Any) -> None:
        required = {
            "rated_voltage", "rated_mechanical_power",
            "conversion_efficiency",
        }
        missing = required - set(overrides)
        if missing:
            raise TypeError(
                f"{type(self).__name__}({name!r}) requires explicit "
                f"calibration for {sorted(missing)}; the underlying "
                f"primitive has no motor shaft from which to infer it")
        super().__init__(name, **overrides)
        who = f"{type(self).__name__}({name!r})"
        for parameter in ("rated_voltage", "rated_mechanical_power",
                          "power_exponent"):
            _positive(self.declared_value(parameter),
                      name=f"{who}.{parameter}")
        efficiency = _finite_scalar(
            self.declared_value("conversion_efficiency"),
            name=f"{who}.conversion_efficiency")
        if not 0.0 < efficiency <= 1.0:
            raise ValueError(f"{who}.conversion_efficiency must be in (0, 1]")
        _positive(self.declared_value("idle_power"),
                  name=f"{who}.idle_power", allow_zero=True)

    def _unpowered_command_fraction(self) -> ca.MX:
        raise NotImplementedError

    def _drive_derating(self, voltage: ca.MX) -> ca.MX:
        voltage_fraction = ca.fmin(
            ca.fmax(voltage, 0.0) / float(self.declared_value("rated_voltage")),
            1.0,
        )
        return (_mx(self.enabled) * self._brownout_gate(voltage)
                * voltage_fraction)

    def _power_terms(self, voltage: ca.MX):
        drive = self._drive_derating(voltage)
        fraction = ca.fabs(self._unpowered_command_fraction()) * drive
        mechanical = (float(self.declared_value("rated_mechanical_power"))
                      * fraction ** float(self.declared_value("power_exponent")))
        gate = _mx(self.enabled) * self._brownout_gate(voltage)
        requested_input = (
            mechanical / float(self.declared_value("conversion_efficiency"))
            + gate * float(self.declared_value("idle_power"))
        )
        current = requested_input / ca.fmax(
            voltage, float(self.declared_value("voltage_floor")))
        input_power = voltage * current
        return input_power, mechanical, input_power - mechanical, drive


class PoweredThruster(_CalibratedPoweredActuator, Thruster):
    """Voltage-derated polynomial thruster with a calibrated power map."""

    def _unpowered_command_fraction(self) -> ca.MX:
        return _mx(self.throttle)

    def _effective_throttle(self):
        return Scalar.from_mx(
            _mx(self.throttle) * self._drive_derating(self._local_voltage()))


class PoweredDuctedPropeller(_CalibratedPoweredActuator, DuctedPropeller):
    """Voltage-derated ducted propeller with calibrated shaft power."""

    def _unpowered_command_fraction(self) -> ca.MX:
        return (_mx(self.thrust_command)
                / float(self.declared_value("max_static_thrust")))

    def _effective_thrust_command(self):
        return Scalar.from_mx(
            _mx(self.thrust_command)
            * self._drive_derating(self._local_voltage()))


class PoweredControlSurface(_CalibratedPoweredActuator, ControlSurface):
    """Control surface whose servo torque and speed fade with rail voltage."""

    def _unpowered_command_fraction(self) -> ca.MX:
        error = _mx(self.deflection_cmd) - _mx(self.deflection)
        stall = float(self.declared_value("stall_torque"))
        if stall == 0.0:
            return ca.MX(0.0)
        effort = ca.fmin(
            ca.fabs(float(self.declared_value("servo_gain")) * error) / stall,
            1.0,
        )
        return effort

    def _servo_torque(self, command_error):
        nominal = super()._servo_torque(command_error)
        return nominal * self._drive_derating(self._local_voltage())


class ConstantPowerElectronicsLoad(ConstantPowerLoad):
    """Compute/electronics load whose consumed power ultimately becomes heat."""

    def _electrical_update(self, ctx):
        values, new_state = super()._electrical_update(ctx)
        consumed = values["input_power"]
        values["output_power"] = ca.MX(0.0)
        values["loss_power"] = consumed
        values["energy_residual"] = ca.MX(0.0)
        return values, new_state

    def dissipated_heat(self) -> ca.MX:
        voltage = self._local_voltage()
        return voltage * self._supply_current(voltage)


__all__ = [
    "ConstantPowerElectronicsLoad",
    "PoweredControlSurface",
    "PoweredDuctedPropeller",
    "PoweredLoadMixin",
    "PoweredMotor",
    "PoweredThruster",
]
