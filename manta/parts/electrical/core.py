"""Bounded lumped-DC electrical networks.

Electrical connectivity is deliberately independent of the mechanical part
tree.  Electrical nodes are still ordinary :class:`~manta.parts.Part`
objects (and therefore participate in simulation, estimation and codegen),
but ``source.connect(load)`` records a separate directed edge.  The supported
topology is a radial forest: every non-source node has exactly one upstream
supply and a supply may have any number of children.

Every rail stores energy in a capacitor and advances its voltage explicitly::

    C dV/dt = I_in - I_out

Series resistance and current/power limits make every edge current bounded.
Loads fade out through a C1 brownout curve before zero volts, so the equations
remain finite during collapse.  Hard physical limits (fuse latch, contactor
edge, current limits) are piecewise differentiable at their switching surface,
which CasADi can differentiate on either side; no state is silently clipped to
hide an unstable integration step.

The model is intentionally not a general circuit solver.  It excludes loops,
parallel sources, charge/reverse-current paths and AC behavior.  Those would
require a different (implicit) solve and are rejected rather than approximated
quietly.
"""

from __future__ import annotations

import math
from typing import Any

import casadi as ca

from ...ir.frames import PartFrame
from ...ir.types import Scalar, Vec3
from ...ir.wrench import Wrench
from .._declarations import (
    Input, Output, Parameter, PartUpdate, State, WhiteNoise,
)
from .._trace import scalar_mx as _mx
from ..base import Part
from ._invariants import (
    bounded_positive as _bounded_positive,
    c1_gate as _c1_gate,
    finite_scalar as _finite_scalar,
    positive as _positive,
    positive_or_inf as _positive_or_inf,
    unit_interval as _unit_interval,
)


def _root_of(part: Part) -> Part:
    node = part
    while node.parent is not None:
        node = node.parent
    return node


def _zero_wrench() -> Wrench:
    zero = Vec3[PartFrame].constant((0.0, 0.0, 0.0))
    return Wrench(force=zero, torque=zero)


class ElectricalPort:
    """Connectivity shared by electrical nodes and powered part variants.

    This is deliberately a plain cooperative mixin rather than a ``Part``.
    A mechanical part can therefore opt into the radial network without
    acquiring a second mechanical identity or duplicating its dynamics.
    """

    _is_source = False
    _can_supply = False

    def __init__(self, name: str, **overrides: Any) -> None:
        super().__init__(name, **overrides)
        self._upstream: ElectricalPort | None = None
        self._electrical_children: list[ElectricalPort] = []

    @property
    def upstream(self) -> "ElectricalPort | None":
        return self._upstream

    @property
    def electrical_children(self) -> tuple["ElectricalPort", ...]:
        return tuple(self._electrical_children)

    def connect(self, child: "ElectricalPort") -> "ElectricalPort":
        """Supply ``child`` from this port and return the child."""
        who = f"{type(self).__name__}({self.name!r}).connect"
        if not isinstance(child, ElectricalPort):
            raise TypeError(
                f"{who}: expected an ElectricalPort, got "
                f"{type(child).__name__}")
        if not self._can_supply:
            raise ValueError(f"{who}: an endpoint load cannot supply children")
        if child._is_source:
            raise ValueError(f"{who}: a source cannot have an upstream supply")
        if child is self:
            raise ValueError(f"{who}: cannot connect a node to itself")
        if child._upstream is not None:
            raise ValueError(
                f"{who}: {child.name!r} already has upstream supply "
                f"{child._upstream.name!r}; parallel source sharing is not "
                f"supported")
        cursor: ElectricalPort | None = self
        while cursor is not None:
            if cursor is child:
                raise ValueError(f"{who}: connection would create a cycle")
            cursor = cursor._upstream
        child._upstream = self
        self._electrical_children.append(child)
        return child

    def _downstream_current(self, voltage: ca.MX) -> ca.MX:
        current = ca.MX(0.0)
        for child in self._electrical_children:
            current = current + child._supply_current(voltage)
        return current

    def on_world_resolve(self, world, craft) -> None:
        """Validate the finished electrical graph before symbolic tracing."""
        super().on_world_resolve(world, craft)
        who = f"{type(self).__name__}({self.name!r})"
        if self._is_source:
            if self._upstream is not None:
                raise ValueError(f"{who}: a source cannot have an upstream")
        elif self._upstream is None:
            raise ValueError(
                f"{who}: every non-source electrical node needs exactly one "
                f"upstream supply")

        expected_root = _root_of(self)
        neighbours = list(self._electrical_children)
        if self._upstream is not None:
            neighbours.append(self._upstream)
        for other in neighbours:
            if _root_of(other) is not expected_root:
                raise ValueError(
                    f"{who}: electrical connection to {other.name!r} crosses "
                    f"crafts; electrical networks are per-craft")
        for child in self._electrical_children:
            if child._upstream is not self:
                raise ValueError(
                    f"{who}: inconsistent edge to {child.name!r}")

        seen: set[int] = set()
        cursor: ElectricalPort | None = self
        while cursor is not None:
            if id(cursor) in seen:
                raise ValueError(f"{who}: electrical topology contains a cycle")
            seen.add(id(cursor))
            cursor = cursor._upstream
        root = self
        while root._upstream is not None:
            root = root._upstream
        if not root._is_source:
            raise ValueError(
                f"{who}: topology does not terminate at an electrical source")


class ElectricalNode(ElectricalPort, Part):
    """Base contract for one node in a radial DC network.

    The diagnostic outputs have identical meanings on every node:

    ``voltage``
        The node terminal/rail voltage (V).
    ``input_current`` / ``input_power``
        Flow entering from the upstream edge (or ideal reservoir for a
        source), in A/W.
    ``output_current`` / ``output_power``
        Flow delivered to children, or consumed as useful endpoint power by
        a load, in A/W.
    ``loss_power``
        Electrical conversion/series loss (W), suitable for thermal coupling.
    ``brownout`` / ``open`` / ``tripped``
        Unit-valued status signals.  Brownout is 1 below the lower voltage and
        0 above the recovery voltage.  ``open`` and ``tripped`` are explicit;
        ordinary buses and loads report zero.
    ``kcl_residual`` / ``energy_residual``
        Equation diagnostics (A and W).  They are symbolically zero for a
        correctly assembled tick, including capacitor storage and injected
        current noise.

    Use :meth:`connect` on the upstream node.  Connectivity need not resemble
    mechanical mounting, but every connected node must ride the same craft.
    """

    voltage = Output()
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

    brownout_voltage: float = Parameter(0.0)
    recovery_voltage: float = Parameter(0.0)

    _can_supply = True

    def __init__(self, name: str, **overrides: Any) -> None:
        super().__init__(name, **overrides)
        who = f"{type(self).__name__}({name!r})"
        low = _positive(self.declared_value("brownout_voltage"),
                        name=f"{who}.brownout_voltage", allow_zero=True)
        high = _positive(self.declared_value("recovery_voltage"),
                         name=f"{who}.recovery_voltage", allow_zero=True)
        if high < low:
            raise ValueError(
                f"{who}.recovery_voltage must be >= brownout_voltage; "
                f"got {high} < {low}")
    def _brownout_gate(self, voltage: ca.MX) -> ca.MX:
        return _c1_gate(
            voltage,
            float(self.declared_value("brownout_voltage")),
            float(self.declared_value("recovery_voltage")),
        )

    def _local_voltage(self) -> ca.MX:
        raise NotImplementedError

    def _supply_current(self, upstream_voltage: ca.MX) -> ca.MX:
        """Current this node requests from its parent (positive into node)."""
        raise NotImplementedError

    def _status(self, voltage: ca.MX) -> tuple[ca.MX, ca.MX, ca.MX]:
        return 1.0 - self._brownout_gate(voltage), ca.MX(0.0), ca.MX(0.0)

    def _electrical_update(self, ctx) -> tuple[dict[str, ca.MX], dict[str, Any]]:
        raise NotImplementedError

    def update(self, ctx) -> PartUpdate:
        values, new_state = self._electrical_update(ctx)
        voltage = values["voltage"]
        brownout, opened, tripped = self._status(voltage)
        values.update(brownout=brownout, open=opened, tripped=tripped)
        return PartUpdate(
            wrench=_zero_wrench(),
            new_state=new_state,
            outputs={name: Scalar.from_mx(values[name])
                     for name in self.output_declarations()},
        )

    def dissipated_heat(self) -> ca.MX:
        """Electrical loss available to ``ThermalMass(source=...)``."""
        raise NotImplementedError



class _CapacitiveRail(ElectricalNode):
    """Shared capacitor state and current-noise plumbing for energized rails."""

    capacitance: float = Parameter(0.1, manifold="R1")
    rail_voltage = State(init=12.0, manifold="R1")
    current_noise = WhiteNoise("R1", sigma=0.0)

    def __init__(self, name: str, **overrides: Any) -> None:
        super().__init__(name, **overrides)
        who = f"{type(self).__name__}({name!r})"
        _positive(self.declared_value("capacitance"),
                  name=f"{who}.capacitance")
        _positive(self.declared_value("rail_voltage"),
                  name=f"{who}.rail_voltage", allow_zero=True)

    def state_declarations(self):
        """Use the configured rail voltage as this instance's initial state.

        ``State.init`` is normally class-scoped in Manta.  Rail nominal
        voltages are inherently per instance, so this follows the existing
        per-instance declaration seam used by variable-I/O parts.
        """
        declarations = dict(super().state_declarations())
        declarations["rail_voltage"] = State(
            init=float(self.declared_value("rail_voltage")), manifold="R1")
        return declarations

    def _local_voltage(self) -> ca.MX:
        return _mx(self.rail_voltage)

    def _rail_balance(self, ctx, *, input_current: ca.MX,
                      input_power: ca.MX, loss_power: ca.MX,
                      output_current: ca.MX | None = None,
                      storage_input_current: ca.MX | None = None,
                      ) -> tuple[dict[str, ca.MX], dict[str, Any]]:
        voltage = self._local_voltage()
        if output_current is None:
            output_current = self._downstream_current(voltage)
        if storage_input_current is None:
            storage_input_current = input_current
        noise = _mx(self.current_noise)
        capacitance = _mx(self.capacitance)
        derivative = (storage_input_current + noise
                      - output_current) / capacitance
        next_voltage = voltage + _mx(ctx.dt) * derivative
        stored_power = capacitance * voltage * derivative
        output_power = voltage * output_current
        values = {
            "voltage": voltage,
            "input_current": input_current,
            "output_current": output_current,
            "input_power": input_power,
            "output_power": output_power,
            "loss_power": loss_power,
            "kcl_residual": (capacitance * derivative
                             - (storage_input_current + noise
                                - output_current)),
            "energy_residual": (input_power + voltage * noise
                                 - output_power - loss_power - stored_power),
        }
        return values, {"rail_voltage": Scalar.from_mx(next_voltage)}

    def stable_timestep_hint(self, *, minimum_voltage: float,
                             maximum_net_current: float) -> float:
        """Conservative positive-voltage Euler step hint in seconds.

        A caller supplies the lowest voltage it intends to resolve and the
        largest possible net discharge current.  Staying below
        ``C*V_min/I_max`` prevents a single explicit step crossing zero from
        that operating point.  This is an inspectable bound, not a hidden
        runtime clamp.
        """
        v = _positive(minimum_voltage, name="minimum_voltage")
        i = _positive(maximum_net_current, name="maximum_net_current")
        return float(self.declared_value("capacitance")) * v / i


class DCSource(_CapacitiveRail):
    """Current-limited Thevenin DC source with terminal capacitance.

    ``open_circuit_voltage`` and ``source_resistance`` describe the ideal
    reservoir.  Current flows only out of the reservoir; A1 deliberately does
    not model charging.  ``enabled`` is a normalized contact command.
    """

    _is_source = True

    open_circuit_voltage: float = Parameter(12.0, manifold="R1")
    source_resistance: float = Parameter(0.05, manifold="R1")
    current_limit: float = Parameter(math.inf, allow_infinite=True)
    enabled: float = Input(default=1.0)

    def __init__(self, name: str, **overrides: Any) -> None:
        if "rail_voltage" not in overrides:
            overrides["rail_voltage"] = overrides.get(
                "open_circuit_voltage",
                type(self)._declarations()["open_circuit_voltage"].default)
        super().__init__(name, **overrides)
        who = f"DCSource({name!r})"
        _positive(self.declared_value("open_circuit_voltage"),
                  name=f"{who}.open_circuit_voltage")
        _positive(self.declared_value("source_resistance"),
                  name=f"{who}.source_resistance")
        _positive_or_inf(self.declared_value("current_limit"),
                         name=f"{who}.current_limit")
        _unit_interval(self.declared_value("enabled"), name=f"{who}.enabled")

    def _internal_current(self) -> ca.MX:
        raw = ((_mx(self.open_circuit_voltage) - self._local_voltage())
               / _mx(self.source_resistance))
        return (_mx(self.enabled)
                * _bounded_positive(raw, float(self.declared_value(
                    "current_limit"))))

    def _supply_current(self, upstream_voltage: ca.MX) -> ca.MX:
        raise RuntimeError("DCSource cannot be connected downstream")

    def _electrical_update(self, ctx):
        voltage = self._local_voltage()
        current = self._internal_current()
        reservoir_voltage = _mx(self.open_circuit_voltage)
        input_power = reservoir_voltage * current
        loss = (reservoir_voltage - voltage) * current
        return self._rail_balance(
            ctx, input_current=current, input_power=input_power,
            loss_power=loss)

    def _status(self, voltage: ca.MX):
        brownout = 1.0 - self._brownout_gate(voltage)
        return brownout, 1.0 - _mx(self.enabled), ca.MX(0.0)

    def dissipated_heat(self) -> ca.MX:
        return ((_mx(self.open_circuit_voltage) - self._local_voltage())
                * self._internal_current())


class ExternalDCSupply(ElectricalNode):
    """Runtime boundary for a simulation-only or hardware DC source.

    ``supplied_voltage`` enters the Manta tick as an ordinary input and the
    aggregate downstream demand leaves as ``output_current``.  A battery
    plant may therefore keep cell, thermal, and fault state in a
    non-differentiable simulator while powered mechanical parts retain their
    normal compiled model.  Source-internal heat remains owned by that plant;
    endpoint conversion loss is still available through each load's
    ``dissipated_heat()``.
    """

    _is_source = True

    supplied_voltage: float = Input(default=0.0)

    def _local_voltage(self) -> ca.MX:
        return _mx(self.supplied_voltage)

    def _supply_current(self, upstream_voltage: ca.MX) -> ca.MX:
        raise RuntimeError("ExternalDCSupply cannot be connected downstream")

    def _electrical_update(self, ctx):
        voltage = self._local_voltage()
        current = self._downstream_current(voltage)
        power = voltage * current
        zero = ca.MX(0.0)
        return {
            "voltage": voltage,
            "input_current": current,
            "output_current": current,
            "input_power": power,
            "output_power": power,
            "loss_power": zero,
            "kcl_residual": zero,
            "energy_residual": zero,
        }, {}

    def dissipated_heat(self) -> ca.MX:
        return ca.MX(0.0)


class ElectricalBus(_CapacitiveRail):
    """Capacitive DC bus fed through a current-limited series edge."""

    series_resistance: float = Parameter(0.01, manifold="R1")
    input_current_limit: float = Parameter(math.inf, allow_infinite=True)

    def __init__(self, name: str, **overrides: Any) -> None:
        super().__init__(name, **overrides)
        who = f"{type(self).__name__}({name!r})"
        _positive(self.declared_value("series_resistance"),
                  name=f"{who}.series_resistance")
        _positive_or_inf(self.declared_value("input_current_limit"),
                         name=f"{who}.input_current_limit")

    def _connection_factor(self) -> ca.MX:
        return ca.MX(1.0)

    def _edge_current(self, upstream_voltage: ca.MX) -> ca.MX:
        raw = ((upstream_voltage - self._local_voltage())
               / _mx(self.series_resistance))
        limited = _bounded_positive(
            raw, float(self.declared_value("input_current_limit")))
        return self._connection_factor() * limited

    def _supply_current(self, upstream_voltage: ca.MX) -> ca.MX:
        return self._edge_current(upstream_voltage)

    def _upstream_voltage(self) -> ca.MX:
        if self._upstream is None:
            raise RuntimeError(
                f"{type(self).__name__}({self.name!r}) has no upstream")
        return self._upstream._local_voltage()

    def _electrical_update(self, ctx):
        upstream_voltage = self._upstream_voltage()
        current = self._edge_current(upstream_voltage)
        voltage = self._local_voltage()
        input_power = upstream_voltage * current
        loss = (upstream_voltage - voltage) * current
        return self._rail_balance(
            ctx, input_current=current, input_power=input_power,
            loss_power=loss)

    def dissipated_heat(self) -> ca.MX:
        upstream_voltage = self._upstream_voltage()
        current = self._edge_current(upstream_voltage)
        return (upstream_voltage - self._local_voltage()) * current


class Contactor(ElectricalBus):
    """Commanded series contactor with downstream hold-up capacitance."""

    closed: float = Input(default=1.0)

    def __init__(self, name: str, **overrides: Any) -> None:
        super().__init__(name, **overrides)
        _unit_interval(self.declared_value("closed"),
                       name=f"Contactor({name!r}).closed")

    def _connection_factor(self) -> ca.MX:
        return _mx(self.closed)

    def _status(self, voltage: ca.MX):
        brownout = 1.0 - self._brownout_gate(voltage)
        return brownout, 1.0 - _mx(self.closed), ca.MX(0.0)


class Fuse(ElectricalBus):
    """Latching I²t fuse with a continuous overload accumulator.

    ``trip_fraction`` integrates normalized I² above the rated current and
    latches at one.  The electrical edge remains closed until the threshold,
    then opens.  This is a deterministic hybrid event and therefore only
    piecewise differentiable at the exact trip surface.
    """

    rated_current: float = Parameter(10.0)
    trip_time: float = Parameter(1.0)
    trip_fraction = State(init=0.0, manifold="R1")

    def __init__(self, name: str, **overrides: Any) -> None:
        super().__init__(name, **overrides)
        who = f"Fuse({name!r})"
        _positive(self.declared_value("rated_current"),
                  name=f"{who}.rated_current")
        _positive(self.declared_value("trip_time"), name=f"{who}.trip_time")
        fraction = _finite_scalar(self.declared_value("trip_fraction"),
                                  name=f"{who}.trip_fraction")
        if not 0.0 <= fraction <= 1.0:
            raise ValueError(f"{who}.trip_fraction must be in [0, 1]")

    def _tripped(self) -> ca.MX:
        return ca.if_else(_mx(self.trip_fraction) >= 1.0, 1.0, 0.0)

    def state_declarations(self):
        declarations = dict(super().state_declarations())
        declarations["trip_fraction"] = State(
            init=float(self.declared_value("trip_fraction")), manifold="R1")
        return declarations

    def _connection_factor(self) -> ca.MX:
        return 1.0 - self._tripped()

    def _electrical_update(self, ctx):
        values, new_state = super()._electrical_update(ctx)
        ratio_sq = (values["input_current"]
                    / float(self.declared_value("rated_current"))) ** 2
        overload = ca.fmax(ratio_sq - 1.0, 0.0)
        next_fraction = ca.fmin(
            1.0,
            _mx(self.trip_fraction)
            + _mx(ctx.dt) * overload / float(self.declared_value("trip_time")),
        )
        new_state["trip_fraction"] = Scalar.from_mx(next_fraction)
        return values, new_state

    def _status(self, voltage: ca.MX):
        tripped = self._tripped()
        brownout = 1.0 - self._brownout_gate(voltage)
        return brownout, tripped, tripped


class DCConverter(_CapacitiveRail):
    """Regulated DC converter with dropout, efficiency and hard ratings.

    A proportional internal regulator charges the output capacitor toward
    ``output_voltage`` through ``control_resistance``.  Delivery is bounded by
    output current, output power and available input power.  Below
    ``minimum_input_voltage`` it fades out through a C1 gate; above that
    voltage the stated efficiency relates input and rail-injection power.
    """

    output_voltage: float = Parameter(12.0, manifold="R1")
    dropout_voltage: float = Parameter(1.0)
    minimum_input_voltage: float = Parameter(2.0)
    input_recovery_voltage: float = Parameter(2.5)
    control_resistance: float = Parameter(0.05)
    efficiency: float = Parameter(0.9)
    output_current_limit: float = Parameter(math.inf, allow_infinite=True)
    output_power_limit: float = Parameter(math.inf, allow_infinite=True)
    input_power_limit: float = Parameter(math.inf, allow_infinite=True)
    quiescent_current: float = Parameter(0.0)
    enabled: float = Input(default=1.0)

    def __init__(self, name: str, **overrides: Any) -> None:
        if "rail_voltage" not in overrides:
            overrides["rail_voltage"] = overrides.get(
                "output_voltage",
                type(self)._declarations()["output_voltage"].default)
        super().__init__(name, **overrides)
        who = f"DCConverter({name!r})"
        for attr in ("output_voltage", "control_resistance",
                     "minimum_input_voltage", "input_recovery_voltage"):
            _positive(self.declared_value(attr), name=f"{who}.{attr}")
        _positive(self.declared_value("dropout_voltage"),
                  name=f"{who}.dropout_voltage", allow_zero=True)
        eta = _finite_scalar(self.declared_value("efficiency"),
                             name=f"{who}.efficiency")
        if not 0.0 < eta <= 1.0:
            raise ValueError(f"{who}.efficiency must be in (0, 1], got {eta}")
        if (float(self.declared_value("input_recovery_voltage"))
                <= float(self.declared_value("minimum_input_voltage"))):
            raise ValueError(
                f"{who}.input_recovery_voltage must be greater than "
                f"minimum_input_voltage")
        for attr in ("output_current_limit", "output_power_limit",
                     "input_power_limit"):
            _positive_or_inf(self.declared_value(attr), name=f"{who}.{attr}")
        _positive(self.declared_value("quiescent_current"),
                  name=f"{who}.quiescent_current", allow_zero=True)
        _unit_interval(self.declared_value("enabled"), name=f"{who}.enabled")

    def _upstream_voltage(self) -> ca.MX:
        if self._upstream is None:
            raise RuntimeError(f"DCConverter({self.name!r}) has no upstream")
        return self._upstream._local_voltage()

    def _conversion(self, upstream_voltage: ca.MX) -> tuple[ca.MX, ca.MX]:
        voltage = self._local_voltage()
        available = _c1_gate(
            upstream_voltage,
            float(self.declared_value("minimum_input_voltage")),
            float(self.declared_value("input_recovery_voltage")),
        ) * _mx(self.enabled)
        attainable = ca.fmax(
            upstream_voltage - float(self.declared_value("dropout_voltage")),
            0.0)
        target = ca.fmin(_mx(self.output_voltage), attainable)
        requested = _bounded_positive(
            (target - voltage) / float(self.declared_value(
                "control_resistance")),
            float(self.declared_value("output_current_limit")),
        )
        delivered = available * requested

        voltage_floor = float(self.declared_value("minimum_input_voltage"))
        rail_floor = max(1e-9, float(self.declared_value("brownout_voltage")))
        output_power_cap = float(self.declared_value("output_power_limit"))
        input_power_cap = float(self.declared_value("input_power_limit"))
        if math.isfinite(output_power_cap):
            delivered = ca.fmin(delivered, output_power_cap / ca.fmax(
                voltage, rail_floor))
        if math.isfinite(input_power_cap):
            delivered = ca.fmin(
                delivered,
                (input_power_cap * float(self.declared_value("efficiency"))
                 / ca.fmax(voltage, rail_floor)),
            )
        rail_power = voltage * delivered
        conversion_input_power = rail_power / float(self.declared_value(
            "efficiency"))
        quiescent_power = (available * upstream_voltage
                           * float(self.declared_value("quiescent_current")))
        input_power = conversion_input_power + quiescent_power
        input_current = input_power / ca.fmax(upstream_voltage, voltage_floor)
        return delivered, input_current

    def _supply_current(self, upstream_voltage: ca.MX) -> ca.MX:
        _, input_current = self._conversion(upstream_voltage)
        return input_current

    def _electrical_update(self, ctx):
        upstream_voltage = self._upstream_voltage()
        delivered, input_current = self._conversion(upstream_voltage)
        input_power = upstream_voltage * input_current
        rail_power = self._local_voltage() * delivered
        loss = input_power - rail_power
        return self._rail_balance(
            ctx, input_current=input_current, input_power=input_power,
            loss_power=loss, storage_input_current=delivered)

    def _status(self, voltage: ca.MX):
        brownout = 1.0 - self._brownout_gate(voltage)
        return brownout, 1.0 - _mx(self.enabled), ca.MX(0.0)

    def dissipated_heat(self) -> ca.MX:
        upstream_voltage = self._upstream_voltage()
        delivered, input_current = self._conversion(upstream_voltage)
        return (upstream_voltage * input_current
                - self._local_voltage() * delivered)


class ElectricalLoad(ElectricalNode):
    """Base endpoint load with brownout and enable semantics."""

    _can_supply = False

    enabled: float = Input(default=1.0)
    heat_fraction: float = Parameter(0.0)

    def __init__(self, name: str, **overrides: Any) -> None:
        super().__init__(name, **overrides)
        fraction = _finite_scalar(
            self.declared_value("heat_fraction"),
            name=f"{type(self).__name__}({name!r}).heat_fraction")
        if not 0.0 <= fraction <= 1.0:
            raise ValueError("heat_fraction must be in [0, 1]")
        _unit_interval(
            self.declared_value("enabled"),
            name=f"{type(self).__name__}({name!r}).enabled")

    def _local_voltage(self) -> ca.MX:
        if self._upstream is None:
            raise RuntimeError(
                f"{type(self).__name__}({self.name!r}) has no upstream")
        return self._upstream._local_voltage()

    def _requested_current(self, voltage: ca.MX) -> ca.MX:
        raise NotImplementedError

    def _supply_current(self, upstream_voltage: ca.MX) -> ca.MX:
        return (_mx(self.enabled) * self._brownout_gate(upstream_voltage)
                * self._requested_current(upstream_voltage))

    def _electrical_update(self, ctx):
        voltage = self._local_voltage()
        current = self._supply_current(voltage)
        power = voltage * current
        zero = ca.MX(0.0)
        return {
            "voltage": voltage,
            "input_current": current,
            "output_current": current,
            "input_power": power,
            "output_power": power,
            "loss_power": zero,
            "kcl_residual": zero,
            "energy_residual": zero,
        }, {}

    def _status(self, voltage: ca.MX):
        brownout = 1.0 - self._brownout_gate(voltage)
        return brownout, 1.0 - _mx(self.enabled), ca.MX(0.0)

    def dissipated_heat(self) -> ca.MX:
        voltage = self._local_voltage()
        return (float(self.declared_value("heat_fraction")) * voltage
                * self._supply_current(voltage))


class ResistiveLoad(ElectricalLoad):
    """Constant-resistance endpoint load."""

    resistance: float = Parameter(10.0, manifold="R1")

    def __init__(self, name: str, **overrides: Any) -> None:
        super().__init__(name, **overrides)
        _positive(self.declared_value("resistance"),
                  name=f"ResistiveLoad({name!r}).resistance")

    def _requested_current(self, voltage: ca.MX) -> ca.MX:
        return ca.fmax(voltage, 0.0) / _mx(self.resistance)


class ConstantCurrentLoad(ElectricalLoad):
    """Constant-current endpoint load above its brownout recovery voltage."""

    current: float = Parameter(1.0, manifold="R1")

    def __init__(self, name: str, **overrides: Any) -> None:
        super().__init__(name, **overrides)
        _positive(self.declared_value("current"),
                  name=f"ConstantCurrentLoad({name!r}).current",
                  allow_zero=True)

    def _requested_current(self, voltage: ca.MX) -> ca.MX:
        return _mx(self.current)


class ConstantPowerLoad(ElectricalLoad):
    """Bounded constant-power endpoint load with a low-voltage floor."""

    power: float = Parameter(10.0, manifold="R1")
    current_limit: float = Parameter(math.inf, allow_infinite=True)
    voltage_floor: float = Parameter(0.1)

    def __init__(self, name: str, **overrides: Any) -> None:
        super().__init__(name, **overrides)
        _positive(self.declared_value("power"),
                  name=f"ConstantPowerLoad({name!r}).power", allow_zero=True)
        _positive_or_inf(self.declared_value("current_limit"),
                         name=f"ConstantPowerLoad({name!r}).current_limit")
        floor = _positive(self.declared_value("voltage_floor"),
                          name=f"ConstantPowerLoad({name!r}).voltage_floor")
        low = float(self.declared_value("brownout_voltage"))
        if low > 0.0 and floor > low:
            raise ValueError(
                f"ConstantPowerLoad({name!r}).voltage_floor must be <= "
                f"brownout_voltage so the load fades before its denominator "
                f"floor")

    def _requested_current(self, voltage: ca.MX) -> ca.MX:
        raw = _mx(self.power) / ca.fmax(
            voltage, float(self.declared_value("voltage_floor")))
        return _bounded_positive(raw, float(self.declared_value("current_limit")))


__all__ = [
    "ConstantCurrentLoad",
    "ConstantPowerLoad",
    "Contactor",
    "DCConverter",
    "DCSource",
    "ElectricalBus",
    "ElectricalLoad",
    "ElectricalNode",
    "ElectricalPort",
    "ExternalDCSupply",
    "Fuse",
    "ResistiveLoad",
]
