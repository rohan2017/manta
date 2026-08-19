"""Deterministic Python battery plant for simulation and fault injection.

Battery and BMS dynamics are intentionally outside Manta's differentiable
``World``.  Cell SOC, fault latches, balancing, and stochastic self-discharge
do not belong in a navigation filter, controller, or generated deployment
model.  A3's ``ExternalDCSupply`` is the boundary to the electrical graph:
drive its supplied-voltage input from this plant, then feed its measured
output current into the next battery step.

The co-simulation boundary is explicit/ZOH by design.  A caller chooses the
ordering and timestep; no wall clock or hidden global state is consulted.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from numbers import Real
from typing import Any, Mapping, Sequence

import numpy as np

from manta._validation import finite_real


def _finite(value: Any, *, name: str) -> float:
    return finite_real(value, name)


def _positive(value: Any, *, name: str, allow_zero: bool = False) -> float:
    result = _finite(value, name=name)
    if result < 0.0 or (result == 0.0 and not allow_zero):
        relation = ">= 0" if allow_zero else "> 0"
        raise ValueError(f"{name} must be {relation}, got {result}")
    return result


def _fraction(value: Any, *, name: str, upper_closed: bool = True) -> float:
    result = _finite(value, name=name)
    valid = 0.0 <= result <= 1.0 if upper_closed else 0.0 <= result < 1.0
    if not valid:
        interval = "[0, 1]" if upper_closed else "[0, 1)"
        raise ValueError(f"{name} must be in {interval}, got {result}")
    return result


def _tuple(values: Sequence[Any], *, name: str) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be a finite scalar sequence")
    return tuple(_finite(value, name=f"{name}[{index}]")
                 for index, value in enumerate(values))


@dataclass(frozen=True)
class OCVCurve:
    """Calibrated piecewise-linear cell OCV curve.

    Evaluation extrapolates outside ``[0, 1]`` rather than clamping.  Normal
    pack stepping rejects an SOC transition outside that range, while direct
    evaluation remains useful for diagnosing corrupt restored state.
    """

    soc_points: tuple[float, ...] = (0.0, 0.1, 0.5, 0.9, 1.0)
    voltage_points: tuple[float, ...] = (3.0, 3.45, 3.68, 4.0, 4.2)

    def __post_init__(self) -> None:
        xs = _tuple(self.soc_points, name="OCVCurve.soc_points")
        ys = _tuple(self.voltage_points, name="OCVCurve.voltage_points")
        if len(xs) < 2 or len(xs) != len(ys):
            raise ValueError("OCV curve needs matching arrays of >= 2 points")
        if xs[0] != 0.0 or xs[-1] != 1.0:
            raise ValueError("OCV SOC points must span exactly [0, 1]")
        if any(right <= left for left, right in zip(xs, xs[1:])):
            raise ValueError("OCV SOC points must be strictly increasing")
        if any(voltage <= 0.0 for voltage in ys):
            raise ValueError("OCV voltage points must all be > 0")
        object.__setattr__(self, "soc_points", xs)
        object.__setattr__(self, "voltage_points", ys)

    def voltage(self, soc: float) -> float:
        soc = _finite(soc, name="soc")
        xs, ys = self.soc_points, self.voltage_points
        index = next((i for i in range(len(xs) - 1) if soc < xs[i + 1]),
                     len(xs) - 2)
        return ys[index] + ((soc - xs[index])
                            * (ys[index + 1] - ys[index])
                            / (xs[index + 1] - xs[index]))

    def integral(self, soc: float) -> float:
        """Integral of OCV dSOC from zero to ``soc`` (V)."""
        soc = _finite(soc, name="soc")
        xs, ys = self.soc_points, self.voltage_points
        if soc < 0.0:
            slope = (ys[1] - ys[0]) / (xs[1] - xs[0])
            return ys[0] * soc + 0.5 * slope * soc * soc
        total = 0.0
        for index in range(len(xs) - 1):
            left, right = xs[index], xs[index + 1]
            if soc <= left:
                break
            end = min(soc, right)
            width = end - left
            slope = (ys[index + 1] - ys[index]) / (right - left)
            total += ys[index] * width + 0.5 * slope * width * width
            if soc <= right:
                return total
        if soc > 1.0:
            width = soc - 1.0
            slope = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
            total += ys[-1] * width + 0.5 * slope * width * width
        return total


@dataclass(frozen=True)
class BatteryCellFaults:
    """Independent, composable fault injection for one simulation step."""

    open_cell: bool = False
    short_cell: bool = False
    high_resistance: bool = False
    capacity_loss_fraction: float = 0.0
    forced_overtemperature: bool = False

    def __post_init__(self) -> None:
        for name in ("open_cell", "short_cell", "high_resistance",
                     "forced_overtemperature"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"BatteryCellFaults.{name} must be bool")
        _fraction(self.capacity_loss_fraction,
                  name="capacity_loss_fraction", upper_closed=False)


@dataclass(frozen=True)
class BatteryCellState:
    soc: float = 1.0

    def __post_init__(self) -> None:
        value = _fraction(self.soc, name="BatteryCellState.soc")
        object.__setattr__(self, "soc", value)


@dataclass(frozen=True)
class BMSState:
    trip_latched: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.trip_latched, bool):
            raise TypeError("BMSState.trip_latched must be bool")


@dataclass(frozen=True)
class BatteryPackState:
    cells: tuple[BatteryCellState, ...]
    bms: BMSState = field(default_factory=BMSState)

    def __post_init__(self) -> None:
        cells = tuple(self.cells)
        if not cells or any(not isinstance(cell, BatteryCellState)
                            for cell in cells):
            raise TypeError(
                "BatteryPackState.cells must be a non-empty sequence of "
                "BatteryCellState")
        if not isinstance(self.bms, BMSState):
            raise TypeError("BatteryPackState.bms must be BMSState")
        object.__setattr__(self, "cells", cells)


@dataclass(frozen=True)
class BatteryStepInput:
    """Complete deterministic input for one battery-plant tick."""

    requested_series_current: float
    cell_temperatures: tuple[float, ...]
    cell_faults: tuple[BatteryCellFaults, ...]
    balance_enabled: tuple[bool, ...]
    contactor_command: bool = True
    trip_command: bool = False
    reset_command: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "requested_series_current", _finite(
            self.requested_series_current,
            name="BatteryStepInput.requested_series_current"))
        temperatures = tuple(_positive(
            value, name="BatteryStepInput.cell_temperature")
            for value in self.cell_temperatures)
        faults = tuple(self.cell_faults)
        balance = tuple(self.balance_enabled)
        if any(not isinstance(value, BatteryCellFaults) for value in faults):
            raise TypeError(
                "BatteryStepInput.cell_faults must contain BatteryCellFaults")
        if any(not isinstance(value, bool) for value in balance):
            raise TypeError("BatteryStepInput.balance_enabled must contain bools")
        for name in ("contactor_command", "trip_command", "reset_command"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"BatteryStepInput.{name} must be bool")
        object.__setattr__(self, "cell_temperatures", temperatures)
        object.__setattr__(self, "cell_faults", faults)
        object.__setattr__(self, "balance_enabled", balance)


@dataclass(frozen=True)
class BatteryTelemetry:
    terminal_voltage: float
    pack_ocv: float
    requested_series_current: float
    series_current: float
    contactor_closed: bool
    tripped: bool
    cell_soc: tuple[float, ...]
    cell_ocv: tuple[float, ...]
    cell_terminal_voltage: tuple[float, ...]
    cell_current: tuple[float, ...]
    balance_current: tuple[float, ...]
    cell_loss_power: tuple[float, ...]
    balance_loss_power: tuple[float, ...]
    pack_loss_power: float
    chemical_power: float
    output_power: float
    energy_residual: float
    minimum_soc: float
    maximum_soc: float
    soc_imbalance: float
    any_open_cell: bool
    any_shorted_cell: bool
    any_overtemperature: bool

    def supply_inputs(self, supply_name: str) -> dict[str, float]:
        """Control overlay for A3's ``ExternalDCSupply`` Manta boundary."""
        return {f"{supply_name}.supplied_voltage": self.terminal_voltage}

    def thermal_inputs(self, thermal_names: Sequence[str]) -> dict[str, float]:
        """Map per-cell losses into existing ``ThermalMass.heat_input``."""
        if len(thermal_names) != len(self.cell_loss_power):
            raise ValueError("thermal_names must contain one name per cell")
        return {f"{name}.heat_input": cell_heat + balance_heat
                for name, cell_heat, balance_heat in zip(
                    thermal_names, self.cell_loss_power,
                    self.balance_loss_power)}


@dataclass(frozen=True)
class BatteryCell:
    """Configuration for one cell; mutable SOC lives in ``BatteryPackState``."""

    usable_capacity_ah: float = 20.0
    internal_resistance: float = 0.003
    short_resistance: float = 0.0005
    high_resistance_multiplier: float = 10.0
    overtemperature_resistance_multiplier: float = 2.0
    reference_temperature: float = 298.15
    maximum_temperature: float = 333.15
    resistance_temperature_coefficient: float = 0.003
    maximum_short_current: float = 1000.0
    self_discharge_current: float = 0.0
    self_discharge_log_sigma: float = 0.0
    ocv_curve: OCVCurve = field(default_factory=OCVCurve)

    def __post_init__(self) -> None:
        for name in ("usable_capacity_ah", "internal_resistance",
                     "short_resistance", "reference_temperature",
                     "maximum_temperature", "maximum_short_current"):
            _positive(getattr(self, name), name=f"BatteryCell.{name}")
        for name in ("resistance_temperature_coefficient",
                     "self_discharge_current", "self_discharge_log_sigma"):
            _positive(getattr(self, name), name=f"BatteryCell.{name}",
                      allow_zero=True)
        for name in ("high_resistance_multiplier",
                     "overtemperature_resistance_multiplier"):
            if _positive(getattr(self, name), name=f"BatteryCell.{name}") < 1.0:
                raise ValueError(f"BatteryCell.{name} must be >= 1")
        if self.maximum_temperature <= self.reference_temperature:
            raise ValueError("maximum_temperature must exceed reference_temperature")
        if not isinstance(self.ocv_curve, OCVCurve):
            raise TypeError("BatteryCell.ocv_curve must be an OCVCurve")

    def ocv(self, state: BatteryCellState) -> float:
        return self.ocv_curve.voltage(state.soc)

    def effective_capacity(self, fault: BatteryCellFaults) -> float:
        return self.usable_capacity_ah * (1.0 - fault.capacity_loss_fraction)

    def overtemperature(self, temperature: float,
                        fault: BatteryCellFaults) -> bool:
        return (fault.forced_overtemperature
                or temperature >= self.maximum_temperature)

    def effective_resistance(self, temperature: float,
                             fault: BatteryCellFaults) -> float:
        temperature = _positive(temperature, name="cell temperature")
        resistance = (self.short_resistance if fault.short_cell
                      else self.internal_resistance)
        resistance *= 1.0 + self.resistance_temperature_coefficient * max(
            temperature - self.reference_temperature, 0.0)
        if fault.high_resistance:
            resistance *= self.high_resistance_multiplier
        if self.overtemperature(temperature, fault):
            resistance *= self.overtemperature_resistance_multiplier
        return resistance

    def stored_energy(self, state: BatteryCellState,
                      fault: BatteryCellFaults) -> float:
        return (3600.0 * self.effective_capacity(fault)
                * self.ocv_curve.integral(state.soc))


@dataclass(frozen=True)
class PassiveBalancer:
    """One BMS-commanded passive shunt channel."""

    cell_index: int
    resistance: float = 100.0
    current_limit: float = math.inf

    def __post_init__(self) -> None:
        if isinstance(self.cell_index, bool) or not isinstance(self.cell_index, int):
            raise TypeError("PassiveBalancer.cell_index must be an integer")
        if self.cell_index < 0:
            raise ValueError("PassiveBalancer.cell_index must be >= 0")
        _positive(self.resistance, name="PassiveBalancer.resistance")
        if math.isnan(float(self.current_limit)) or self.current_limit <= 0.0:
            raise ValueError("PassiveBalancer.current_limit must be > 0")


class BMSPlant:
    """Commanded contactor and trip latch, without protection policy."""

    @staticmethod
    def transition(state: BMSState, *, contactor_command: bool,
                   trip_command: bool, reset_command: bool) -> BMSState:
        for name, value in (("contactor_command", contactor_command),
                            ("trip_command", trip_command),
                            ("reset_command", reset_command)):
            if not isinstance(value, bool):
                raise TypeError(f"{name} must be bool")
        latched = state.trip_latched
        if trip_command:
            latched = True
        elif reset_command:
            latched = False
        return BMSState(trip_latched=latched)

    @staticmethod
    def contactor_closed(state: BMSState, command: bool) -> bool:
        if not isinstance(command, bool):
            raise TypeError("contactor_command must be bool")
        return command and not state.trip_latched


class SeriesBatteryPack:
    """Simulation-only series battery pack with explicit snapshot/replay."""

    def __init__(self, cells: Sequence[BatteryCell], *,
                 initial_soc: Sequence[float] | float = 1.0,
                 balancers: Sequence[PassiveBalancer] = (),
                 seed: int = 0) -> None:
        if isinstance(cells, (str, bytes)) or not isinstance(cells, Sequence):
            raise TypeError("cells must be a BatteryCell sequence")
        self.cells = tuple(cells)
        if not self.cells or any(not isinstance(cell, BatteryCell)
                                 for cell in self.cells):
            raise ValueError("cells must contain at least one BatteryCell")
        self.balancers = tuple(balancers)
        if any(not isinstance(item, PassiveBalancer) for item in self.balancers):
            raise TypeError("balancers must all be PassiveBalancer")
        if any(item.cell_index >= len(self.cells) for item in self.balancers):
            raise ValueError("balancer cell index is outside the pack")
        indices = [item.cell_index for item in self.balancers]
        if len(indices) != len(set(indices)):
            raise ValueError("only one passive balancer per cell is supported")
        if isinstance(initial_soc, Real) and not isinstance(initial_soc, bool):
            socs = (float(initial_soc),) * len(self.cells)
        else:
            socs = _tuple(initial_soc, name="initial_soc")
        if len(socs) != len(self.cells):
            raise ValueError("initial_soc must contain one value per cell")
        self._initial_state = BatteryPackState(
            cells=tuple(BatteryCellState(soc) for soc in socs))
        self._state = self._initial_state
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("seed must be an integer")
        self._seed = seed
        self._rng = np.random.default_rng(seed)
        self._last: BatteryTelemetry | None = None

    @property
    def state(self) -> BatteryPackState:
        return self._state

    @property
    def telemetry(self) -> BatteryTelemetry | None:
        return self._last

    def reset(self, *, seed: int | None = None) -> None:
        if seed is not None:
            if isinstance(seed, bool) or not isinstance(seed, int):
                raise TypeError("seed must be an integer")
            self._seed = seed
        self._state = self._initial_state
        self._rng = np.random.default_rng(self._seed)
        self._last = None

    def checkpoint(self) -> dict[str, Any]:
        return {
            "version": 1,
            "soc": [state.soc for state in self._state.cells],
            "trip_latched": self._state.bms.trip_latched,
            "rng_state": copy.deepcopy(self._rng.bit_generator.state),
        }

    def restore(self, checkpoint: Mapping[str, Any]) -> None:
        if checkpoint.get("version") != 1:
            raise ValueError("unsupported battery checkpoint version")
        soc = checkpoint.get("soc")
        if not isinstance(soc, Sequence) or len(soc) != len(self.cells):
            raise ValueError("checkpoint SOC count does not match pack")
        trip = checkpoint.get("trip_latched")
        next_state = BatteryPackState(
            cells=tuple(BatteryCellState(value) for value in soc),
            bms=BMSState(trip_latched=trip))
        try:
            next_rng = np.random.default_rng()
            next_rng.bit_generator.state = copy.deepcopy(
                checkpoint["rng_state"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid battery checkpoint RNG state") from exc
        self._state = next_state
        self._rng = next_rng
        self._last = None

    def _validate_input(self, inputs: BatteryStepInput) -> None:
        if not isinstance(inputs, BatteryStepInput):
            raise TypeError("inputs must be BatteryStepInput")
        _positive(inputs.requested_series_current,
                  name="requested_series_current", allow_zero=True)
        if len(inputs.cell_temperatures) != len(self.cells):
            raise ValueError("cell_temperatures needs one value per cell")
        for temperature in inputs.cell_temperatures:
            _positive(temperature, name="cell temperature")
        if (len(inputs.cell_faults) != len(self.cells)
                or any(not isinstance(fault, BatteryCellFaults)
                       for fault in inputs.cell_faults)):
            raise ValueError("cell_faults needs one BatteryCellFaults per cell")
        if len(inputs.balance_enabled) != len(self.balancers):
            raise ValueError("balance_enabled needs one bool per balancer")
        if any(not isinstance(value, bool) for value in inputs.balance_enabled):
            raise TypeError("balance_enabled entries must be bool")
        for name in ("contactor_command", "trip_command", "reset_command"):
            if not isinstance(getattr(inputs, name), bool):
                raise TypeError(f"{name} must be bool")

    def _evaluate(self, inputs: BatteryStepInput, *, advance: bool,
                  dt: float = 0.0) -> BatteryTelemetry:
        self._validate_input(inputs)
        if advance:
            dt = _positive(dt, name="dt")
        next_bms = BMSPlant.transition(
            self._state.bms,
            contactor_command=inputs.contactor_command,
            trip_command=inputs.trip_command,
            reset_command=inputs.reset_command) if advance else self._state.bms
        contactor_closed = BMSPlant.contactor_closed(
            next_bms, inputs.contactor_command)
        any_open = any(fault.open_cell for fault in inputs.cell_faults)
        series_current = (inputs.requested_series_current
                          if contactor_closed and not any_open else 0.0)

        ocvs = tuple(cell.ocv(state) for cell, state in zip(
            self.cells, self._state.cells))
        resistances = tuple(cell.effective_resistance(temp, fault)
                            for cell, temp, fault in zip(
                                self.cells, inputs.cell_temperatures,
                                inputs.cell_faults))
        effective_ocvs = tuple(0.0 if fault.short_cell else ocv
                               for ocv, fault in zip(ocvs, inputs.cell_faults))
        terminal_cells = tuple(voltage - series_current * resistance
                               for voltage, resistance in zip(
                                   effective_ocvs, resistances))
        terminal_voltage = (sum(terminal_cells)
                            if contactor_closed and not any_open else 0.0)
        if terminal_voltage < 0.0:
            raise ValueError(
                "requested series current exceeds the pack's present "
                "short-circuit capability; terminal voltage would be "
                f"negative ({terminal_voltage})")

        balance_by_cell = [0.0] * len(self.cells)
        balance_loss_by_cell = [0.0] * len(self.cells)
        for balancer, enabled in zip(self.balancers, inputs.balance_enabled):
            index = balancer.cell_index
            if enabled and not inputs.cell_faults[index].short_cell:
                current = min(
                    effective_ocvs[index]
                    / (resistances[index] + balancer.resistance),
                    balancer.current_limit)
                balance_by_cell[index] = current
                # Everything not dissipated in the cell's internal resistance
                # is heat in the shunt/current limiter. This remains energy-
                # conserving when current_limit is active.
                balance_loss_by_cell[index] = max(
                    effective_ocvs[index] * current
                    - current * current * resistances[index], 0.0)

        parasitic = []
        for cell in self.cells:
            if cell.self_discharge_current == 0.0:
                parasitic.append(0.0)
            else:
                z = float(self._rng.normal()) if advance else 0.0
                parasitic.append(cell.self_discharge_current * math.exp(
                    cell.self_discharge_log_sigma * z
                    - 0.5 * cell.self_discharge_log_sigma**2))
        short_currents = tuple(
            min(ocv / (cell.internal_resistance + cell.short_resistance),
                cell.maximum_short_current) if fault.short_cell else 0.0
            for cell, ocv, fault in zip(self.cells, ocvs, inputs.cell_faults))
        currents = tuple(series_current + balance + leak + short
                         for balance, leak, short in zip(
                             balance_by_cell, parasitic, short_currents))
        soc_currents = tuple(
            (0.0 if fault.short_cell else series_current)
            + balance + leak + short
            for fault, balance, leak, short in zip(
                inputs.cell_faults, balance_by_cell, parasitic, short_currents))

        cell_losses = []
        for index, (ocv, resistance) in enumerate(zip(ocvs, resistances)):
            series_loss = series_current * series_current * resistance
            balance_internal = balance_by_cell[index]**2 * resistance
            parasitic_loss = ocv * parasitic[index]
            short_loss = ocv * short_currents[index]
            cell_losses.append(series_loss + balance_internal
                               + parasitic_loss + short_loss)
        chemical_power = sum(
            effective_ocv * series_current
            + ocv * (balance + leak + short)
            for effective_ocv, ocv, balance, leak, short in zip(
                effective_ocvs, ocvs, balance_by_cell, parasitic,
                short_currents))
        output_power = terminal_voltage * series_current
        pack_loss = sum(cell_losses) + sum(balance_loss_by_cell)
        energy_residual = chemical_power - output_power - pack_loss

        if advance:
            next_cells = []
            for index, (cell, state, fault, current) in enumerate(zip(
                    self.cells, self._state.cells, inputs.cell_faults,
                    soc_currents)):
                capacity = cell.effective_capacity(fault)
                next_soc = state.soc - dt * current / (3600.0 * capacity)
                if not math.isfinite(next_soc) or not 0.0 <= next_soc <= 1.0:
                    raise ValueError(
                        f"cell {index} SOC would leave [0, 1] ({next_soc}); "
                        "reduce dt, reduce current, or terminate the run")
                next_cells.append(BatteryCellState(next_soc))
            self._state = BatteryPackState(tuple(next_cells), next_bms)

        socs = tuple(state.soc for state in self._state.cells)
        result = BatteryTelemetry(
            terminal_voltage=terminal_voltage,
            pack_ocv=sum(effective_ocvs),
            requested_series_current=inputs.requested_series_current,
            series_current=series_current,
            contactor_closed=contactor_closed,
            tripped=next_bms.trip_latched,
            cell_soc=socs,
            cell_ocv=ocvs,
            cell_terminal_voltage=terminal_cells,
            cell_current=currents,
            balance_current=tuple(balance_by_cell),
            cell_loss_power=tuple(cell_losses),
            balance_loss_power=tuple(balance_loss_by_cell),
            pack_loss_power=pack_loss,
            chemical_power=chemical_power,
            output_power=output_power,
            energy_residual=energy_residual,
            minimum_soc=min(socs), maximum_soc=max(socs),
            soc_imbalance=max(socs) - min(socs),
            any_open_cell=any_open,
            any_shorted_cell=any(fault.short_cell
                                 for fault in inputs.cell_faults),
            any_overtemperature=any(
                cell.overtemperature(temp, fault)
                for cell, temp, fault in zip(
                    self.cells, inputs.cell_temperatures, inputs.cell_faults)),
        )
        if advance:
            self._last = result
        return result

    def preview(self, inputs: BatteryStepInput) -> BatteryTelemetry:
        """Evaluate terminal conditions without changing state or RNG."""
        return self._evaluate(inputs, advance=False)

    def step(self, dt: float, inputs: BatteryStepInput) -> BatteryTelemetry:
        """Advance exactly one explicit simulation tick."""
        rng_state = copy.deepcopy(self._rng.bit_generator.state)
        try:
            return self._evaluate(inputs, advance=True, dt=dt)
        except Exception:
            # Invalid steps are atomic: neither physical state nor stochastic
            # stream advances, so correcting the input and retrying replays.
            self._rng.bit_generator.state = rng_state
            raise
