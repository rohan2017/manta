# Lumped-DC electrical networks

Manta's electrical model is a bounded lumped-DC plant for vehicle power
systems. It is intended to answer questions such as:

- Does a rail brown out when two loads start together?
- Does a regulator hit its current or power rating?
- How long does downstream hold-up capacitance survive an open contactor?
- Which part's electrical loss should heat a `ThermalMass`?

It is not SPICE. AC behavior, arbitrary circuit loops, parallel source
sharing, reverse current and charging require an implicit circuit solve and
are outside this model.

## Two independent trees

Every electrical node is an ordinary `Part`, so its states and equations pass
through the same Sim, EKF/UKF, Fit and code-generation pipeline as the rigid
body. Electrical edges are separate from mechanical mounting:

```python
from manta.parts import (
    ConstantPowerLoad, DCConverter, DCSource, ElectricalBus,
)

source = craft.add(DCSource(
    "battery_terminal", open_circuit_voltage=50.4,
    source_resistance=0.06, capacitance=5.0, current_limit=80.0,
    brownout_voltage=34.0, recovery_voltage=38.0,
))
bus = craft.add(ElectricalBus(
    "main_bus", rail_voltage=48.0, capacitance=0.5,
    series_resistance=0.01, input_current_limit=60.0,
))
regulator = craft.add(DCConverter(
    "computer_regulator", output_voltage=12.0, capacitance=0.2,
    efficiency=0.92, output_current_limit=15.0,
    output_power_limit=160.0, input_power_limit=180.0,
    brownout_voltage=9.0, recovery_voltage=10.0,
))
computer = craft.add(ConstantPowerLoad(
    "computer", power=120.0, current_limit=15.0,
    voltage_floor=1.0, brownout_voltage=8.0, recovery_voltage=10.0,
))

source.connect(bus)
bus.connect(regulator)
regulator.connect(computer)
```

The mechanical parts may be siblings, nested composites, or mounted far apart.
`connect` alone defines the electrical graph. A transform snapshot rejects:

- a non-source without exactly one upstream supply;
- a second upstream supply;
- a cycle;
- a source used as a child;
- an endpoint load used as a supply; and
- an electrical edge between two craft.

These constraints make the topology a directed radial forest. Demand can be
evaluated from leaves to roots without an algebraic loop, keeping the generated
tick compact and deterministic.

## Equations and diagnostics

Each energized rail owns a capacitor state:

\[
C\dot V = I_{in} + I_{noise} - I_{out}.
\]

`DCSource` charges its terminal capacitor from a current-limited Thevenin
reservoir. `ElectricalBus`, `Contactor`, and `Fuse` receive current through a
series resistance. `DCConverter` injects bounded current into its output
capacitor while respecting dropout, efficiency, input power, output power and
output current limits.

Endpoint loads are available as constant resistance, current and power. Every
load has a C1 brownout gate: it is exactly off below `brownout_voltage`, exactly
on above `recovery_voltage`, and smoothly transitions between them. Constant
power loads also have a denominator floor and optional current limit, so their
current cannot become singular during voltage collapse.

Every node exposes the same outputs:

| Output | Meaning |
| --- | --- |
| `voltage` | Local terminal or output-rail voltage, V |
| `input_current`, `input_power` | Flow entering from the parent/reservoir, A/W |
| `output_current`, `output_power` | Flow delivered to children or consumed by a load, A/W |
| `loss_power` | Series/conversion loss, W |
| `brownout`, `open`, `tripped` | Explicit unit-valued condition signals |
| `kcl_residual` | Capacitor current-balance residual, A |
| `energy_residual` | Input minus output, loss and stored power, W |

The residual outputs are useful assertions in simulation and Monte Carlo runs.
`dissipated_heat()` returns the node's loss for
`ThermalMass(source=electrical_part)`. Loads also accept a `heat_fraction` for
the share of useful endpoint power that becomes local heat.

Fuse state is a normalized I²t accumulator. Once it reaches one the fuse
latches open. A contactor's `closed` input and a source/converter's `enabled`
input are normalized commands. These switching surfaces and current limits are
piecewise differentiable: CasADi provides the derivative on either side, while
the derivative at the exact hybrid event is not physically meaningful.

## Timestep and positivity

Rail voltage uses explicit Euler. Manta does not silently clip a negative
voltage or capacitor energy: that would conceal an invalid integration step and
produce a false conservation result. Choose `dt` against the fastest electrical
time constant.

For a rail with capacitance `C`, lowest resolved voltage `V_min`, and maximum
net discharge `I_max`, the conservative one-step positivity condition is:

\[
dt < C V_{min} / I_{max}.
\]

`rail.stable_timestep_hint(minimum_voltage=..., maximum_net_current=...)`
calculates that local bound. For a linear RC mode, explicit Euler's absolute
stability limit is `dt < 2 R C`; staying below `R C` also avoids a one-step
sign change. Interconnected rails must use the fastest effective edge time
constant, not merely the slowest load.

The representative 50.4 V (12-series-equivalent) network benchmark uses a
1 ms step. Faster switching detail is intentionally averaged into regulator
efficiency and capacitance; modeling PWM edges belongs in a different tool.

Electrical diagnostic outputs are plant observables, not noisy device sensors.
Construct an EKF/UKF with `sensors=[]` (or an explicit real sensor subset) when
estimating rail states. Add a separate voltage/current sensor part when a
physical measurement and covariance are required.
