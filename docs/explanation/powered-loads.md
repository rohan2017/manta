# Powered mechanical parts

Manta's powered variants are opt-in endpoints on the radial DC network. The
ordinary `Motor`, `Thruster`, `DuctedPropeller`, and `ControlSurface` keep their
existing direct command behavior. Selecting a powered class is therefore an
explicit vehicle-model migration, not a global change in actuator semantics.

```python
from manta.parts import DCSource, PoweredMotor

pack = craft.add(DCSource("pack", open_circuit_voltage=48.0))
fin = craft.add(PoweredMotor(
    "fin_motor",
    torque_constant=0.08,
    resistance=0.6,
    brownout_voltage=30.0,
    recovery_voltage=34.0,
))
pack.connect(fin)
```

`PoweredMotor` preserves the motor's signed `voltage` command. Its winding
voltage is clipped to the available rail and fades through the configured
brownout interval. Armature current, geared torque, useful shaft power, winding
heat, controller loss, and non-regenerative braking heat come from the same DC
motor equations. `input_power = mechanical_power + loss_power`; connect one
`ThermalMass(source=motor)` to `dissipated_heat()` to capture the loss. Do not
also add winding loss as a separate `heat_input`.

`PoweredThruster`, `PoweredDuctedPropeller`, and `PoweredControlSurface` use a
calibrated power-map seam because their underlying primitives contain no motor
shaft speed, winding current, gear ratio, or ESC model. They require explicit
`rated_voltage`, `rated_mechanical_power`, and `conversion_efficiency` values.
They do not infer power from thrust or hinge torque: doing so would invent an
advance ratio or servo gear train that the source model does not contain.
Voltage and brownout derate the physical command, while the calibrated power
map supplies electrical current and loss accounting. Replace these adapters
with a resolved motor/propeller model when shaft data is available.

`ConstantPowerElectronicsLoad` is the compute/device specialization. It uses
the bounded constant-power load law, reports zero mechanical output, and sends
all consumed power through the thermal-source seam.

`ExternalDCSupply` is the boundary for a simulation-only battery or a hardware
source. Its `supplied_voltage` input feeds the compiled electrical/load model;
its `output_current` reports aggregate demand back to the external plant. Cell,
thermal, degradation, and fault states can consequently remain ordinary
simulator state rather than entering Manta's differentiable state vector.

## Boundaries and limitations

- A1 networks are unidirectional. A powered motor absorbs regenerative or
  dynamic-braking energy as heat instead of returning it to the bus.
- Converter overload acts through its current/power limits and output
  capacitor. The resulting rail sag continuously derates powered parts; there
  is no separate mission-policy cutoff hidden in the actuator.
- The calibrated propulsor/servo power output is motor-side useful mechanical
  power, not hydrodynamic propulsive efficiency or aerodynamic work at the
  vehicle. Those require shaft-resolved primitives.
- ESC/PWM protocols, vendor calibration, arming, watchdogs, and failsafe policy
  belong in Shiver. Manta models only the physical plant.
