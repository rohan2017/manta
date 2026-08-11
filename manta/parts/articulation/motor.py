"""Motor — a voltage-commanded DC motor driving a revolute DOF.

A `Motor` is a `RevoluteDOF`: kinematically identical to a
`RevoluteJoint` (children ride its rotating output frame, the
joint-space solve produces its q̈), but its generalized DOF torque comes
from the classic quasi-static DC-motor electrical model instead of a
direct torque command:

    ω_m   = gear_ratio · rate                    (motor-shaft speed)
    i     = (voltage − k·ω_m) / resistance       (armature current)
    i     ← clamp(i, ±current_limit)             (driver/thermal limit)
    τ_out = gear_ratio · k · i                   (shaft torque, geared)

with one SI motor constant `k = torque_constant` serving as both the
torque constant (N·m/A) and the back-EMF constant (V·s/rad) — equal in
SI units for an ideal machine. The model captures the linear
torque-speed curve every real motor has and a direct torque command
does not:

  * stall torque `k·V/R` at zero speed,
  * zero torque at the no-load speed `ω_nl = V/k`,
  * regenerative/dynamic braking past it (and at `voltage = 0`, where
    the shorted winding brakes the shaft with `τ = −k²·ω_m/R` — the
    behavior of a PWM driver's off-phase, so an unpowered Motor spins
    down instead of freewheeling).

Quasi-static means the winding inductance is neglected (`L·di/dt ≈ 0`):
the electrical time constant of small machines is orders of magnitude
below the mechanical one, so current is an algebraic function of state
and no extra (stiff) electrical state slot is needed.

`rate` is the joint rate — rotor *relative to* the mounting body —
which is exactly what the physical back-EMF sees (rotor vs stator), so
the model stays correct on a tumbling craft.

`torque_constant` and `resistance` are promotable (`manifold="R1"`) —
both are classic system-ID targets (`Fit(world,
parameters=["m.torque_constant", ...])`).

There is no `mode`: the electrical model is always live. Friction is
the inherited `damping` (viscous, at the output shaft).
"""

from __future__ import annotations

import math

import casadi as ca

from .._declarations import Input, Parameter
from .._trace import scalar_mx as _mx
from .joint import RevoluteDOF


class Motor(RevoluteDOF):
    """Voltage-commanded DC motor on a revolute DOF.

    Parameters:
        axis            — input-frame unit vector along the rotation
                          axis. Default (0, 0, 1).
        torque_constant — SI motor constant k: torque constant (N·m/A)
                          and back-EMF constant (V·s/rad). Promotable.
                          Default 0.05.
        resistance      — winding resistance R (Ω). Promotable.
                          Default 1.0.
        current_limit   — armature-current clamp magnitude (A), the
                          driver/thermal limit. Default inf (no limit).
        gear_ratio      — motor-shaft turns per output-shaft turn
                          (reduction ratio ≥ 1 gears torque up).
                          Default 1.0 (direct drive).
        damping         — viscous friction at the output shaft
                          (N·m·s/rad). Default 0.

    Inputs:
        voltage         — terminal voltage V (signed; negative reverses).

    State:
        angle           — output-shaft angle, rad.
        rate            — output-shaft rate relative to the mounting
                          body, rad/s.
    """

    torque_constant: float = Parameter(0.05, manifold="R1")
    resistance:      float = Parameter(1.0, manifold="R1")
    current_limit:   float = Parameter(math.inf, allow_infinite=True)
    gear_ratio:      float = Parameter(1.0)
    voltage:         float = Input(default=0.0)

    def __init__(self, name: str, **overrides) -> None:
        if "mode" in overrides:
            raise TypeError(
                f"Motor {name!r}: no `mode` — the electrical model is "
                f"always live (voltage=0 gives dynamic braking through "
                f"the winding). Use a RevoluteJoint for passive/"
                f"saturating torque semantics.")
        super().__init__(name, **overrides)
        for attr in ("torque_constant", "resistance", "current_limit",
                     "gear_ratio"):
            if float(self.declared_value(attr)) <= 0.0:
                raise ValueError(
                    f"Motor {name!r}: {attr} must be > 0, got "
                    f"{self.declared_value(attr)!r}")

    def _current_mx(self) -> ca.MX:
        """Armature current i = (V − k·ω_m)/R, clamped to
        ±current_limit, as a raw MX scalar."""
        _, rate_name = self.dof_state_names()
        rate_mx = _mx(getattr(self, rate_name))
        k = _mx(self.torque_constant)
        R = _mx(self.resistance)
        G = float(self.declared_value("gear_ratio"))
        i = (_mx(self.voltage) - k * (G * rate_mx)) / R
        i_max = float(self.declared_value("current_limit"))
        if math.isfinite(i_max):
            i = ca.fmin(ca.fmax(i, -i_max), i_max)
        return i

    def applied_dof_force(self):
        """Electrical shaft torque on top of the base viscous damping, as
        a raw MX scalar — this joint's generalized-force row entry (see
        the base class for the reaction-bookkeeping argument)."""
        k = _mx(self.torque_constant)
        G = float(self.declared_value("gear_ratio"))
        return super().applied_dof_force() + G * k * self._current_mx()

    def dissipated_heat(self) -> ca.MX:
        """Winding copper loss i²·R (W) — the `ThermalMass` heat-source
        protocol (`ThermalMass("winding", source=motor)`). Everything
        the electrical model wastes: at stall the full V²/R, at no-load
        speed ~0."""
        i = self._current_mx()
        return i * i * _mx(self.resistance)
