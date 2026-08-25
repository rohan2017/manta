"""Tick compilation — model → CasADi tick function.

This package is the model→IR boundary: it traces a `World`'s crafts (the
declarative part trees) into a single CasADi tick `ca.Function`, then lets
callers read that function's signature back out.

  * `world_tick.compile_world_tick` — the sole tick compiler. Runs the
    unified kinematic pass + symbolic inertia rollup over every craft +
    coupling and assembles the Newton-Euler tick.
  * `kinematics.kinematic_pass` / `inertia.symbolic_inertia_rollup` — the
    two symbolic sub-passes it composes (per-part frame-indexed kinematics;
    mass-weighted inertia aggregation).
  * `tick_signature.walk_tick_signature` — classifies a compiled tick's
    inputs/outputs (Input vs Noise vs state vs dt/t; sensors), the shared
    metadata the EKF / LQR / backends route on.

Downstream, the IR transforms (`Sim`, `EKF`, `LQR`) consume the compiled
tick; they live a layer up, not here.
"""

from .tick_signature import walk_tick_signature
from .world_tick import compile_world_tick

__all__ = ["compile_world_tick", "walk_tick_signature"]
