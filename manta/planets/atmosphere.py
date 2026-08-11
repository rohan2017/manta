"""Re-export shim — the fluid-column physics moved to
`manta.fields.fluid_props`.

The equations here are planet-independent (its own docstring always said
so): pure symbolic/numeric scalar relations that planet-free fluid
disturbances (`UniformFluid`, `FlatOcean`) need just as much as the
planet regimes do. Hosting them under `manta.planets` forced
`manta.fields.fluid` into a lazy runtime import to keep the
fields↔planets dependency one-directional, so the content now lives in
`manta.fields.fluid_props` and this module only re-exports the public
names for existing importers.
"""

from __future__ import annotations

from ..fields.fluid_props import (
    LAPSE_ISA, MU_REF_AIR, P0_ISA, R_AIR, RHO0_ISA, S_SUTHERLAND_AIR,
    T0_ISA, T_REF_SUTHERLAND,
    hydrostatic_pressure, ideal_gas_density, isa_pressure, isa_temperature,
    sutherland_viscosity,
)

__all__ = [
    "LAPSE_ISA", "MU_REF_AIR", "P0_ISA", "R_AIR", "RHO0_ISA",
    "S_SUTHERLAND_AIR", "T0_ISA", "T_REF_SUTHERLAND",
    "hydrostatic_pressure", "ideal_gas_density", "isa_pressure",
    "isa_temperature", "sutherland_viscosity",
]
