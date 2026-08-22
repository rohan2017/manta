"""Tick-signature walker — classify a compiled world tick's I/O.

The world tick (`compile_world_tick`) exposes flat-prefixed inputs and
outputs. Every transform needs the *same* classification of that
signature against the model: which inputs are live Part Inputs (→ the
control vector `u`), which are stochastic Noise drivers (→ process
noise), and which Part Outputs are candidate measurements. This module
is the single place that walks the signature; its result is what
`LinearizedSystem` consumes.

`spec` membership answers "is this input a state slot?"; everything else
is a per-owner Input, a Noise driver, or (for outputs) a sensor. Owners
are resolved from the `<owner>.<sub>` prefix — a craft name, or a
state-bearing field disturbance name.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class InputChannel:
    """A live Part Input → one entry of the control vector `u`."""
    full:    str      # tick input name, "<craft>.<part>.<input>"
    owner:   Any      # the owning Part
    name:    str      # the Input sub-name
    default: float    # part-instance default value


@dataclass(frozen=True)
class NoiseChannel:
    """A stochastic Noise driver feeding the tick (white, random-walk, or
    Gauss–Markov driver)."""
    full:  str        # tick input name
    owner: Any        # owning Part or Disturbance
    name:  str        # the driver-input sub-name
    dim:   int        # signal dimension
    sigma: float      # per-channel std-dev (read from the owner)


@dataclass(frozen=True)
class SensorOutput:
    """A Part Output → a candidate measurement."""
    full:        str  # "<craft>.<part>.<output>"
    craft:       Any
    part:        Any
    output_name: str


@dataclass(frozen=True)
class ParameterChannel:
    """A promoted (tunable) Parameter feeding the tick as a live input —
    one block of the parameter vector `p` (system ID)."""
    full:  str        # tick input name, "<craft>.<part>.<param>"
    owner: Any        # the owning Part
    name:  str        # the Parameter sub-name
    dim:   int        # ambient dimension (1 scalar, 3 vec)
    value: Any        # declared numeric value (np.ndarray, flat)


@dataclass(frozen=True)
class TickSignature:
    """The classified I/O of a compiled world tick.

    `inputs` / `noise` / `params` are in cf-input-signature order;
    `sensors` in `world.crafts → parts → outputs` order. The consumers
    (`EKF`, the C++ extractor) build their `u` vector / noise layout /
    parameter vector / sensor table from these in order."""
    inputs:  list[InputChannel]
    noise:   list[NoiseChannel]
    sensors: list[SensorOutput]
    params:  list[ParameterChannel]

    @property
    def input_names(self) -> list[str]:
        return [ic.full for ic in self.inputs]

    @property
    def input_defaults(self) -> dict[str, float]:
        return {ic.full: ic.default for ic in self.inputs}

    @property
    def sensor_names(self) -> list[str]:
        return [s.full for s in self.sensors]

    @property
    def param_names(self) -> list[str]:
        return [p.full for p in self.params]


def _dist_by_name(world) -> dict:
    """Index every state-bearing field disturbance by name."""
    from ..fields.base import Disturbance
    out: dict[str, Any] = {}
    for field in world.fields:
        for d in field._disturbances:
            if isinstance(d, Disturbance):
                out[d.name] = d
    return out


def _match_noise(owner, sub: str):
    """`(nname, ndecl)` for the noise channel whose driver input equals
    `sub`, else None. Polymorphic on the Noise subclass — each declares
    its own driver-input naming via `driver_input_name(name)`."""
    for nname, ndecl in owner.noise_declarations().items():
        if ndecl.driver_input_name(nname) == sub:
            return nname, ndecl
    return None


def walk_tick_signature(cf, world, spec) -> TickSignature:
    """Classify a compiled world tick's inputs + outputs against `world`.

    Each non-state, non-`dt`/`t` input routes to a live Part Input (in
    cf-signature order → `u`) or a Noise driver; `spec` membership marks
    state slots (skipped). Every Part Output is enumerated as a candidate
    sensor. Raises if an input matches neither an Input nor a Noise
    channel, or references an unknown owner/part.
    """
    import numpy as np

    dist_by_name = _dist_by_name(world)
    inputs: list[InputChannel] = []
    noise:  list[NoiseChannel] = []
    params: list[ParameterChannel] = []

    for i in range(cf.n_in()):
        name = cf.name_in(i)
        if name in ("dt", "t") or name in spec:
            continue
        head, _, rest = name.partition(".")
        craft = next((c for c in world.crafts if c.name == head), None)
        if craft is not None:
            # Per-craft Input or Noise → `<craft>.<part>.<sub>`.
            if "." not in rest:
                raise RuntimeError(
                    f"tick input {name!r} not in spec and not in the "
                    f"`<craft>.<part>.<sub>` shape.")
            part_name, sub = rest.split(".", 1)
            part = next((p for p in craft.parts if p.name == part_name), None)
            if part is None:
                raise RuntimeError(
                    f"tick input {name!r}: unknown part {part_name!r} "
                    f"on craft {craft.name!r}.")
            if sub in part.input_declarations():
                inputs.append(InputChannel(
                    full=name, owner=part, name=sub,
                    default=float(getattr(part, sub))))
                continue
            pdecls = part.promotable_parameter_declarations()
            if sub in pdecls:
                params.append(ParameterChannel(
                    full=name, owner=part, name=sub,
                    dim=pdecls[sub].manifold.ambient_dim,
                    value=np.atleast_1d(np.asarray(
                        part.declared_value(sub), dtype=float)).ravel()))
                continue
            owner, owner_label = part, f"craft '{craft.name}'"
        else:
            # Per-disturbance Noise (state slots already handled by spec).
            dist = dist_by_name.get(head)
            if dist is None:
                raise RuntimeError(
                    f"tick input {name!r} doesn't match any craft or "
                    f"registered disturbance.")
            owner, sub, owner_label = dist, rest, f"disturbance '{dist.name}'"

        hit = _match_noise(owner, sub)
        if hit is None:
            raise RuntimeError(
                f"tick input {name!r} on {owner_label} is neither an Input "
                f"nor a recognized Noise channel.")
        nname, ndecl = hit
        noise.append(NoiseChannel(
            full=name, owner=owner, name=sub,
            dim=ndecl.signal_manifold.ambient_dim,
            sigma=float(getattr(owner, f"{nname}_sigma"))))

    sensors: list[SensorOutput] = []
    for craft in world.crafts:
        for part in craft.parts:
            for out_name in part.output_declarations():
                sensors.append(SensorOutput(
                    full=f"{craft.name}.{part.name}.{out_name}",
                    craft=craft, part=part, output_name=out_name))

    return TickSignature(inputs=inputs, noise=noise, sensors=sensors,
                         params=params)
