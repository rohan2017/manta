"""ThermalMass — a lumped thermal node with conduction links.

Lumped-capacitance heat transfer on the part tree — a pure *network*
model with no spatial aspect: every heat path is a named coefficient,
not a field query. Each `ThermalMass` is one thermal node holding a
`temperature` State (K) behind a heat capacity C (J/K). Heat flows into
the node from four places, and the temperature advances by explicit
Euler each tick:

    C · dT/dt =   heat_input                          (external, W — an Input)
                + Σ_links  k_link · (T_other − T)      (conduction network)
                + g_amb · (ambient_temperature − T)    (loss to surroundings)
                + source.dissipated_heat()             (a heat-generating part)
                + heat_noise                           (white W, EKF auto-Q)

**Conduction network** — `a.connect(b, conductance=k)` registers a
symmetric link (W/K): each side gains `k·(T_other − T_self)`, so the
exchange conserves energy by construction. The coefficient *is* the
joint: a metal bracket is a high-k link, a plastic standoff a low-k
link, a heat pipe an extremely-high-k link (give the pipe its own
`ThermalMass` in the chain if its capacitance matters). One `connect`
call wires both directions; parallel heat paths between the same pair
are extra `connect` calls *from the same endpoint* (a reciprocal
`b.connect(a, …)` raises — it would silently double the conductance).
Links join nodes on the *same craft* (thermal structure is craft
structure; validated at `World.finalize()`).

**Ambient exchange** — with `ambient_conductance > 0` the node leaks to
a boundary temperature chosen by `ambient=`:

  * `"input"` (default) — an `ambient_temperature` **Input** (K): a
    plain boundary condition, no world field involved. As an Input it
    persists between steps and can be rewritten from the driving loop
    (`sim.state["c"]["n.ambient_temperature"] = ...`) — script it
    against depth or altitude if the environment matters.
  * `"fluid"` — the world FluidField's `temperature` component,
    evaluated at the part's own world position: a hull node feels the
    water it is actually in, and `Earth`'s ocean/atmosphere regimes
    already carry mutually-consistent temperatures. Opt-in on purpose:
    a density-only fluid (declared just for drag) leaves `temperature`
    at its unset 0 K, and coupling to that silently freezes the model —
    so make sure every regime the craft traverses declares a real
    temperature, and that regimes cover the trajectory (outside all
    memberships the composite tends to 0 K).

With `ambient_conductance == 0` (default, insulated), or with
`ambient="fluid"`, the `ambient_temperature` input is not plumbed into
the graph at all.

**Heat generation** — any part implementing `dissipated_heat() -> MX`
(watts, symbolic; `Motor` ships one: the winding's i²R loss) can be
passed as `source=`: the node reads the loss straight from the source's
traced state, so an electrical model and its thermal consequence stay
one graph — stall the motor and the winding node heats at V²/R with no
extra wiring.

Stability: explicit Euler is exact enough when `dt` is small against
every node's time constant (C / Σk of its conductances) — the same
discipline the rigid-body integrator already imposes.

Temperatures are estimable for free: each node's `temperature` is an R1
state slot, and `heat_noise` (σ in W) gives the EKF its process-noise
channel. Observing them needs a sensor part with a temperature Output
(not shipped yet).
"""

from __future__ import annotations

from typing import Any

import casadi as ca

from ...ir.frames import PartFrame
from ...ir.types import Scalar, Vec3
from ...ir.wrench import Wrench
from .._declarations import Input, Parameter, PartUpdate, State, WhiteNoise
from .._trace import scalar_mx as _mx
from ..base import Part


def _root_of(part: Part) -> Part:
    node = part
    while node.parent is not None:
        node = node.parent
    return node


class ThermalMass(Part):
    """A lumped thermal node: heat capacity + temperature state +
    conduction/boundary/generation heat flows.

    Parameters:
        heat_capacity       — C, J/K. Promotable (a classic sysid
                              target). Default 100.
        ambient_conductance — g_amb, W/K leak to the ambient boundary.
                              0 (default) = insulated: no ambient term,
                              and the ambient input is not plumbed.
        ambient             — where the boundary temperature comes from:
                              "input" (the ambient_temperature Input) or
                              "fluid" (the world FluidField's
                              temperature at the part's position —
                              requires a registered FluidField whose
                              regimes declare temperatures).

    Inputs:
        heat_input          — external heat into the node, W (signed).
        ambient_temperature — boundary temperature the node leaks to, K.
                              Only bound when ambient_conductance > 0
                              AND ambient="input". Default 293.15;
                              rewrite per tick from the driving loop to
                              script an environment.

    State:
        temperature         — node temperature, K. Init 293.15; set a
                              per-run value via
                              `add_craft(..., **{"node.temperature": T0})`.

    Noise:
        heat_noise          — white heat-flow noise, W (σ default 0 —
                              inert). Gives the EKF an auto-Q channel
                              for the temperature slot.

    Construction:
        source              — optional Part implementing
                              `dissipated_heat() -> MX` (W); its loss is
                              added to this node's balance each tick.

    `connect(other, conductance=k)` links two nodes with a k W/K
    conductance (symmetric; both must ride the same craft).
    """

    heat_capacity:       float = Parameter(100.0, manifold="R1")
    ambient_conductance: float = Parameter(0.0)
    ambient:             str   = Parameter("input")

    heat_input:          float = Input(default=0.0)
    ambient_temperature: float = Input(default=293.15)
    temperature = State(init=293.15, manifold="R1")
    heat_noise = WhiteNoise("R1", sigma=0.0)

    def __init__(self, name: str, *, source: Part | None = None,
                 **overrides: Any) -> None:
        super().__init__(name, **overrides)
        who = f"ThermalMass {name!r}"
        if float(self.declared_value("heat_capacity")) <= 0.0:
            raise ValueError(
                f"{who}: heat_capacity must be > 0, got "
                f"{self.declared_value('heat_capacity')!r}")
        if float(self.declared_value("ambient_conductance")) < 0.0:
            raise ValueError(
                f"{who}: ambient_conductance must be ≥ 0, got "
                f"{self.declared_value('ambient_conductance')!r}")
        if self.declared_value("ambient") not in ("input", "fluid"):
            raise ValueError(
                f"{who}: ambient must be 'input' or 'fluid', got "
                f"{self.declared_value('ambient')!r}")
        if source is not None and not callable(
                getattr(source, "dissipated_heat", None)):
            raise TypeError(
                f"{who}: source {type(source).__name__}"
                f"('{getattr(source, 'name', '?')}') has no "
                f"dissipated_heat() — a heat source must implement "
                f"`dissipated_heat() -> MX` (watts).")
        self._source = source
        # (other_node, conductance, initiated) triples; each link is
        # registered on BOTH endpoints so each side's balance sees the
        # exchange, and `initiated` records which side made the
        # `connect` call — the reciprocal-call guard keys on it.
        self._links: list[tuple["ThermalMass", float, bool]] = []

    # ---- per-instance I/O (camera precedent) --------------------------

    def input_declarations(self):
        """The ambient_temperature input exists only when it is the
        live boundary: an insulated node (ambient_conductance == 0) has
        no ambient term, and a fluid-coupled node reads the FluidField
        instead — drop it so it never reaches the u port."""
        decls = dict(super().input_declarations())
        if (float(self.declared_value("ambient_conductance")) <= 0.0
                or self.declared_value("ambient") == "fluid"):
            decls.pop("ambient_temperature", None)
        return decls

    # ---- conduction network -------------------------------------------

    def connect(self, other: "ThermalMass", *,
                conductance: float) -> "ThermalMass":
        """Register a symmetric conduction link to `other` (W/K). One
        call wires BOTH directions — do not also call
        `other.connect(self, ...)`; the natural-looking reciprocal call
        would silently double the conductance, so it raises instead.
        Deliberate parallel heat paths are still expressed by calling
        `connect` again *from the same endpoint*. Returns self for
        chaining."""
        who = f"ThermalMass {self.name!r}.connect"
        if not isinstance(other, ThermalMass):
            raise TypeError(
                f"{who}: expected a ThermalMass, got "
                f"{type(other).__name__}")
        if other is self:
            raise ValueError(f"{who}: cannot connect a node to itself")
        k = float(conductance)
        if k <= 0.0:
            raise ValueError(
                f"{who}: conductance must be > 0 W/K, got {conductance!r}")
        if any(o is other and not initiated
               for o, _, initiated in self._links):
            raise ValueError(
                f"{who}: {other.name!r} already connected this pair from "
                f"its side — one connect() call wires both directions, "
                f"and a reciprocal call would double the conductance. "
                f"For a deliberate parallel path, call connect() again "
                f"from the same endpoint that made the first call.")
        self._links.append((other, k, True))
        other._links.append((self, k, False))
        return self

    @property
    def links(self) -> tuple[tuple["ThermalMass", float], ...]:
        return tuple((o, k) for o, k, _ in self._links)

    # ---- compile-time validation --------------------------------------

    def _require_same_craft(self, other: Part, what: str) -> None:
        if _root_of(other) is not _root_of(self):
            raise ValueError(
                f"ThermalMass '{self.name}': {what} "
                f"'{getattr(other, 'name', '?')}' rides a different part "
                f"tree. Thermal links and heat sources must live on the "
                f"same craft — cross-craft heat exchange isn't modeled.")

    def on_world_finalize(self, world, craft) -> None:
        """Validate the thermal network's structure once, at finalize —
        before any tracing — instead of erroring mid-trace: every linked
        node and the heat source must ride this node's craft."""
        for other, _, _ in self._links:
            self._require_same_craft(other, "linked node")
        if self._source is not None:
            self._require_same_craft(self._source, "heat source")

    # ---- tick ---------------------------------------------------------

    def update(self, ctx) -> PartUpdate:
        T = _mx(self.temperature)
        C = _mx(self.heat_capacity)

        q_dot = _mx(self.heat_input) + _mx(self.heat_noise)

        # Structure (same-craft links/source) was validated by
        # `on_world_finalize` — the trace only builds the balance.
        for other, k, _ in self._links:
            q_dot = q_dot + k * (_mx(other.temperature) - T)

        if self._source is not None:
            q_dot = q_dot + self._source.dissipated_heat()

        g_amb = float(self.declared_value("ambient_conductance"))
        if g_amb > 0.0:
            if self.declared_value("ambient") == "fluid":
                from ...fields.fluid import FluidField
                from ...ir.frames import WorldFrame
                if not ctx.has_field(FluidField):
                    raise ValueError(
                        f"ThermalMass '{self.name}': ambient='fluid' "
                        f"reads the world FluidField's temperature, but "
                        f"no FluidField is registered. add_field one "
                        f"(with regimes that declare temperatures), or "
                        f"use ambient='input'.")
                fluid = ctx.field(FluidField)
                # A registered-but-empty field composites to 0 K —
                # exactly as wrong as no field. Demand ≥ 1 regime.
                if not fluid.disturbances:
                    raise ValueError(
                        f"ThermalMass '{self.name}': ambient='fluid', "
                        f"but the registered FluidField has no fluid "
                        f"regimes — its temperature composite would be "
                        f"0 K. Add a regime that declares a "
                        f"temperature, or use ambient='input'.")
                T_amb = fluid.value_at_sym(
                    ctx.position[WorldFrame], ctx.t).temperature
            else:
                T_amb = _mx(self.ambient_temperature)
            q_dot = q_dot + g_amb * (T_amb - T)

        T_next = T + _mx(ctx.dt) * q_dot / C

        zero = Vec3[PartFrame].constant((0.0, 0.0, 0.0))
        return PartUpdate(wrench=Wrench(force=zero, torque=zero),
                          new_state={"temperature": Scalar(T_next)})
