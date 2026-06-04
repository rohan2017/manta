"""Sim — the forward-dynamics transform of a World.

`Sim(world)` validates the model and linearizes the compiled world tick
(via `LinearizedSystem`); `sim.module(noise=…)` emits the typed `Module` a
backend lowers. The with/without-noise choice is made HERE, at IR
construction — the two are different Modules and lowering just lowers:

* ``module(noise=True)`` — the **oracle** (simulation truth): one `step`
  entry, the full forward tick — it advances the state AND returns every
  sensor reading, all from one noise draw (so a driven run's state and
  readings share the same realization; pass zeros for a noiseless oracle)::

      step(x; u, noise, dt, t) -> x', readings…

* ``module(noise=False)`` — the **deploy** shape (what runs on a robot
  against real sensors): noiseless forward map, per-sensor measurement
  models, and their Jacobians::

      predict(x; u, dt, t) -> x'           predict_jacobian -> F
      measure_<s>(x; u, t) -> reading      measure_<s>_jacobian -> H

State is THREADED (the caller owns it; nothing is held). Run it::

    sim   = TargetNumpy(Sim(w))               # lowers module(noise=True)
    state = sim.initial_state()
    state = sim.step(state, dt=0.01)           # next state (truth)
    sim.outputs()                              # this step's readings

    TargetCpp(Sim(w).module(noise=False), out, class_name="Drone")
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import casadi as ca
import numpy as np

from .ir.module import (
    EntryPoint, Hosting, Module, Port, PortField, PortRef, Role, StateField,
    StateLayout, StateRef,
)
from .ir.state_spec import flatten_nested
from .linearized_system import LinearizedSystem

if TYPE_CHECKING:
    from .world import World


class Sim:
    """Forward-dynamics transform: model validation + the linearized tick,
    emitting oracle/deploy Modules."""

    def __init__(self, world: "World") -> None:
        # Model validation (planet prep, requires_fields/requires_planet,
        # craft back-pointers) happens inside LinearizedSystem — the one
        # choke point every transform passes through.
        self._sys = LinearizedSystem(world)     # full state, all sensors
        self.world = world
        self.crafts = self._sys.crafts

    # ------------------------------------------------------------------

    @property
    def sys(self) -> LinearizedSystem:
        return self._sys

    @property
    def tick(self):
        """The compiled world tick (named CasADi I/O)."""
        return self._sys.tick

    def module(self, noise: bool = True) -> Module:
        """Emit the typed Module — oracle (`noise=True`) or deploy."""
        sys = self._sys
        spec = sys.spec
        init_flat = flatten_nested(self.world._initial_state_dict())
        x0 = spec.pack_any(init_flat)
        x_field = StateField("x", "manifold", (spec.ambient_dim,),
                             init=x0, manifold=spec)
        # A command's declared default is the MODEL's initial value: an
        # `add_craft(..., **{"t.throttle": x0})` override wins over the
        # Part-declared default.
        u_port = Port("u", Role.CONTROL, (len(sys.input_names),),
                      fields=tuple(
                          PortField(n, 1, float(np.asarray(init_flat.get(
                              n, sys.input_defaults[n])).ravel()[0]),
                                    rate=sys.sample_rates.get(n))
                          for n in sys.input_names))
        dtp, tp = Port("dt", Role.TIMESTEP), Port("t", Role.TIME)
        meas_ports = [Port(full, Role.MEASUREMENT, (s.dim,),
                           rate=sys.sample_rates.get(full))
                      for full, s in sys.sensors.items()]
        sensor_fulls = list(sys.sensors)

        if noise:
            # Oracle: the full forward tick, one noise draw → state + readings.
            noise_port = Port(
                "noise", Role.NOISE, (sys.n_noise,),
                fields=tuple(PortField(c.full, c.dim, 0.0, sigma=c.sigma)
                             for c in sys.noise_specs))
            step_fn = ca.Function(
                "step",
                [sys.x_sym, sys.u_sym, sys.n_sym, sys.dt_sym, sys.t_sym],
                [sys.x_new_noisy] + [
                    ca.reshape(sys.sensors[f].h_noisy_sym,
                               sys.sensors[f].dim, 1) for f in sensor_fulls],
                ["x", "u", "noise", "dt", "t"],
                ["x_new"] + [f.replace(".", "_") for f in sensor_fulls])
            return Module(
                name=self.world.name, state=StateLayout((x_field,)),
                ports=(u_port, noise_port, dtp, tp, *meas_ports),
                functions={"step": step_fn},
                entry_points=(EntryPoint(
                    "step", "step",
                    (StateRef("x"), PortRef("u"), PortRef("noise"),
                     PortRef("dt"), PortRef("t")),
                    writes=("x",), returns=tuple(sensor_fulls)),),
                hosting=Hosting.THREADED)

        # Deploy: noiseless forward map + measurement models + Jacobians.
        # A measurement is dt-independent — dt is eliminated at construction,
        # so the measure kernels honestly take (x, u, t).
        tan = spec.tangent_dim
        zero_dt = ca.MX.zeros(1, 1)
        functions = {"predict": sys.predict_fn, "predict_jacobian": sys.F_fn}
        ports = [u_port, dtp, tp, *meas_ports,
                 Port("F", Role.MATRIX, (tan, tan))]
        entries = [
            EntryPoint("predict", "predict",
                       (StateRef("x"), PortRef("u"), PortRef("dt"),
                        PortRef("t")), writes=("x",)),
            EntryPoint("predict_jacobian", "predict_jacobian",
                       (StateRef("x"), PortRef("u"), PortRef("dt"),
                        PortRef("t")), returns=("F",)),
        ]
        margs = [sys.x_sym, sys.u_sym, sys.t_sym]
        margn = ["x", "u", "t"]
        for full, s in sys.sensors.items():
            ident = full.replace(".", "_")
            h = ca.substitute(s.h_sym, sys.dt_sym, zero_dt)
            H = ca.substitute(s.H_sym, sys.dt_sym, zero_dt)
            functions[f"measure_{ident}"] = ca.Function(
                f"h_{ident}", margs, [h], margn, ["h"])
            functions[f"measure_{ident}_jacobian"] = ca.Function(
                f"H_{ident}", margs, [H], margn, ["H"])
            ports.append(Port(f"H_{ident}", Role.MATRIX, (s.dim, tan)))
            entries.append(EntryPoint(
                f"measure_{ident}", f"measure_{ident}",
                (StateRef("x"), PortRef("u"), PortRef("t")),
                returns=(full,)))
            entries.append(EntryPoint(
                f"measure_{ident}_jacobian", f"measure_{ident}_jacobian",
                (StateRef("x"), PortRef("u"), PortRef("t")),
                returns=(f"H_{ident}",)))
        return Module(
            name=self.world.name, state=StateLayout((x_field,)),
            ports=tuple(ports), functions=functions,
            entry_points=tuple(entries), hosting=Hosting.THREADED)

    def __repr__(self) -> str:
        names = ", ".join(c.name for c in self.crafts)
        return f"<Sim crafts=[{names}]>"
