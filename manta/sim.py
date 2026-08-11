"""Sim — the forward-dynamics transform of a World.

`Sim(world)` validates the model and linearizes the compiled world tick
(via `LinearizedSystem`); it emits two structurally different Modules,
chosen HERE, at IR construction — lowering just lowers:

* ``module()`` — the **oracle** (simulation truth): one `step` entry,
  the full forward tick — it advances the state AND returns every
  sensor reading, all from one noise draw (so a driven run's state and
  readings share the same realization; pass zeros for a noiseless oracle)::

      step(x; u, noise, dt, t) -> x', readings…

* ``deploy_module()`` — the **deploy** shape (what runs on a robot
  against real sensors): noiseless forward map, per-sensor measurement
  models, and their Jacobians::

      predict(x; u, dt, t) -> x'           predict_jacobian -> F
      measure_<s>(x; u, t) -> reading      measure_<s>_jacobian -> H

`Sim(world, parameters=[...])` promotes the named promotable Parameters
(thruster gains, mounts, masses — see `parts._declarations.Parameter`)
to a live `params` port threaded into every kernel above (`step(x; u, noise,
params, dt, t)`, …). Passing the port's declared defaults reproduces
the baked model exactly; `manta.fit` optimizes over it for system ID.

The Module's state is THREADED (a kernel maps state in → state out; a
C++ caller owns its `State` struct). The numpy view (`NumpySim`) holds
the nested state dict for you::

    sim = TargetNumpy(Sim(w))                  # lowers the oracle module()
    sim.state["drone"]["t.throttle"] = 14.7    # mutate the held state
    sim.step(0.01)                             # advance truth
    sim.outputs()                              # this step's readings

    TargetCpp(Sim(w).deploy_module(), out, class_name="Drone")
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import casadi as ca
import numpy as np

from .ir.module import (
    EntryPoint, Hosting, Module, Port, PortField, PortRef, Role, StateField,
    StateLayout, StateRef, entry_ident,
)
from .ir.state_spec import flatten_nested
from .linearization import LinearizedSystem

if TYPE_CHECKING:
    from .world import World


class Sim:
    """Forward-dynamics transform: model validation + the linearized tick,
    emitting oracle/deploy Modules."""

    def __init__(self, world: "World", *,
                 discretization: str = "exact",
                 parameters: list[str] | None = None) -> None:
        # Model validation (planet prep, requires_fields/requires_planet,
        # craft back-pointers) happens inside LinearizedSystem — the one
        # choke point every transform passes through. `discretization`
        # selects how predict_jacobian discretizes F ("exact" | "euler" —
        # see LinearizedSystem; "euler" trades an O(dt²) jacobian
        # difference for much smaller generated deploy code).
        # `parameters` promotes the named promotable Parameters to a live
        # `params` port on every emitted Module (system ID — `manta.fit`);
        # passing the port's declared defaults reproduces the baked model
        # bit-for-bit.
        self._sys = LinearizedSystem(           # full state, all sensors
            world, discretization=discretization, parameters=parameters)
        self.world = self._sys.world
        self.crafts = self._sys.crafts

    # ------------------------------------------------------------------

    @property
    def sys(self) -> LinearizedSystem:
        return self._sys

    @property
    def tick(self):
        """The compiled world tick (named CasADi I/O)."""
        return self._sys.tick

    def _module_scaffold(self):
        """The port/state pieces both Module shapes share."""
        sys = self._sys
        spec = sys.spec
        init_flat = flatten_nested(self.world._initial_state_dict())
        x0 = spec.pack_any(init_flat)
        x_field = StateField("x", "manifold", (spec.ambient_dim,),
                             init=x0, spec=spec)
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
        return x_field, u_port, dtp, tp, meas_ports

    def _param_port(self) -> Port | None:
        """The promoted-parameter port, or None when nothing was promoted.
        Field defaults are the DECLARED values — pass them through and
        the kernel reproduces the baked-constant model exactly."""
        sys = self._sys
        if sys.n_param == 0:
            return None
        return Port("params", Role.PARAMETER, (sys.n_param,),
                    fields=tuple(PortField(p.full, p.dim,
                                           np.asarray(p.value, dtype=float))
                                 for p in sys.param_specs))

    def module(self) -> Module:
        """The **oracle** Module (simulation truth): one `step` entry —
        the full forward tick, one live noise draw → state + readings."""
        sys = self._sys
        x_field, u_port, dtp, tp, meas_ports = self._module_scaffold()
        sensor_fulls = list(sys.sensors)
        noise_port = Port(
            "noise", Role.NOISE, (sys.n_noise,),
            fields=tuple(PortField(c.full, c.dim, 0.0, sigma=c.sigma)
                         for c in sys.noise_specs))
        p_port = self._param_port()
        kargs, kargn = [sys.x_sym, sys.u_sym, sys.n_sym], ["x", "u", "noise"]
        eargs = [StateRef("x"), PortRef("u"), PortRef("noise")]
        if p_port is not None:
            kargs.append(sys.p_sym); kargn.append("params")
            eargs.append(PortRef("params"))
        kargs += [sys.dt_sym, sys.t_sym]; kargn += ["dt", "t"]
        eargs += [PortRef("dt"), PortRef("t")]
        step_fn = ca.Function(
            "step", kargs,
            [sys.x_new_noisy] + [
                ca.reshape(sys.sensors[f].h_noisy_sym,
                           sys.sensors[f].dim, 1) for f in sensor_fulls],
            kargn,
            ["x_new"] + [entry_ident(f) for f in sensor_fulls])
        ports = [u_port, noise_port, dtp, tp, *meas_ports]
        if p_port is not None:
            ports.insert(2, p_port)
        return Module(
            name=self.world.name, state=StateLayout((x_field,)),
            ports=tuple(ports),
            functions={"step": step_fn},
            entry_points=(EntryPoint(
                "step", "step", tuple(eargs),
                writes=("x",), returns=tuple(sensor_fulls)),),
            hosting=Hosting.THREADED)

    def deploy_module(self) -> Module:
        """The **deploy** Module (runs on a robot against real sensors):
        noiseless forward map + per-sensor measurement models + Jacobians."""
        sys = self._sys
        spec = sys.spec
        x_field, u_port, dtp, tp, meas_ports = self._module_scaffold()
        # A measurement is dt-independent — dt is eliminated at construction,
        # so the measure kernels honestly take (x, u, t).
        tan = spec.tangent_dim
        zero_dt = ca.MX.zeros(1, 1)
        functions = {"predict": sys.predict_fn, "predict_jacobian": sys.F_fn}
        p_port = self._param_port()
        p_ref = () if p_port is None else (PortRef("params"),)
        ports = [u_port, *((p_port,) if p_port is not None else ()),
                 dtp, tp, *meas_ports,
                 Port("F", Role.MATRIX, (tan, tan))]
        entries = [
            EntryPoint("predict", "predict",
                       (StateRef("x"), PortRef("u"), *p_ref, PortRef("dt"),
                        PortRef("t")), writes=("x",)),
            EntryPoint("predict_jacobian", "predict_jacobian",
                       (StateRef("x"), PortRef("u"), *p_ref, PortRef("dt"),
                        PortRef("t")), returns=("F",)),
        ]
        margs = [sys.x_sym, sys.u_sym, sys.t_sym]
        margn = ["x", "u", "t"]
        if p_port is not None:
            margs.insert(2, sys.p_sym)
            margn.insert(2, "p")
        for full, s in sys.sensors.items():
            ident = entry_ident(full)
            h = ca.substitute(s.h_sym, sys.dt_sym, zero_dt)
            H = ca.substitute(s.H_sym, sys.dt_sym, zero_dt)
            functions[f"measure_{ident}"] = ca.Function(
                f"h_{ident}", margs, [h], margn, ["h"])
            functions[f"measure_{ident}_jacobian"] = ca.Function(
                f"H_{ident}", margs, [H], margn, ["H"])
            ports.append(Port(f"H_{ident}", Role.MATRIX, (s.dim, tan)))
            entries.append(EntryPoint(
                f"measure_{ident}", f"measure_{ident}",
                (StateRef("x"), PortRef("u"), *p_ref, PortRef("t")),
                returns=(full,)))
            entries.append(EntryPoint(
                f"measure_{ident}_jacobian", f"measure_{ident}_jacobian",
                (StateRef("x"), PortRef("u"), *p_ref, PortRef("t")),
                returns=(f"H_{ident}",)))
        return Module(
            name=self.world.name, state=StateLayout((x_field,)),
            ports=tuple(ports), functions=functions,
            entry_points=tuple(entries), hosting=Hosting.THREADED)

    def __repr__(self) -> str:
        names = ", ".join(c.name for c in self.crafts)
        return f"<Sim crafts=[{names}]>"
