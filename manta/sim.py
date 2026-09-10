"""Sim — the forward-dynamics transform of a World.

`Sim(world)` validates the model and linearizes the compiled world tick
(via `LinearizedSystem`); it emits three structurally different Modules,
chosen HERE, at IR construction — lowering just lowers:

* ``module()`` — the **scheduled oracle** (simulation truth): one plant
  `step` plus a separate kernel for each rate-limited acquisition group. The
  runtime calls only due measurement kernels and holds their last values.
  Measurements without a declared rate remain in `step`::

      step(x; u, noise, dt, t) -> x', per_tick_readings…
      sample_group_<n>(x; u, noise, dt, t) -> readings…

  This is a dependency partition, not a symbolic ``if due`` inside one
  monolithic graph: a measurement-only subgraph cannot consume plant-tick
  compute when it is not acquired.

* ``inline_module()`` — the **inline oracle**: one `step` returns every
  reading every tick. This is the explicit smooth/batched artifact for
  callers such as differentiable rollouts that own their own sampling::

      step(x; u, noise, dt, t) -> x', all_readings…

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
    EntryPoint,
    Hosting,
    Module,
    ModuleKind,
    Port,
    PortField,
    PortRef,
    Role,
    StateField,
    StateLayout,
    StateRef,
    entry_ident,
)
from .ir.state_spec import flatten_nested
from .linearization import LinearizedSystem

if TYPE_CHECKING:
    from .world import World


class Sim:
    """Forward-dynamics transform: model validation + the linearized tick,
    emitting oracle/deploy Modules."""

    def __init__(self, world: World, *,
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
        self.model = self._sys.model

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
        x0 = spec.pack_projected(init_flat)
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
        """The scheduled simulation-truth Module.

        Measurements with the same declared positive rate are emitted as one
        ``sample_group_*`` entry and omitted from the plant ``step`` outputs.
        Backends can therefore schedule the kernel without evaluating its
        symbolic dependencies on every physics tick. Measurements with no
        rate remain inline and are evaluated every tick.
        """
        return self._oracle_module(schedule_rate_limited=True)

    def inline_module(self) -> Module:
        """The all-inline simulation oracle for smooth/batched callers.

        Every measurement is returned by ``step`` regardless of declared
        rate. Rate metadata remains present, but this artifact deliberately
        performs no acquisition scheduling.
        """
        return self._oracle_module(schedule_rate_limited=False)

    def _oracle_module(self, *, schedule_rate_limited: bool) -> Module:
        sys = self._sys
        x_field, u_port, dtp, tp, meas_ports = self._module_scaffold()
        sensor_fulls = list(sys.sensors)
        rate_limited = [
            full for full in sensor_fulls
            if schedule_rate_limited and sys.sample_rates.get(full) is not None
        ]
        opened_x = sys.x_new_noisy
        opened_sensors = {
            full: sys.sensors[full].h_noisy_sym for full in sensor_fulls
        }
        plant_coupled = []
        scheduled = []
        if rate_limited:
            opened_x, opened_sensors = sys.inline_simulation_expressions(
                sensor_fulls)
            # The world compiler marks readings that consume its body/joint
            # acceleration placeholders. Specific force is the canonical
            # example: it must share the plant solve even on an unactuated
            # craft. Other observations are independent acquisition kernels.
            for full in rate_limited:
                if full in sys.plant_coupled_outputs:
                    plant_coupled.append(full)
                else:
                    scheduled.append(full)
        inline = [full for full in sensor_fulls if full not in scheduled]
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
        functions = {}
        entries = []
        noise_offsets = []
        noise_offset = 0
        for channel in sys.noise_specs:
            noise_offsets.append((channel, noise_offset))
            noise_offset += channel.dim

        def noise_dependencies(expressions) -> tuple[str, ...]:
            if not noise_offsets:
                return ()
            pattern = np.asarray(ca.DM(
                ca.jacobian(ca.vertcat(*expressions), sys.n_sym).sparsity()
            ), dtype=bool).any(axis=0)
            return tuple(
                channel.full
                for channel, offset in noise_offsets
                if pattern[offset:offset + channel.dim].any()
            )

        noise_contract = []
        step_expressions = [opened_x] + [opened_sensors[f] for f in inline]
        functions["step"] = ca.Function(
            "step", kargs,
            [opened_x] + [
                ca.reshape(opened_sensors[f], sys.sensors[f].dim, 1)
                for f in inline],
            kargn,
            ["x_new"] + [entry_ident(f) for f in inline])
        entries.append(EntryPoint(
            "step", "step", tuple(eargs),
            writes=("x",), returns=tuple(inline)))
        noise_contract.append(("step", noise_dependencies(step_expressions)))
        schedule_contract = []
        groups: dict[float, list[str]] = {}
        for full in scheduled:
            groups.setdefault(float(sys.sample_rates[full]), []).append(full)
        for group_index, (rate, fulls) in enumerate(groups.items()):
            method = f"sample_group_{group_index}"
            expressions = [opened_sensors[full] for full in fulls]
            selected = [
                (arg, name, ref)
                for arg, name, ref in zip(kargs, kargn, eargs, strict=True)
                if any(ca.depends_on(expression, arg)
                       for expression in expressions)
            ]
            functions[method] = ca.Function(
                method,
                [arg for arg, _name, _ref in selected],
                expressions,
                [name for _arg, name, _ref in selected],
                [entry_ident(full) for full in fulls])
            entries.append(EntryPoint(
                method, method,
                tuple(ref for _arg, _name, ref in selected),
                returns=tuple(fulls)))
            schedule_contract.append((method, rate, tuple(fulls)))
            noise_contract.append((method, noise_dependencies(expressions)))
        ports = [u_port, noise_port, dtp, tp, *meas_ports]
        if p_port is not None:
            ports.insert(2, p_port)
        return Module(
            name=self.world.name, state=StateLayout((x_field,)),
            ports=tuple(ports),
            functions=functions,
            entry_points=tuple(entries),
            kind=ModuleKind.SIMULATOR,
            hosting=Hosting.THREADED,
            metadata=self.model.transform_metadata({
                "transform": "simulator",
                "sensor_scheduling": (
                    "dependency" if schedule_rate_limited else "inline"
                ),
                "scheduled_measurement_groups": tuple(schedule_contract),
                "plant_coupled_measurements": tuple(plant_coupled),
                "noise_dependencies": (
                    tuple(noise_contract) if schedule_rate_limited else None
                ),
                "discretization": sys.discretization,
                "parameters": tuple(p.full for p in sys.param_specs),
            }))

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
            entry_points=tuple(entries), kind=ModuleKind.KERNEL,
            hosting=Hosting.THREADED,
            metadata=self.model.transform_metadata({
                "transform": "deploy_model",
                "discretization": sys.discretization,
                "parameters": tuple(p.full for p in sys.param_specs),
            }))

    def __repr__(self) -> str:
        names = ", ".join(c.name for c in self.crafts)
        return f"<Sim crafts=[{names}]>"
