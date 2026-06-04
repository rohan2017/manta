"""RecurrenceBlock — the shared base for stateful dataflow blocks.

A *recurrence* is the single runtime kind that covers every freestanding
signal-processing block whose whole step is pure dataflow:

    x' = step(x, u, dt, t)        # next ambient state
    y  = out(x, u, dt, t)         # readout

PID, the Madgwick/Mahony attitude filter, and the IMU strapdown integrator
are all this shape. Crucially the manifold integration (e.g. SO(3) for an
attitude filter) is baked **into the symbolic kernel** — `update_fn`
returns the next *ambient* state directly — so the runtime that drives it
never does a boxplus: it just evaluates one `ca.Function` and stores the
result. That is what lets the one generic runtime per backend serve every
block of this kind (see `manta.codegen.target`).

A subclass builds its symbolic recurrence in `__init__` and hands it to
`_build_recurrence(...)`:

    class PID(RecurrenceBlock):
        def __init__(self, kp, ...):
            def rec(x, u, dt, t):
                err = u["setpoint"] - u["measurement"]
                integ = x["integral"] + err * dt
                cmd = kp * err + ki * integ
                return {"integral": integ, ...}, {"command": cmd}
            self._build_recurrence(
                name="pid",
                state=[("integral", R3Manifold()), ...],
                inputs=[("setpoint", n), ("measurement", n)],
                outputs=[("command", n)],
                x0={"integral": np.zeros(n), ...},
                recurrence=rec)

The author's `recurrence` callback receives the state sliced into a
`{slot_name: MX}` dict and the inputs into a `{port_name: MX}` dict, and
returns `(next_state_dict, output_dict)` — raw CasADi MX, frame-blind (the
IR boundary is the kernel, so no frame tags here, same as the EKF/codegen
symbolic layer).
"""

from __future__ import annotations

from dataclasses import dataclass

import casadi as ca

from .ir.state_spec import StateSpec


@dataclass(frozen=True)
class Port:
    """One named exogenous input or readout of a recurrence block."""
    name: str
    dim: int


class RecurrenceBlock:
    """Base for a stateful manifold recurrence. Subclasses call
    `_build_recurrence(...)`; the base assembles the single `step` kernel
    and emits the typed `Module` (held state `x`; named input/output ports)
    via `module()` — like every transform, it lowers through the one
    generic backend path."""

    def _build_recurrence(self, *, name, state, inputs, outputs, x0,
                          recurrence) -> None:
        """Assemble the block contract from a symbolic recurrence callback.

        Args:
            name       — codegen basename / default class name.
            state      — ordered `[(slot_name, Manifold), …]` state layout.
            inputs     — ordered `[(port_name, dim), …]` exogenous inputs,
                         concatenated into the kernel's `u` in this order.
            outputs    — ordered `[(port_name, dim), …]` readouts.
            x0         — `{slot_name: value}` initial ambient state.
            recurrence — `f(x, u, dt, t) -> (x_next, y)` over MX dicts;
                         see the module docstring.
        """
        spec = StateSpec.from_layout(state)
        self.name    = name
        self.spec    = spec
        self.inputs  = [Port(n, int(d)) for n, d in inputs]
        self.outputs = [Port(n, int(d)) for n, d in outputs]
        if not self.inputs:
            raise ValueError(
                f"{name}: a recurrence block needs at least one input port.")
        if not self.outputs:
            raise ValueError(
                f"{name}: a recurrence block needs at least one output port.")
        self.input_dim  = sum(p.dim for p in self.inputs)
        self.output_dim = sum(p.dim for p in self.outputs)
        self.x0 = spec.pack(dict(x0))

        # Symbolic step: (x, u, dt, t) -> (x_next, y), assembled once.
        x_sym  = ca.MX.sym("x",  spec.ambient_dim, 1)
        u_sym  = ca.MX.sym("u",  self.input_dim,   1)
        dt_sym = ca.MX.sym("dt")
        t_sym  = ca.MX.sym("t")

        xd = {s.name: x_sym[s.ambient_offset:s.ambient_offset + s.ambient_dim] for s in spec.slots}
        ud, off = {}, 0
        for p in self.inputs:
            ud[p.name] = u_sym[off:off + p.dim]
            off += p.dim

        xnext_d, yd = recurrence(xd, ud, dt_sym, t_sym)

        missing_x = [s.name for s in spec.slots if s.name not in xnext_d]
        if missing_x:
            raise ValueError(
                f"{name}: recurrence did not return next-state for slots "
                f"{missing_x}.")
        missing_y = [p.name for p in self.outputs if p.name not in yd]
        if missing_y:
            raise ValueError(
                f"{name}: recurrence did not return outputs {missing_y}.")

        x_next = ca.vertcat(*[ca.reshape(xnext_d[s.name], s.ambient_dim, 1)
                              for s in spec.slots])
        y = ca.vertcat(*[ca.reshape(yd[p.name], p.dim, 1)
                         for p in self.outputs])
        self.update_fn = ca.Function(
            f"{name}_recurrence", [x_sym, u_sym, dt_sym, t_sym], [x_next, y],
            ["x", "u", "dt", "t"], ["x_next", "y"])

    def module(self):
        """The typed `Module` IR a backend lowers: held state `x`, one
        `step(x; u, dt, t) -> y` entry, named input/output port fields."""
        from .ir.module import (
            EntryPoint, Hosting, Module, Port, PortField, PortRef, Role,
            StateField, StateLayout, StateRef,
        )
        spec = self.spec
        return Module(
            name=self.name,
            state=StateLayout((StateField(
                "x", "manifold", (spec.ambient_dim,),
                init=self.x0.copy(), manifold=spec),)),
            ports=(
                Port("u", Role.CONTROL, (self.input_dim,), fields=tuple(
                    PortField(p.name, p.dim, 0.0) for p in self.inputs)),
                Port("y", Role.OUTPUT, (self.output_dim,), fields=tuple(
                    PortField(p.name, p.dim, 0.0) for p in self.outputs)),
                Port("dt", Role.TIMESTEP), Port("t", Role.TIME),
            ),
            functions={"step": self.update_fn},
            entry_points=(EntryPoint(
                "step", "step",
                (StateRef("x"), PortRef("u"), PortRef("dt"), PortRef("t")),
                writes=("x",), returns=("y",)),),
            hosting=Hosting.HELD)

    def __repr__(self) -> str:
        ins = ",".join(p.name for p in self.inputs)
        outs = ",".join(p.name for p in self.outputs)
        return (f"<{type(self).__name__} state={self.spec.ambient_dim} "
                f"in=({ins}) out=({outs})>")
