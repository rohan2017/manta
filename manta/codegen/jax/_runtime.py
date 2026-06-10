"""JaxModule — the JAX view over a typed Module.

Where `TargetNumpy` gives a stateful runtime, the JAX target gives the
FUNCTIONAL artifacts a training loop wants: every kernel as a jitted
pure function, and (for a sim oracle) a `lax.scan` rollout you can
`jax.grad` / `jax.vmap` straight through.

    jm = TargetJax(Sim(world, parameters=[...]))
    step = jm.kernel("step")                  # jitted (x,u,noise,p,dt,t)
    rollout = jm.make_rollout()               # jitted scan over a window
    Xs, readings = rollout(x0, U, NOISE, params, dt, t0)
    loss = lambda p: jnp.sum((rollout(x0, U, N0, p, dt, 0.0)[1][s] - Z)**2)
    g = jax.grad(loss)(params)                # exact, end-to-end

Limitations: kernels must SX-expand — flat (joint-free) crafts only;
a jointed craft's joint-space solve keeps a runtime-pivoting Linsol
node and raises at lowering.
"""

from __future__ import annotations

import numpy as np

import jax
import jax.numpy as jnp

from ...ir.module import Module, PortRef, Role, StateRef
from ._translate import translate


class JaxModule:
    """Lowered Module: jitted kernels by name + rollout builder."""

    def __init__(self, x) -> None:
        module = x if isinstance(x, Module) else x.module()
        if not isinstance(module, Module):
            raise TypeError(
                f"TargetJax: expected a Module or a transform with "
                f".module(), got {type(x).__name__}")
        self.module = module
        self._kernels = {name: translate(fn)
                         for name, fn in module.functions.items()}

    def kernel(self, name: str):
        """The jitted kernel, positional args in the CasADi signature
        order, returning a tuple of dense arrays."""
        if name not in self._kernels:
            raise KeyError(
                f"{self.module.name}: no kernel {name!r} "
                f"(have {sorted(self._kernels)}).")
        return self._kernels[name]

    def call(self, method: str, **values):
        """Run one entry point with named values (state fields and ports
        by name; TIME defaults to 0, PARAMETER to its declared values).
        Functional: returns `(writes_dict, returns_dict)` — nothing is
        held."""
        ep = self.module.entry(method)
        args = []
        for a in ep.args:
            if isinstance(a, StateRef):
                if a.name not in values:
                    raise KeyError(f"{method}: missing state {a.name!r}.")
                args.append(values[a.name])
            else:
                port = self.module.port(a.name)
                if a.name in values:
                    args.append(values[a.name])
                elif port.role is Role.TIME:
                    args.append(0.0)
                elif port.role is Role.PARAMETER:
                    args.append(self.param_defaults())
                else:
                    raise KeyError(f"{method}: missing port {a.name!r}.")
        outs = self.kernel(ep.fn)(*args)
        writes = {w: outs[i] for i, w in enumerate(ep.writes)}
        rets = {r: outs[len(ep.writes) + i]
                for i, r in enumerate(ep.returns)}
        return writes, rets

    # ---- convenience vectors -------------------------------------------

    def initial_state(self) -> jnp.ndarray:
        """The Module's packed initial state field `x`."""
        return jnp.asarray(
            np.asarray(self.module.state.field("x").init, dtype=float))

    def param_defaults(self) -> jnp.ndarray:
        """The PARAMETER port's declared values (empty if none)."""
        ports = self.module.ports_by_role(Role.PARAMETER)
        if not ports:
            return jnp.zeros(0)
        return jnp.concatenate([
            jnp.asarray(np.asarray(f.default, dtype=float).ravel())
            for f in ports[0].fields])

    # ---- rollout ---------------------------------------------------------

    def make_rollout(self):
        """A jitted `lax.scan` rollout of the oracle `step` entry::

            rollout(x0, u_seq, noise_seq[, params], dt, t0)
                -> (x_traj, {sensor: trace})

        `u_seq`/`noise_seq` are (K, n) per-step rows; `params` appears
        only when the Module has a PARAMETER port (promoted via
        `Sim(world, parameters=[...])`) and is a differentiable input —
        `jax.grad` through the whole window works. Readings row k is
        produced by the step taken FROM state k (the manta data
        convention); `x_traj` rows are the K post-step states."""
        ep = self.module.entry("step")
        step = self.kernel(ep.fn)
        has_params = any(
            isinstance(a, PortRef)
            and self.module.port(a.name).role is Role.PARAMETER
            for a in ep.args)
        roles = []
        for a in ep.args:
            roles.append("x" if isinstance(a, StateRef)
                         else self.module.port(a.name).role)
        sensor_names = list(ep.returns)

        def rollout(x0, u_seq, noise_seq, *rest):
            if has_params:
                params, dt, t0 = rest
            else:
                (dt, t0) = rest
                params = None
            K = u_seq.shape[0]
            ts = t0 + dt * jnp.arange(K)

            def body(x, per):
                u_k, n_k, t_k = per
                args = []
                for a, role in zip(ep.args, roles):
                    if role == "x":
                        args.append(x)
                    elif role is Role.CONTROL:
                        args.append(u_k)
                    elif role is Role.NOISE:
                        args.append(n_k)
                    elif role is Role.PARAMETER:
                        args.append(params)
                    elif role is Role.TIMESTEP:
                        args.append(dt)
                    else:                          # TIME
                        args.append(t_k)
                outs = step(*args)
                return outs[0][:, 0], outs   # carry 1-D; kernel returns (n,1)

            _, traj = jax.lax.scan(body, x0, (u_seq, noise_seq, ts))
            x_traj = traj[0][..., 0]               # (K, amb, 1) → (K, amb)
            readings = {name: traj[1 + i][..., 0]
                        for i, name in enumerate(sensor_names)}
            return x_traj, readings

        return jax.jit(rollout)

    def __repr__(self) -> str:
        return (f"<JaxModule over {self.module!r} "
                f"kernels={sorted(self._kernels)}>")
