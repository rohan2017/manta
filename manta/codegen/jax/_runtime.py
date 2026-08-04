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

Jointed crafts are supported: a kernel whose joint-space solve keeps a
runtime-pivoting Linsol node (and so cannot SX-expand whole) is cut at
the solve nodes and recomposed around `jnp.linalg.solve` — see
`_translate.py`. Translation stays lazy per kernel.
"""

from __future__ import annotations

import numpy as np

import jax
import jax.numpy as jnp

from ...ir.module import PortRef, Role, StateRef
from ..target import as_module, for_role, resolve_args
from ._translate import translate


def _rollout_port_arg(role: Role, *, u_k, n_k, params, dt, t_k):
    """The per-step `lax.scan` argument for one step-entry PortRef, by
    Role."""
    return for_role(role, {
        Role.CONTROL: lambda: u_k,
        Role.NOISE: lambda: n_k,
        Role.PARAMETER: lambda: params,
        Role.TIMESTEP: lambda: dt,
        Role.TIME: lambda: t_k,
    }, who="_rollout_port_arg")


class JaxModule:
    """Lowered Module: jitted kernels by name (translated lazily, cached
    on first use) + rollout builder."""

    def __init__(self, x) -> None:
        module = as_module(x, "TargetJax")
        self.module = module
        self._kernels: dict = {}        # name → jitted fn, on first use

    def kernel(self, name: str):
        """The jitted kernel (translated and cached on first use),
        positional args in the CasADi signature order, returning a tuple
        of dense arrays."""
        fn = self._kernels.get(name)
        if fn is None:
            if name not in self.module.functions:
                raise KeyError(
                    f"{self.module.name}: no kernel {name!r} "
                    f"(have {sorted(self.module.functions)}).")
            fn = self._kernels[name] = translate(self.module.functions[name])
        return fn

    def call(self, method: str, **values):
        """Run one entry point with named values (state fields and ports
        by name; TIME defaults to 0, PARAMETER to its declared values).
        Functional: returns `(writes_dict, returns_dict)` — nothing is
        held."""
        ep = self.module.entry(method)

        def _state(name):
            if name not in values:
                raise KeyError(
                    f"{self.module.name}.{method}: missing state {name!r}.")
            return values[name]

        args = resolve_args(self.module, ep, values, state_lookup=_state,
                            param_default=self.param_defaults)
        outs = self.kernel(ep.fn)(*args)
        writes = {w: outs[i] for i, w in enumerate(ep.writes)}
        rets = {r: outs[len(ep.writes) + i]
                for i, r in enumerate(ep.returns)}
        return writes, rets

    # ---- convenience vectors -------------------------------------------

    def initial_state(self) -> jnp.ndarray:
        """The Module's packed initial manifold state."""
        x0 = self.module.initial_x
        if x0 is None:
            raise ValueError(
                f"{self.module.name}: module carries no manifold state — "
                f"initial_state() is undefined.")
        return jnp.asarray(np.asarray(x0, dtype=float))

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
                args = [x if role == "x" else
                        _rollout_port_arg(role, u_k=u_k, n_k=n_k,
                                          params=params, dt=dt, t_k=t_k)
                        for role in roles]
                outs = step(*args)
                return outs[0][:, 0], outs   # carry 1-D; kernel returns (n,1)

            _, traj = jax.lax.scan(body, x0, (u_seq, noise_seq, ts))
            n_w = len(ep.writes)
            x_traj = traj[0][..., 0]               # (K, amb, 1) → (K, amb)
            readings = {name: traj[n_w + i][..., 0]
                        for i, name in enumerate(sensor_names)}
            return x_traj, readings

        return jax.jit(rollout)

    def __repr__(self) -> str:
        return (f"<JaxModule over {self.module!r} "
                f"kernels={sorted(self.module.functions)}>")
