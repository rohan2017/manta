"""Backend Role-dispatch exhaustiveness.

`manta.ir.module.ARG_ROLES` is the contract: every Role except the
return-only OUTPUT and DIAGNOSTIC roles may appear as an EntryPoint argument,
and each backend's
argument dispatch must cover all of them. Growing the `Role` enum makes
these tests fail at the contract, instead of surfacing as a
NotImplementedError deep inside a backend at lowering time.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from manta.ir.module import ARG_ROLES, Port, PortRef, Role
from manta.codegen.cpp import module_emit as cpp_emit
from manta.codegen.numpy._sim import _stepn_port_arg


def _port_for(role: Role) -> Port:
    if role is Role.MATRIX:
        return Port("Q", role, (3, 3))
    return Port("p", role, (3,))


def _fake_ctx(port: Port) -> SimpleNamespace:
    return SimpleNamespace(
        x_port=port,
        port=lambda name, _p=port: _p,
        m=None,
        held=True,
    )


def test_arg_roles_excludes_return_only_roles():
    assert ARG_ROLES == frozenset(Role) - {Role.OUTPUT, Role.DIAGNOSTIC}


def test_cpp_param_dispatch_covers_every_arg_role():
    for role in sorted(ARG_ROLES, key=lambda r: r.value):
        port = _port_for(role)
        assert cpp_emit._param_for(port, True, _fake_ctx(port)) is not None


def test_cpp_arg_dispatch_covers_every_arg_role():
    for role in sorted(ARG_ROLES, key=lambda r: r.value):
        port = _port_for(role)
        expr = cpp_emit._arg_expr(PortRef(port.name), _fake_ctx(port))
        assert isinstance(expr, str) and expr


#: Roles a sim-oracle ``step`` entry may take as a PortRef argument. The
#: folded-substep and scan rollouts only build these; anything else must
#: raise a NotImplementedError NAMING the role — never fall through to a
#: silent TIME catch-all.
STEP_ARG_ROLES = {Role.CONTROL, Role.NOISE, Role.PARAMETER,
                  Role.TIMESTEP, Role.TIME}


def _stepn(role: Role):
    return _stepn_port_arg(role, hold=lambda v: v, u=np.zeros(2),
                           noise=np.zeros(1), params=np.zeros(1),
                           dt=0.1, t=0.0, n=3)


def test_numpy_stepn_dispatch_handles_or_names_every_arg_role():
    for role in sorted(ARG_ROLES, key=lambda r: r.value):
        if role in STEP_ARG_ROLES:
            assert _stepn(role) is not None
        else:
            with pytest.raises(NotImplementedError, match=role.name):
                _stepn(role)


def test_jax_rollout_dispatch_handles_or_names_every_arg_role():
    pytest.importorskip("jax")
    from manta.codegen.jax._runtime import _rollout_port_arg
    for role in sorted(ARG_ROLES, key=lambda r: r.value):
        if role in STEP_ARG_ROLES:
            assert _rollout_port_arg(role, u_k=1.0, n_k=2.0, params=3.0,
                                     dt=0.1, t_k=0.5) is not None
        else:
            with pytest.raises(NotImplementedError, match=role.name):
                _rollout_port_arg(role, u_k=1.0, n_k=2.0, params=3.0,
                                  dt=0.1, t_k=0.5)
