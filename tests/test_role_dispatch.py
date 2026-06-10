"""Backend Role-dispatch exhaustiveness.

`manta.ir.module.ARG_ROLES` is the contract: every Role except OUTPUT
(return-only) may appear as an EntryPoint argument, and each backend's
argument dispatch must cover all of them. Growing the `Role` enum makes
these tests fail at the contract, instead of surfacing as a
NotImplementedError deep inside a backend at lowering time.
"""

from types import SimpleNamespace

from manta.ir.module import ARG_ROLES, Port, PortRef, Role
from manta.codegen.cpp import module_emit as cpp_emit


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


def test_arg_roles_is_role_minus_output():
    assert ARG_ROLES == frozenset(Role) - {Role.OUTPUT}


def test_cpp_param_dispatch_covers_every_arg_role():
    for role in sorted(ARG_ROLES, key=lambda r: r.value):
        port = _port_for(role)
        assert cpp_emit._param_for(port, True, _fake_ctx(port)) is not None


def test_cpp_arg_dispatch_covers_every_arg_role():
    for role in sorted(ARG_ROLES, key=lambda r: r.value):
        port = _port_for(role)
        expr = cpp_emit._arg_expr(PortRef(port.name), _fake_ctx(port))
        assert isinstance(expr, str) and expr
