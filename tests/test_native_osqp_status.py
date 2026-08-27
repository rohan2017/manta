from manta.control._osqp import _is_bridge_failure


def test_osqp_negative_solve_status_is_not_a_bridge_failure() -> None:
    # OSQP -2 is maximum iterations reached. The native wrapper must preserve
    # that solver result rather than misreporting a numeric update failure.
    assert not _is_bridge_failure(-2)
    assert not _is_bridge_failure(1)
    assert _is_bridge_failure(-201)
