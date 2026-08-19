"""R^n: the `VecN` carrier and the `RnManifold` that rides it.

The parameterized-dimension Euclidean pair — one class per concept,
with the dimension as instance data ("dimension is data, not type").
It is the production path for a fitted coefficient block travelling as
a flat vector (FossenDamping's 6x6 damping tensor as R36) and for
`Noise(signal_manifold="R7")`-style declarations.

This file exists because the pair shipped with zero coverage: a missing
module-level numpy import made `VecN[n].constant` — and with it
`VecN[n].coerce` and `RnManifold.ir_zero` — raise NameError on first call,
and the whole IR suite still passed. The tests below exercise every
entry point that touches numpy, plus the `"R<n>"` shortcut grammar and
the manifold identities.
"""

import numpy as np
import pytest

from manta import ir
from manta.ir.frames import CraftFrame
from manta.ir.manifold import (
    R3Manifold, RnManifold, ScalarManifold, manifold_from_shortcut,
)
from manta.ir.types import VecN


# ---------------------------------------------------------------------------
# VecN constructors
# ---------------------------------------------------------------------------

def test_constant_from_sequence_infers_dim():
    v = VecN[4].constant([1.0, 2.0, 3.5, -4.0])
    assert v.dim == 4
    assert np.allclose(np.asarray(v._mx.to_DM()).reshape(-1),
                       [1.0, 2.0, 3.5, -4.0])


def test_constant_with_matching_dim_is_accepted():
    v = VecN[7].constant(np.zeros(7))
    assert v.dim == 7


def test_constant_with_wrong_dim_raises():
    with pytest.raises(ValueError, match="expected 3 entries"):
        VecN[3].constant([1.0, 2.0])


def test_class_getitem_grammar_builds_dimensioned_inputs():
    with ir.Graph():
        a = VecN[5].input("a")
        b = VecN[5].input("b")
    assert a.dim == b.dim == 5
    assert a._mx.shape == b._mx.shape == (5, 1)


@pytest.mark.parametrize("bad", [0, -1, 2.0, True, "3"])
def test_class_getitem_rejects_non_dimensions(bad):
    with pytest.raises(TypeError, match="dim must be an int"):
        VecN[bad]


def test_constructor_rejects_a_mismatched_mx_shape():
    with ir.Graph():
        mx = VecN[3].input("v")._mx
        with pytest.raises(ValueError, match=r"expected shape \(4,1\)"):
            VecN(mx, 4)


def test_coerce_passes_a_vecn_through_and_wraps_a_plain_value():
    """The promotable-Parameter idiom: a promoted VecN survives, a plain
    Python sequence becomes a constant of the declared dim."""
    with ir.Graph():
        promoted = VecN[6].input("D")
        assert VecN[6].coerce(promoted) is promoted
    wrapped = VecN[6].coerce([0.0] * 6)
    assert isinstance(wrapped, VecN) and wrapped.dim == 6


# ---------------------------------------------------------------------------
# RnManifold
# ---------------------------------------------------------------------------

def test_dims_are_instance_data():
    assert RnManifold(7).ambient_dim == 7
    assert RnManifold(7).tangent_dim == 7
    assert RnManifold(36).storage_shape == (36,)
    # One class, two dims — not two types.
    assert type(RnManifold(7)) is type(RnManifold(36))


def test_dim_below_one_raises():
    with pytest.raises(ValueError, match="dim must be >= 1"):
        RnManifold(0)


def test_default_value_and_ir_zero_agree():
    m = RnManifold(5)
    assert np.allclose(m.default_value(), np.zeros(5))
    zero = m.ir_zero()
    assert zero.dim == 5
    assert np.allclose(np.asarray(zero._mx.to_DM()).reshape(-1), np.zeros(5))


def test_boxplus_boxminus_roundtrip():
    """Euclidean: (x ⊞ δ) ⊟ x == δ, exactly."""
    m = RnManifold(4)
    with ir.Graph() as g:
        x = m.ir_input("x")
        d = m.ir_input("d")
        g.output(m.boxminus(m.boxplus(x, d), x), "back")
    out = g.compile()(x=[1.0, -2.0, 3.0, 0.5], d=[0.1, 0.2, -0.3, 4.0])
    assert np.allclose(np.asarray(out["back"]).reshape(-1),
                       [0.1, 0.2, -0.3, 4.0])


def test_boxplus_rejects_a_wrong_carrier():
    m = RnManifold(3)
    with ir.Graph():
        v3 = ir.Vec3[CraftFrame].input("v")
        with pytest.raises(TypeError, match="RnManifold.boxplus"):
            m.boxplus(v3, v3)


def test_ir_add_takes_a_raw_tangent_mx():
    """`ir_add` is the codegen-side boxplus: a value plus a bare MX
    tangent (what a filter's correction arrives as)."""
    m = RnManifold(3)
    with ir.Graph() as g:
        x = m.ir_input("x")
        d = VecN[3].input("d")
        g.output(m.ir_add(x, d._mx), "sum")
    out = g.compile()(x=[1.0, 2.0, 3.0], d=[0.5, 0.5, 0.5])
    assert np.allclose(np.asarray(out["sum"]).reshape(-1), [1.5, 2.5, 3.5])


# ---------------------------------------------------------------------------
# The "R<n>" shortcut grammar
# ---------------------------------------------------------------------------

def test_r1_and_r3_resolve_to_their_specialized_classes():
    """n = 1 and n = 3 are NOT RnManifold: they keep the scalar and the
    frame-checked 3-vector, which is why the grammar can be open-ended
    without shadowing them."""
    assert isinstance(manifold_from_shortcut("R1"), ScalarManifold)
    assert isinstance(manifold_from_shortcut("R3", frame=CraftFrame),
                      R3Manifold)


@pytest.mark.parametrize("n", [2, 4, 7, 36, 100])
def test_any_other_rn_resolves_to_the_parameterized_class(n):
    m = manifold_from_shortcut(f"R{n}")
    assert isinstance(m, RnManifold) and m.dim == n


def test_a_manifold_instance_passes_through():
    m = RnManifold(9)
    assert manifold_from_shortcut(m) is m


def test_a_frame_on_a_frame_free_shortcut_raises():
    """A silently dropped frame would resurface as an unchecked frame
    bug far from the declaration."""
    with pytest.raises(ValueError, match="frame="):
        manifold_from_shortcut("R7", frame=CraftFrame)


@pytest.mark.parametrize("bad", ["R", "R0", "R01", "R-2", "Rx", "SE3"])
def test_malformed_shortcuts_raise(bad):
    with pytest.raises(ValueError):
        manifold_from_shortcut(bad)
