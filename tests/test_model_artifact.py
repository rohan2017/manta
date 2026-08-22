"""`ModelArtifact` identity is a pure function of the model and its
derivation payload — never of a `repr()` that carries a memory address."""

from __future__ import annotations

import enum
from dataclasses import dataclass

import numpy as np
import pytest

from manta import Craft, Sim, World
from manta.fields import GravityField
from manta.model import canonical_derivation_bytes
from manta.parts import Mass


class _Kind(enum.Enum):
    HELD_OUT = "held_out"


@dataclass(frozen=True)
class _Report:
    objective: float
    values: tuple[tuple[str, float], ...]
    residual: np.ndarray
    kind: _Kind


class _AddressRepr:
    """Default `object.__repr__` — `<... at 0x7f...>` — differs per process."""


def _model():
    craft = Craft("c")
    craft.add(Mass("body", mass=1.0, moi=(0.1, 0.1, 0.1)))
    world = World("artifact").add_field(GravityField(g=(0.0, 0.0, -9.81)))
    world.add_craft(craft)
    return Sim(world).model


def _report(seed: float = 1.0) -> _Report:
    return _Report(objective=seed, values=(("mass", 2.0 * seed),),
                   residual=np.arange(3.0) * seed, kind=_Kind.HELD_OUT)


def test_equal_payloads_built_separately_share_an_identity():
    first = _model().with_derivation("fit", _report())
    second = _model().with_derivation("fit", _report())
    assert first.model_id == second.model_id
    assert first.artifact_id == second.artifact_id
    assert first.artifact_id != first.model_id
    changed = _model().with_derivation("fit", _report(seed=1.5))
    assert changed.artifact_id != first.artifact_id
    # Identity depends on the encoded value, not on how the payload was
    # constructed (tuple vs list, numpy vs Python scalars).
    a = canonical_derivation_bytes({"r": {"x": (1, 2.5), "y": np.float64(3)}})
    b = canonical_derivation_bytes({"r": {"x": [1, 2.5], "y": 3.0}})
    assert a == b


def test_payloads_without_a_canonical_encoding_are_refused():
    model = _model()
    with pytest.raises(TypeError, match=r"fit\['evidence'\].*_AddressRepr.*repr"):
        model.with_derivation(
            "fit", {"evidence": _AddressRepr()})
    with pytest.raises(TypeError, match="lambda|function"):
        model.with_derivation("fit", {"hook": lambda: None})
    with pytest.raises(TypeError, match="mapping key"):
        model.with_derivation("fit", {("tuple", "key"): 1.0})
    with pytest.raises(TypeError, match="canonical"):
        model.with_derivations({"fit": object()})
    # Nothing was attached by a refused call.
    assert dict(model.derivation) == {}
