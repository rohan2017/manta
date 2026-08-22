"""Validated, immutable identity for one resolved executable model revision."""

from __future__ import annotations

import copy
import dataclasses
import enum
import json
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from hashlib import sha256
from types import MappingProxyType
from typing import Any

import numpy as np


def _canonical(value: Any, *, path: str) -> Any:
    """Lower a derivation payload to a JSON-canonical form or refuse it.

    Artifact identity must be a pure function of the payload's *value*. A
    default ``repr`` carries a memory address and would make two identical
    fit reports hash differently per process, so only value types with an
    unambiguous encoding are accepted: JSON scalars, numpy scalars/arrays
    (dtype + shape + values), mappings with scalar keys, sequences, enums,
    bytes, and dataclasses (by qualified type name and field values).
    Anything else raises ``TypeError`` naming where in the payload it sits.
    """
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value
    if isinstance(value, bytes):
        return {"__bytes__": value.hex()}
    if isinstance(value, np.generic):
        return _canonical(value.item(), path=path)
    if isinstance(value, np.ndarray):
        return {"__ndarray__": str(value.dtype), "shape": list(value.shape),
                "values": value.tolist()}
    if isinstance(value, enum.Enum):
        return {"__enum__": type(value).__qualname__, "name": value.name}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            "__dataclass__": type(value).__qualname__,
            "fields": {
                f.name: _canonical(getattr(value, f.name),
                                   path=f"{path}.{f.name}")
                for f in dataclasses.fields(value)
            },
        }
    if isinstance(value, Mapping):
        out = {}
        for key, item in value.items():
            if not isinstance(key, (str, int, float, bool)) or key is None:
                raise TypeError(
                    f"derivation payload at {path!r}: mapping key {key!r} of "
                    f"type {type(key).__qualname__} has no canonical "
                    "serialization")
            out[f"{type(key).__qualname__}:{key!r}"] = _canonical(
                item, path=f"{path}[{key!r}]")
        return {"__mapping__": dict(sorted(out.items()))}
    if isinstance(value, (list, tuple, set, frozenset)):
        items = sorted(value, key=repr) if isinstance(
            value, (set, frozenset)) else value
        return [_canonical(item, path=f"{path}[{index}]")
                for index, item in enumerate(items)]
    raise TypeError(
        f"derivation payload at {path!r}: {type(value).__module__}."
        f"{type(value).__qualname__} has no canonical serialization; "
        "artifact identity refuses objects whose only encoding is repr()")


def canonical_derivation_bytes(derivation: Mapping[str, Any]) -> bytes:
    """Deterministic encoding of a derivation map for artifact identity."""
    encoded = {
        str(name): _canonical(report, path=str(name))
        for name, report in derivation.items()
    }
    return json.dumps(encoded, sort_keys=True, separators=(",", ":"),
                      allow_nan=True, ensure_ascii=True).encode()


@dataclass(frozen=True)
class ModelValidationReport:
    """Structural checks completed before a transform may use a model."""

    checks: tuple[str, ...]
    craft_names: tuple[str, ...]
    coupling_names: tuple[str, ...]

    @property
    def valid(self) -> bool:
        return True


@dataclass(frozen=True)
class ModelArtifact:
    """One resolved World revision shared by all artifacts of a transform.

    The authoring World is never frozen. Constructing another transform from
    it captures a new revision, so users may add or remove parts between a
    Sim, EKF, UKF, or controller build. ``_world`` is the transform-owned
    snapshot and is intentionally not an authoring surface.
    """

    name: str
    model_id: str
    artifact_id: str
    state_spec: Any = field(repr=False, compare=False)
    input_names: tuple[str, ...]
    sensor_names: tuple[str, ...]
    parameter_names: tuple[str, ...]
    validation: ModelValidationReport
    _world: Any = field(repr=False, compare=False)
    _authoring_world: Any = field(repr=False, compare=False)
    derivation: Mapping[str, Any] = field(
        default_factory=lambda: MappingProxyType({}), repr=False, compare=False)

    @classmethod
    def from_compiled(cls, world, state_spec, compiled_function, signature,
                      *, parameter_names=(),
                      authoring_world=None,
                      derivation: Mapping[str, Any] | None = None):
        digest = sha256()
        digest.update(b"manta-model-v1\0")
        digest.update(world.name.encode())
        digest.update(compiled_function.serialize().encode())
        for slot in state_spec.slots:
            digest.update(repr((slot.name, slot.manifold)).encode())
        initial = world._initial_state_dict()
        for owner, values in sorted(initial.items()):
            for name, value in sorted(values.items()):
                array = np.asarray(value, dtype=float)
                digest.update(repr((owner, name, array.shape)).encode())
                digest.update(array.tobytes())
        coupling_names = tuple(coupling.name for coupling in world.couplings)
        for name in coupling_names:
            digest.update(name.encode())
        report = ModelValidationReport(
            checks=("resolved", "ownership", "requirements", "state_layout",
                    "tick_signature"),
            craft_names=tuple(craft.name for craft in world.crafts),
            coupling_names=coupling_names,
        )
        model_id = digest.hexdigest()
        derivation = MappingProxyType(dict(derivation or {}))
        return cls(
            name=world.name,
            model_id=model_id,
            artifact_id=cls._artifact_identity(model_id, derivation),
            state_spec=state_spec,
            input_names=tuple(signature.input_names),
            sensor_names=tuple(signature.sensor_names),
            parameter_names=tuple(parameter_names),
            validation=report,
            derivation=derivation,
            _world=world,
            _authoring_world=(copy.deepcopy(world) if authoring_world is None
                              else authoring_world),
        )

    @staticmethod
    def _artifact_identity(model_id: str, derivation: Mapping[str, Any]) -> str:
        digest = sha256()
        digest.update(b"manta-artifact-v2\0")
        digest.update(model_id.encode())
        digest.update(canonical_derivation_bytes(derivation))
        return digest.hexdigest()

    def with_derivation(self, name: str, report: Any) -> "ModelArtifact":
        """Return this physical model with immutable provenance attached."""
        if not isinstance(name, str) or not name:
            raise ValueError("derivation name must be a non-empty string")
        base = name
        index = 2
        while name in self.derivation:
            name = f"{base}_{index}"
            index += 1
        derivation = MappingProxyType({**self.derivation, name: report})
        return replace(
            self,
            artifact_id=self._artifact_identity(self.model_id, derivation),
            derivation=derivation,
        )

    def with_derivations(
            self, derivation: Mapping[str, Any]) -> "ModelArtifact":
        """Carry provenance forward onto a newly derived physical model."""
        if self.derivation:
            raise ValueError("with_derivations requires a fresh model artifact")
        owned = MappingProxyType(dict(derivation))
        return replace(
            self,
            artifact_id=self._artifact_identity(self.model_id, owned),
            derivation=owned,
        )

    def __repr__(self) -> str:
        return (f"<ModelArtifact {self.name!r} "
                f"revision={self.artifact_id[:12]} "
                f"crafts={self.validation.craft_names}>")

    def world_copy(self):
        """Return an editable authoring copy of this model revision."""
        return copy.deepcopy(self._authoring_world)

    def _resolved_world_copy(self):
        """Private transform input that must not rerun deferred hooks."""
        return copy.deepcopy(self._world)
