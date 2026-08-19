"""Immutable provenance attached to model revisions derived by fitting."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class FitDerivationReport:
    method: str
    source_artifact_id: str
    objective: float
    values: tuple[tuple[str, Any], ...]
    validation: tuple[tuple[str, Any], ...]
    accepted: bool


def derivation_report(method: str, source_artifact_id: str, objective: float,
                      values: Mapping[str, Any], validation) -> FitDerivationReport:
    def freeze(value):
        if isinstance(value, np.ndarray):
            return tuple(freeze(item) for item in value.tolist())
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, Mapping):
            return tuple(sorted((str(key), freeze(item))
                                for key, item in value.items()))
        if isinstance(value, (list, tuple)):
            return tuple(freeze(item) for item in value)
        return value

    evidence = dict(validation or {})
    accepted = evidence.get("accepted") is True
    return FitDerivationReport(
        method=method,
        source_artifact_id=source_artifact_id,
        objective=float(objective),
        values=tuple(sorted((name, freeze(value))
                            for name, value in values.items())),
        validation=tuple(sorted((str(name), freeze(value))
                                for name, value in evidence.items())),
        accepted=accepted,
    )
