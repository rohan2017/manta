"""Immutable provenance attached to model revisions derived by fitting."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from ._evidence import FitEvidence


@dataclass(frozen=True)
class FitDerivationReport:
    """Provenance of one derived model revision.

    ``evidence`` is the typed held-out artifact (`FitEvidence`) or ``None``
    for an exploratory derivation that computed none; there is no untyped
    form. ``accepted`` is the evidence's own criteria-derived decision and
    is never set by a caller — a report without evidence is not accepted.
    """

    method: str
    source_artifact_id: str
    objective: float
    values: tuple[tuple[str, Any], ...]
    evidence: FitEvidence | None

    def __post_init__(self) -> None:
        if self.evidence is not None and not isinstance(self.evidence,
                                                        FitEvidence):
            raise TypeError(
                "FitDerivationReport.evidence must be a FitEvidence (from "
                "FitResult.evidence(...) / NoiseFitResult.evidence(...) / "
                "held_out_evidence(...)) or None; got "
                f"{type(self.evidence).__name__}")

    @property
    def accepted(self) -> bool:
        return self.evidence is not None and self.evidence.accepted


def derivation_report(method: str, source_artifact_id: str, objective: float,
                      values: Mapping[str, Any],
                      evidence: FitEvidence | None) -> FitDerivationReport:
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

    return FitDerivationReport(
        method=method,
        source_artifact_id=source_artifact_id,
        objective=float(objective),
        values=tuple(sorted((name, freeze(value))
                            for name, value in values.items())),
        evidence=evidence,
    )
