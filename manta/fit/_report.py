"""Immutable provenance attached to model revisions derived by fitting."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from ._common import DEFAULT_FILL_POLICY_ID, FitDefaultFill
from ._evidence import FitEvidence


@dataclass(frozen=True)
class FitDerivationReport:
    """Provenance of one derived model revision.

    ``evidence`` is the typed held-out artifact (`FitEvidence`) or ``None``
    for an exploratory derivation that computed none; there is no untyped
    form. ``accepted`` is the evidence's own criteria-derived decision and
    is never set by a caller — a report without evidence is not accepted.
    ``default_fills`` records every model initial-state/control value used for
    omitted window data; exploratory reports retain their training fills too.
    """

    method: str
    source_artifact_id: str
    objective: float
    values: tuple[tuple[str, Any], ...]
    evidence: FitEvidence | None
    default_fill_policy_id: str = DEFAULT_FILL_POLICY_ID
    default_fills: tuple[FitDefaultFill, ...] = ()

    def __post_init__(self) -> None:
        if self.evidence is not None and not isinstance(self.evidence,
                                                        FitEvidence):
            raise TypeError(
                "FitDerivationReport.evidence must be a FitEvidence (from "
                "FitResult.evidence(...) / NoiseFitResult.evidence(...) / "
                "held_out_evidence(...)) or None; got "
                f"{type(self.evidence).__name__}")
        if self.default_fill_policy_id != DEFAULT_FILL_POLICY_ID:
            raise ValueError(
                "FitDerivationReport.default_fill_policy_id is unsupported"
            )
        fills = tuple(self.default_fills)
        if not all(isinstance(fill, FitDefaultFill) for fill in fills):
            raise TypeError(
                "FitDerivationReport.default_fills must contain "
                "FitDefaultFill records"
            )
        fills = tuple(sorted(
            fills,
            key=lambda fill: (
                fill.dataset_role, fill.window_digest, fill.source, fill.name
            ),
        ))
        if len(set(fills)) != len(fills):
            raise ValueError(
                "FitDerivationReport.default_fills contains duplicates"
            )
        if self.evidence is not None and fills != self.evidence.default_fills:
            raise ValueError(
                "FitDerivationReport.default_fills must match its evidence"
            )
        if self.evidence is None and any(
            fill.dataset_role != "training" for fill in fills
        ):
            raise ValueError(
                "a report without evidence may contain only training "
                "default fills"
            )
        object.__setattr__(self, "default_fills", fills)

    @property
    def accepted(self) -> bool:
        return self.evidence is not None and self.evidence.accepted


def derivation_report(method: str, source_artifact_id: str, objective: float,
                      values: Mapping[str, Any],
                      evidence: FitEvidence | None,
                      default_fills: tuple[FitDefaultFill, ...] = (),
                      ) -> FitDerivationReport:
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
        default_fills=default_fills,
    )
