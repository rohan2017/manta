"""Linearization — the shared plumbing every transform reads.

`LinearizedSystem` (in `system`) orchestrates; `TickLinearizer` (in
`engine`) differentiates; `partition` analyzes structure. Only the two
names a transform actually consumes are exported — everything else is
internal to this package.
"""

from .engine import SensorModel
from .system import LinearizedSystem

__all__ = [
    "LinearizedSystem",
    "SensorModel",
]
