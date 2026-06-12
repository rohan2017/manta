"""Linearization — the shared plumbing every transform reads.

`LinearizedSystem` (in `system`) orchestrates; `TickLinearizer` (in
`engine`) differentiates; `partition` analyzes structure; `names` holds
the name-resolution helpers used across the runtimes.
"""

from .engine import SensorModel, TickLinearizer
from .names import freeze_complement, slot_of_tangent_index
from .partition import dependency_closure, partition_blocks
from .system import LinearizedSystem

__all__ = [
    "LinearizedSystem",
    "SensorModel",
    "TickLinearizer",
    "dependency_closure",
    "freeze_complement",
    "partition_blocks",
    "slot_of_tangent_index",
]
