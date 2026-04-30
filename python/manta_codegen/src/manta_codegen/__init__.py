"""manta-codegen: pure-Python descriptor library + emitter for manta C++ crafts.

Public surface — what users typically import:
"""
from .core import (
    Craft,
    PartDescriptor,
    FieldDescriptor,
    StaticTransform,
    tf,
)
from .emit import emit

__all__ = [
    "Craft",
    "PartDescriptor",
    "FieldDescriptor",
    "StaticTransform",
    "tf",
    "emit",
]
