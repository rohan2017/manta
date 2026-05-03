"""manta-codegen: pure-Python descriptor library + emitter for manta C++ crafts.

Public surface — what users typically import:
"""
from .core import (
    Craft,
    PartDescriptor,
    FieldDescriptor,
    PlanetDescriptor,
    StaticTransform,
    Tether,
    World,
    tf,
)
from .manifest import MantaConfig, Target
from .bindings import connect, publish, subscribe
from .estimation import EKF
from .emit import emit

__all__ = [
    "Craft",
    "PartDescriptor",
    "FieldDescriptor",
    "PlanetDescriptor",
    "StaticTransform",
    "Tether",
    "World",
    "MantaConfig",
    "Target",
    "EKF",
    "tf",
    "connect",
    "publish",
    "subscribe",
    "emit",
]
