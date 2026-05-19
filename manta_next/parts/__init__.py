"""Stock parts. Each part is a Python class subclassing `Part`, declares
its parameters at class scope, and implements `update(ctx)` to contribute
a wrench to the craft.

Public surface re-exports the part classes for ergonomic imports::

    from manta_next.parts import Part, Parameter, Mass
"""

from .base import Part, Parameter, Input
from .wrench import Wrench
from .structure.mass import Mass

__all__ = ["Part", "Parameter", "Input", "Wrench", "Mass"]
