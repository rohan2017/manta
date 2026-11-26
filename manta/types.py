"""Public, frame-checked symbolic values for authoring Parts and Fields.

These are the same types used internally by Manta; no adapters or copies are
introduced. Import them here when writing model equations. Compiler graphs,
manifolds, module layouts and private numerical routines remain in ``manta.ir``.

For numerical orientation calculations outside model equations, use
``scipy.spatial.transform.Rotation`` with ``scalar_first=True`` when reading
or writing Manta's ``[w, x, y, z]`` quaternions. SciPy is not a core dependency.
It cannot operate on symbolic CasADi values inside ``Part.update``.

``Quat[A, B]`` and ``Mat3[A, B]`` map vector components from B into A.
For example, ``Quat[WorldFrame, PartFrame].apply(v_part)`` returns a world
vector. A craft's orientation similarly maps body vectors into world axes.
"""

from .ir.frames import (
    CraftFrame,
    Frame,
    FrameError,
    ParentFrame,
    PartFrame,
    PlanetFrame,
    WorldFrame,
)
from .ir.types import Mat3, Quat, Scalar, Vec3, VecN
from .ir.wrench import Wrench

__all__ = [
    "CraftFrame", "Frame", "FrameError", "ParentFrame", "PartFrame",
    "PlanetFrame", "WorldFrame", "Mat3", "Quat", "Scalar", "Vec3", "VecN",
    "Wrench",
]
