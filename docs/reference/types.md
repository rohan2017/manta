# Model authoring types

Use `manta.types` for symbolic Part and Field equations:

```python
from manta.types import PartFrame, WorldFrame, Scalar, Vec3, Wrench
```

These public exports retain Manta's frame checking and CasADi differentiation.
They are aliases of the internal implementation, so existing models keep their
behavior. Model authors do not need to import `manta.ir`; that module remains
available for compatibility and compiler/backend work.

`Quat[A, B]` and `Mat3[A, B]` convert vector components **from B into A**.
Craft orientation is body-to-world; quaternion order is `[w, x, y, z]`.

For numerical orientation manipulation outside symbolic model equations, use
[SciPy Rotation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html):

```python
from scipy.spatial.transform import Rotation

body_to_world = Rotation.from_quat(state["orientation"], scalar_first=True)
vector_world = body_to_world.apply(vector_body)
q_wxyz = body_to_world.as_quat(scalar_first=True)
```

Always specify `scalar_first=True`; SciPy's default is `[x, y, z, w]`.
Composition `a * b` applies `b` first, then `a`, matching Manta and
`a.as_matrix() @ b.as_matrix()`. The inverse is `a.inv()`. Both `q` and `-q`
represent the same attitude; compare matrices or rotated vectors when checking
round trips. Supply unit quaternions to Manta; SciPy normalizes inputs, whereas
not every Manta entry point does.

The INS handoff tests verify SciPy 1.18.0 against Manta for quaternion order,
matrix conversion, vector action, composition, inverse, and initialization/output
frame conversions. SciPy is a handoff dependency, not a core Manta dependency.
Its numerical Rotation objects must not be used on symbolic values inside
`Part.update`; use `ctx.orientation.apply(...)` / `.conjugate()` there.

::: manta.types
