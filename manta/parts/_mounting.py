"""Rest-pose mount walks — numeric composition of static mount poses.

The ONE implementation of "where does this part sit, at rest, relative
to an ancestor frame". Every hop from a part to its parent composes the
child's static pose into the parent's output frame:

    off_parent_frame = mount_offset_p + R(mount_orientation_p) · off_child
    R_parent_frame   = R(mount_orientation_p) · R_child

Summing raw `mount_offset`s (the historical bug this module replaces)
is only correct when every intermediate part is mounted upright — a
90°-rotated bracket would translate its children along the wrong axis
and leave their inertia tensors unrotated, silently misplacing the
subtree inertia that gates whether a revolute DOF is live (`I_axial`).

"Rest pose" means every joint DOF on the path is at zero (angle 0,
displacement 0), so a joint contributes only its own static mount pose
— exactly the snapshot semantics `ArticulatedJoint.I_joint_tensor` and
`FieldSource`'s `cumulative_offset` document. Reads go through
`declared_attr`, so a promoted (symbolic) mount parameter never leaks
into these numeric snapshots.
"""

from __future__ import annotations

import numpy as np

from ..ir._rotation import quat_to_rotmat_np
from ._trace import declared_attr


def rest_rotation(part) -> np.ndarray:
    """3×3 rotation of `part`'s own axes into its parent's output frame
    at rest — just the static `mount_orientation`, as a matrix."""
    q = np.asarray(declared_attr(part, "mount_orientation",
                                 (1.0, 0.0, 0.0, 0.0)), dtype=float)
    return quat_to_rotmat_np(q)


def fold_rest_pose(chain) -> tuple[np.ndarray, np.ndarray]:
    """Compose the static mount poses of `chain` — parts ordered
    DESCENDANT-FIRST, each the parent of its predecessor — into the
    topmost part's PARENT output frame. Returns `(offset, R)`:
    the descendant origin's rest position, and the rotation taking the
    descendant's axes into that ancestor frame (for R·I·Rᵀ inertia
    lifts)."""
    off = np.zeros(3)
    R = np.eye(3)
    for p in chain:
        R_p = rest_rotation(p)
        off = (np.asarray(declared_attr(p, "mount_offset", (0.0, 0.0, 0.0)),
                          dtype=float)
               + R_p @ off)
        R = R_p @ R
    return off, R


def rest_pose_from(ancestor, descendant) -> tuple[np.ndarray, np.ndarray]:
    """`(offset, R)` of `descendant` relative to `ancestor`'s output
    frame at rest: the pose composition over the path from `ancestor`
    (exclusive) down to `descendant` (inclusive). Raises loudly when
    `descendant` does not hang under `ancestor` — a silent zero here
    would fake a part sitting at the ancestor origin."""
    chain: list = []
    cur = descendant
    while cur is not None and cur is not ancestor:
        chain.append(cur)
        cur = cur.parent
    if cur is not ancestor:
        raise ValueError(
            f"rest_pose_from: '{descendant.name}' is not a descendant of "
            f"'{ancestor.name}'.")
    return fold_rest_pose(chain)
