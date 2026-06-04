"""Shared rerun visualization helpers for the manta examples.

Thin wrappers over the rerun SDK, set up to match manta's WorldFrame:
right-handed, **z-up**, metres. Manta quaternions are ``[w, x, y, z]``;
rerun wants ``[x, y, z, w]``, so :meth:`Viz.pose` converts for you.

rerun is imported lazily, so examples that don't visualize (``quickstart``)
don't need it installed. A demo creates one :class:`Viz`, calls
:meth:`Viz.t` once per logged frame to stamp the timeline, then logs
geometry under stable entity paths::

    viz = Viz("manta/quadcopter")
    viz.box("world/quad/body", (0.1, 0.1, 0.03), color=(80, 140, 220))   # once
    for ...:
        viz.t(t)
        viz.pose("world/quad", pos, quat_wxyz)        # moves the body frame
        viz.trail("world/quad/trail", pos)

Entity paths are hierarchical: a child (``world/quad/rotor0``) inherits its
parent's :meth:`pose`, so nested frames (a gimbal on a hull) compose by
logging a local transform at each level — exactly manta's frame chain.
"""

from __future__ import annotations

import numpy as np


def require_rerun():
    """Import rerun or exit with an actionable message."""
    try:
        import rerun as rr
    except ImportError as exc:  # pragma: no cover - depends on env
        raise SystemExit(
            "This demo needs the rerun SDK for visualization:\n"
            "    .venv/bin/pip install rerun-sdk\n"
            "(or run the non-visual examples like quickstart)."
        ) from exc
    return rr


def _vec3(v) -> np.ndarray:
    return np.asarray(v, dtype=float).ravel()[:3]


class Viz:
    """A small stateful wrapper around one rerun recording.

    Constructing one opens the rerun viewer; headless runs (``--no-viz``)
    skip creating a ``Viz`` entirely.
    """

    def __init__(self, app_id: str) -> None:
        rr = require_rerun()
        self.rr = rr
        rr.init(app_id, spawn=True)
        rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        self._trails: dict[str, list[np.ndarray]] = {}

    # ---- timeline ------------------------------------------------------

    def t(self, seconds: float) -> None:
        """Stamp subsequent logs at sim-time ``seconds`` on the timeline."""
        self.rr.set_time("sim_time", duration=float(seconds))

    # ---- transforms + geometry ----------------------------------------

    def pose(self, path: str, position, quat_wxyz=None) -> None:
        """Log the transform of an entity frame (translation + rotation).

        ``quat_wxyz`` is manta's ``[w, x, y, z]`` (converted to rerun's
        xyzw). Children of ``path`` inherit this transform.
        """
        kw: dict = {"translation": _vec3(position)}
        if quat_wxyz is not None:
            w, x, y, z = np.asarray(quat_wxyz, dtype=float).ravel()
            kw["quaternion"] = self.rr.Quaternion(xyzw=[x, y, z, w])
        self.rr.log(path, self.rr.Transform3D(**kw))

    def box(self, path, half_sizes, *, center=(0.0, 0.0, 0.0),
            color=(150, 150, 150), static: bool = True) -> None:
        """A solid box at ``center`` in the entity's frame (default origin)."""
        self.rr.log(
            path,
            self.rr.Boxes3D(centers=[_vec3(center)],
                            half_sizes=[_vec3(half_sizes)], colors=[color],
                            fill_mode="solid"),
            static=static)

    def arrow(self, path, origin, vector, *, color=(235, 80, 80),
              radius: float | None = None) -> None:
        """A single arrow ``origin → origin+vector`` (in the path's frame)."""
        self.rr.log(
            path,
            self.rr.Arrows3D(origins=[_vec3(origin)], vectors=[_vec3(vector)],
                             colors=[color],
                             radii=None if radius is None else [radius]))

    def line(self, path, points, *, color=(220, 220, 90),
             radius: float | None = None) -> None:
        """A polyline through ``points`` (Nx3)."""
        pts = np.asarray(points, dtype=float).reshape(-1, 3)
        self.rr.log(
            path,
            self.rr.LineStrips3D([pts], colors=[color],
                                 radii=None if radius is None else radius))

    def point(self, path, position, *, color=(255, 220, 60),
              radius: float = 0.05) -> None:
        """A single marker point."""
        self.rr.log(
            path,
            self.rr.Points3D([_vec3(position)], colors=[color], radii=[radius]))

    def trail(self, path, position, *, color=(80, 160, 255),
              max_len: int = 4000) -> None:
        """Append ``position`` to a growing world-frame trail polyline."""
        buf = self._trails.setdefault(path, [])
        buf.append(_vec3(position).copy())
        if len(buf) > max_len:
            del buf[0]
        if len(buf) >= 2:
            self.rr.log(path, self.rr.LineStrips3D(
                [np.asarray(buf)], colors=[color]))

    def plane(self, path, *, z: float = 0.0, size: float = 20.0,
              color=(70, 110, 70, 160)) -> None:
        """A flat (thin-box) reference plane at height ``z`` — ground/water.

        ``color`` may be RGBA; an alpha < 255 renders it semi-transparent.
        Logged static (call once).
        """
        self.rr.log(
            path,
            self.rr.Boxes3D(centers=[[0.0, 0.0, z]],
                            half_sizes=[[size, size, 1e-3]],
                            colors=[color], fill_mode="solid"),
            static=True)
