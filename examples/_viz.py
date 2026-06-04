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

import os
import sys

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
    # The repo convention is `.venv/bin/python -m examples...` without
    # activating the venv, so the `rerun` viewer executable pip installed
    # next to the interpreter may not be on PATH — and `rr.init(spawn=True)`
    # finds the viewer via PATH. Prepend the interpreter's bin dir.
    venv_bin = os.path.dirname(sys.executable)
    path = os.environ.get("PATH", "")
    if (os.path.exists(os.path.join(venv_bin, "rerun"))
            and venv_bin not in path.split(os.pathsep)):
        os.environ["PATH"] = venv_bin + os.pathsep + path
    return rr


def _vec3(v) -> np.ndarray:
    return np.asarray(v, dtype=float).ravel()[:3]


class Viz:
    """A small stateful wrapper around one rerun recording.

    Constructing one opens the rerun viewer; headless runs (``--no-viz``)
    skip creating a ``Viz`` entirely. Pass ``addr`` (``host[:port]`` or a
    full ``rerun+http://…/proxy`` URL) to stream to an already-running
    viewer instead of spawning one — e.g. from WSL to a GPU-rendered
    Windows-native viewer, instead of the software-rasterized WSLg one.
    """

    def __init__(self, app_id: str, addr: str | None = None) -> None:
        rr = require_rerun()
        self.rr = rr
        if addr:
            if "://" not in addr:
                if ":" not in addr:
                    addr += ":9876"
                addr = f"rerun+http://{addr}/proxy"
            rr.init(app_id)
            rr.connect_grpc(addr)
        else:
            rr.init(app_id, spawn=True)
        rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        self._trails: dict[str, list[np.ndarray]] = {}
        self._cams: set[str] = set()
        self._grid_idx: dict[tuple, np.ndarray] = {}

    # ---- timeline ------------------------------------------------------

    def t(self, seconds: float) -> None:
        """Stamp subsequent logs at sim-time ``seconds`` on the timeline."""
        self.rr.set_time("sim_time", duration=float(seconds))

    # ---- chase camera ----------------------------------------------------

    def chase(self, path: str, eye, target) -> None:
        """Pose a chase-camera entity at ``eye`` looking at ``target``
        (world coords, z-up). Log it every frame to fly the camera; pair
        with :meth:`track` to lock the 3-D view onto it.

        The first call also logs the camera intrinsics. Orientation uses
        rerun's RDF convention (x=right, y=down, z=forward).
        """
        rr = self.rr
        if path not in self._cams:
            self._cams.add(path)
            rr.log(path, rr.Pinhole(fov_y=0.9, aspect_ratio=16 / 9,
                                    camera_xyz=rr.ViewCoordinates.RDF,
                                    image_plane_distance=0.5), static=True)
        fwd = _vec3(target) - _vec3(eye)
        fwd = fwd / max(float(np.linalg.norm(fwd)), 1e-9)
        right = np.cross(fwd, np.array([0.0, 0.0, 1.0]))
        n = float(np.linalg.norm(right))
        if n < 1e-6:                      # looking straight up/down
            right = np.array([1.0, 0.0, 0.0])
            n = 1.0
        right = right / n
        down = np.cross(fwd, right)
        rr.log(path, rr.Transform3D(
            translation=_vec3(eye),
            mat3x3=np.column_stack([right, down, fwd])))

    def track(self, path: str) -> None:
        """Track the entity at ``path`` with the 3-D view's eye.

        For a camera entity (see :meth:`chase`) the view takes the camera
        pose verbatim; for a plain entity the orbit CENTRE follows it and
        the user orbits/zooms around it normally. Send once, AFTER the
        entity has data on the timeline (the viewer resolves the tracked
        entity against existing data). Click another entity to detach.
        """
        import rerun.blueprint as rrb  # local: rerun is lazily imported
        self.rr.send_blueprint(rrb.Spatial3DView(
            origin="/",
            eye_controls=rrb.archetypes.EyeControls3D(tracking_entity=path),
        ))

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
              max_len: int = 4000, min_dist: float = 0.0) -> None:
        """Append ``position`` to a growing world-frame trail polyline.

        ``position`` is in WORLD coords, so ``path`` must NOT sit under an
        entity that gets :meth:`pose`\\ d (children inherit the parent
        transform — the whole trail would ride the moving body).

        Every log re-sends the whole polyline, so for live demos keep it
        cheap: ``min_dist`` skips points closer than that to the last kept
        one (call as often as you like — most calls become no-ops), and
        ``max_len`` caps the strip length.
        """
        buf = self._trails.setdefault(path, [])
        pos = _vec3(position).copy()
        if buf and min_dist > 0.0 \
                and float(np.linalg.norm(pos - buf[-1])) < min_dist:
            return
        buf.append(pos)
        if len(buf) > max_len:
            del buf[0]
        if len(buf) >= 2:
            self.rr.log(path, self.rr.LineStrips3D(
                [np.asarray(buf)], colors=[color]))

    def heightfield(self, path, X, Y, Z, *, color=(60, 120, 180)) -> None:
        """A triangle-mesh surface ``z = f(x, y)`` from grid arrays.

        ``X``/``Y``/``Z`` are equal-shape 2-D arrays (``np.meshgrid``
        style). Re-log every frame to animate (e.g. a water surface);
        the triangle indices are cached per grid shape. Normals come
        from the height gradient so the surface shades properly.
        """
        X = np.asarray(X, float); Y = np.asarray(Y, float)
        Z = np.asarray(Z, float)
        n_r, n_c = Z.shape
        key = (path, n_r, n_c)
        if key not in self._grid_idx:
            r, c = np.meshgrid(np.arange(n_r - 1), np.arange(n_c - 1),
                               indexing="ij")
            v00 = (r * n_c + c).ravel()
            v01, v10 = v00 + 1, v00 + n_c
            v11 = v10 + 1
            self._grid_idx[key] = np.concatenate([
                np.column_stack([v00, v01, v11]),
                np.column_stack([v00, v11, v10])])
        verts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
        dzdr, dzdc = np.gradient(Z)
        dx = np.gradient(X, axis=1); dy = np.gradient(Y, axis=0)
        normals = np.column_stack([
            (-dzdc / np.where(dx == 0, 1, dx)).ravel(),
            (-dzdr / np.where(dy == 0, 1, dy)).ravel(),
            np.ones(Z.size)])
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)
        self.rr.log(path, self.rr.Mesh3D(
            vertex_positions=verts,
            triangle_indices=self._grid_idx[key],
            vertex_normals=normals,
            albedo_factor=color))

    def disc(self, path, radius: float, *, color=(70, 110, 70, 160),
             thickness: float = 1e-3, static: bool = True) -> None:
        """A flat solid disc of ``radius`` in the entity's xy-plane —
        e.g. a water surface that follows a vehicle (log once, then
        :meth:`pose` the entity each frame)."""
        self.rr.log(
            path,
            self.rr.Ellipsoids3D(half_sizes=[[radius, radius, thickness]],
                                 colors=[color], fill_mode="solid"),
            static=static)

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
