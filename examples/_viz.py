"""Shared rerun visualization helpers for the manta examples.

Thin wrappers over the rerun SDK, set up to match manta's WorldFrame:
right-handed, **z-up**, metres. Manta quaternions are ``[w, x, y, z]``;
rerun wants ``[x, y, z, w]``, so :meth:`Viz.pose` converts for you.

rerun is imported lazily, so examples that don't visualize (``quickstart``)
don't need it installed. A demo creates one :class:`Viz`, logs static geometry ONCE before the loop,
then inside the loop gates visualization with :meth:`Viz.due` (so the
control/physics loop runs at its own high rate but the viewer sees ~30 Hz),
stamps the timeline with :meth:`Viz.t`, and re-logs the moving geometry::

    viz = Viz("manta/quadcopter")
    viz.box("world/quad/body", (0.1, 0.1, 0.03), color=(80, 140, 220))   # once
    for ...:
        sim.step(dt)
        if viz is not None and viz.due(t):            # ~30 Hz, not every step
            viz.t(t)
            viz.pose("world/quad", pos, quat_wxyz)    # moves the body frame
            viz.trail("world/trail", pos)             # WORLD path (see below)

Entity paths are hierarchical: a child (``world/quad/rotor0``) inherits its
parent's :meth:`pose`, so nested frames (a gimbal on a hull) compose by
logging a local transform at each level — exactly manta's frame chain. The
flip side: a world-frame overlay (a trail, a ground-truth marker) must be a
SIBLING of a posed entity, never a child, or it rides the moving body —
hence ``world/trail``, not ``world/quad/trail``.
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


# Default viz frame rate. Demos integrate far faster (50–500 Hz) but log
# geometry to the viewer at most this often — see `Viz.due`.
VIZ_HZ = 30.0


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

    def __init__(self, app_id: str, addr: str | None = None,
                 save: str | None = None) -> None:
        rr = require_rerun()
        self.rr = rr
        if save:
            # Record to a .rrd file instead of a live viewer — the run can
            # then go full speed (no real-time pacing) and you scrub it in the
            # viewer afterwards (`rerun FILE.rrd`). Best path on WSL.
            rr.init(app_id)
            rr.save(save)
        elif addr:
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
        self._last_due: float | None = None

    # ---- timeline ------------------------------------------------------

    def t(self, seconds: float) -> None:
        """Stamp subsequent logs at sim-time ``seconds`` on the timeline."""
        self.rr.set_time("sim_time", duration=float(seconds))

    def due(self, t: float, *, hz: float = VIZ_HZ) -> bool:
        """Frame-rate gate for viz logging. Returns True at most once per
        ``1/hz`` of sim-time, so a control/physics loop running far faster
        than the display can call it every step and only log ~``hz`` frames.

        This is the one place every demo throttles its visualization — the
        loop integrates at its own (high) rate, but re-sending geometry to
        the viewer faster than ~30 Hz only floods the stream without adding
        anything the eye can see. Use as ``if viz is not None and viz.due(t)``.
        Time may jump backwards between runs sharing a Viz; a backward step
        always logs (and re-arms the clock).
        """
        last = self._last_due
        if last is None or t < last or t - last >= 1.0 / hz - 1e-9:
            self._last_due = t
            return True
        return False

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

    def box(self, path, size, *, center=(0.0, 0.0, 0.0),
            color=(150, 150, 150), static: bool = True) -> None:
        """A solid box at ``center`` in the entity's frame (default origin).

        ``size`` is the box's FULL extent ``(dx, dy, dz)`` — e.g. a wing of
        chord 1.49 m passes 1.49, not 0.745. (rerun's `Boxes3D` wants
        half-extents, so this halves internally — matching `split_cylinder`,
        which likewise takes a full height.)
        """
        self.rr.log(
            path,
            self.rr.Boxes3D(centers=[_vec3(center)],
                            half_sizes=[0.5 * _vec3(size)], colors=[color],
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

    def split_cylinder(self, path, radius: float, height: float, *,
                       colors=((230, 80, 80), (235, 220, 100)),
                       static: bool = True) -> None:
        """A solid cylinder (z-axis, centred) split into two coloured
        half-cylinders (the ±x sides of the entity frame) — pose the
        entity about its z to make a spin visible (two closed meshes,
        logged once)."""
        n = 24
        hz = height / 2.0
        for half, (a0, color) in enumerate(zip((-np.pi / 2, np.pi / 2),
                                               colors)):
            ang = np.linspace(a0, a0 + np.pi, n)
            xs, ys = radius * np.cos(ang), radius * np.sin(ang)
            verts = np.vstack([
                [0.0, 0.0, +hz], [0.0, 0.0, -hz],
                np.column_stack([xs, ys, np.full(n, +hz)]),   # top arc
                np.column_stack([xs, ys, np.full(n, -hz)])])  # bottom arc
            T = lambda i: 2 + i           # noqa: E731 - tiny index helpers
            B = lambda i: 2 + n + i       # noqa: E731
            tris = []
            for i in range(n - 1):
                tris.append([0, T(i), T(i + 1)])          # top face fan
                tris.append([1, B(i + 1), B(i)])          # bottom face fan
                tris.append([T(i), B(i), T(i + 1)])       # side wall
                tris.append([T(i + 1), B(i), B(i + 1)])
            # Diametral cut face closing the half cylinder.
            tris.append([T(0), B(0), T(n - 1)])
            tris.append([T(n - 1), B(0), B(n - 1)])
            self.rr.log(f"{path}/{half}", self.rr.Mesh3D(
                vertex_positions=verts, triangle_indices=np.asarray(tris),
                albedo_factor=color), static=static)

    def disc(self, path, radius: float, *, center=(0.0, 0.0, 0.0),
             color=(70, 110, 70, 160), thickness: float = 1e-3,
             static: bool = True) -> None:
        """A flat solid disc of ``radius`` in the entity's xy-plane at
        ``center`` — e.g. a water surface that follows a vehicle (log
        once, then :meth:`pose` the entity each frame)."""
        self.rr.log(
            path,
            self.rr.Ellipsoids3D(centers=[_vec3(center)],
                                 half_sizes=[[radius, radius, thickness]],
                                 colors=[color], fill_mode="solid"),
            static=static)

    def checker_sphere(self, path, radius: float, *,
                       colors=((25, 25, 25), (245, 205, 40)),
                       n_lat: int = 16, n_lon: int = 32,
                       static: bool = True) -> None:
        """A UV-sphere checkered per octant in the two ``colors`` —
        the classic black/yellow tracking-target marker. Centred on the
        entity origin; logged once. ``n_lat`` should be even and
        ``n_lon`` divisible by 4 so the grid lines land exactly on the
        octant boundaries (crisp colour edges)."""
        lat = np.linspace(0.0, np.pi, n_lat + 1)
        lon = np.linspace(0.0, 2 * np.pi, n_lon + 1)
        verts: list = []
        vcols: list = []
        tris: list = []
        for i in range(n_lat):
            for j in range(n_lon):
                quad = []
                for a, b in ((i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)):
                    th, ph = lat[a], lon[b]
                    quad.append([radius * np.sin(th) * np.cos(ph),
                                 radius * np.sin(th) * np.sin(ph),
                                 radius * np.cos(th)])
                c = np.mean(quad, axis=0)
                color = colors[int(c[0] > 0) ^ int(c[1] > 0) ^ int(c[2] > 0)]
                k = len(verts)
                verts.extend(quad)            # per-face verts: flat colours
                vcols.extend([color] * 4)
                tris.extend([[k, k + 1, k + 2], [k, k + 2, k + 3]])
        verts = np.asarray(verts)
        self.rr.log(path, self.rr.Mesh3D(
            vertex_positions=verts, triangle_indices=np.asarray(tris),
            vertex_normals=verts / radius,
            vertex_colors=np.asarray(vcols)), static=static)

    def cov_ellipsoid(self, path, center, cov, *, nsigma: float = 3.0,
                      color=(235, 235, 255, 70),
                      min_half: float = 0.0) -> None:
        """An uncertainty ellipsoid at ``center`` (world coords): half-axes
        ``nsigma·√λᵢ`` along the eigenvectors of the 3×3 covariance
        ``cov``. Re-log every frame to animate a filter's confidence;
        ``min_half`` floors the axes so a tight filter stays visible."""
        cov = np.asarray(cov, dtype=float).reshape(3, 3)
        w, V = np.linalg.eigh(cov)
        half = np.maximum(nsigma * np.sqrt(np.maximum(w, 0.0)), min_half)
        if np.linalg.det(V) < 0.0:    # eigh may hand back a reflection
            V[:, 0] = -V[:, 0]
        self.rr.log(path, self.rr.Transform3D(translation=_vec3(center),
                                              mat3x3=V))
        self.rr.log(path, self.rr.Ellipsoids3D(
            centers=[[0.0, 0.0, 0.0]], half_sizes=[half], colors=[color],
            fill_mode="solid"))

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
