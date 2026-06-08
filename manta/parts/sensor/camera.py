"""Projective cameras over the OpticalField — `BBoxCamera` + `CentroidCamera`.

Both enumerate the `OpticalField`'s semantic ellipsoids (every other
vehicle's `OpticalSource`, plus any scenery), project each through a pinhole,
and emit per-target image measurements. They differ only in WHAT they report:

  * `BBoxCamera` → the axis-aligned bounding box ``(xmin, ymin, xmax, ymax)``
    — the 2-D detection a perception stack consumes. The box SIZE encodes
    range (given the target's known semi-axes), so even ONE bbox camera is a
    range sensor — but a *size-dependent* one (a bigger-but-farther object
    looks identical).

  * `CentroidCamera` → just the projected centre ``(u, v)`` — a pure BEARING,
    INDEPENDENT of target size. One centroid pins the target to a ray; the
    EKF recovers range by triangulating several spaced cameras' rays (and the
    Kalman update IS that noise-weighted ray intersection — no triangulation
    code). Size-free, and as good as the baseline you give it.

Both project with the dual-quadric / point projection through
``P = K·[R_cw | −R_cw·C]`` (looks down its own +z; image +x right, +y down;
intrinsics from a horizontal FOV with a centred principal point). The output
set is fixed at compile time: `World._register_field_sources` fills
`self._targets` with every ellipsoid not on the camera's own craft before the
tick is traced, and both `output_declarations` and `update` iterate it.

EKF use: a target's measurement is a differentiable function of BOTH the
camera craft's pose AND the target's pose, so selecting the measurement
outputs as EKF sensors couples the two craft blocks into one joint filter.
Set ``noise`` (`bbox_sigma` / `pixel_sigma`) > 0 to declare the per-component
white noise the filter needs for a nonzero R; the ``_vis`` flag carries no
noise and must be excluded from ``sensors=[...]``.
"""

from __future__ import annotations

from typing import ClassVar

import casadi as ca

from ...fields.optical import OpticalField
from ...ir.frames import PartFrame, WorldFrame
from ...ir.types import Scalar, Vec3
from ...ir.wrench import Wrench
from ..base import Output, Parameter, Part, PartUpdate, WhiteNoise


class ProjectiveCamera(Part):
    """Base for a pinhole camera that measures `OpticalField` ellipsoids.

    Construct with the image size and horizontal field of view (the focal
    length and a centred principal point are derived), or override any
    intrinsic explicitly. Subclasses set `_COMPONENTS` (the per-target scalar
    measurement names, excluding ``vis``) and implement `_project`.

    Parameters:
        width, height — image size in pixels.
        fx, fy, cx, cy — intrinsics (focal lengths + principal point, px).
        rate — optional capture rate (Hz); `None` ⇒ every tick.
        noise_sigma — per-component pixel measurement σ (subclasses expose it
            under a friendlier name). 0 ⇒ noiseless oracle, no noise channels
            (byte-identical to a camera with none), and not EKF-usable.
    """

    requires_fields = [OpticalField]

    #: Per-target scalar measurement names (excludes the ``vis`` flag).
    _COMPONENTS: ClassVar[tuple[str, ...]] = ()

    width:  float = Parameter(640.0)
    height: float = Parameter(480.0)
    fx:     float = Parameter(0.0)
    fy:     float = Parameter(0.0)
    cx:     float = Parameter(0.0)
    cy:     float = Parameter(0.0)
    rate:   float = Parameter(None)
    noise_sigma: float = Parameter(0.0)

    def __init__(self, name: str, *, width: float = 640.0, height: float = 480.0,
                 hfov_deg: float = 70.0, fx=None, fy=None, cx=None, cy=None,
                 rate=None, noise_sigma: float = 0.0,
                 transform=(0.0, 0.0, 0.0)) -> None:
        import math
        w, h = float(width), float(height)
        f = (w / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
        super().__init__(
            name, width=w, height=h,
            fx=float(fx) if fx is not None else f,
            fy=float(fy) if fy is not None else (float(fx) if fx is not None else f),
            cx=float(cx) if cx is not None else w / 2.0,
            cy=float(cy) if cy is not None else h / 2.0,
            rate=rate, noise_sigma=float(noise_sigma), transform=transform)
        # Filled by World._register_field_sources (every ellipsoid not on
        # this camera's craft). Drives output/noise declarations and update.
        self._targets: list = []

    # The output set is per-instance (one detection per visible source), so
    # we override the class-level declaration walk with a dynamic one.
    def output_declarations(self) -> dict:
        out: dict = {}
        for e in self._targets:
            for suffix in (*self._COMPONENTS, "vis"):
                out[f"{e.name}_{suffix}"] = Output()
        return out

    # Per-instance noise: one independent white-noise channel per measurement
    # component per target, so the EKF can build a nonzero R. Mirrors
    # `output_declarations`; inert (byte-identical) when noise_sigma == 0.
    def noise_declarations(self) -> dict:
        sigma = float(self.noise_sigma)
        if sigma <= 0.0 or not self._targets:
            return {}
        decls: dict = {}
        for e in self._targets:
            for suffix in self._COMPONENTS:
                name = f"{e.name}_{suffix}_noise"
                decls[name] = WhiteNoise("R1", sigma=sigma)
                # The tick-signature walk reads `<name>_sigma` off the owner.
                setattr(self, f"{name}_sigma", sigma)
        return decls

    def _camera_matrix(self, ctx):
        """The 3×4 pinhole matrix P (+ R_cw, C, W, H) at this tick."""
        C = ctx.position[WorldFrame]._mx
        R_cw = ctx.orientation.conjugate().to_rotmat()._mx
        fx, fy = float(self.fx), float(self.fy)
        cx, cy = float(self.cx), float(self.cy)
        W, H = float(self.width), float(self.height)
        K = ca.DM([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
        P = K @ ca.horzcat(R_cw, -R_cw @ C)
        return P, R_cw, C, W, H

    def update(self, ctx) -> PartUpdate:
        zero = Vec3[PartFrame].constant((0.0, 0.0, 0.0))
        if not self._targets:
            return PartUpdate(wrench=Wrench(force=zero, torque=zero))
        P, R_cw, C, W, H = self._camera_matrix(ctx)
        noisy = float(self.noise_sigma) > 0.0
        outputs: dict = {}
        for e in self._targets:
            c = e.center_world()._mx                  # 3×1 world center
            M = e.shape_world_mx()                    # 3×3 world shape
            comps, vis = self._project(P, R_cw, C, c, M, W, H)
            tag = e.name
            for suffix in self._COMPONENTS:
                val = Scalar(comps[suffix])
                if noisy:
                    val = val + getattr(self, f"{tag}_{suffix}_noise")
                outputs[f"{tag}_{suffix}"] = ctx.sample(val, rate=self.rate)
            outputs[f"{tag}_vis"] = ctx.sample(Scalar(vis), rate=self.rate)
        return PartUpdate(wrench=Wrench(force=zero, torque=zero),
                          outputs=outputs)

    def _project(self, P, R_cw, C, c, M, W, H):
        """Return ``(components: dict[str, MX], vis: MX)`` for one target —
        `components` keyed by `_COMPONENTS`. Subclass implements."""
        raise NotImplementedError(
            f"{type(self).__name__}: must implement _project().")


class BBoxCamera(ProjectiveCamera):
    """Pinhole camera emitting per-object image-frame bounding boxes.

    Outputs per visible source ``S``: ``<S>_xmin/_ymin/_xmax/_ymax`` (pixel
    box, clamped to the image) and ``<S>_vis`` (1 when in front and a real
    ellipse, else 0). The box size encodes range given the target's semi-axes.

        BBoxCamera("cam", width=640, height=480, hfov_deg=70)
        BBoxCamera("cam", width=1280, height=720, bbox_sigma=2.0)   # EKF-usable
    """

    _COMPONENTS = ("xmin", "ymin", "xmax", "ymax")

    def __init__(self, name: str, *, bbox_sigma: float = 0.0, **kw) -> None:
        super().__init__(name, noise_sigma=bbox_sigma, **kw)

    def _project(self, P, R_cw, C, c, M, W, H):
        box, vis = _project_box(P, R_cw, C, c, M, W, H)
        return {"xmin": box[0], "ymin": box[1],
                "xmax": box[2], "ymax": box[3]}, vis


class CentroidCamera(ProjectiveCamera):
    """Pinhole camera emitting per-object image-frame CENTROIDS ``(u, v)``.

    A centroid is the projection of the target's centre — a pure bearing,
    independent of the target's size. Outputs per visible source ``S``:
    ``<S>_u``, ``<S>_v`` (pixels) and ``<S>_vis``. One camera fixes a ray;
    space several apart and select their ``_u``/``_v`` as EKF sensors and the
    filter triangulates the target's 3-D position (the wider the baseline, the
    better the range — size never enters).

        CentroidCamera("c0", width=1280, height=720, hfov_deg=40,
                       pixel_sigma=1.0)
    """

    _COMPONENTS = ("u", "v")

    def __init__(self, name: str, *, pixel_sigma: float = 0.0, **kw) -> None:
        super().__init__(name, noise_sigma=pixel_sigma, **kw)

    def _project(self, P, R_cw, C, c, M, W, H):
        u, v, vis = _project_point(P, R_cw, C, c, W, H)
        return {"u": u, "v": v}, vis


def _project_point(P, R_cw, C, c, W, H):
    """Project a world point `c` through camera `P` to an (unclamped) image
    centre ``(u, v)`` and a 0/1 visibility flag (in front + inside the image
    rectangle). All args are CasADi MX/DM. No clamping — an out-of-frame
    centroid is simply flagged not-visible (and never folded), so the
    in-frame measurement Jacobian stays clean."""
    x = P @ ca.vertcat(c, ca.DM(1.0))                # 3×1 homogeneous
    depth = x[2]
    u = x[0] / depth
    v = x[1] / depth
    in_front = (R_cw @ (c - C))[2] > 1e-6
    in_frame = (u > 0.0) * (u < W) * (v > 0.0) * (v < H)
    return u, v, in_front * in_frame


def _project_box(P, R_cw, C, c, M, W, H):
    """Project the ellipsoid (center `c`, shape `M`) through camera `P` to an
    axis-aligned image box (clamped to [0,W]×[0,H]) and a 0/1 visibility flag.
    All args are CasADi MX/DM."""
    Mc = M @ c
    Q = ca.vertcat(                                  # 4×4 point quadric
        ca.horzcat(M, -Mc),
        ca.horzcat(-Mc.T, c.T @ M @ c - 1.0))
    Cstar = P @ ca.inv(Q) @ P.T                      # 3×3 image dual conic
    c00, c02, c22 = Cstar[0, 0], Cstar[0, 2], Cstar[2, 2]
    c11, c12 = Cstar[1, 1], Cstar[1, 2]
    disc_u = c02 * c02 - c00 * c22
    disc_v = c12 * c12 - c11 * c22
    su = ca.sqrt(ca.fmax(disc_u, 1e-12))
    sv = ca.sqrt(ca.fmax(disc_v, 1e-12))
    u0, u1 = (c02 - su) / c22, (c02 + su) / c22
    v0, v1 = (c12 - sv) / c22, (c12 + sv) / c22
    ux0, ux1 = ca.fmin(u0, u1), ca.fmax(u0, u1)      # raw box (unclamped)
    vy0, vy1 = ca.fmin(v0, v1), ca.fmax(v0, v1)
    # Report the box clamped to the image rectangle.
    xmin, xmax = ca.fmax(ux0, 0.0), ca.fmin(ux1, W)
    ymin, ymax = ca.fmax(vy0, 0.0), ca.fmin(vy1, H)
    # Visible iff the center is in front of the camera, the projection is a
    # real ellipse (both discriminants positive), AND the box overlaps the
    # image rectangle (an off-frame object is not detected).
    depth = (R_cw @ (c - C))[2]
    in_frame = (ux1 > 0.0) * (ux0 < W) * (vy1 > 0.0) * (vy0 < H)
    vis = (depth > 1e-6) * (disc_u > 0.0) * (disc_v > 0.0) * in_frame
    return (xmin, ymin, xmax, ymax), vis
