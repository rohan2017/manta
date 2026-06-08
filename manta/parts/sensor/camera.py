"""Camera — a pinhole sensor that bounding-boxes semantic ellipsoids.

The camera enumerates the `OpticalField`'s ellipsoids (every other
vehicle's `OpticalSource`, plus any scenery), projects each quadric to
the image plane, and outputs the axis-aligned bounding box — exactly the
2-D detections a perception stack consumes.

The projection is the dual-quadric trick, fully differentiable:

  * an ellipsoid ``(x−c)ᵀM(x−c) ≤ 1`` is the 4×4 point quadric
    ``Q = [[M, −Mc], [−(Mc)ᵀ, cᵀMc − 1]]``; its dual is ``Q* = Q⁻¹``;
  * the 3×4 camera ``P = K·[R_cw | −R_cw·C]`` maps it to the image dual
    conic ``C* = P·Q*·Pᵀ`` (3×3);
  * the box edges are the conic's tangent vertical/horizontal lines:
    ``u = (C*₀₂ ± √(C*₀₂² − C*₀₀·C*₂₂)) / C*₂₂`` and the v-analogue.

Optical convention: the camera looks down its own +z axis, image +x
right, +y down. Intrinsics are a focal length (from horizontal FOV) and
the principal point at the image center.

Outputs (per visible source ``S``, five scalars):
  ``<S>_xmin``, ``<S>_ymin``, ``<S>_xmax``, ``<S>_ymax`` — pixel box,
  clamped to the image; ``<S>_vis`` — 1 when the object is in front of
  the camera and projects to a real ellipse, else 0 (the box is
  meaningless then). ``S`` is the source disturbance's name, so a box on
  craft ``rover`` carrying ``OpticalSource("beacon")`` is ``rover_beacon_*``.

The set of outputs is fixed at compile time: `World._register_field_sources`
fills `self._targets` with every ellipsoid not on the camera's own craft
before the tick is traced, and both `output_declarations` and `update`
iterate it.
"""

from __future__ import annotations

import casadi as ca

from ...fields.optical import OpticalField
from ...ir.frames import PartFrame, WorldFrame
from ...ir.types import Scalar, Vec3
from ...ir.wrench import Wrench
from ..base import Output, Parameter, Part, PartUpdate


class Camera(Part):
    """Pinhole camera emitting per-object image-frame bounding boxes.

    Construct with the image size and horizontal field of view (the focal
    length and a centered principal point are derived), or override any
    intrinsic explicitly::

        Camera("cam", width=640, height=480, hfov_deg=70)
        Camera("cam", width=1280, height=720, fx=900, fy=900, rate=30)

    Parameters:
        width, height — image size in pixels.
        fx, fy, cx, cy — intrinsics (focal lengths + principal point, px).
        rate — optional capture rate (Hz); `None` ⇒ every tick.
    """

    requires_fields = [OpticalField]

    width:  float = Parameter(640.0)
    height: float = Parameter(480.0)
    fx:     float = Parameter(0.0)
    fy:     float = Parameter(0.0)
    cx:     float = Parameter(0.0)
    cy:     float = Parameter(0.0)
    rate:   float = Parameter(None)

    def __init__(self, name: str, *, width: float = 640.0, height: float = 480.0,
                 hfov_deg: float = 70.0, fx=None, fy=None, cx=None, cy=None,
                 rate=None) -> None:
        import math
        w, h = float(width), float(height)
        f = (w / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
        super().__init__(
            name, width=w, height=h,
            fx=float(fx) if fx is not None else f,
            fy=float(fy) if fy is not None else (float(fx) if fx is not None else f),
            cx=float(cx) if cx is not None else w / 2.0,
            cy=float(cy) if cy is not None else h / 2.0,
            rate=rate)
        # Filled by World._register_field_sources (every ellipsoid not on
        # this camera's craft). Drives both output_declarations and update.
        self._targets: list = []

    # The output set is per-instance (one box per visible source), so we
    # override the class-level declaration walk with a dynamic one.
    def output_declarations(self) -> dict:
        out: dict = {}
        for e in self._targets:
            for suffix in ("xmin", "ymin", "xmax", "ymax", "vis"):
                out[f"{e.name}_{suffix}"] = Output()
        return out

    def update(self, ctx) -> PartUpdate:
        zero = Vec3[PartFrame].constant((0.0, 0.0, 0.0))
        if not self._targets:
            return PartUpdate(wrench=Wrench(force=zero, torque=zero))

        # Camera pose: C = world position; R_cw = world→camera rotation
        # (the camera looks down its own +z). ctx.orientation is
        # Quat[World, Part]; its conjugate's matrix is R_cam_from_world.
        C = ctx.position[WorldFrame]._mx
        R_cw = ctx.orientation.conjugate().to_rotmat()._mx
        fx, fy = float(self.fx), float(self.fy)
        cx, cy = float(self.cx), float(self.cy)
        W, H = float(self.width), float(self.height)
        K = ca.DM([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
        P = K @ ca.horzcat(R_cw, -R_cw @ C)          # 3×4 camera matrix

        outputs: dict = {}
        for e in self._targets:
            c = e.center_world()._mx                  # 3×1 world center
            M = e.shape_world_mx()                    # 3×3 world shape
            box, vis = _project_box(P, R_cw, C, c, M, W, H)
            tag = e.name
            vals = {"xmin": box[0], "ymin": box[1],
                    "xmax": box[2], "ymax": box[3], "vis": vis}
            for suffix, mx in vals.items():
                outputs[f"{tag}_{suffix}"] = ctx.sample(
                    Scalar(mx), rate=self.rate)
        return PartUpdate(wrench=Wrench(force=zero, torque=zero),
                          outputs=outputs)


def _project_box(P, R_cw, C, c, M, W, H):
    """Project the ellipsoid (center `c`, shape `M`) through camera `P`
    to an axis-aligned image box (clamped to [0,W]×[0,H]) and a 0/1
    visibility flag. All args are CasADi MX/DM."""
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
