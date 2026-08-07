"""AddedMass — the inertia of the fluid a body drags along with it.

Accelerating a submerged body also accelerates the surrounding fluid,
which reacts with a force proportional to ACCELERATION — not velocity
(drag) and not gravity (mass). For aircraft the entrained air is ~1/800
of the vehicle's density and everyone ignores it; for a neutrally
buoyant underwater vehicle the entrained water is comparable to the
vehicle's own mass: a slender hull's transverse added mass is roughly
100 % of its displaced mass (it accelerates sideways as if it weighed
twice its dry mass), axial is 5–15 %.

The effect splits into two physically distinct halves, and this part
deliberately implements only one of them here:

* **Acceleration-proportional inertia** — `(m·I + A)·ν̇ = F`. This
  CANNOT be a wrench: a wrench that depends on acceleration would make
  the dynamics implicit (the tick validates against exactly that). The
  tick compiler collects `AddedMass` parts and augments the linear
  solve and the rotational inertia directly (see
  `tick/world_tick.py`); the gyroscopic couple of the added rotational
  inertia, `−ω×(B·ω)`, then emerges from the ordinary Euler term and
  MUST NOT be emitted here (it would be double-counted).
* **Velocity-product forces** — the added-mass Coriolis terms, which
  ARE ordinary state-dependent wrenches and live in `update()`:

      F = A(ω×ν_rel) − ω×(A·ν_rel)      (vanishes for isotropic A)
      τ = −ν_rel×(A·ν_rel)              (the Munk moment)

  The Munk moment is the destabilizing couple that turns a slender
  body broadside to the flow — the reason bare torpedo hulls are
  directionally unstable and fins exist. It falls out of the added-mass
  anisotropy; nothing extra is modelled.

Restrictions, deliberate for v1: the tensors are DIAGONAL in the
part's own frame (exact for bodies with three planes of symmetry —
any hull of revolution; `mount_orientation` rotates them for an
off-axis appendage), and they are referenced about the craft's COM —
mount this part at (or near) the COM; `mount_offset` only moves the
velocity sampling point. Off-diagonal linear↔angular coupling blocks
and joint-space linear coupling are out of scope until something
needs them.

`translational` and `rotational` are promotable, so `manta.fit` can
identify added mass from recorded maneuvers like any other physical
parameter.

Not in `structure/` with `Mass` because added mass is a FLUID effect:
it contributes no weight, never shifts the COM, and does not exist
without a `FluidField` — `contributes_inertia` stays False and the
gravity/COM rollup never sees it.
"""

from __future__ import annotations

import casadi as ca

from ...fields import FluidField
from ...ir.frames import PartFrame, WorldFrame
from ...ir.types import Vec3
from .._declarations import Parameter, PartUpdate
from ..base import Part
from ...ir.wrench import Wrench


class AddedMass(Part):
    """Diagonal added mass (kg) and added rotational inertia (kg·m²),
    in the part's own frame, about the craft's COM.

    Parameters:
        translational — (Ax, Ay, Az) kg: extra effective mass per
                        part-frame axis. Slender body along +x:
                        Ax ≪ Ay ≈ Az.
        rotational    — (Bx, By, Bz) kg·m²: extra effective rotational
                        inertia per part-frame axis. Bx ≈ 0 for a hull
                        of revolution (fluid slips around the roll
                        axis).
    """

    requires_fields = [FluidField]

    translational: "tuple[float, float, float]" = Parameter(
        (0.0, 0.0, 0.0), manifold="R3", frame=PartFrame)
    rotational: "tuple[float, float, float]" = Parameter(
        (0.0, 0.0, 0.0), manifold="R3", frame=PartFrame)

    def __init__(self, name: str, **overrides) -> None:
        super().__init__(name, **overrides)
        for label, value in (("translational", self.translational),
                             ("rotational", self.rotational)):
            vals = tuple(float(x) for x in value)
            if len(vals) != 3 or any(x < 0.0 for x in vals):
                raise ValueError(
                    f"{type(self).__name__} {name!r}: {label} must be "
                    f"three non-negative diagonal entries, got {value!r}")

    def update(self, ctx) -> PartUpdate:
        # Fluid-relative velocity at the mount point, in the part's own
        # frame — the DragSurface pattern exactly. ctx.velocity already
        # includes the rotational lever arm; mounting at the COM makes
        # it ν_com, which is the reference this part documents.
        p_world = ctx.position[WorldFrame]
        fluid = ctx.field(FluidField).value_at_sym(p_world, ctx.t)
        v_rel_world = ctx.velocity[WorldFrame] - fluid.velocity

        R_part_from_world = ctx.orientation.conjugate()
        nu = R_part_from_world.apply(v_rel_world)._mx
        omega = R_part_from_world.apply(
            ctx.angular_velocity[WorldFrame])._mx

        # Diagonal A applied element-wise; `coerce` reads the parameter
        # whether it is a plain tuple or promoted to a live input.
        a = Vec3[PartFrame].coerce(self.translational)._mx
        a_nu = a * nu

        # F = A(ω×ν) − ω×(Aν): the linear added-momentum Coriolis pair.
        # Identically zero for isotropic A — an isotropic added mass is
        # just extra mass, and the tick's inertia side carries all of
        # that.
        force_mx = a * ca.cross(omega, nu) - ca.cross(omega, a_nu)

        # Munk moment: τ = −ν×(Aν). Anisotropy again — a slender hull
        # at incidence is torqued toward broadside.
        torque_mx = -ca.cross(nu, a_nu)

        # NOTE deliberately absent: −ω×(B·ω). The rotational added
        # inertia is folded into I_com by the tick, so the ordinary
        # Euler gyroscopic term already produces it.
        return PartUpdate(wrench=Wrench(
            force=Vec3[PartFrame].from_mx(force_mx),
            torque=Vec3[PartFrame].from_mx(torque_mx)))
