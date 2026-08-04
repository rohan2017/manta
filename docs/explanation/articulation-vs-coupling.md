# Articulation vs coupling — connecting bodies

manta gives you two ways to let parts of a system move relative to one
another:

- an **articulation** — an [`ArticulatedJoint`][manta.parts.RevoluteJoint]
  ([`RevoluteJoint`][manta.parts.RevoluteJoint] /
  [`PrismaticJoint`][manta.parts.PrismaticJoint]) — a moving DOF *inside a
  single craft*; and
- a **coupling** — a [`Coupling`][manta.Coupling] (e.g.
  [`Tether`][manta.couplings.Tether]) — a force exchanged *between two
  separate crafts*.

They look similar from a distance (both let one piece move against
another), but they are different mechanisms at different layers, and the
choice changes how the dynamics are solved. This page is the decision
guide.

## The one-sentence rule

> If the connection is a **rigid mechanical joint within one rigid-body
> assembly**, use an **articulation**. If it is a **force passed between
> two independent bodies that each keep their own full pose**, use a
> **coupling**.

A gimbal, a control-surface hinge, a reaction wheel, a landing-gear
slider, a pan–tilt camera mount — all articulations: one craft, an
internal DOF. A towed glider, a docking spring, a mooring line, a
grappling cable — all couplings: two crafts exchanging a wrench.

## What each one *is*

### Articulation — a Part in the craft tree

An [`ArticulatedJoint`](../reference/parts.md) is a `CompositePart`. You
`add()` it to a craft like any other part, and you hang a subtree of
children off it (`Mass`, sensors, thrusters, even nested joints). It
contributes exactly **one mechanical degree of freedom** — a rotation
about (`RevoluteJoint`) or a slide along (`PrismaticJoint`) its `axis` —
and the whole subtree rides that DOF rigidly. Off the DOF, the child is
geometrically *locked* to its mount: it has no independent free pose.

The joint adds one position-like state and one rate to the craft (e.g.
`angle`/`rate`). The craft is still **one rigid-body assembly** with one
6-DOF root pose plus its joint coordinates.

```python
craft = Craft("airplane")
craft.add(Mass("fuselage", mass=8.0, moi=(0.5, 1.2, 1.4)))
hinge = RevoluteJoint("aileron_L", mode="saturating",
                      stall_torque=5.0, axis=(0, 1, 0),
                      transform=(0.0, 1.5, 0.0))
hinge.add(Aerofoil("surf", ...))   # the surface rides the hinge
craft.add(hinge)
```

### Coupling — a separate object joining two crafts

A [`Coupling`][manta.Coupling] is **not** a part and does not live in any
craft tree. It names two crafts (`craft_a`, `craft_b`) and produces the
wrench pair they exchange (`compute_wrenches_sym`). You register it on the
world, after both crafts:

```python
world.add_craft(base)
world.add_craft(bob)
world.add_coupling(Tether(base, "hook", bob, "hook",
                          stiffness=5e3, damping=20.0, rest_length=1.0))
```

Both crafts keep their **own full 6-DOF state**. The coupling adds no
constraint and removes no DOF — it just feeds a force into each body's net
wrench every tick.

## Why they are different — the mechanics

This is the part that actually matters for your model's behavior.

| | Articulation | Coupling |
|---|---|---|
| Lives in | the craft tree (a Part) | the world (`add_coupling`) |
| Connects | parent ↔ child *within one craft* | *two separate crafts* |
| Constraint | hard, kinematic — removes DOF down to 1 | none — a soft force |
| DOF added | 1 (the joint coordinate) | 0 |
| How it's solved | inside the craft's joint-space mass-matrix solve | as a wrench added to each craft's net force |
| Connection stiffness | rigid (exact) | whatever you set (`stiffness`/`damping`) |
| Compile unit | already one craft = one tick | **forces both crafts into the same tick** |

**An articulation is a constraint.** The child cannot drift off its axis,
ever — the joint is exact and rigid. The dynamics emerge from a single
combined solve: the world tick assembles the craft's generalized mass
matrix over `[body ω; all joint rates]`, the Hamel bias, and the
virtual-work generalized forces, and solves the body angular acceleration
and every joint `q̈` *together*. Gyroscopic couples, Coriolis joint
torques, nested-gimbal inertia coupling, prismatic centrifugal flinging,
and recoil all fall out of that one solve — you write no coupling math.
The joint class itself supplies no dynamics formula beyond its actuator
clamp and viscous damping. (See [parts](parts.md) and
`manta/tick/joint_space.py`.)

**A coupling is a force.** There is no constraint — the two bodies are
free, and the only thing holding them together is the wrench you compute.
A `Tether` is a tension-only spring-damper — taut it pulls, slack it
exerts nothing, and it can never push. If you make the spring very stiff
it *approximates* a rigid cable, but it is never exactly rigid, and a
stiff spring is a stiff ODE (small `dt`, possible ringing). In exchange
you get two genuinely independent bodies that can separate, swing
freely, wrap, go slack — things a 1-DOF joint cannot represent. Because the wrench is
evaluated from both crafts' states at once, a coupling **fuses the two
crafts into one compiled tick** (and, downstream, into one joint EKF block
if you filter across them).

## The litmus test: count the degrees of freedom

Ask: *how many ways can the moving piece move relative to its mount?*

- **Exactly one**, and it can never come off → articulation. A hinge
  rotates and only rotates; a slider slides and only slides.
- **Six** (it keeps a full free pose, and the link only pulls/pushes) →
  coupling. A towed body can pitch, yaw, swing, and recede; the cable just
  applies tension along its line.

If you need **two or three** joint DOFs (a pan–tilt gimbal, a 3-axis
mount), you don't reach for a coupling — you **nest articulations**: stack
`RevoluteJoint`s, each carrying the next. The joint-space solve handles
the inter-joint inertia coupling for you.

## The same system, modeled both ways

The [Foucault pendulum example](../tutorials/index.md) is a clean
illustration that the choice is sometimes genuinely yours. A bob swinging
from an apex can be modeled as:

- a **`Tether`** (coupling) — base craft and bob craft, a stiff wire
  between them. This is the physically faithful suspension: a real
  Foucault pendulum hangs on a wire with a ball pivot, and the bob is free
  to swing in any direction and recede slightly.
- a **two-axis `RevoluteJoint` gimbal** (articulation) — one craft, the
  bob rigidly pivoting about a fixed apex on two stacked hinges.

Both reproduce the precession, because the inter-body Coriolis coupling
that drives it is carried either way — by the wrench exchange in the
coupling, or by the joint-space solve in the articulation. The wire wins
here only because it matches the real instrument and lets the bob recede;
the gimbal would impose an exact spherical constraint the real pendulum
doesn't have.

The general lesson: **prefer an articulation when the real connection is a
rigid mechanism** (a hinge, a bearing, a rail) — it's exact, cheap, and
adds no stiffness. **Reach for a coupling when the bodies are genuinely
separate** and connected by something compliant or detachable (a cable, a
spring, a contact), or when you need each body to keep its own full pose.

## See also

- [Parts and the declaration model](parts.md) — what a `Part` is; how
  joints sit in the craft tree
- [The three-layer pipeline](architecture.md) — where couplings enter the
  model layer
- Reference: [Model — World, Craft, Coupling](../reference/model.md),
  [Parts](../reference/parts.md) (`RevoluteJoint`, `PrismaticJoint`,
  `TetherEndpoint`)
- Code: `manta/parts/articulation/joint.py`,
  `manta/couplings/`, `manta/tick/joint_space.py`
