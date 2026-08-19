# The three-layer pipeline

manta is organized as three layers with an explicit boundary at every
step. The model is **declarative**; the transforms are **siblings** over
it, each owning its math and emitting a typed `Module` IR; a `Target*`
lowers any Module to a backend.

```
Model            Transform            Target              Result
─────────────────────────────────────────────────────────────────
World        →   Sim(world)       →   TargetNumpy(sim) →  NumpyRuntime
                  (linearized tick)                        .step() / .outputs()

World        →   EKF(world)       →   TargetNumpy(ekf) →  NumpyRuntime
                  (Kalman recursion,                       .update() / .predict()
                   baked kernels)                          (you own the loop)

World        →   LQR(world, …)    →   TargetNumpy(lqr) →  NumpyRuntime
                  (Riccati → gain K)                       .control(state) → {input: u}

any of these →   .module()        →   TargetCpp(x, …)  →  <basename>.cpp/.hpp
                                                           + flat-C kernels + CMake
```

## Layer 1 — Model

A [`World`][manta.World] holds [`Craft`][manta.Craft]s,
[`Planet`][manta.Planet]s, [`Coupling`][manta.Coupling]s, and shared
[`Field`](fields.md)s. A craft is a tree of [`Part`](parts.md)s, each
declaring its channels at class scope:

- `Parameter` — frozen at construction, baked into the graph.
- `State` — mutable per-tick state, with a manifold (`R1`, `R3`, `SO3`).
- `Input` — per-tick user-supplied value (e.g. throttle).
- `Output` — per-tick observable (a sensor reading).
- `Noise` — white noise or a random-walk bias state.

The model is pure description: nothing executes at this layer.

### Authoring models and transform snapshots

`World`, `Craft`, fields, and parts are editable authoring objects. Constructing
a transform takes a private snapshot and resolves deferred model behavior on
that copy: planets register their disturbances, field-source parts emit their
sources, camera targets are discovered, and planet-frame initial state is
lowered to the world frame. A failed resolution discards the private copy and
does not partially mutate the authoring model.

Existing transforms never change when the authoring model is edited. A later
`Sim(world)` or `EKF(world)` captures the later revision, which makes iterative
model comparison possible without a lock/finalize lifecycle.

Every transform exposes that resolved revision as an immutable
[`ModelArtifact`][manta.ModelArtifact] in its `.model` attribute. The artifact
has stable content identity, a validated state/input/sensor layout, and an
editable `world_copy()`. It can be passed directly to another transform. This
keeps the common workflow deliberately open-ended:

```python
sim = Sim(world)                 # revision A
world.crafts[0].remove("camera")
ekf = EKF(world)                 # revision B
world.crafts[0].add(camera)
ukf = UKF(world)                 # revision C

# Rebuild from exactly A, independently of later authoring edits.
replay_filter = EKF(sim.model)
```

The validated artifact is therefore a boundary between revisions, not a
one-way `finalize()` operation on the authoring objects.

## Layer 2 — Transform

`Sim(world)`, `EKF(world)`, and `LQR(world, …)` are **pure compile-time**.
Each writes its math symbolically over the shared `LinearizedSystem`
(manifold-aware F / B / H / L over the compiled world tick) and emits a
typed `Module` — a state layout plus named CasADi kernels plus typed
entry points. A Module is **not directly callable**: it is data
describing the computation.

The three transforms are siblings — none is privileged, none knows about
the others, and each owns exactly its own math (forward dynamics for
`Sim`; the Kalman recursion for `EKF`; the Riccati solve for `LQR`).

## Layer 3 — Target

A `Target*` lowers a Module to a backend:

- [`TargetNumpy`][manta.TargetNumpy] → the one native-Python
  `NumpyRuntime`. Its surface is *derived from the Module's shape* — a
  Sim Module yields `.step()`/`.outputs()`, an EKF Module yields
  `.update()`/`.predict()`, an LQR Module yields `.control()`.
- [`TargetCpp`][manta.TargetCpp] → a typed Eigen C++ class over flat-C
  kernels, plus a CMake project, for embedded deployment.
- [`TargetJax`][manta.TargetJax] → a jitted JAX rollout.

Backends contain **no per-transform code**: each implements exactly one
generic lowering of a Module. This is what keeps the numpy and C++ paths
behaving identically — you own the same driving loop in both.

## Why error-state, why CasADi

The rigid-body state lives on a manifold (orientation is `SO(3)`), so the
EKF is an **error-state** filter: covariance and updates live in the
tangent space, and the model carries manifold-correct `boxplus`/`boxminus`
so attitude never leaves the unit-quaternion sheet. CasADi gives manta
the symbolic graph + autodiff it needs to assemble F/B/H/L (and the
process/measurement noise) automatically from the declared model, and to
lower the same graph to C.

## See also

- [Parts and the declaration model](parts.md)
- [Articulation vs coupling — connecting bodies](articulation-vs-coupling.md)
- [State estimation (error-state EKF)](estimation.md)
- [Codegen and backends](codegen.md)
