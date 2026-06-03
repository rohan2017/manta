# manta — Target Architecture: Module IR + Generic Backends

**Status:** design, not yet implemented. Written to be handed to a fresh
session as the implementation brief. Self-contained — assumes only that you
can read the current tree.

**One-line thesis:** everything manta produces is a **Module** = (typed
**State** + named **Functions**). A backend implements exactly two things —
*translate a `ca.Function`* and *lower one Module generically* — and then
**every** feature (Sim, EKF, LQR, the recurrence filters, anything future)
becomes lowerable to that backend with **zero feature-specific backend
code**. This is the `evaluator` unification finished, the EKF folded in, and
the door opened to torch/jax.

---

## 0. Vocabulary (the whole ontology)

Three concepts cover 100% of what manta provides:

| Concept | Definition | Examples |
|---|---|---|
| **Function** | a pure `ca.Function`: `out = f(in…)` | the world tick `x'=f(x,u,dt,t)`; a Jacobian `F`; a sensor model `h`; an EKF `predict`; an LQR law `u=g(x)` |
| **State** | a typed, ordered layout of values carried across calls | a craft's rigid-body + bias state; an EKF's `(x, P)`; a PID's integral term |
| **Module** | a stateful wrapper = State + named Functions, exposing methods (`step(dt)`, `update(z)`) so the user never touches `x` | `Sim`, `EKF`, `LQR`, `PID`, `Madgwick` … |

CasADi is the canonical low-level op IR (`+ × matmul transpose solve sin
cos sqrt atan2 reshape concat slice diag …`, ~30 ops). Manta builds *thin*
abstractions on top — manifolds (`Manifold`/`_rotation.py`), frame-tagged
value types (`ir/types.py`) — to keep the math structured. Those are
authoring conveniences; they bottom out in `ca.Function`s.

**A backend = (1) a way to run/translate a `ca.Function`, (2) one generic
Module lowering.** Nothing else. The backend never knows "EKF" exists.

---

## 1. Where we are now (and why it's not that)

### The pipeline (good, keep)
```
declarative model (World/Craft/Part/Field)
   │  manta/tick/world_tick.py :: compile_world_tick   ← THE key transform
   ▼
recurrence Function  x_i = f(x_{i-1}, u, dt, t)  + a StateSpec   (a CompiledGraph)
   │  transforms: Sim / EKF / LQR / RecurrenceBlock
   ▼
IR "blocks" (each carries ca.Functions + a state layout)
   │  manta/codegen/  Target.lower_block dispatch on RUNTIME_KIND
   ▼
backend artifact (numpy runtime object | emitted C++)
```
The tick compiler already *is* "declarative model → recurrence function."
That framing is correct and load-bearing; everything downstream consumes
that one Function + StateSpec. Keep it central.

### What's wrong
1. **Per-(backend × block) runtime classes.** numpy alone has `NumpyWorld`,
   `NumpyEKF`, `NumpyLQR`, `NumpyRecurrence`
   (`manta/codegen/numpy/__init__.py`). Each hand-writes state threading +
   method plumbing. C++ has the parallel set. Adding a backend means
   re-writing all of them. **This is the thing to kill.**
2. **The EKF is a special kind.** `RUNTIME_KIND` has `KIND_EVALUATOR`
   (Sim/LQR/recurrence) *and* `KIND_EKF` (`manta/codegen/block.py`). A
   backend must implement `lower_ekf` separately. The only reason was the
   `(x,P)` two-part state + the measurement bus — both *separable* (see §3).
3. **CasADi is hard-wired into the IR.** `ir/types.py` wraps `ca.MX`;
   `linearization.py` calls `ca.jacobian`/`ca.solve`. Fine for numpy/C++;
   blocks torch/jax until we add a *translation* backend (§7).
4. **`ekf.py` mixes clean Kalman math with ~150 lines of slot/sensor/subset
   machinery.** The math (`Linearization.kalman_functions`) is already
   pypose-clean; the machinery around it obscures the file (§6).

### What's already right (build on it)
- The `evaluator` kind **already** lowers Sim/LQR/recurrence through *one*
  generic path: `EvaluatorSpec` (`manta/codegen/evaluator.py`) + the cpp
  `evaluator_wrapper.py` emitter + numpy's `lower_evaluator`. This is 75% of
  the Module model. The target generalizes `EvaluatorSpec` → `Module` and
  makes it the **only** kind.
- The EKF's linear algebra now lives **once**, symbolically, in
  `Linearization.kalman_functions` (`manta/linearization.py`) — both
  backends already just *evaluate* it (commit `682532d`). So the EKF is
  *already* "functions + state"; it's just not yet expressed as a generic
  Module.
- `manta/linearized_world.py` (`resolve_suffix`/`flatten_nested`/
  `freeze_complement`) is the first slice of the `LinearizedSystem` (§6).

---

## 2. The Module IR (target data model)

Replace the `(block, RUNTIME_KIND)` split with a single uniform IR object
that every transform emits. This is `EvaluatorSpec` generalized along three
axes: **(a)** State can hold plain tensors (the EKF's `P`), not just
manifold slots; **(b)** entry points can take extra *runtime* inputs (a
measurement `z`), not just `STATE/INPUTS/DT/T`; **(c)** an entry point can
write *multiple* State fields (`x` and `P`).

```python
# manta/ir/module.py  (new — backend-neutral)

@dataclass(frozen=True)
class StateField:
    name: str
    kind: str            # "manifold" | "matrix" | "vector" | "scalar"
    manifold: Manifold | None   # for kind=="manifold" (StateSpec slot)
    shape: tuple[int, ...]      # for matrix/vector/scalar
    init: Any            # default value (ndarray / scalar / identity)

@dataclass(frozen=True)
class StateLayout:
    fields: tuple[StateField, ...]
    # helpers: total ambient/tangent dims, pack/unpack, manifold-aware
    #   boxplus over the manifold fields (matrix/vector fields are Euclidean).
    # NOTE: this SUBSUMES estimation/state_spec.py::StateSpec. Migrate
    #   StateSpec to be the manifold-only special case, or fold it in here.

@dataclass(frozen=True)
class Port:
    name: str            # "u", "z", "dt", "t", "Q", …
    shape: tuple[int, ...]
    # a runtime input to a method (not stored in State)

@dataclass(frozen=True)
class EntryPoint:
    method: str                      # e.g. "predict", "update_gps", "step"
    fn: str                          # key into Module.functions
    reads_state: tuple[str, ...]     # State fields the kernel consumes
    reads_ports: tuple[str, ...]     # Ports the kernel consumes (ordered)
    writes_state: tuple[str, ...]    # State fields the kernel result writes
    returns: tuple[str, ...]         # extra outputs returned to the caller
    # arg_order: how to lay out the kernel's flat positional args from
    #   reads_state + reads_ports (+ implicit dt/t). Mirrors today's
    #   EntryPoint.kernel_args / ArgSource but generalized to named ports.

@dataclass
class Module:
    name: str
    state: StateLayout
    ports: tuple[Port, ...]
    functions: dict[str, ca.Function]    # the only thing a backend translates
    entry_points: tuple[EntryPoint, ...]
    # Analysis-only functions exposed but NOT lowered as methods (e.g. the
    # EKF's F/H for observability). Keep separate so the backend ignores them.
    analysis: dict[str, ca.Function] = field(default_factory=dict)
```

Every transform produces a `Module`:

| Transform | State | Ports | functions / entry_points |
|---|---|---|---|
| `Sim` | world `x` (manifold fields) | `u, dt, t` | `step(state,u,dt,t)→state` |
| recurrence (PID/Madgwick/…) | `x` | `u, dt, t` | `step → (state, readout)` |
| `LQR` | — (empty State) | `x_full` | `control(x_full)→u` (stateless) |
| `EKF` | `x` (manifold) + `P` (matrix) | `u, dt, t, z, Q` | `predict(x,P,Q,u,dt,t)→(x,P)`, `update_<s>(x,P,z,u,t)→(x,P)` |

The EKF is **not** special — it's a Module with a two-field State and a few
entry points. That's the whole point.

---

## 3. The generic backend contract

```python
# manta/codegen/target.py  (target shape)

class Target(ABC):
    def translate(self, fn: ca.Function) -> Any:
        """Make `fn` runnable/emittable in this backend.
           numpy: return fn itself (CasADi evaluates it).
           cpp:   register it for CodeGenerator emission; return its C name.
           torch: walk the op tape → a torch callable (§7).
           jax:   walk the op tape → a jax callable (§7)."""

    def lower_module(self, m: Module, **opts) -> Any:
        """The ONE generic lowering. Knows nothing about EKF/Sim/LQR.
           Allocate storage for m.state; for each entry point, build a
           method that packs (reads_state + reads_ports) into the kernel's
           flat args, calls translate(m.functions[ep.fn]), and scatters the
           result into writes_state / returns."""
```

`lower_block` collapses to `lower_module` — **one** dispatch, no
`RUNTIME_KIND` switch. `manta/codegen/block.py`'s `KIND_*` constants are
deleted. Delete `lower_ekf` / `lower_evaluator` split.

Per-backend `lower_module` realizations:
- **numpy**: one `NumpyModule` class — holds `{field.name: ndarray}`,
  exposes a method per entry point that calls the `ca.Function` and updates
  the dict. `NumpyWorld`/`NumpyEKF`/`NumpyLQR`/`NumpyRecurrence` **all
  collapse into this one class** parameterized by the Module. (~80 lines
  replaces ~700.)
- **cpp**: generalize `evaluator_wrapper.py` to handle matrix State fields +
  extra-input entries → emits one struct + methods. `ekf_wrapper.py`,
  `extract_ekf.py`, `_emit_ekf_cpp` **collapse into the evaluator path**.
- **torch**: `lower_module` → an `nn.Module` whose State fields are buffers
  (or `nn.Parameter` for tunables, §8), methods call the translated kernels.
- **jax**: `lower_module` → a `(params, init_state, {method: fn})` bundle;
  methods are pure `(state, *ports) → (state, out)` — `scan`/`vmap`/`grad`
  friendly.

### Separable concerns — lift these OUT of the runtime classes
These currently live inside `NumpyEKF` and make it look "special." They are
**not** part of lowering; they layer generically over any Module:

- **Measurement bus** (`feed`/`step`/staleness/ZOH, the `Signal` wiring):
  a backend-agnostic Python loop that, given a Module with `update_*` entry
  points + per-sensor sample rates, calls the right method when a sample
  arrives. Lives once, above Modules (extend `manta/signal.py`). Works on a
  numpy Module today, a torch Module tomorrow.
- **`observability` / `consistency`**: IR-level analyses over the EKF's
  `F`/`H` (already in `Module.analysis`). Not runtime methods. Keep in
  `manta/estimation/{observability,consistency}.py` but have them consume
  `Module.analysis`, not a backend object.
- **`reset` / `state_dict`**: generic State get/set on `NumpyModule` (any
  Module has a State).
- **Low-level `update(h_sym, z, R)`** (custom runtime measurement): this is
  the ONE thing that genuinely needs a runtime-supplied `h` and so can't be
  a baked kernel. Keep it as a thin numpy-only helper on `NumpyModule`
  (a `joseph_update(x, P, h, H, R)` utility), NOT a reason for a separate
  class. cpp/torch/jax don't offer it.

---

## 4. How each block becomes a Module (migration map)

- **`Sim`** (`manta/sim.py`): already emits a single tick `ca.Function` +
  StateSpec. Wrap as `Module(state=manifold fields, functions={"step": tick},
  entry_points=[step])`. The `build_evaluator_spec` Sim branch
  (`cpp/evaluator_spec.py`) becomes the generic Module builder.
- **`RecurrenceBlock`** (`manta/recurrence.py`): already (state, update_fn,
  ports). Direct map; it's the cleanest existing example of the Module shape.
- **`LQR`** (`manta/control/lqr.py`): empty State, one stateless `control`
  entry. Direct map.
- **`EKF`** (`manta/estimation/ekf.py`): State = `[x: manifold fields,
  P: matrix(tan,tan)]`; functions = `{predict, process_noise, update_<s>…}`
  (already built by `Linearization.kalman_functions`); entry_points =
  `predict` (reads x,P + ports u,dt,t,Q → writes x,P), one `update_<s>` per
  sensor (reads x,P + port z,u,t → writes x,P). `analysis = {F, H_<s>}` for
  observability. **`extract_ekf.py` + `ekf_wrapper.py` + `lower_ekf` + the
  whole `NumpyEKF` class are deleted**; the generic Module lowering produces
  the runtime.

After this, `block.py`/`RUNTIME_KIND` are gone, `evaluator.py`'s
`EvaluatorSpec` is replaced by `Module`, and there is exactly one lowering
path per backend.

---

## 5. Verification strategy (keep the suite green throughout)

Every step is behavior-preserving and roundtrip-checkable:
- The full suite (409 tests, ~2.5 min) is the backstop. Run targeted EKF +
  codegen subsets per step; full suite per milestone.
- **Roundtrip parity is the gold standard** and already exists: e.g.
  `tests/test_codegen_ekf_cpp.py` compiles + runs the generated C++ and
  compares to numpy to ~1e-7; `tests/test_codegen_roundtrip.py` likewise.
  Keep these green — they prove the generic lowering matches the old
  bespoke one bit-for-bit.
- When torch/jax land, add the analogous roundtrip: numpy Module ==
  translated-torch Module on the same op sequence.

---

## 6. Making the EKF/LQR pypose-readable (`LinearizedSystem`)

The Kalman *math* is already ~60 clean lines in
`Linearization.kalman_functions`. What clutters `ekf.py` is the *machinery*:
state-subset closure (`dependency_closure`), sensor table assembly, input
routing, freezing untracked slots. Push it **down** (as you said — into the
linearizer / tick compiler) behind a pypose-`System`-shaped object:

```python
# manta/linearized_system.py  (new — generalizes manta/linearized_world.py)

class LinearizedSystem:
    """The pypose `System` analog: a linearized recurrence + measurement
    model over a (sub)state, with everything indexed by name.

    Owns: the StateLayout (subset + frozen complement already applied),
    the input routing, the sensor set, and the Jacobian Functions. Built by
    the tick compiler + Linearization from a World — ALL slot/sensor/subset
    machinery lives here, NOT in the filter."""
    spec:    StateLayout
    f:       ca.Function   # x' = f(x,u,dt,t)
    F:       ca.Function   # ∂f/∂δ   (symbolic — manta's differentiator)
    sensors: dict[str, Sensor]   # each: h, H, R-builder, dim
    # process noise: Q = L Σ Lᵀ as a Function
```

Then the filters read like pypose — pure math over the System, no slot
plumbing:

```python
# what ekf.py SHOULD look like (~40 lines)
def ekf_predict(sys, x, P, Q, u, dt, t):
    return sys.f(x,u,dt,t), sys.F(x,u,dt,t) @ P @ sys.F(...).T + Q

def ekf_update(sys, sensor, x, P, z, u, t):
    h, H, R = sensor.h(x,u,t), sensor.H(x,u,t), sensor.R(x,u,t)
    S = H @ P @ H.T + R
    K = P @ H.T @ solve(S)                 # ldl; see gotcha (§9)
    x = sys.spec.boxplus(x, K @ (z - h))   # manifold-correct
    IKH = I - K @ H
    return x, IKH @ P @ IKH.T + K @ R @ K.T
```

`LinearizedSystem` is also exactly what `LQR` consumes (`A=F`, `B`), so both
transforms become thin readable math over one shared System. This is the
home for the `linearized_world.py` helpers + the EKF `__init__` machinery.

**Substrate-agnostic linearization.** `F = ∂f/∂δ` is `ca.jacobian`
symbolically, `torch.func.jacrev` eagerly, `jax.jacfwd` in jax. `sys.F` is
one interface; the substrate decides. For your stated experiment you want
the *symbolic* (CasADi) F baked in — see §7.

---

## 7. Lowering to torch / jax (the new capability)

**Decision: CasADi stays the *authoring + symbolic-linearization*
substrate; torch/jax are *translation targets*.** This is not a compromise —
it is what your experiment requires. You want to compare pypose's
*autodiff* EKF against a manta EKF with *symbolic* linearization; the
symbolic `F` only exists because CasADi derived it. So: author in CasADi
(get symbolic `F` baked into the kernels), then translate the resulting
graph to torch/jax.

### The translator (Path A)
A `ca.Function` is an SSA instruction tape. Walk it and replay into
torch/jax ops:
```python
def translate(fn: ca.Function, xp):   # xp = jnp or torch
    # fn.n_instructions(); for k: op = fn.instruction_id(k);
    #   ins = fn.instruction_input(k); out = fn.instruction_output(k)
    # map CasADi OP_* → xp ops:
    #   OP_ADD/SUB/MUL/DIV/NEG, OP_MTIMES→matmul, OP_SOLVE→cholesky_solve,
    #   OP_SIN/COS/SQRT/SIGN/ATAN2, OP_RESHAPE, OP_HORZCAT/VERTCAT→concat,
    #   OP_GETNONZEROS/SETNONZEROS→slice/scatter, OP_TRANSPOSE, OP_CONST…
    # ~30–50 OP codes. Returns a pure callable.
```
The op surface is small and bounded; there is **no exotic op** in the stack.
The manifold kernels (`so3_exp`/`quat_mul`) are pure elementwise+trig — they
translate trivially and (because they're branch-free / eps-regularized — see
`ir/_rotation.py`) are smooth and `grad`-friendly.

### torch backend
`translate` → torch callable; `lower_module` → an `nn.Module`:
- State manifold/matrix fields → registered buffers (or `nn.Parameter`, §8).
- Each entry point → a method calling the translated kernel.
- A `Sim` Module lowered to torch is a pypose-compatible `System`
  (provides `f` and, via the *symbolic* `F` you translated, `A`).

### jax backend
Same translator, jax op map. `lower_module` → a functional bundle
`(params, init_state, {method: (state,*ports)->(state,out)})`. Rollouts are
`lax.scan` over `step`; batch with `vmap`; differentiate with `grad`. Fixed
shapes + no control-flow-in-IR (the design rule "convergence loops stay out
of the IR") make this the XLA happy path.

### Validation targets (your experiments)
1. Lower a manta `Sim` → torch; wrap as a pypose NLS; run **pypose's** EKF.
2. Lower the manta `EKF` (symbolic F) → torch; run it.
3. Compare accuracy + speed (symbolic-baked-Jacobian vs pypose runtime
   autodiff). Roundtrip-check both against the numpy manta EKF.

---

## 8. Parameter lifting → systemID (torch/jax optimization)

Today a `Parameter` (e.g. `Mass.mass`) is baked into the tick as a constant.
For fitting, lift tunable parameters to **Function inputs** (a new Port) so
they're differentiable:
- `Parameter(default, tunable=True)` (the `manta/tuning/` placeholder's
  first concrete step — its docstring already specs this).
- The tick compiler emits the tunable params as extra kernel inputs instead
  of `ca.DM` constants.
- The torch `lower_module` registers those Ports as `nn.Parameter`; jax
  exposes them as a pytree leaf.
- SystemID loop: roll out the recurrence (`scan`/Python loop), accumulate
  `Σ‖z_pred − z_obs‖²`, backprop, step a torch/optax optimizer. Gradients
  flow through the whole physics + sensor model because the lowered Module
  is differentiable end-to-end.

This is the same mechanism whether you fit through a Sim (dynamics ID) or a
Sim+EKF (end-to-end filter tuning).

---

## 9. Invariants & gotchas discovered (don't relearn these)

- **`ca.solve(S, ·, "symbolicqr")` does NOT codegen** (`#error SymbolicQr
  does not support code generation`). Use **`"ldl"`** (Cholesky-family,
  natural for the SPD innovation `S`) — it codegens to dependency-free C and
  translates to `cholesky_solve` in torch/jax. `"qr"`/`"lsqr"` also codegen.
  Never explicit `inverse()`.
- **Dense predict == block-diagonal predict, numerically.** The
  independent-subsystem partition makes `F` exactly block-diagonal, so
  `F P Fᵀ` keeps cross-block `P` at structural zero. The current code's
  per-block optimization is *only* a speed trick; the symbolic kernel uses
  dense and matches bit-for-bit (verified). `n_blocks` stays as metadata.
- **Measurement Jacobians are evaluated at `dt=0`** (a measurement is
  dt-independent) — `kalman_functions` substitutes `dt→0` in `h/H/L_h`.
  Preserve this for parity; `t` is kept as a port (numpy passes 0 today).
- **Manifold math is branch-free on purpose** (`_so3_exp`/`_so3_log` in
  `ir/_rotation.py`, eps-regularized, no `if_else`) — needed for CasADi
  autodiff *and* a happy accident for jax/torch `grad`. Don't "fix" it into
  a conditional.
- **`StateSlot` uses `ambient_offset`/`ambient_dim` vs
  `tangent_offset`/`tangent_dim`** — covariance/Jacobians index by the
  *tangent* pair, pack/unpack by the *ambient* pair. Mixing them is a
  classic bug; the names are symmetric to make it visible. The new
  `StateLayout` must keep this distinction (manifold fields have both; plain
  tensor fields have ambient==tangent).
- **Q-override semantics**: predict takes `Q` as an explicit Port; the
  default `Q = L Σ Lᵀ` comes from a separate `process_noise` Function (it's
  state-dependent). A runtime override just skips that Function.
- **`t=0` for measurements** and the **low-level custom-h update** are the
  two numpy-only behaviors; everything else is substrate-portable.
- Manifold/value types (`ir/types.py`) currently *are* CasADi-wrapped — the
  translator works at the `ca.Function` boundary (post-trace), so you do
  NOT need to abstract `Vec3`/`Quat` to hit torch/jax. (That would be the
  heavier "native tracing" path; not needed for the stated goals.)

---

## 10. Sequencing (each step backend-agnostic + suite-green)

1. **Module IR + generic numpy `lower_module`.** Introduce `Module`/
   `StateLayout`/`EntryPoint` (`manta/ir/module.py`). Port Sim + recurrence
   + LQR (the current `evaluator` clients) onto it; collapse `NumpyWorld`/
   `NumpyLQR`/`NumpyRecurrence` into one `NumpyModule`. Keep `EvaluatorSpec`
   working in parallel until parity, then delete it. *No new capability;
   structural.*
2. **Fold the EKF into the Module model.** Express EKF as a Module
   (`(x,P)` State + predict/update entries). Delete `lower_ekf`, `NumpyEKF`,
   `extract_ekf.py`, `ekf_wrapper.py`; the generic numpy + cpp lowering
   produce it. Roundtrip tests must stay green. *This deletes the per-backend
   EKF you dislike — highest-leverage step.*
3. **Lift the bus / observability / reset** out of the (now-deleted) EKF
   class into generic layers over Modules.
4. **Extract `LinearizedSystem`**; rewrite `ekf.py`/`lqr.py` as readable math
   over it. Move the slot/sensor/subset machinery into it (absorbing
   `linearized_world.py`).
5. **cpp generic `lower_module`** — generalize `evaluator_wrapper.py` to
   matrix State fields + extra-input entries; delete the cpp EKF specifics.
6. **torch backend** — the `ca.Function` translator + `lower_module →
   nn.Module`. Validate against pypose.
7. **jax backend** — same translator, jax op map.
8. **Parameter lifting** — tunable `Parameter` → Port → `nn.Parameter`;
   systemID demo (torch + jax).

Steps 1–5 are pure structural cleanup that make "a backend = ops + one
Module lowering" literally true and the EKF pypose-readable — valuable even
if torch/jax never ship. Steps 6–8 are the new capability.

---

## 11. File-level map (current → target)

| Current | Fate |
|---|---|
| `codegen/block.py` (`RUNTIME_KIND`, `KIND_*`) | **delete** — one kind |
| `codegen/evaluator.py` (`EvaluatorSpec`) | **replace** with `ir/module.py` `Module` |
| `codegen/numpy/__init__.py` `NumpyWorld/EKF/LQR/Recurrence` | **collapse** into one `NumpyModule` |
| `codegen/numpy` `lower_evaluator` + `lower_ekf` | **merge** into `lower_module` |
| `codegen/cpp/evaluator_spec.py` + `evaluator_wrapper.py` | **generalize** to the Module builder/emitter |
| `codegen/cpp/extract_ekf.py` + `ekf_wrapper.py` + `_emit_ekf_cpp` | **delete** — EKF is a Module |
| `estimation/ekf.py` (machinery) | **shrink** to Kalman math over `LinearizedSystem` |
| `estimation/state_spec.py::StateSpec` | **fold into** `StateLayout` (manifold-only special case) |
| `linearized_world.py` | **absorb into** `LinearizedSystem` |
| `linearization.py::kalman_functions` | **keep** (the EKF math, once) — may move next to the readable filter |
| `tick/world_tick.py::compile_world_tick` | **keep** — the model→recurrence transform, unchanged |
| `ir/_rotation.py`, `ir/manifold.py`, `ir/types.py` | **keep** — the thin math abstractions; translator works post-trace |

---

## 12. The acid test (when it's done)

> Adding a backend is: implement `translate(ca.Function)` + one
> `lower_module(Module)`. You write **zero** lines that mention "EKF",
> "Sim", "Kalman", "Riccati", or "PID". Every current and future manta
> feature lowers to your backend automatically. The numpy backend turns the
> EKF Module into a stateful object with `predict`/`update_*` methods and
> has no idea it's a filter — it just works.
