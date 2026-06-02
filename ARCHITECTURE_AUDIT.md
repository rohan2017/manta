# manta — Architecture & Code-Quality Audit (2026-06-01)

Full-codebase audit after the recent additions (control/LQR, recurrence blocks,
signal bus, C++ backend, manifold unification). Focus: structure, organization,
dead code, redundancy, abstraction balance, naming. **No code was changed.**

Method: six parallel subagents (estimation, codegen, IR, model/tick,
parts/fields/planets/couplings, control/recurrence), each reading its subsystem
fully and grepping usage across `manta/`, `tests/`, `examples/`. Two highest-stakes
claims were re-verified by hand (the rotation-kernel duplication and the LQR
closure inconsistency).

**Headline:** the three-layer design (model → IR → lowering) is sound, the
`RUNTIME_KIND` dispatch seam is genuinely good, and no correctness defects exist
in the physics. The debt is *accumulated-leftover* debt: a half-finished manifold
unification, rotation math copied across four files, an EKF/LQR prologue
reimplemented twice, an abstraction (`EvaluatorSpec`) that only one backend uses,
and a respectable pile of dead code from the `compile_tick`→`world_tick` /
`World.compile`→`Sim` / cascade-rewrite churn.

---

## Cross-cutting themes (ranked by value)

### A. Rotation/quaternion math is duplicated across 4 modules — the clearest defect
`ir/manifold.py` docstring claims "ONE underlying math definition per manifold …
no parallel implementation to drift." **Verified false:**

- `_quat_mul_mx` defined **twice**: `ir/manifold.py:104` and `kinematics.py:221`.
- `_R_from_axis_angle_mx` defined **twice**: `inertia.py:172` and `kinematics.py:600`.
- Hamilton product **also** hand-inlined in `types.py:493` (`Quat.__mul__`) and a
  **third** numpy copy inside `manifold.py:350` (`boxplus_num`).
- SO(3) exp **twice**: `manifold.py:60` (`_so3_exp`, CasADi) and `manifold.py:341`
  (numpy reimpl inside `boxplus_num`, with a *different* eps placement → genuine
  drift hazard).
- quat-conjugate inlined in `manifold.py:338` (`boxminus_sym`) and `types.py:516`.
- `kinematics.py` also owns `_quat_from_axis_angle_mx` (207) and
  `_rotate_vec_by_quat_mx` (628); `types.py` owns the only quat→rotmat (554).

**Fix:** one dependency-free `ir/_rotation.py` (pure-MX + numpy kernels):
`_so3_exp`, `_so3_log`, `_quat_mul`, `_quat_conj`, `_quat_from_axis_angle`,
`_rotate_vec_by_quat`, `_R_from_axis_angle`, `_quat_to_rotmat`, plus numpy
twins (`_so3_exp_np`, `_quat_mul_np`). `manifold.py`, `types.py`, `kinematics.py`,
`inertia.py` all import from it. This breaks the current `manifold.py ↔ types.py`
import cycle cleanly (kernels move below both) and makes the SSOT claim true.
**Priority: HIGH.**

### B. The manifold unification is incomplete — per-`kind` if/elif ladders survive
The memory note says "all 14 isinstance dispatch sites collapsed," but the
*IR-construction* polymorphism was never added. Three `if kind=="scalar"/"vec"`
ladders remain, with a self-aware comment admitting "the two dispatch sites should
stay aligned":

- `parts/base.py:282-319` — `_ir_input_for_manifold` / `_ir_zero_for_manifold` /
  `_ir_add_for_manifold` (handle scalar/vec only — would `NotImplementedError` on quat).
- `world_tick.py:316-330` — the per-State ladder (handles scalar/vec/**quat** — so
  it's *already out of sync* with base.py).
- `world_tick.py:234-239` — disturbance-state ladder.

**Fix:** add `Manifold.ir_input(name)`, `Manifold.ir_zero()`, `Manifold.ir_add(a,b)`
to the ABC, one impl per subclass (the typed factories `Vec3[frame].input`,
`Scalar.input`, `Quat[...].input` already exist). Delete all three ladders. The
R3 `frame or CraftFrame` default moves into `R3Manifold.ir_input`. **Priority: HIGH.**
This is the change that actually finishes the unification the codebase claims to have.

### C. Three competing vocabularies for the same three manifolds
- `State(manifold=…)`: `"R1"/"R3"/"SO3"` (via `manifold_from_shortcut`, manifold.py:365).
- `Noise(…)` / `Output(shape=…)`: `"scalar"/"vec3"/"vec4"` (via a **separate**
  `_noise_manifold_from_shortcut`, parts/base.py:263, whose docstring admits it
  exists only for "ergonomic continuity with the prior `shape=` kwarg" — a compat
  artifact, contra the repo's own `no-compat-aliases` rule).
- Backend internal `kind`: `"scalar"/"vec"/"quat"`.

So a user writes `State(manifold="R3")` but `WhiteNoise("vec3", …)` for the
identical `R3Manifold`. **Fix:** delete `_noise_manifold_from_shortcut`, route Noise
and Output through `manifold_from_shortcut`, converge user vocab on `R1/R3/SO3`,
keep `kind` as the internal backend key only. **Priority: MEDIUM-HIGH.**

### D. EKF and LQR reimplement the same "subset + linearize a world" prologue
Both: build the tick, `StateSpec.from_world`, `walk_tick_signature`, resolve inputs
by suffix, flatten the nested init dict, optionally subset + freeze, then
`Linearization(...)`. Concrete duplications (all verified):

- **Suffix name-resolution: 8 near-identical copies** — `ekf.py:317` & `:340`,
  `lqr.py:150`, `numpy/__init__.py:238/503/628/915/1066`, + `state_spec.py:333`.
  Each is "exact match → unique `.`-suffix match → raise ambiguous/unknown" with
  subtly different error strings.
- **Nested→flat flatten: 5 copies** — `lqr.py:79` (`_flatten_nested`), `ekf.py:208`,
  `numpy/__init__.py:376/441/828`.
- **Frozen-untracked loop: byte-identical** — `ekf.py:235-239` ≡ `lqr.py:176-180`.
- **Pack-with-defaults: 3-4 copies** — `observability.py:108`, `consistency.py:189`,
  plus the EKF/LQR init packs.

Plus a **semantic inconsistency**: EKF closes the `track` set over the dynamics
(`_closure`, ekf.py:291 — reads F sparsity, pulls in any slot a tracked slot
depends on); **LQR does not** (`kept = set(track)`, lqr.py:172 — freezes the rest at
`x_ref`). Same `track=` keyword, two meanings. For LQR, freezing a *dynamically
coupled* state at its operating point silently drops that coupling column from
`A`/`B`, so the Riccati solve optimizes a reduced plant the user didn't explicitly
ask for. Defensible as "reduced-order LQR," but it must be either shared with EKF's
closure or *documented as a deliberate divergence with a warning when a frozen slot
is structurally depended-on by a tracked one*. (Not flatly a bug — at the
equilibrium the frozen states equal their refs — but a sharp edge for an
underactuated craft.)

**Fix:** a shared `manta/linearized_world.py` (or factory on `Linearization`)
exposing `prepare(world, *, track, inputs, frozen_init, control=) -> (spec, frozen,
sig, cf)`; promote `_closure` onto `Linearization`; one module-level
`resolve_suffix(name, candidates, *, label)`; one `StateSpec.pack_with_defaults`.
EKF and LQR become thin callers. **Priority: HIGH** (subsumes ~6 dup findings + the
closure inconsistency).

### E. `StateSlot` ambient/tangent naming asymmetry (a known footgun)
`state_spec.py:113-124`: ambient layout is bare `offset`/`dim`; tangent layout is
`tangent_offset`/`tangent_dim`. So `slot.offset` (ambient) sits next to
`slot.tangent_offset` — exactly the asymmetry the memory note already records as a
past bug ("index covariance/Jacobians by `tangent_offset` NOT `.offset`"). Flagged
independently by two agents. **Fix:** rename `offset→ambient_offset`, `dim→ambient_dim`
(symmetric, self-documenting). ~6 call sites, mechanical. **Priority: MEDIUM.**

### F. `EvaluatorSpec` is a cpp-only abstraction — numpy has a fully parallel path
`codegen/evaluator.py` sits at the *top level* of codegen presented as the
"backend-agnostic shape of an evaluator block," but grep confirms **zero** imports
outside `codegen/cpp/`. The numpy `lower_evaluator` (numpy/__init__.py:1091)
ignores it entirely and isinstance-dispatches to hand-written `NumpyWorld`/`NumpyLQR`/
`NumpyRecurrence`. The unification the docstrings sell is real only inside cpp.
**Fix (choose):** (1) *demote* — move `evaluator.py` into `codegen/cpp/` and stop
implying cross-backend neutrality (low effort, honest); or (2) *promote* — make the
numpy runtimes consume `EvaluatorSpec` too (its entry-points/field/pack arithmetic
is exactly what numpy hand-rolls), deleting most of `NumpyLQR.control`/
`NumpyRecurrence._pack_u` (higher effort, abstraction finally earns its keep).
**Priority: MEDIUM-HIGH** (decide direction first).

Related: **`numpy/__init__.py` is a 1121-line monolith** holding five unrelated
runtimes. Split per-runtime (`numpy/world.py`, `ekf.py`, `lqr.py`, `recurrence.py`,
`driver.py`) to mirror the cpp layout. **Priority: MEDIUM.**

### G. Part and Disturbance duplicate the entire declaration-host machinery
`fields/base.py:98-139` re-implements `_apply_declarations` + the five
`*_declarations` classmethods + the noise-sigma override routing — byte-for-byte
from `parts/base.py:555-615`. **Fix:** extract a `DeclarationHost` mixin (new
`manta/declarations.py` or in parts/base.py); `Part` and `Disturbance` both inherit.
~80 lines collapse to one. **Priority: MEDIUM.**

---

## Folder-structure assessment

The tree is **mostly sensible**. Concrete issues:

1. **Recurrence blocks are split incoherently.** PID lives in `control/`;
   Madgwick/Mahony/IMUIntegrator live in `estimation/` — yet all four are the same
   `RecurrenceBlock` / `RUNTIME_KIND="evaluator"` peers, re-exported flat from the
   top-level `__init__`. Nothing keys on the directory. **Proposal:** `manta/blocks/`
   for the four freestanding recurrence blocks; keep `LQR` in `control/` (it's a
   *world transform*, not a freestanding block) and `EKF` in `estimation/`. The split
   becomes "world-transforms (sim/ekf/control) vs freestanding-blocks," which is
   real, instead of the current "estimation vs control," which isn't.
2. **`manta/tuning/`** is an empty documented placeholder (`__all__ = []`, never
   imported). Intentional, but it's dead weight in the tree — fine to keep given the
   docstring is a genuine design note, but worth a conscious keep/drop decision.
3. **`ir/manifold.py` owns the project's only quaternion-multiply primitive** — odd
   for a "metadata" module; the kernel extraction in Theme A relocates it.
4. `types.py` (571 lines) is large but **cohesive** (the four frame-tagged value
   types + shared plumbing) — do **not** split it.
5. cpp's 12 files are **mostly justified** (extract vs wrapper is a real seam); the
   problem is mislocated shared helpers (`_call` lives in `ekf_wrapper.py` but is
   imported by `evaluator_wrapper.py`; `_densify` lives in `extract_ekf.py` but is
   imported by the LQR builder). Move both to a shared `cpp/_casadi.py`.

---

## Dead / unused code inventory (consolidated, grep-verified)

| Item | Location | Note |
|------|----------|------|
| `EKF._craft_of_part` | ekf.py:137-142 | built every compile, **never read** |
| 4 unused EKF IR attrs `_u_sym`/`_n_sym`/`_n_noise`/`_lin` | ekf.py:246-250 | assigned, never read (`_x_sym` IS used — keep) |
| `_wrench_to_craft` | craft.py:341-378 | pre-cascade fossil, zero callers |
| `KinematicState` fields `velocity_body_in_craft`, `acceleration_body`, `angular_acceleration` + feeders | kinematics.py:116,122-124,279,486,509,513 | never read (live values ride `frame_views`); ~8 lines of symbolic work/compile |
| `t` param threaded through kinematic pass | kinematics.py:242,339,352; world_tick.py:387 | no kinematic quantity uses it |
| `g_ctx` param of `_trace_craft_pass1` | world_tick.py:269,146 | never touched |
| `emit_kernels` + `kernel_function_names` | cpp/kernels.py:28,74 | test-only; prod uses `emit_kernel_list` |
| `emit_wrapper` shim (whole file) | cpp/wrapper.py | test-only compat shim (contra no-alias rule) |
| `CompiledGraph.generate_c` | ir/graph.py:254 | zero callers; CasADi flat-C path unused |
| `_check_frame` | ir/frames.py:119 | zero callers (frame checks inlined elsewhere) |
| `_try_current_graph` | ir/graph.py:38 | zero callers |
| `cpp_class` ClassVar | parts/base.py:532 | never read |
| `Earth.height_above_sea_level` | planets/earth.py:154 | zero callers |
| `LQR.P` | lqr.py:205 | set, never read (keep only if intentionally public) |
| `ReturnSpec.state_res`/`out_res` | codegen/evaluator.py:82-83 | always default (0/1) |
| `EvaluatorSpec.funcs` | codegen/evaluator.py:108 | pass-through `Any` grab-bag for cpp result only |
| `ArgSource.ZERO_DT` | codegen/evaluator.py:56 | C-ism leaked into "neutral" IR |
| `_h_supports` parallel array | linearization.py:195 | derivable from `outputs[*].observed_cols` |
| `Output(shape="vec4")` branch | parts/base.py:88-92 | latent dead — no quat output wired |
| `resolved_signal_manifold` frame=None branch | parts/base.py:207-211 | no frameless noise exists |
| `requires_fields`/`requires_planet` verification | sim.py:73-88 | **vacuous** — no part ever sets them (decide: wire or delete) |

---

## Verbosity: "does ekf.py need to be that verbose?"

Largely **no, but the logic is tight** — of 409 lines, ~150 are executable; the rest
is docstrings (some triplicated: the EKF-as-IR story is retold in module + class +
`__init__` + two inline comments; ekf.py:276-279 is a verbatim subset of 375-384).
Concrete trims beyond Theme D: the `_sensors` table is a stringly-typed
`dict[tuple, dict[str,Any]]` read by literal keys in three places (ekf.py:262-274) —
replace with a frozen `Sensor` dataclass (the cpp side already has `EkfSensorSpec`
for the same data). Net achievable: ekf.py drops well under 300 lines with no loss
of capability.

---

## Smaller findings worth a pass

- **Sensor zero-wrench boilerplate** repeated ~9× (`Vec3[PartFrame].constant((0,0,0))`
  then a zero `Wrench`): make `PartUpdate.wrench` optional (default `Wrench.zero`);
  pure sensors return `PartUpdate(outputs={…})`. (parts/sensor/*, attachment/*)
- **`ctx.orientation.conjugate().apply(world_vec)`** + identical 4-line comment copied
  ~7× → add `ctx.to_part_frame(vec)` helper. (parts/sensor/*, aero/*)
- **`Coupling` ABC** is a plain class with the *real* method (`compute_wrenches_sym`)
  undeclared and duck-typed; either make it a true `abc.ABC` with an `@abstractmethod`
  or fold `base.py` into `tether.py` until a 2nd coupling lands.
- **`Mat3[CraftFrame, CraftFrame]` frame-typing leak**: `R_craft_from_input` is really
  `[CraftFrame, PartFrame]` (world_tick.py:434 even retags it), defeating frame-checking
  at that seam. (kinematics.py:120,281,377)
- **`pack_inputs` emitted by two functions** (`_structs.py:130` scalar vs
  `evaluator_wrapper.py:245` vector-aware) — keep only the general one.
- `_trace_craft_pass1` is a **344-line monolith** (world_tick.py:269-612) — followable
  via banners, but extractable into ~4 named phase helpers (rebind-IO, run-updates,
  cascade, com-relative).

---

## Proposed remediation sequencing

Ordered to minimize churn collisions (each builds on the prior):

1. **Kernels** (Theme A) — `ir/_rotation.py`; pure dedup, unblocks the import-cycle.
2. **Finish the manifold unification** (Theme B) — `Manifold.ir_input/ir_zero/ir_add`;
   delete the 3 ladders. Do C in the same pass (one resolver, one vocab).
3. **Dead-code sweep** (the table) — low-risk deletions; do before refactors so you
   refactor less surface.
4. **`StateSlot` rename** (Theme E) — mechanical, kills a known footgun.
5. **Shared `linearized_world`** (Theme D) — the big structural win; collapses EKF/LQR
   prologue, the 8 suffix-resolvers, the 5 flatteners, and resolves the LQR-closure
   inconsistency in one place.
6. **`DeclarationHost` mixin** (Theme G).
7. **EvaluatorSpec direction + numpy split** (Theme F) — decide demote-vs-promote first.
8. **Folder move** (`manta/blocks/`) — last, since it touches the most imports
   (per the no-alias rule, fix every import site).

Each step is independently shippable and test-covered (419 tests today).
