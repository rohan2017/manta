# manta — Refactor Implementation Plan

Executing the approved audit fixes. Baseline: **409 tests green** (150s).
Sequenced to minimize churn collisions; each step ends green before the next.
Full suite at every milestone; targeted test files during a step.

Approved scope: A, B, C, D, E, F, G + dead-code sweep (keep
`requires_fields`/`requires_planet`) + a folder reorg for machinery modules.

---

## Step A — Unified rotation kernels (`ir/_rotation.py`)  [DONE-marker per step]
New dependency-free module (casadi + numpy only — sits *below* `types.py`/
`frames.py`, breaks the manifold↔types cycle). Holds the canonical MX kernels
+ numpy twins:
- `so3_exp(omega)` / `so3_log(q)`  (from manifold.py:60/83)
- `quat_mul(a, b)`                 (canonical; dup at manifold.py:104 + kinematics.py:221)
- `quat_conj(q)`                   (new; inline in manifold.boxminus_sym + types.conjugate)
- `quat_from_axis_angle(axis, ang)`(from kinematics.py:207)
- `rotate_vec_by_quat(q, v)`       (from kinematics.py:628)
- `R_from_axis_angle(axis, ang)`   (canonical, with kinematics' defensive shape
                                     handling; dup at kinematics.py:600 + inertia.py:172)
- `quat_to_rotmat(q)`              (from types.py:554 `_rotmat_mx`)
- numpy: `so3_exp_np(omega)`, `quat_mul_np(a, b)`  (replace boxplus_num's inline copies)

Rewire: `manifold.py`, `types.py`, `kinematics.py`, `inertia.py` import from it
and delete their local copies. **Verified identical** by hand (same Hamilton
product, same Rodrigues, same R(q)). Risk: low (behavior-preserving).
Tests: test_manifold, test_so3_state, test_rigid_body, test_jacobian, then full.

## Step B — Finish manifold unification (kill the kind ladders)
Add to the `Manifold` ABC + each subclass:
- `ir_input(name) -> Scalar|Vec3|Quat`  (typed IR input symbol)
- `ir_zero() -> <IR value>`
- `ir_add(a_val, b_mx) -> <IR value>`    (type-preserving sum for RW)
Delete `_ir_input_for_manifold`/`_ir_zero_for_manifold`/`_ir_add_for_manifold`
(parts/base.py) and the per-State `if kind==` ladder (world_tick.py:316-330) →
`manifold.ir_input(name)`. R3's `frame or CraftFrame` default moves into
`R3Manifold.ir_input`. Risk: medium (touches world_tick state construction).
Tests: test_part_state, test_r3_state, test_so3_state, test_process_noise,
test_noise_driver, then full.

## Step C — One manifold vocabulary (R1/R3/SO3)
Delete `_noise_manifold_from_shortcut` (parts/base.py:263). Route `Noise` through
`manifold_from_shortcut` (add a `vec`-frame path). Update all stock-part noise
decls from `"vec3"`/`"scalar"` → `"R3"`/`"R1"`. Fold `Output.shape` onto the same
vocab (accept `"R1"/"R3"`, or carry a Manifold). Keep backend `kind` internal.
Risk: medium (call-site sweep across parts/). Done in same pass as B where they
overlap. Tests: test_sensor, test_process_noise, full.

## Step D — Shared `linearized_world` helper (subsumes EKF/LQR dup + closure fix)
New `manta/linearized_world.py` (or methods on `Linearization`):
- `resolve_suffix(name, candidates, *, label)` — the 8× suffix resolver.
- `flatten_nested(d)` / `StateSpec.pack_with_defaults(world, overlay)`.
- `freeze_complement(full_spec, kept, init_flat)`.
- promote `_closure` (ekf.py:291) → `Linearization.dependency_closure(seed)`.
- a `prepare_subset(world, *, track, ...)` that both EKF and LQR call.
**LQR closure fix:** LQR applies the same dynamics closure as EKF (was `kept =
set(track)` verbatim). Document the behavior change. Risk: medium-high (core to
both transforms). Tests: test_ekf*, test_lqr, test_observability, full.

## Step E — `StateSlot` ambient naming (`offset→ambient_offset`, `dim→ambient_dim`)
Mechanical rename in state_spec.py + ~6 call sites (linearization.py:318,
lqr.py:213, evaluator_spec.py:70, numpy packers). Risk: low (compiler catches
misses — but it's attribute access, so grep carefully). Tests: full.

## Step F — `EvaluatorSpec` truly backend-neutral (numpy consumes it too)
Make `NumpyWorld`/`NumpyLQR`/`NumpyRecurrence` build from the shared
`EvaluatorSpec` (entry-points/fields/pack arithmetic) instead of hand-rolling.
Remove C-isms from the neutral IR (`ArgSource.ZERO_DT`, `ReturnSpec.state_res/
out_res`, `EvaluatorSpec.funcs`). Move `evaluator.py` stays top-level (now genuinely
shared). Split `numpy/__init__.py` per-runtime. Move cpp `_call`/`_densify` to
`cpp/_casadi.py`. Risk: high (largest). Tests: test_codegen*, test_end_to_end, full.

## Step G — `DeclarationHost` mixin (Part/Disturbance)
Extract `_apply_declarations` + the five `*_declarations` + sigma-routing into a
mixin both `Part` and `Disturbance` inherit. Risk: low-medium.
Tests: test_fields, test_combining_modes, test_wind_bubble, full.

## Dead-code sweep (interleaved, mostly after A/B)
Delete (grep-verified zero prod callers), KEEPING requires_fields/requires_planet:
`EKF._craft_of_part`, EKF `_u_sym/_n_sym/_n_noise/_lin`, `_wrench_to_craft`,
4 KinematicState fields + feeders, kinematics `t` param, `_trace_craft_pass1`
`g_ctx`, cpp `emit_kernels`/`kernel_function_names`, cpp/wrapper.py shim,
`CompiledGraph.generate_c`, `frames._check_frame`, `graph._try_current_graph`,
`cpp_class`, `Earth.height_above_sea_level`, `LQR.P`, `_h_supports`,
`Output(vec4)`, `resolved_signal_manifold` frame=None branch. Update the 2-3
tests that pin test-only helpers (wrapper/kernels).

## Folder reorg (last — most import churn)
Group machinery modules to signal purpose. Proposal (decide before doing):
- `manta/ir/` already good.
- Consider `manta/runtime/` or keep flat. Discuss with user before moving the
  top-level machinery (`world_tick.py`, `kinematics.py`, `inertia.py`,
  `tick_signature.py`, `linearization.py`, `signal.py`, `recurrence.py`).
Per the no-alias rule: move + fix every import, no shims.

---

### Progress log
- [x] Baseline: 409 green.
- [x] **A** — `ir/_rotation.py` unified kernels; manifold/types/kinematics/inertia
  import it; `imu_integrator` off private kernels. Full suite green.
- [x] **B** — `Manifold.ir_input/ir_zero/ir_add`; deleted the 3 `_ir_*_for_manifold`
  ladders + the world_tick State ladder. The unification is now actually finished.
- [x] **C** — one vocab `R1/R3/SO3`; deleted `_noise_manifold_from_shortcut`;
  `Output` shape → `R1/R3/SO3` (+ `_SHAPE_DIM`); swept all part/field/test call sites.
- [x] **dead-code sweep** — all listed items removed (kept `requires_fields`/
  `requires_planet` per request): `_craft_of_part`, 4 EKF IR attrs, `_wrench_to_craft`,
  4 KinematicState fields + feeders, kinematics `t` param, `g_ctx` param,
  `emit_kernels`/`kernel_function_names`, `wrapper.py`, `generate_c`, `_check_frame`,
  `_try_current_graph`, `cpp_class`, `Earth.height_above_sea_level`, `Output(vec4)`.
- [x] **E** — `StateSlot.offset→ambient_offset`, `dim→ambient_dim` (symmetric with the
  tangent pair); swept ~10 files + tests, discriminating slot `.dim` from FieldSpec/
  Port/Noise `.dim`. Full suite green.
- [x] **G** — `DeclarationHost` mixin in parts/base.py; `Part` and `Disturbance` both
  inherit; deleted Disturbance's byte-identical copy.
- [x] **D** — `manta/linearized_world.py` (`resolve_suffix`, `flatten_nested`,
  `freeze_complement`) + `Linearization.dependency_closure`. EKF + LQR + the 4 numpy
  suffix-resolvers all route through them. **LQR closure: investigated applying EKF's
  dynamics-closure to LQR — it breaks the core underactuated case (closure pulls the
  frozen attitude back in because body-frame thrust couples velocity↔orientation, so
  the reduced system is no longer stabilizable and Riccati diverges). Reverted to
  verbatim `track` with an explicit docstring on why LQR deliberately diverges from
  EKF.** This is the "pursue a fix, find it doesn't fit, report" case.
- [x] **F (partial)** — made `EvaluatorSpec` *genuinely* backend-neutral: removed the
  `ArgSource.ZERO_DT` C-ism (emitter now synthesizes the zero dt) and the always-default
  `ReturnSpec.state_res/out_res`. Moved the cross-file-coupled `_call`/`_densify` into a
  shared `cpp/_casadi.py` (`emit_kernel_call`/`densify`). Roundtrip C++ compile-and-run
  tests green.
  - **DEFERRED / RECOMMEND-AGAINST: "numpy also consumes EvaluatorSpec."** On inspection
    the numpy runtimes' value is the live *signal-bus* (Signal ports, latched ZOH
    commands, estimate-gather-by-layout, `wire()`), which `EvaluatorSpec` neither models
    nor should — it's a *source-emission* IR (typed structs + entry-point methods).
    Forcing numpy onto it shares only the trivial pack→call→unpack core while coupling
    two dissimilar runtime models. Conclusion: the audit smell ("neutral-looking but
    cpp-only") is resolved by making the spec *honestly* neutral (done) — numpy
    legitimately doesn't emit source, so it doesn't consume it. Awaiting user call on
    whether to force the coupling anyway.
  - **DEFERRED: numpy per-runtime file split** — pure organization (the monolith is
    smaller now after the dedup); offered as a follow-up.
