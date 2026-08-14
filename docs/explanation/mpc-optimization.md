# RTI MPC optimization roadmap

!!! note "Design note — 2026-08-13"
    This records promising follow-on work after the first direct-RTI
    performance pass. It is a roadmap, not a commitment to a particular
    solver, precision, learned component, or controller cascade.

## Objective

Make direct actuator-level RTI reliable at 20 Hz on vehicle hardware while
leaving enough compute for navigation, sonar, cameras, autonomy, logging, and
communications. Optimize tail latency and interference tolerance, not just an
idle-workstation median.

Performance work must preserve the reasons to use MPC:

- one controller implementation for fin-controlled, six-DOF, surface, and
  future vehicles;
- direct optimization of the physical actuators declared by the model;
- model-aware motion planning over a roughly ten-second preview;
- hard actuator and scheduled state constraints;
- replanning after an actuator is removed or its authority changes;
- multi-craft worlds and coupled dynamics; and
- unconstrained orientation unless guidance explicitly supplies an attitude
  objective or envelope.

Guidance remains responsible for paths, speed schedules, bank schedules, and
mission intent. Vehicle configuration and reduced-model artifacts remain
Shiver concerns. Manta owns the generic optimizer and vehicle mathematics.

## Current implementation

Manta currently runs one tangent-space, direct-multiple-shooting RTI update per
control tick. It shifts the nonlinear actuator plan, rolls out and linearizes
the complete reduced world, assembles one sparse QP, solves it through either
OSQP or HPIPM, then rolls out the accepted actuator plan.

The current optimized path has:

- `-O3 -march=native` compiled horizon dynamics and derivative kernels;
- compiled horizon objective and accepted-plan diagnostic kernels;
- persistent NumPy assembly and native solver workspaces;
- a sparse OSQP reference backend with split update/iteration diagnostics;
- an HPIPM optimal-control backend with optional partial condensing;
- an augmented HPIPM stage state that preserves the exact actuator-slew cost;
- no per-stage HPIPM heap allocation in steady state; and
- shifted primal and dual warm starts.

The numerical path is currently double precision. NumPy arrays use
`float64`, and the HPIPM bridge uses its `d_ocp_qp` API. The controller is
compiled, but it is not an F32 implementation.

### Reference measurements

The following measurements are useful for comparing subsequent changes, but
are not release thresholds. They were collected on the development machine
with 40 nodes, 0.25 s prediction spacing, a ten-second preview, compiled
kernels, HPIPM, and `condense_to=8`:

| Workload | Median | p99 | Worst | Result |
| --- | ---: | ---: | ---: | --- |
| Matched-model, 20 s | 5.35 ms | 6.97 ms | 8.30 ms | no deadline misses |
| Full-truth helix, 145 s | 4.7 ms | 6.9 ms | 19.6 ms | 99.8% complete |
| Full-truth lawnmower, 130 s | 5.1 ms | 7.3 ms | 20.1 ms | 100% complete |

The recorded path runs used a temporary Mako reduction admitted at a 0.22
validation threshold because the deterministic fit scored 0.216 against the
normal 0.20 gate. They demonstrate controller execution, not promotion of
that artifact or a canonical tracking comparison. Future performance records
must name and fingerprint the accepted model artifact.

The approximate matched-workload median is currently divided among:

- rollout and linearization: 1.5 ms;
- objective and QP assembly: 0.75 ms;
- HPIPM update plus iterations: 2.7 ms; and
- accepted rollout, diagnostics, warm-start advancement, and result ownership:
  the remaining roughly 0.4 ms.

This breakdown should be remeasured on every target computer. It identifies
where an optimization can help; it is not a portable timing model.

## Benchmark discipline

Before changing the formulation again, establish reproducible baselines on
the actual vehicle computer. Every result should record:

- CPU model, frequency policy, thermal state, compiler, native flags, and
  library versions;
- model-artifact fingerprint and controller configuration;
- control rate, prediction grid, solver settings, and active constraints;
- median, p95, p99, p99.9, maximum, and deadline misses;
- kernel, assembly, numeric-update, iteration, and post-solve times;
- solver iterations, residuals, regularization, and status;
- process CPU time, wall time, memory high-water mark, and allocations;
- tracking, constraint violation, saturation, effort, and completion; and
- whether camera, sonar, navigation, logging, and communications workloads
  were running concurrently.

Use a quiet-core microbenchmark to locate algorithmic cost, then repeat under
representative system load. Pinning MPC to a reserved core or CPU set may be a
valid deployment choice, but the benchmark must say when it does so. Camera
and sonar pipelines should use their natural accelerators and bounded queues
instead of being allowed to create unbounded CPU contention.

The required behavioral set includes the feasible helix and lawnmower, a
six-DOF position-plus-terminal-attitude transfer, scheduled roll lock, fin and
thruster failures, a surface craft, and a multi-craft/coupled world. A speedup
does not pass if it weakens any declared constraint or materially changes
tracking without an understood reason.

## Prioritized work

### 1. Finish the native numeric data path

The remaining NumPy assembly is small but still crosses Python and ctypes
boundaries and initializes sparse arrays every tick. Profile before moving it.
If it remains material on vehicle hardware:

1. Move fixed-shape QP scatter/update logic into the native HPIPM bridge.
2. Precompute every constant stage block and update only dynamics, state cost,
   bounds, and active scheduled constraints.
3. Replace broad `set_all` calls with field-specific HPIPM updates if that
   avoids copies without making the bridge fragile.
4. Keep a readable NumPy/reference assembly path for numerical equivalence
   tests.

The gate is exact QP-data equivalence followed by matching first commands and
closed-loop behavior. Do not trade inspectability for a sub-millisecond change
unless the target-hardware profile justifies it.

### 2. Sweep HPIPM structure on target hardware

Partial condensing is workload- and CPU-dependent. Benchmark uncondensed and
several fixed condensed horizons rather than assuming eight is universal.
Sweep tolerance, iteration limit, regularization, and warm-start mode while
recording residuals and closed-loop behavior.

Promote solver defaults per deployment class only after the same settings pass
the constraint and failure fixtures. OSQP should remain the numerical
reference until HPIPM equivalence is routine across vehicle types.

### 3. Use a nonuniform prediction grid

Reducing the number of shooting nodes offers a larger structural saving than
micro-optimizing a fixed 40-node problem. Preserve the ten-second preview with
dense nodes near the vehicle and progressively coarser nodes farther away, for
example:

```text
0–1 s       fine dynamics and actuator response
1–4 s       maneuver-scale spacing
4–10 s      coarse planning and terminal intent
```

This requires stage-specific integration intervals, correctly integrated
cost weights, guidance sampling at the exact node times, and slew penalties
that account for interval length. It must be evaluated on fast fin dynamics,
six-DOF strafing versus reorientation, hard maneuver envelopes, and actuator
loss. Do not shorten the preview merely to report a faster solve.

Prefer a small set of precompiled grids over rebuilding solver structure at
runtime. Adaptive selection can later switch among those profiles at explicit
mission boundaries or well-defined controller states.

### 4. Split RTI preparation from feedback latency

Classic RTI can prepare the next trajectory linearization and QP data after a
solve, while waiting for the next state estimate. When the estimate arrives,
the feedback phase updates the initial-condition-dependent data and solves the
prepared QP.

This does not reduce total CPU work, but it can move rollout and assembly off
the state-to-actuator critical path and onto a reserved worker. The design must
carry the plan revision, model/failure revision, expected state time, and
constraint schedule through both phases. A missed or stale preparation falls
back to the ordinary synchronous tick.

Preparation must not become the previously rejected policy of blindly using
old Jacobians. It is a pipelined computation for the next predicted nominal
trajectory, with explicit validity checks and deterministic fallback.

### 5. Improve warm starts before adding learning

The shifted previous primal/dual solution is already a strong default. Further
deterministic candidates include:

- projecting the shifted actuator plan through new failure/bound changes;
- resetting duals associated with constraints whose activity changed;
- preserving duals only when the model and schedule revisions match;
- using a cheap KKT-residual test to choose between shifted and neutral warm
  starts; and
- maintaining separately tuned warm-start policies for OSQP and HPIPM.

Measure iteration tails, not just medians. A heuristic that saves one easy-case
iteration but makes failure transitions worse should be rejected.

### 6. Advisory learned warm starts

A small learned policy may be useful only as a proposal to the optimizer. It
may predict an actuator trajectory correction, active-set hint, or dual warm
start from the current error, reference summary, actuator availability, and
previous solution. It must never replace the model, constraint evaluation, or
QP solve.

The safe boundary is:

```text
state + reference + model revision
              │
              ▼
      optional learned proposal
              │
       validate shape/bounds/revision
              │
 compare cheap residual with shifted warm start
              │
              ▼
      ordinary RTI QP solve and checks
```

Requirements:

- train from versioned simulation/replay data with actuator-failure coverage;
- include an out-of-distribution score and deterministic rejection path;
- bound inference time and memory, preferably as one small compiled network;
- log proposal identity, acceptance, residual, and actual iteration saving;
- retain the shifted deterministic warm start as fallback; and
- demonstrate lower p99 compute without worse constraints or tracking.

Learning is most promising if profiling shows solver iterations dominate and
deterministic warm starts fail on repeatable classes of maneuvers. It cannot
repair an infeasible path or inaccurate reduced model.

### 7. Evaluate mixed precision deliberately

Single-precision dynamics kernels and HPIPM's `s_ocp_qp` API could reduce
memory traffic and improve SIMD throughput on some embedded CPUs. On others,
double precision may be nearly as fast and much easier to validate.

Start with mixed precision rather than converting everything at once:

1. benchmark F32 rollout/linearization against the current F64 kernels;
2. measure accumulated manifold, constraint, and derivative error;
3. compare an F32 QP solve with F64 residual verification; and
4. test poorly scaled six-DOF and failure cases, not only Mako path tracking.

The artifact schema must declare numerical precision and scaling. F32 is
acceptable only if constraint margins and repeatability remain adequate.

### 8. Exploit derivative and model structure

The linearization kernel currently differentiates the complete reduced world.
Potential improvements include generating only state/input blocks required by
the shooting transcription, exploiting known block sparsity across uncoupled
craft, and selecting forward or reverse derivative construction from the
actual state/input dimensions.

For multi-craft worlds, preserve coupling blocks when a tether, wake, contact,
or other modeled interaction exists. Block-diagonal assumptions must come from
the world graph, never from a single-craft shortcut in MPC.

### 9. Hardware and process isolation

Algorithm work cannot compensate for unbounded competing workloads. On the
vehicle computer, evaluate:

- a reserved CPU/core and real-time scheduling policy for the control process;
- bounded queues and backpressure in camera and sonar processing;
- GPU/NPU/DSP use where available for perception;
- pre-faulted/locked controller memory where the platform supports it;
- avoiding filesystem writes and model compilation in the live loop; and
- thermal throttling during sustained mission-length tests.

This is Shiver deployment policy, not Manta controller mathematics. The same
MPC API and model artifacts must work with or without such isolation.

## Experiments already rejected or deferred

### Blind every-other-tick linearization reuse

A prototype reused dynamics Jacobians for one extra tick. On the matched
40-node workload it reduced median time only from 5.36 ms to 4.99 ms because
HPIPM required more iterations. Peak bank increased from 10.5° to 16.0°, and
the closed-loop trajectory changed materially. It is not implemented.

Revisit only if a validity metric can predict safe reuse cheaply and broad
trajectory tests show a clear advantage. Avoid adding several thresholds that
make the controller harder to diagnose.

### Learned replacement controller or learned plant

A network that directly commands actuators without the optimizer's constraint
check is outside this design. A learned plant cannot silently replace the
accepted Manta reduced model. Either would undermine failure handling,
constraints, replayability, and the one-controller/every-vehicle goal.

## Recommended sequence

1. Establish idle and fully loaded baselines on the actual vehicle computer.
2. Profile and, if justified, finish the native HPIPM numeric update path.
3. Sweep HPIPM condensing and solver settings on each hardware class.
4. Implement and validate one or two fixed nonuniform prediction grids.
5. Improve deterministic warm starts and failure transitions.
6. Try an advisory learned warm start only if iteration tails still dominate.
7. Evaluate mixed precision after numerical scaling tests exist.

Each step should land independently with a reproducible benchmark report and
an easy fallback to the previous controller. Benchmark artifacts belong with
Shiver's operational tooling; generic solver and kernel improvements belong in
Manta.
