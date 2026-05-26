"""Parameter tuning — fitting Craft Parameters against logged data.

This package is **empty placeholder** as of 2026-05-20. The design
target is a small, opinionated parameter-fit loop:

  * Author a Craft as usual. Mark some `Parameter(default, tunable=True)`
    declarations as fit candidates (Parameter sentinel grows a tunable
    flag, defaulting False).
  * Provide a logged trajectory: per-tick `{u_t, z_t}` where `u_t` is
    the commanded inputs and `z_t` are the observed sensor outputs.
  * Call `tuning.fit(craft, log, loss="output_mse", ...)`. Internally:
      1. Build the per-tick CasADi graph parameterized on the tunable
         Parameters (lifted from constants to free variables).
      2. Roll out N ticks symbolically, accumulating Σ‖z_pred − z_obs‖².
      3. Hand the symbolic loss to IPOPT (already a CasADi dependency
         via the EKF Jacobian path); get optimal Parameter values back.
      4. Return a new Craft instance with the fitted Parameters baked in.

The CasADi substrate makes this almost-free conceptually — gradients
flow through the entire physics + sensor model. The work is the
plumbing: Parameter-as-free-variable lifting, log-loading convenience,
sane defaults for solver options, convergence diagnostics.

Why a separate folder: parameter tuning sits orthogonal to the
sim/estimation split. It consumes both (compiled tick for forward
sim, plus the same gradient infrastructure the EKF uses for F/H) but
has its own surface area (loss functions, log adapters, solver wrappers).

Not yet implemented. The Parameter sentinel currently has no `tunable`
flag — adding it is the first concrete step.
"""

__all__: list[str] = []
