"""UKF — Python descriptor for an Unscented Kalman Filter.

Mirrors `EKF` (same wrap-a-World contract, same per-sensor measurement
list, same BoundSignal output surface) but generates a
`manta::estimation::UKF<EstCraft<double>, MeasDim>` instance
instead of `EKF<EstCraftT, MeasDim>`.

Two structural differences from EKF:

  * The wrapped craft does NOT need to be `scalar_templated`. UKF only
    ever calls `evaluate(x, dt)` with `double` scalars (the unscented
    transform doesn't take Jacobians), so plain non-templated crafts
    work too. Templated crafts work as well — the codegen instantiates
    the `<double>` form.

  * The constructor takes (alpha, beta, kappa) sigma-point tuning
    knobs. Defaults match the standard formulation:
      alpha = 1e-3   (sigma-point spread; smaller = closer)
      beta  = 2.0    (optimal for Gaussian state)
      kappa = 0.0    (secondary scaling)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..signal import BoundSignal, Signal

if TYPE_CHECKING:
    from ..core import World


# Sentinel for UKF-output signals — accessor_for() recognizes the
# `$ukf/<ukf_id>` prefix and routes to the UKF's C++ accessors instead
# of any craft.
UKF_SENTINEL_PREFIX = "$ukf/"


def _state_slice_signal(name: str, accessor_method: str, n: int,
                         craft_idx: int = 0) -> Signal:
    return Signal(
        name=name,
        direction="out",
        n_floats=n,
        cpp_read_exprs=tuple(
            f"{{accessor}}.{accessor_method}({craft_idx})({i})" for i in range(n)
        ),
    )


class _CraftUkfSignals:
    """Per-craft BoundSignal bundle for `ukf.crafts['<name>'].position` etc.
    Mirrors EKF's _CraftEkfSignals."""
    def __init__(self, name: str):
        self.name = name
        self.position           = None
        self.orientation        = None
        self.vel_linear         = None
        self.vel_angular        = None
        self.position_stddev    = None
        self.orientation_stddev = None
        self.vel_linear_stddev  = None
        self.vel_angular_stddev = None


@dataclass
class UKF:
    """Unscented Kalman Filter wrapping a World.

    Args:
        world: the World whose system model this UKF estimates. The
               world should contain exactly one Craft plus the fields /
               planets that the craft's parts need to access during
               predict.
        measurements: list of sensor-part descriptors the UKF observes.
        q_jitter: numerical diagonal floor on Q each tick — keeps the
               covariance strictly positive-definite for the kernel's
               LLT cholesky. NOT a model-noise knob; declare
               `Noise<*>` on parts (per-part σ on the descriptor) to
               capture physical process noise via auto-Q assembly. The
               default 1e-9 is safe for typical state magnitudes; raise
               it if LLT begins to fail in long-horizon runs.
        initial_covariance: scalar — P_0 is `initial_covariance * I`.
        alpha, beta, kappa: sigma-point tuning knobs (see UKF<...>
               docstring). Defaults to the standard van-der-Merwe form.
    """
    world: "World"
    measurements: list = field(default_factory=list)
    q_jitter: float = 1e-9
    initial_covariance: float = 1.0

    # Per-craft initial-state overrides — see EKF for the contract.
    # None ⇒ inherit from `world.add_craft(...)` defaults.
    initial_position:         "tuple | list | None" = None
    initial_orientation:      "tuple | list | None" = None
    initial_velocity:         "tuple | list | None" = None
    initial_angular_velocity: "tuple | list | None" = None

    # Per-block initial-variance overrides (None ⇒ initial_covariance).
    initial_position_var:         float | None = None
    initial_attitude_var:         float | None = None
    initial_velocity_var:         float | None = None
    initial_angular_velocity_var: float | None = None
    alpha: float = 1e-3
    beta:  float = 2.0
    kappa: float = 0.0

    STATE_DIM: int = 13

    def __post_init__(self) -> None:
        global _ukf_id_counter
        try:
            _ukf_id_counter += 1
        except NameError:
            _ukf_id_counter = 0
        self._ukf_id = _ukf_id_counter
        self._world = self.world

        if not self.world.crafts:
            raise ValueError(
                "UKF: world must have at least one craft "
                "(world.add_craft(c) before constructing the UKF).")
        # Multi-craft worlds work — the joint sigma-point propagation
        # runs the full World.update for every sigma vector; the per-
        # craft slice lives at state[craft_idx*13 : (craft_idx+1)*13].

        # Filter targets need scalar_templated crafts (the value-side
        # WorldT<double> requires every craft as `<double>`). Templating
        # is a codegen detail — set it automatically rather than making
        # users write boilerplate.
        world_crafts = [e.craft for e in self.world.crafts]
        for c in world_crafts:
            c.scalar_templated = True

        from ..core import PartDescriptor
        for m in self.measurements:
            if not isinstance(m, PartDescriptor):
                raise TypeError(
                    f"UKF.measurements: expected PartDescriptor instances, "
                    f"got {type(m).__name__} ({m!r}).")
            if getattr(m, "_craft", None) not in world_crafts:
                names = [c.name for c in world_crafts]
                raise ValueError(
                    f"UKF: measurement part {m.name!r} is not attached to "
                    f"any craft in the UKF's wrapped world (crafts: {names}).")

        # Per-craft signal tree (see ekf.py for the design). Indexed by
        # craft name: `ukf.crafts["my_drone"].position`.
        self.crafts: dict[str, _CraftUkfSignals] = {}
        for idx, c in enumerate(world_crafts):
            cs = _CraftUkfSignals(c.name)
            cs.position           = self._make_craft_signal(idx, "position",           "position",           3)
            cs.orientation        = self._make_craft_signal(idx, "orientation",        "orientation",        4)
            cs.vel_linear         = self._make_craft_signal(idx, "vel_linear",         "vel_linear",         3)
            cs.vel_angular        = self._make_craft_signal(idx, "vel_angular",        "vel_angular",        3)
            cs.position_stddev    = self._make_craft_signal(idx, "position_stddev",    "position_stddev",    3)
            cs.orientation_stddev = self._make_craft_signal(idx, "orientation_stddev", "orientation_stddev", 4)
            cs.vel_linear_stddev  = self._make_craft_signal(idx, "vel_linear_stddev",  "vel_linear_stddev",  3)
            cs.vel_angular_stddev = self._make_craft_signal(idx, "vel_angular_stddev", "vel_angular_stddev", 3)
            self.crafts[c.name] = cs

        # Single-craft shortcuts at the top level.
        sd = self.STATE_DIM
        first_cs = self.crafts[world_crafts[0].name]
        self.position           = first_cs.position
        self.orientation        = first_cs.orientation
        self.vel_linear         = first_cs.vel_linear
        self.vel_angular        = first_cs.vel_angular
        self.position_stddev    = first_cs.position_stddev
        self.orientation_stddev = first_cs.orientation_stddev
        self.vel_linear_stddev  = first_cs.vel_linear_stddev
        self.vel_angular_stddev = first_cs.vel_angular_stddev
        self.full_state         = self._make_full_state_signal(sd)

    def _make_craft_signal(self, craft_idx: int, name: str,
                           accessor_method: str, n: int) -> BoundSignal:
        sig = _state_slice_signal(name, accessor_method, n,
                                  craft_idx=craft_idx)
        return BoundSignal(
            part_name=f"{UKF_SENTINEL_PREFIX}{self._ukf_id}",
            signal=sig,
            craft_ref=self,
        )

    def _make_full_state_signal(self, n: int) -> BoundSignal:
        sig = Signal(
            name="full_state",
            direction="out",
            n_floats=n,
            cpp_read_exprs=tuple(
                f"{{accessor}}.full_state()({i})" for i in range(n)
            ),
        )
        return BoundSignal(
            part_name=f"{UKF_SENTINEL_PREFIX}{self._ukf_id}",
            signal=sig,
            craft_ref=self,
        )

    def cpp_var_name(self) -> str:
        return f"ukf_{self._ukf_id}"

    def measurement_dim(self) -> int:
        total = 0
        for m in self.measurements:
            for sig in getattr(type(m), "signals", []) or []:
                if sig.direction == "out" and sig.name.startswith("last_"):
                    total += sig.n_floats
        return total
