"""MantaConfig + Target — top-level shape for a manta system spec.

A `MantaConfig` describes one or more `Target`s. Each Target is a
codegen output unit — produces one C++ main() that drives one or more
Worlds (and, soon, EKFs / UKFs / other tick-driven things) in lockstep
at the Target's chosen dt.

The entry point in user code is:

    def make_config() -> MantaConfig:
        sim_world = World("ex5_sim")
        sim_world.add_craft(sim).add_field(sim_grav)

        est_world = World("ex5_est")
        est_world.add_craft(est).add_field(est_grav)
        ekf = EKF(est_world, measurements=[est.imu])

        # In-process wire — both endpoints in the same target → connect().
        connect(sim.imu.last_accel, est.imu.set_measurement_accel)

        return MantaConfig(targets=[
            Target("ex5", drives=[sim_world, ekf], dt=0.001),
        ])

Cross-target signal flow uses explicit `publish(...) / subscribe(...)`
on Zenoh — cadence partitioning and protocol choice are user concerns
once the worlds live in different processes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence


# A Driveable is anything that has a tick semantic — for now just `World`,
# but `EKF` / `UKF` / future controllers slot in here. The codegen drives
# each Driveable in a Target's main loop at the Target's dt.
Driveable = object   # structural type; concrete classes live in core.py / estimation.py


@dataclass
class Target:
    """One C++ main() the codegen will emit. Holds the Driveables this
    binary ticks (worlds, estimators) plus its tick cadence.

    Args:
        name: identifier — becomes the C++ binary / target name.
        drives: heterogeneous list of things to tick each step. Order
                matters: drives are ticked in sequence within a tick.
                A Target must have at least one World (directly in
                `drives` or owned transitively by an EKF).
        dt:    sim step in seconds. All drives in this Target tick at
                this rate.
        sim_rate_mult: ratio of sim seconds to wall seconds (1.0 = realtime).
    """
    name: str
    drives: list = field(default_factory=list)
    dt: float = 0.001
    sim_rate_mult: float = 1.0

    def __post_init__(self) -> None:
        if not self.name.isidentifier():
            raise ValueError(f"Target name {self.name!r} must be a valid identifier")
        if not self.drives:
            raise ValueError(f"Target {self.name!r}: drives list cannot be empty")
        # Validate at least one World transitively present. Deferred —
        # circular-import-free: walk drives at codegen time instead.


@dataclass
class MantaConfig:
    """Top-level codegen input. Holds a list of Targets, each producing
    one C++ main(). Bindings (`publish`, `subscribe`, `connect`) are
    recorded globally and partitioned across targets at codegen time
    based on which target each signal's owning world lives in.
    """
    targets: list[Target] = field(default_factory=list)

    def __post_init__(self) -> None:
        names = [t.name for t in self.targets]
        if len(set(names)) != len(names):
            dups = sorted({n for n in names if names.count(n) > 1})
            raise ValueError(f"MantaConfig: duplicate target name(s): {dups}")
        if not self.targets:
            raise ValueError("MantaConfig: must declare at least one Target")
