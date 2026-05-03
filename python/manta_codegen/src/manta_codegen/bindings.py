"""Module-level binding API: `connect`, `publish`, `subscribe`.

These are thin free-function wrappers that find the owning World by
walking the signal's `craft_ref` and dispatch to the existing
World-level methods. Hands the user a flat, reads-naturally API:

    connect(sim.imu.last_accel, est.imu.set_measurement_accel)
    publish(ekf.position, "manta/ex5/est/p")
    subscribe(craft.thrust.set_throttle, "manta/ex5/cmd")

For multi-world setups the binding is recorded on the world owning
the source signal. Codegen iterates all worlds (across all Targets)
to find them.
"""

from __future__ import annotations

from .signal import BoundSignal


def _world_for_signal(sig: BoundSignal, kind: str) -> "World":
    if not isinstance(sig, BoundSignal):
        raise TypeError(
            f"{kind}: expected a BoundSignal, got {type(sig).__name__}")
    if sig.craft_ref is None:
        raise ValueError(
            f"{kind}: signal {sig.name!r} is not attached to any craft "
            f"(call c.add(part) first).")
    craft = sig.craft_ref
    # Find the World holding this craft. The World stamps craft.world_back
    # at add_craft time (added below in core.py). Without it, we walk the
    # registered worlds; for now require an explicit back-reference.
    world = getattr(craft, "_world", None)
    if world is None:
        raise ValueError(
            f"{kind}: craft {craft.name!r} for signal {sig.name!r} is not "
            f"in any world (call world.add_craft(craft) first).")
    return world


def connect(source: BoundSignal, sink: BoundSignal) -> None:
    """Wire an out-direction signal to an in-direction signal in the same
    target's main loop. Signals must live in worlds that share a Target;
    cross-target connect is an error (use publish/subscribe for cross-
    binary flow)."""
    world = _world_for_signal(source, "connect")
    world.connect(source, sink)


def publish(what, topic: str = None, protocol: str = "zenoh",
            encoding: str = "json") -> None:
    """Publish a single signal or bundled-struct dict over Zenoh."""
    if isinstance(what, dict):
        if not what:
            raise ValueError("publish: empty struct dict")
        first = next(iter(what.values()))
        world = _world_for_signal(first, "publish")
    else:
        world = _world_for_signal(what, "publish")
    world.publish(what, topic=topic, protocol=protocol, encoding=encoding)


def subscribe(what, topic: str = None, protocol: str = "zenoh",
              encoding: str = "json") -> None:
    """Subscribe a Zenoh topic into an in-direction signal (or bundled struct)."""
    if isinstance(what, dict):
        if not what:
            raise ValueError("subscribe: empty struct dict")
        first = next(iter(what.values()))
        world = _world_for_signal(first, "subscribe")
    else:
        world = _world_for_signal(what, "subscribe")
    world.subscribe(what, topic=topic, protocol=protocol, encoding=encoding)
