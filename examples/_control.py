"""Keyboard control with a self-running scripted fallback.

The vehicle demos take live keyboard input, but should also run
unattended (CI, no display, a quick "does it work?"). So both share one
tiny interface — :meth:`Controller.held` — with two implementations:

  * :class:`KeyboardController` — a `pynput` global listener tracking the
    set of currently-held keys. Used when a demo is run with
    ``--keyboard``.
  * :class:`ScriptedController` — replays a timed script of held-key sets
    as a function of sim-time. The default, so every demo is
    self-contained and headless-testable.

Keys are normalized to lowercase characters (``"w"``, ``"a"`` …) plus the
special names ``"up" "down" "left" "right" "space" "shift"``. A demo asks
``ctrl.held("w")`` each tick; for edge-triggered actions it can poll
``ctrl.pressed("g")`` (true once per fresh press).

`pynput` is only imported (and a backend only opened) when
:class:`KeyboardController` is actually constructed — the scripted default
never touches it, so it works with no display.
"""

from __future__ import annotations

import argparse


# --------------------------------------------------------------------------
# Common interface
# --------------------------------------------------------------------------

class Controller:
    def held(self, key: str) -> bool:
        raise NotImplementedError

    def pressed(self, key: str) -> bool:
        """Edge-trigger: True once per fresh press of ``key``."""
        held = self.held(key)
        was = self._edge.get(key, False)
        self._edge[key] = held
        return held and not was

    _edge: dict[str, bool] = {}

    def update(self, t: float) -> None:
        """Advance any time-driven state (no-op for live keyboard)."""

    def stop(self) -> None:
        """Release any OS resources (listener thread)."""


# --------------------------------------------------------------------------
# Scripted (default) — held-key sets over time windows
# --------------------------------------------------------------------------

class ScriptedController(Controller):
    """Replays ``segments`` — a list of ``(t_start, t_end, {keys…})``.

    Any number of overlapping segments may hold a key; ``held(k)`` is true
    if any active segment contains ``k`` at the current time.
    """

    def __init__(self, segments: list[tuple[float, float, set[str]]]) -> None:
        self._segments = [(a, b, set(ks)) for a, b, ks in segments]
        self._t = 0.0
        self._edge = {}

    def update(self, t: float) -> None:
        self._t = t

    def held(self, key: str) -> bool:
        return any(a <= self._t < b and key in ks
                   for a, b, ks in self._segments)


# --------------------------------------------------------------------------
# Live keyboard via pynput
# --------------------------------------------------------------------------

_SPECIAL = {
    "up": "up", "down": "down", "left": "left", "right": "right",
    "space": "space", "shift": "shift", "shift_r": "shift",
}


class KeyboardController(Controller):
    """Tracks currently-held keys with a background `pynput` listener."""

    def __init__(self) -> None:
        from pynput import keyboard  # local: only when live control is asked
        self._keyboard = keyboard
        self._down: set[str] = set()
        self._edge = {}
        self._listener = keyboard.Listener(
            on_press=self._on_press, on_release=self._on_release)
        self._listener.daemon = True
        self._listener.start()

    def _norm(self, k) -> str | None:
        kb = self._keyboard
        if isinstance(k, kb.KeyCode) and k.char:
            return k.char.lower()
        if isinstance(k, kb.Key):
            return _SPECIAL.get(k.name)
        return None

    def _on_press(self, k) -> None:
        n = self._norm(k)
        if n:
            self._down.add(n)

    def _on_release(self, k) -> None:
        n = self._norm(k)
        if n:
            self._down.discard(n)

    def held(self, key: str) -> bool:
        return key in self._down

    def stop(self) -> None:
        self._listener.stop()


# --------------------------------------------------------------------------
# Wiring
# --------------------------------------------------------------------------

def make_controller(keyboard: bool,
                    script: list[tuple[float, float, set[str]]] | None = None
                    ) -> Controller:
    """A live keyboard controller (``keyboard=True``) or a scripted one."""
    if keyboard:
        return KeyboardController()
    return ScriptedController(script or [])


def common_args(description: str) -> argparse.ArgumentParser:
    """Standard demo CLI: ``--keyboard``, ``--no-viz``, ``--duration``."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--keyboard", action="store_true",
                   help="drive with live keyboard input (else a scripted run)")
    p.add_argument("--no-viz", action="store_true",
                   help="run headless: no rerun viewer")
    p.add_argument("--duration", type=float, default=None,
                   help="override the run length in seconds")
    return p
