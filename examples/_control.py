"""Keyboard control with a self-running scripted fallback.

The vehicle demos take live keyboard input, but should also run
unattended (CI, no display, a quick "does it work?"). So all share one
tiny interface — :meth:`Controller.held` — with three implementations:

  * :class:`KeyboardController` — a `pynput` global listener tracking the
    set of currently-held keys. Needs a display server that actually sees
    your keystrokes (a native Linux desktop).
  * :class:`TerminalController` — raw-stdin reader: keys typed into the
    terminal the demo was launched from. Works everywhere a tty does
    (WSL, ssh, headless boxes) — on WSLg the Windows-side terminal never
    reaches X11, so a global listener hears nothing there.
  * :class:`ScriptedController` — replays a timed script of held-key sets
    as a function of sim-time. The default, so every demo is
    self-contained and headless-testable.

``--keyboard`` picks the live backend automatically (terminal on WSL or
with no display, global otherwise); ``--keyboard terminal`` /
``--keyboard global`` forces one.

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
import os
import select
import sys
import time


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
# Live keyboard via the launching terminal (raw stdin)
# --------------------------------------------------------------------------

class TerminalController(Controller):
    """Reads keys from the terminal the demo was launched from.

    A terminal only reports presses (refreshed by OS auto-repeat), never
    releases — so ``held`` treats a key as down until no byte arrives for
    ``HOLD_S``, sized to bridge the auto-repeat initial delay. The feel:
    a tap gives ~``HOLD_S`` of deflection, holding sustains it. The
    terminal only auto-repeats the most recent key, so chords don't
    register — fly one control at a time. ``shift`` can't be seen at all.
    """

    HOLD_S = 0.65
    _ESC = {"[A": "up", "[B": "down", "[C": "right", "[D": "left"}

    def __init__(self) -> None:
        import termios
        import tty
        if not sys.stdin.isatty():  # pragma: no cover - depends on env
            raise SystemExit(
                "--keyboard terminal needs an interactive terminal (stdin "
                "is not a tty).")
        self._termios = termios
        self._fd = sys.stdin.fileno()
        self._saved = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        self._last: dict[str, float] = {}
        self._edge = {}

    def update(self, t: float) -> None:
        # Drain raw bytes from the fd (not sys.stdin — its text wrapper
        # buffers ahead of select), then scan for keys + arrow sequences.
        buf = b""
        while select.select([self._fd], [], [], 0)[0]:
            buf += os.read(self._fd, 64)
        s, i, now = buf.decode(errors="ignore"), 0, time.monotonic()
        while i < len(s):
            ch = s[i]
            if ch == "\x1b" and s[i + 1:i + 3] in self._ESC:
                key, i = self._ESC[s[i + 1:i + 3]], i + 3
            else:
                key, i = ("space" if ch == " " else ch.lower()), i + 1
            self._last[key] = now

    def held(self, key: str) -> bool:
        return time.monotonic() - self._last.get(key, -1e9) < self.HOLD_S

    def stop(self) -> None:
        self._termios.tcsetattr(self._fd, self._termios.TCSADRAIN,
                                self._saved)


# --------------------------------------------------------------------------
# Real-time pacing (live demos: keep sim time ≈ wall clock)
# --------------------------------------------------------------------------

class Pacer:
    """Sleep a live loop so sim time tracks the wall clock.

    The sim integrates several times faster than real time; uncapped,
    live control is unflyable and viz logs flood the viewer's channel.
    Call :meth:`pace` once per tick with the current sim time.
    """

    def __init__(self) -> None:
        self._t0: float | None = None

    def pace(self, t: float) -> None:
        now = time.monotonic()
        if self._t0 is None:
            self._t0 = now - t
        lag = (self._t0 + t) - now
        if lag > 0:
            time.sleep(lag)


# --------------------------------------------------------------------------
# Wiring
# --------------------------------------------------------------------------

def _wsl() -> bool:
    try:
        with open("/proc/version") as f:
            return "microsoft" in f.read().lower()
    except OSError:  # pragma: no cover - non-Linux
        return False


def make_controller(keyboard: str | None,
                    script: list[tuple[float, float, set[str]]] | None = None
                    ) -> Controller:
    """A live keyboard controller (``keyboard`` mode string from
    ``--keyboard``) or the scripted fallback (``keyboard=None``)."""
    if not keyboard:
        return ScriptedController(script or [])
    mode = keyboard
    if mode == "auto":
        # WSLg's display server never sees Windows-terminal keystrokes,
        # and with no display pynput can't even start — read stdin there.
        native_display = (os.environ.get("DISPLAY")
                          or os.environ.get("WAYLAND_DISPLAY"))
        mode = "terminal" if _wsl() or not native_display else "global"
    if mode == "terminal":
        return TerminalController()
    return KeyboardController()


def common_args(description: str) -> argparse.ArgumentParser:
    """Standard demo CLI: ``--keyboard [mode]``, ``--no-viz``,
    ``--duration``."""
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--keyboard", nargs="?", const="auto", default=None,
                   choices=("auto", "terminal", "global"),
                   help="drive with live keyboard input (else a scripted "
                        "run); mode 'terminal' reads the launching tty, "
                        "'global' uses a pynput listener, default auto")
    p.add_argument("--no-viz", action="store_true",
                   help="run headless: no rerun viewer")
    p.add_argument("--duration", type=float, default=None,
                   help="override the run length in seconds")
    return p
