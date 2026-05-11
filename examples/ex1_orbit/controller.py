"""Keyboard controller for ex1_orbit (terminal-focused, WSL-compatible).

Reads keys from the terminal in raw mode (no OS-level keyboard hook),
so pynput-style "held key" signals aren't available. Instead we use an
auto-repeat + decay model: a press boosts the input toward 1.0, and the
input decays back to 0.0 over ~0.3 s with no further press. On most
terminals the OS already auto-repeats a held key at ~30 Hz, which keeps
the input alive while you hold; releasing the key lets it decay.

The controller terminal must have focus for the keys to register.

    Key              Action            Affects
    ---              ------            -------
    W / S            ±Y translation    ty_xp, ty_xn   (both equal)
    A / D            ∓X translation    tx_zp, tx_zn   (both equal)
    Q / E            ±Z translation    tz_yp, tz_yn   (both equal)
    ↑ / ↓            ±Y rotation       tx_zp / tx_zn  (opposite)
    ← / →            ±Z rotation       ty_xp / ty_xn  (opposite)
    Z / X            ±X rotation       tz_yp / tz_yn  (opposite)
    SPACE            zero all thrusters
    q (lowercase q is used for -Z — use Ctrl-C to quit)

Run:  .venv/bin/python examples/ex1_orbit/controller.py
Quit: Ctrl-C
"""

import json
import math
import os
import select
import signal
import sys
import termios
import time
import tty

import zenoh

PUB_HZ          = 50.0
PUB_PERIOD      = 1.0 / PUB_HZ
THRUSTERS       = ["tx_zp", "tx_zn", "ty_xp", "ty_xn", "tz_yp", "tz_yn"]

# Per-DOF intensity follows an impulse-with-decay model. A keystroke
# adds IMPULSE to the relevant DOF (clamped to ±1). Each tick, the
# DOF magnitude decays toward zero with time-constant DECAY_TAU.
IMPULSE         = 1.0           # peak input per keystroke
DECAY_TAU       = 0.2           # seconds — half-life ≈ 140 ms


# ---------------------------------------------------------------------
# Terminal key reader. Reads stdin in raw mode, decodes single-byte
# printable keys + 3-byte arrow-key escape sequences.

ESC = '\x1b'

# Map raw token → list of (dof_name, sign) contributions.
KEY_TO_DOFS: dict[str, list[tuple[str, float]]] = {
    'w':       [('uy',     +1.0)],
    's':       [('uy',     -1.0)],
    'a':       [('ux',     -1.0)],
    'd':       [('ux',     +1.0)],
    'e':       [('uz',     +1.0)],
    'q':       [('uz',     -1.0)],
    'z':       [('uroll',  -1.0)],
    'x':       [('uroll',  +1.0)],
    'UP':      [('upitch', +1.0)],
    'DOWN':    [('upitch', -1.0)],
    'LEFT':    [('uyaw',   -1.0)],
    'RIGHT':   [('uyaw',   +1.0)],
    ' ':       [('zero', 1.0)],
}


def _drain_keys() -> list[str]:
    """Read all available keystrokes from stdin. Returns a list of
    tokens — single chars or 'UP'/'DOWN'/'LEFT'/'RIGHT' for arrows."""
    out: list[str] = []
    while select.select([sys.stdin], [], [], 0)[0]:
        ch = sys.stdin.read(1)
        if not ch:
            break
        if ch != ESC:
            out.append(ch)
            continue
        # Possible CSI arrow sequence: ESC [ A/B/C/D
        seq = ch
        for _ in range(2):
            if not select.select([sys.stdin], [], [], 0)[0]:
                break
            seq += sys.stdin.read(1)
        arrow = {ESC + '[A': 'UP', ESC + '[B': 'DOWN',
                 ESC + '[C': 'RIGHT', ESC + '[D': 'LEFT'}.get(seq)
        if arrow:
            out.append(arrow)
        # Bare ESC or unknown seq → ignored.
    return out


# ---------------------------------------------------------------------
# Mixer: DOF inputs → per-thruster throttle.

def mix(ux: float, uy: float, uz: float,
        upitch: float, uyaw: float, uroll: float) -> dict[str, float]:
    cmd = {
        "tx_zp": ux + upitch,
        "tx_zn": ux - upitch,
        "ty_xp": uy + uyaw,
        "ty_xn": uy - uyaw,
        "tz_yp": uz + uroll,
        "tz_yn": uz - uroll,
    }
    return {k: max(-1.0, min(1.0, v)) for k, v in cmd.items()}


# ---------------------------------------------------------------------
# Main loop.

def main() -> int:
    if not sys.stdin.isatty():
        sys.exit("controller.py: stdin must be a TTY (run in a terminal).")

    running = True
    def on_signal(*_):
        nonlocal running
        running = False
    signal.signal(signal.SIGINT,  on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    cfg = zenoh.Config()
    session = zenoh.open(cfg)
    pubs = {name: session.declare_publisher(f"manta/ex1/{name}/cmd")
            for name in THRUSTERS}

    print("ex1 controller: WASD/QE = translate, arrows = pitch/yaw, "
          "Z/X = roll, SPACE = zero, Ctrl-C = quit.")
    print("Focus this terminal for keys to register. Keys auto-repeat "
          "while held; releasing decays the input over ~0.2 s.")

    # Switch the terminal to cbreak mode so single keystrokes are read
    # without enter, but signals (Ctrl-C) still fire.
    fd = sys.stdin.fileno()
    saved = termios.tcgetattr(fd)
    tty.setcbreak(fd)
    # Also stop newline/echo from the OS for cleanliness.
    last_pub: dict[str, float] = {n: math.nan for n in THRUSTERS}

    dofs = {'ux': 0.0, 'uy': 0.0, 'uz': 0.0,
            'upitch': 0.0, 'uyaw': 0.0, 'uroll': 0.0}
    try:
        last_t = time.monotonic()
        while running:
            now = time.monotonic()
            dt  = now - last_t
            last_t = now

            # Decay all DOFs toward zero.
            decay = math.exp(-dt / DECAY_TAU)
            for k in dofs:
                dofs[k] *= decay

            # Apply any new keystrokes.
            for tok in _drain_keys():
                if tok == '\x03':       # Ctrl-C — defensive (signal handler usually catches)
                    running = False
                    break
                actions = KEY_TO_DOFS.get(tok)
                if not actions:
                    continue
                if actions == [('zero', 1.0)]:
                    for k in dofs:
                        dofs[k] = 0.0
                    continue
                for dof_name, sign in actions:
                    dofs[dof_name] = max(-1.0, min(1.0, dofs[dof_name] + sign * IMPULSE))

            cmd = mix(dofs['ux'], dofs['uy'], dofs['uz'],
                      dofs['upitch'], dofs['uyaw'], dofs['uroll'])
            for name, v in cmd.items():
                if abs(v - last_pub[name]) > 1e-4:
                    pubs[name].put(json.dumps([v]).encode("utf-8"))
                    last_pub[name] = v

            time.sleep(PUB_PERIOD)
    finally:
        # Zero all thrusters on exit so the craft stops accelerating.
        for pub in pubs.values():
            pub.put(json.dumps([0.0]).encode("utf-8"))
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)
        session.close()
        print("\ncontroller: stopped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
