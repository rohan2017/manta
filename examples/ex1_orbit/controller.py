"""Keyboard controller for ex1_orbit — bang-bang, WSL-compatible.

Reads keys from this terminal in raw mode (no OS-level keyboard hook,
so it works under WSL where pynput's hook silently fails). Each
thruster command is bang-bang: +1, -1, or 0. The user feels the
command saturate; no analog blending.

"Held key" semantics without OS release events: every keystroke updates
that key's last-seen timestamp. The OS auto-repeats a physically held
key at ~30 Hz, so a key is considered held if it was seen in the last
HOLD_MS milliseconds. Releasing the key stops the repeat stream and the
"held" state expires shortly after.

The controller terminal must have focus for keys to register.

    Key              Action            Affects
    ---              ------            -------
    W / S            ±Y translation    ty_xp, ty_xn   (both equal)
    A / D            ∓X translation    tx_zp, tx_zn   (both equal)
    Q / E            ±Z translation    tz_yp, tz_yn   (both equal)
    J / L            ±X rotation (roll, about A/D axis)         (opposite)
    I / K            ±Y rotation (pitch, about W/S axis)        (opposite)
    U / O            ±Z rotation (yaw, about Q/E axis)          (opposite)
    SPACE            zero all + clear held state
    Ctrl-C           quit
"""

import json
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

# A keystroke marks the key as "held"; the key stays held for HOLD_S
# seconds after the last keystroke. Linux's default terminal auto-repeat
# is ~25–35 Hz when a key is physically held, so 150 ms is comfortably
# longer than the repeat interval but short enough that releasing a key
# feels instant.
HOLD_S          = 0.15

ESC = '\x1b'

# Map raw key token → (dof_name, sign). One token per DOF contribution.
# Rotation keys use the convention "rotation about the AD/WS/QE axis":
#   J/L  → roll  about ±X  (the A/D translation axis)
#   I/K  → pitch about ±Y  (the W/S translation axis)
#   U/O  → yaw   about ±Z  (the Q/E translation axis)
KEY_TO_DOF: dict[str, tuple[str, float]] = {
    'w':     ('uy',     +1.0),
    's':     ('uy',     -1.0),
    'a':     ('ux',     -1.0),
    'd':     ('ux',     +1.0),
    'e':     ('uz',     +1.0),
    'q':     ('uz',     -1.0),
    'l':     ('uroll',  +1.0),
    'j':     ('uroll',  -1.0),
    'i':     ('upitch', +1.0),
    'k':     ('upitch', -1.0),
    'o':     ('uyaw',   +1.0),
    'u':     ('uyaw',   -1.0),
}


def _drain_keys() -> list[str]:
    """Read all available keystrokes from stdin. Single-byte printable
    tokens pass through (lowercased). Any ESC-prefixed sequence is
    swallowed silently so arrow keys etc. don't accidentally trigger
    other actions."""
    out: list[str] = []
    while select.select([sys.stdin], [], [], 0)[0]:
        ch = sys.stdin.read(1)
        if not ch:
            break
        if ch == ESC:
            # Consume any trailing CSI bytes so they don't leak into
            # the next read.
            for _ in range(2):
                if not select.select([sys.stdin], [], [], 0)[0]:
                    break
                sys.stdin.read(1)
            continue
        out.append(ch.lower())
    return out


def sign(x: float) -> float:
    return 1.0 if x > 0.0 else (-1.0 if x < 0.0 else 0.0)


def mix(ux: float, uy: float, uz: float,
        upitch: float, uyaw: float, uroll: float) -> dict[str, float]:
    """Sum DOF contributions per thruster, then bang-bang via sign(sum)."""
    raw = {
        "tx_zp": ux + upitch,
        "tx_zn": ux - upitch,
        "ty_xp": uy + uyaw,
        "ty_xn": uy - uyaw,
        "tz_yp": uz + uroll,
        "tz_yn": uz - uroll,
    }
    return {k: sign(v) for k, v in raw.items()}


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

    print("ex1 controller (bang-bang). WASD/QE = translate, arrows = "
          "pitch/yaw, Z/X = roll, SPACE = zero, Ctrl-C = quit.")
    print("This terminal must have focus. Keys auto-repeat while held; "
          "releasing decays held-state in ~150 ms.")

    fd = sys.stdin.fileno()
    saved = termios.tcgetattr(fd)
    tty.setcbreak(fd)

    last_seen: dict[str, float] = {}     # token → monotonic timestamp
    last_pub:  dict[str, float] = {n: float('nan') for n in THRUSTERS}

    try:
        while running:
            now = time.monotonic()
            for tok in _drain_keys():
                if tok == ' ':
                    last_seen.clear()
                    continue
                if tok in KEY_TO_DOF:
                    last_seen[tok] = now

            # Held = seen in the last HOLD_S seconds.
            held = {tok for tok, t in last_seen.items() if now - t < HOLD_S}
            dofs = {'ux': 0.0, 'uy': 0.0, 'uz': 0.0,
                    'upitch': 0.0, 'uyaw': 0.0, 'uroll': 0.0}
            for tok in held:
                name, sgn = KEY_TO_DOF[tok]
                dofs[name] += sgn

            cmd = mix(dofs['ux'], dofs['uy'], dofs['uz'],
                      dofs['upitch'], dofs['uyaw'], dofs['uroll'])
            for name, v in cmd.items():
                if v != last_pub[name]:
                    pubs[name].put(json.dumps([v]).encode("utf-8"))
                    last_pub[name] = v

            time.sleep(PUB_PERIOD)
    finally:
        for pub in pubs.values():
            pub.put(json.dumps([0.0]).encode("utf-8"))
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)
        session.close()
        print("\ncontroller: stopped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
