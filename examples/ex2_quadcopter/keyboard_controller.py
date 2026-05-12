"""Keyboard controller for ex2 quadcopter — terminal-focused, WSL-compatible.

Reads keys from this terminal in raw mode. Each press updates a per-key
last-seen timestamp; a key is treated as "held" while it's been seen
within HOLD_S seconds, which works under WSL where pynput-style OS
keyboard hooks silently fail. The OS auto-repeats a physically held
key at ~30 Hz, keeping the state alive.

Wire format on `manta/ex2/cmd` is a 4-float JSON array:
    [thr, roll_rate, pitch_rate, yaw_rate]
    thr      ∈ [0, 1]   collective throttle (held)
    *_rate   ∈ rad/s    body-frame rate setpoint (released ⇒ 0)

Bindings:
    I / K         throttle up / down   (held = ramping at THROTTLE_RAMP/s)
    W / S         pitch nose-down / nose-up
    A / D         roll left / right
    Q / E         yaw left / right
    SPACE         zero rates (keep throttle)
    X             zero everything
    Ctrl-C        quit
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

PUB_HZ        = 50.0
PUB_PERIOD    = 1.0 / PUB_HZ
HOLD_S        = 0.15
RATE_CMD      = 1.0    # rad/s while a rotation key is held
THROTTLE_RAMP = 0.5    # throttle units per second while I/K is held

ESC = '\x1b'

# Rotation keys map to (axis_name, sign). Signs are right-hand-rule body
# rates (+ω about the axis), matching the cmd convention the sim expects.
# Released → axis goes to 0.
ROTATION_KEYS: dict[str, tuple[str, float]] = {
    'w': ('pitch', +1.0),  # +ω_y = nose-down (W = forward tilt)
    's': ('pitch', -1.0),  # −ω_y = nose-up
    'a': ('roll',  -1.0),  # −ω_x = left bank (left wing dips)
    'd': ('roll',  +1.0),  # +ω_x = right bank
    'q': ('yaw',   +1.0),  # +ω_z = yaw left (CCW from above)
    'e': ('yaw',   -1.0),  # −ω_z = yaw right
}

# Throttle keys (continuous integration while held).
THROTTLE_KEYS: dict[str, float] = {
    'i': +1.0,
    'k': -1.0,
}


def _drain_keys() -> list[str]:
    """Single-byte printable tokens (lowercased). ESC + CSI sequences
    (e.g. arrow keys) are silently consumed."""
    out: list[str] = []
    while select.select([sys.stdin], [], [], 0)[0]:
        ch = sys.stdin.read(1)
        if not ch:
            break
        if ch == ESC:
            for _ in range(2):
                if not select.select([sys.stdin], [], [], 0)[0]:
                    break
                sys.stdin.read(1)
            continue
        out.append(ch.lower())
    return out


def main() -> int:
    if not sys.stdin.isatty():
        sys.exit("keyboard_controller.py: stdin must be a TTY.")

    running = True
    def on_signal(*_):
        nonlocal running
        running = False
    signal.signal(signal.SIGINT,  on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    cfg = zenoh.Config()
    session = zenoh.open(cfg)
    pub = session.declare_publisher("manta/ex2/cmd")

    print("ex2 quadcopter controller.")
    print("  I / K       throttle up / down (held = continuous ramp)")
    print("  W / S       pitch (nose-down / nose-up)")
    print("  A / D       roll  (left / right)")
    print("  Q / E       yaw   (left / right)")
    print("  SPACE       zero rates (keep throttle)")
    print("  X           zero everything")
    print("  Ctrl-C      quit")
    print("Held-key auto-repeat keeps rate commands active; releasing "
          "decays to zero in ~150 ms.")

    fd = sys.stdin.fileno()
    saved = termios.tcgetattr(fd)
    tty.setcbreak(fd)

    last_seen: dict[str, float] = {}
    throttle = 0.0
    last_pub_t = time.monotonic()
    last_payload = None
    try:
        while running:
            now = time.monotonic()
            dt = now - last_pub_t
            last_pub_t = now

            for tok in _drain_keys():
                if tok == ' ':
                    # Drop all rotation keys' held state.
                    for k in list(last_seen.keys()):
                        if k in ROTATION_KEYS:
                            del last_seen[k]
                    continue
                if tok == 'x':
                    throttle = 0.0
                    last_seen.clear()
                    continue
                if tok in ROTATION_KEYS or tok in THROTTLE_KEYS:
                    last_seen[tok] = now

            # Held = seen within HOLD_S.
            held = {k for k, t in last_seen.items() if now - t < HOLD_S}

            # Throttle: continuous integration while I/K held.
            for k in THROTTLE_KEYS:
                if k in held:
                    throttle += THROTTLE_KEYS[k] * THROTTLE_RAMP * dt
            throttle = max(0.0, min(1.0, throttle))

            # Rates: ±RATE_CMD per held key; opposite-direction keys cancel.
            rates = {'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0}
            for k in held:
                if k in ROTATION_KEYS:
                    axis, sign = ROTATION_KEYS[k]
                    rates[axis] += sign * RATE_CMD

            payload = [throttle, rates['roll'], rates['pitch'], rates['yaw']]
            if payload != last_payload:
                pub.put(json.dumps(payload).encode("utf-8"))
                last_payload = payload

            time.sleep(PUB_PERIOD)
    finally:
        pub.put(json.dumps([0.0, 0.0, 0.0, 0.0]).encode("utf-8"))
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)
        session.close()
        print("\nkeyboard_controller: stopped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
