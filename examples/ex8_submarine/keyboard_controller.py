"""Keyboard controller for ex8 submarine — held-key WSL-compatible.

Bipolar throttle on each of the two thrusters:

    W / S       forward / reverse on thrust_x  (body +x is the sub's nose)
    I / K       rise / dive on thrust_z       (body +z is up)
    SPACE       zero both thrusters
    Ctrl-C      quit

Throttle is integrated continuously while a key is "held" (auto-repeat
keeps it alive within HOLD_S seconds), then decays back to 0 once
released. Range is clamped to [-1, +1].

Wire format on `manta/ex8/thrust_x/cmd` and `manta/ex8/thrust_z/cmd`
is a single-element JSON float array: `[throttle]` ∈ [-1, +1].
"""

import json
import select
import signal
import sys
import termios
import time
import tty

import zenoh


PUB_HZ      = 50.0
PUB_PERIOD  = 1.0 / PUB_HZ
HOLD_S      = 0.15
RAMP_PER_S  = 1.0      # throttle change per second while held
DECAY_PER_S = 2.0      # throttle decay per second when released

ESC = '\x1b'

# Per-thruster keymap. Each entry maps a key to (channel, sign).
KEYS: dict[str, tuple[str, float]] = {
    'w': ('x', +1.0),  # forward
    's': ('x', -1.0),  # reverse
    'i': ('z', +1.0),  # rise
    'k': ('z', -1.0),  # dive
}


def _drain_keys() -> list[str]:
    """Read all available single-byte tokens. Drops ESC + CSI sequences."""
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


def _step_toward(value: float, target: float, max_step: float) -> float:
    """Move `value` toward `target` by at most `max_step`."""
    if value < target:
        return min(target, value + max_step)
    return max(target, value - max_step)


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
    pub_x = session.declare_publisher("manta/ex8/thrust_x/cmd")
    pub_z = session.declare_publisher("manta/ex8/thrust_z/cmd")

    print("ex8 submarine controller.")
    print("  W / S       forward / reverse  (thrust_x)")
    print("  I / K       rise / dive        (thrust_z)")
    print("  SPACE       zero both thrusters")
    print("  Ctrl-C      quit")
    print("Held-key auto-repeat ramps the throttle; release lets it decay.")

    fd = sys.stdin.fileno()
    saved = termios.tcgetattr(fd)
    tty.setcbreak(fd)

    last_seen: dict[str, float] = {}
    throttle = {'x': 0.0, 'z': 0.0}
    last_pub_t = time.monotonic()
    last_payload = {'x': None, 'z': None}

    try:
        while running:
            now = time.monotonic()
            dt = now - last_pub_t
            last_pub_t = now

            for tok in _drain_keys():
                if tok == ' ':
                    throttle['x'] = 0.0
                    throttle['z'] = 0.0
                    last_seen.clear()
                    continue
                if tok in KEYS:
                    last_seen[tok] = now

            # Which keys are currently "held" (auto-repeat alive).
            held = {k for k, t in last_seen.items() if now - t < HOLD_S}

            # Per-channel target: sum of (sign × ramp · dt) for held keys.
            # If no keys for a channel are held, decay toward 0.
            for ch in ('x', 'z'):
                signs = [s for k, (c, s) in KEYS.items() if c == ch and k in held]
                if signs:
                    throttle[ch] += sum(signs) * RAMP_PER_S * dt
                else:
                    throttle[ch] = _step_toward(throttle[ch], 0.0,
                                                DECAY_PER_S * dt)
                throttle[ch] = max(-1.0, min(1.0, throttle[ch]))

            # Publish each channel separately (only on change to keep the
            # wire quiet — the sim picks up the latest sample on each tick).
            for ch, pub in (('x', pub_x), ('z', pub_z)):
                payload = [throttle[ch]]
                if payload != last_payload[ch]:
                    pub.put(json.dumps(payload).encode("utf-8"))
                    last_payload[ch] = payload

            time.sleep(PUB_PERIOD)
    finally:
        # Best-effort zero on exit.
        pub_x.put(json.dumps([0.0]).encode("utf-8"))
        pub_z.put(json.dumps([0.0]).encode("utf-8"))
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)
        session.close()
        print("\nkeyboard_controller: stopped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
