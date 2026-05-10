"""Keyboard controller for ex1_orbit.

Publishes per-thruster throttle commands on `manta/ex1/<name>/cmd` based on
held keys. The thruster layout (see config.py) is six bipolar thrusters
in three offset pairs; this script maps 6 keyboard inputs to the 6
thrusters via a fixed mixer:

    Key              Action            Affects
    ---              ------            -------
    W / S            ±Y translation    ty_xp, ty_xn   (both equal)
    A / D            ∓X translation    tx_zp, tx_zn   (both equal)
    Q / E            ±Z translation    tz_yp, tz_yn   (both equal)
    ↑ / ↓            ±Y rotation       tx_zp / tx_zn  (opposite)
    ← / →            ±Z rotation       ty_xp / ty_xn  (opposite)
    Z / X            ±X rotation       tz_yp / tz_yn  (opposite)
    SPACE            zero all thrusters
    Esc / Ctrl-C     quit

Each held key contributes a unit input; the per-thruster sum is clamped to
[-1, 1] before publishing. Releasing a key drops its contribution to zero.

Requires `pynput`:  python -m pip install pynput zenoh
"""

import json
import math
import signal
import sys
import threading
import time

import zenoh

try:
    from pynput import keyboard
except ImportError:
    sys.exit("controller.py: install pynput first: python -m pip install pynput")


PUB_HZ          = 50.0
PUB_PERIOD      = 1.0 / PUB_HZ
THRUSTERS       = ["tx_zp", "tx_zn", "ty_xp", "ty_xn", "tz_yp", "tz_yn"]


# ---------------------------------------------------------------------
# Input state — one float per logical DOF, set from keyboard callbacks.

class Input:
    """Tracks which keys are currently held; converts to 6 DOF inputs."""

    def __init__(self) -> None:
        self.held: set = set()
        self.lock   = threading.Lock()

    def on_press(self, key) -> None:
        with self.lock:
            self.held.add(_normalize(key))

    def on_release(self, key) -> None:
        with self.lock:
            self.held.discard(_normalize(key))

    def dofs(self) -> tuple[float, float, float, float, float, float]:
        """Return (u_x, u_y, u_z, u_pitch, u_yaw, u_roll) each in {-1, 0, +1}."""
        with self.lock:
            held = set(self.held)
        ux = _axis(held, 'd', 'a')           # WASD: D = +X, A = -X
        uy = _axis(held, 'w', 's')           # W = +Y, S = -Y
        uz = _axis(held, 'e', 'q')           # E = +Z, Q = -Z
        u_pitch = _axis(held, 'up', 'down')
        u_yaw   = _axis(held, 'right', 'left')
        u_roll  = _axis(held, 'x', 'z')
        return ux, uy, uz, u_pitch, u_yaw, u_roll


def _normalize(key) -> str:
    """Map a pynput key to a stable lowercase string token."""
    if hasattr(key, 'char') and key.char is not None:
        return key.char.lower()
    name = str(key).removeprefix('Key.')
    return name.lower()


def _axis(held: set, positive: str, negative: str) -> float:
    """+1 if `positive` is held, -1 if `negative` is held, else 0."""
    p = positive in held
    n = negative in held
    return float(p) - float(n)


# ---------------------------------------------------------------------
# Mixer: DOF inputs → per-thruster throttle.

def mix(ux: float, uy: float, uz: float,
        u_pitch: float, u_yaw: float, u_roll: float) -> dict[str, float]:
    """Map 6 DOF inputs to 6 thruster throttles, each clamped to [-1, 1].

    Translation: both thrusters in a pair get the same throttle.
    Rotation:    differential throttle (one +, the other -)."""
    cmd = {
        # X-pair offset in Z → translation X, differential ⇒ pitch (about Y).
        "tx_zp": ux + u_pitch,
        "tx_zn": ux - u_pitch,
        # Y-pair offset in X → translation Y, differential ⇒ yaw (about Z).
        "ty_xp": uy + u_yaw,
        "ty_xn": uy - u_yaw,
        # Z-pair offset in Y → translation Z, differential ⇒ roll (about X).
        "tz_yp": uz + u_roll,
        "tz_yn": uz - u_roll,
    }
    return {k: max(-1.0, min(1.0, v)) for k, v in cmd.items()}


# ---------------------------------------------------------------------
# Main loop.

def main() -> int:
    running = threading.Event()
    running.set()

    def on_signal(*_):
        running.clear()
    signal.signal(signal.SIGINT,  on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    inp = Input()
    listener = keyboard.Listener(on_press=inp.on_press, on_release=inp.on_release)
    listener.start()

    cfg = zenoh.Config()
    session = zenoh.open(cfg)
    pubs = {name: session.declare_publisher(f"manta/ex1/{name}/cmd")
            for name in THRUSTERS}

    print("ex1 controller: WASD/QE = translate, arrows = pitch/yaw, "
          "Z/X = roll, SPACE = zero, Esc = quit.")

    last_cmd: dict[str, float] = {n: math.nan for n in THRUSTERS}
    while running.is_set():
        ux, uy, uz, up, uyaw, ur = inp.dofs()
        if 'space' in inp.held:
            ux = uy = uz = up = uyaw = ur = 0.0
        if 'esc' in inp.held:
            running.clear()
            break
        cmd = mix(ux, uy, uz, up, uyaw, ur)
        # Publish only when something changed to keep the wire quiet.
        for name, v in cmd.items():
            if abs(v - last_cmd[name]) > 1e-4:
                pubs[name].put(json.dumps([v]).encode("utf-8"))
                last_cmd[name] = v
        time.sleep(PUB_PERIOD)

    # Zero all thrusters on exit so the craft stops accelerating.
    for name, pub in pubs.items():
        pub.put(json.dumps([0.0]).encode("utf-8"))
    listener.stop()
    session.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
