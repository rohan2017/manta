"""Keyboard controller for ex0 free-flight craft (codegen-binary version).

Each thruster has its own Zenoh command topic, so we publish a single float
per topic:
    manta/ex0/tx_p/cmd  ← [throttle ∈ [0,1]]
    manta/ex0/tx_n/cmd
    manta/ex0/ty_p/cmd
    manta/ex0/ty_n/cmd
    manta/ex0/tz_p/cmd
    manta/ex0/tz_n/cmd

Bindings (hold key to fire thruster, release lets the throttle decay):
    w/s : +x / -x
    a/d : +y / -y
    r/f : +z / -z
    q   : quit (or ESC)

Throttle decays toward 0 each tick when no key is pressed for that thruster.
"""

import json
import select
import signal
import sys
import termios
import time
import tty

import zenoh


THRUSTERS = ["tx_p", "tx_n", "ty_p", "ty_n", "tz_p", "tz_n"]
KEY_TO_INDEX = {
    "w": 0, "s": 1,   # ±x
    "a": 2, "d": 3,   # ±y
    "r": 4, "f": 5,   # ±z
}


def main() -> None:
    cfg = zenoh.Config()
    session = zenoh.open(cfg)
    publishers = {
        name: session.declare_publisher(f"manta/ex0/{name}/cmd")
        for name in THRUSTERS
    }

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    def restore() -> None:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    signal.signal(signal.SIGINT, lambda *_: (restore(), session.close(), sys.exit(0)))
    tty.setcbreak(fd)

    cmd = [0.0] * 6
    impulse = 1.0
    decay   = 0.85   # multiplicative decay per tick when key not pressed
    floor   = 0.02   # clamp tiny values to zero

    print("controls: w/s=±x  a/d=±y  r/f=±z  q/ESC=quit. Hold key to fire.")
    try:
        running = True
        while running:
            seen: set[int] = set()
            # Drain pending keys.
            while select.select([sys.stdin], [], [], 0)[0]:
                ch = sys.stdin.read(1)
                if ch == "q" or ch == "\x1b":
                    running = False
                    break
                idx = KEY_TO_INDEX.get(ch)
                if idx is not None:
                    seen.add(idx)
            if not running:
                break

            # Decay all thrusters; boost the ones whose key was seen this tick.
            for i, name in enumerate(THRUSTERS):
                if i in seen:
                    cmd[i] = impulse
                else:
                    cmd[i] *= decay
                    if cmd[i] < floor:
                        cmd[i] = 0.0
                publishers[name].put(json.dumps([cmd[i]]))

            time.sleep(0.05)   # 20 Hz
    finally:
        restore()
        session.close()


if __name__ == "__main__":
    main()
