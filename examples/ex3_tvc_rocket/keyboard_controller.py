"""Keyboard controller for ex3 TVC rocket.

Sends [throttle, pitch_rate_sp, yaw_rate_sp] to 'manta/ex3/cmd'.

Bindings:
  i / k : throttle up / down
  w / s : pitch nose-up / nose-down (rate)
  a / d : yaw left / right (rate)
  space : zero rates (keep throttle)
  x     : zero everything (cut engine)
  ESC   : quit
"""

import json
import select
import signal
import sys
import termios
import time
import tty

import zenoh


def main() -> None:
    cfg = zenoh.Config()
    session = zenoh.open(cfg)
    pub = session.declare_publisher("manta/ex3/cmd")

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    def restore() -> None:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    signal.signal(signal.SIGINT, lambda *_: (restore(), session.close(), sys.exit(0)))
    tty.setcbreak(fd)

    thr = 0.0
    pitch = yaw = 0.0
    rate_step = 0.5
    thr_step  = 0.05

    print("controls: i/k=thr  w/s=pitch  a/d=yaw  space=zero rates  x=cut  ESC=quit")
    try:
        while True:
            while select.select([sys.stdin], [], [], 0)[0]:
                ch = sys.stdin.read(1)
                if ch == "\x1b":
                    return
                elif ch == "i": thr   = min(1.0, thr + thr_step)
                elif ch == "k": thr   = max(0.0, thr - thr_step)
                elif ch == "w": pitch += rate_step
                elif ch == "s": pitch -= rate_step
                elif ch == "a": yaw   -= rate_step
                elif ch == "d": yaw   += rate_step
                elif ch == " ": pitch = yaw = 0.0
                elif ch == "x": thr = pitch = yaw = 0.0
            pub.put(json.dumps([thr, pitch, yaw]))
            time.sleep(0.05)
    finally:
        restore()
        session.close()


if __name__ == "__main__":
    main()
