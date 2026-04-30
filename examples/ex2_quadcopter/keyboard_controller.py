"""Keyboard controller for ex2 quadcopter.

Sends [thr, roll_rate, pitch_rate, yaw_rate] to 'manta/ex2/cmd':
  thr     ∈ [0, 1]  collective throttle
  *_rate  ∈ rad/s   rate setpoint

Bindings:
  i / k : throttle up / down
  w / s : pitch nose-up / nose-down (rate)
  a / d : roll left / right (rate)
  q / e : yaw left / right (rate)
  space : zero rates (keep throttle)
  x     : zero everything
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
    pub = session.declare_publisher("manta/ex2/cmd")

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    def restore() -> None:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    signal.signal(signal.SIGINT, lambda *_: (restore(), session.close(), sys.exit(0)))
    tty.setcbreak(fd)

    thr  = 0.0
    roll = pitch = yaw = 0.0
    rate_step = 0.5   # rad/s per keypress
    thr_step  = 0.05

    print("controls: i/k=thr  w/s=pitch  a/d=roll  q/e=yaw  space=zero rates  x=zero all  ESC=quit")
    try:
        while True:
            while select.select([sys.stdin], [], [], 0)[0]:
                ch = sys.stdin.read(1)
                if ch == "\x1b":   # ESC
                    return
                elif ch == "i":   thr   = min(1.0, thr + thr_step)
                elif ch == "k":   thr   = max(0.0, thr - thr_step)
                elif ch == "w":   pitch += rate_step
                elif ch == "s":   pitch -= rate_step
                elif ch == "a":   roll  -= rate_step
                elif ch == "d":   roll  += rate_step
                elif ch == "q":   yaw   -= rate_step
                elif ch == "e":   yaw   += rate_step
                elif ch == " ":   roll = pitch = yaw = 0.0
                elif ch == "x":   thr = roll = pitch = yaw = 0.0

            pub.put(json.dumps([thr, roll, pitch, yaw]))
            time.sleep(0.05)
    finally:
        restore()
        session.close()


if __name__ == "__main__":
    main()
