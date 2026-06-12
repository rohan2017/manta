"""ProcessNoise — white wrench noise, the model's "things I didn't model".

A part whose *only* contribution is a zero-mean white force/torque on the
craft. Physically it is Langevin forcing — random buffeting from
everything below the model's resolution (micro-turbulence, surface
chop, cable strum). A white force makes the velocity a random walk (a
Wiener process — Brownian motion in velocity; under drag it relaxes to an
Ornstein–Uhlenbeck process), and the torque channel does the same for
angular velocity.

Why you'd add one: process noise is what keeps a filter honest. Noise on
a *sensor* output builds the EKF's `R`; noise entering the *wrench*
propagates through the dynamics into the next state, so the EKF
auto-builds `Q` from it — and a `NoiseDriver` jitters the truth by the
same channel, so `nees` consistency holds by construction. Actuators
already carry such channels (`Thruster.force_noise`); `ProcessNoise` is
for crafts with no noisy actuator — a drifting buoy, a glider, a moored
instrument — where the model would otherwise declare `Q = 0`, its
covariance would collapse, and slow information channels (e.g.
gyrocompassing) would be down-weighted into uselessness.

    craft.add(ProcessNoise("pn", force_noise_sigma=0.05,
                           torque_noise_sigma=0.005))

Channels (per-tick white, in the part's own frame; σ defaults to 0 =
inert):
    force_noise  — N
    torque_noise — N·m

Mounted off the craft origin (Part.transform), the force also produces
the corresponding body torque, as with any part wrench.
"""

from __future__ import annotations

from ...ir.frames import PartFrame
from ...ir.types import Vec3
from .._declarations import WhiteNoise
from ..base import Part
from ...ir.wrench import Wrench


class ProcessNoise(Part):
    """White force/torque wrench — model uncertainty as Langevin forcing.

    Set σ to engage a channel (`force_noise_sigma=` / `torque_noise_sigma=`
    on construction); both default to 0 (a perfectly modeled craft). The
    EKF assembles `Q` from the engaged channels and `NoiseDriver` excites
    the truth identically.
    """

    force_noise  = WhiteNoise("R3", frame=PartFrame, sigma=0.0)
    torque_noise = WhiteNoise("R3", frame=PartFrame, sigma=0.0)

    def update(self, ctx):
        zero = Vec3[PartFrame].constant((0.0, 0.0, 0.0))
        return Wrench(force=zero + self.force_noise,
                      torque=zero + self.torque_noise)
