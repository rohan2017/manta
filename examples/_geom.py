"""Shared numpy quaternion / rotation helpers for the examples.

manta uses wxyz quaternions and the WorldFrame ENU convention throughout.
These are the small geometry utilities the example *drivers* need for viz
and readout — the library keeps its own symbolic versions in `manta.ir`.
They live here so every example agrees on the formulas instead of each
re-deriving them.
"""
from __future__ import annotations

import numpy as np

from manta.planets import Earth


def quat(axis, angle):
    """wxyz quaternion for a rotation of `angle` (rad) about `axis`."""
    h = 0.5 * angle
    ax = np.asarray(axis, dtype=float)
    return (np.cos(h), *(np.sin(h) * ax))


def quat_mul(a, b):
    """Hamilton product `a ⊗ b` of two wxyz quaternions."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([aw * bw - ax * bx - ay * by - az * bz,
                     aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw])


def rotmat(q):
    """3x3 body→world rotation matrix from a wxyz quaternion."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)]])


def euler_zyx(q):
    """ZYX yaw/pitch/roll (rad) from a wxyz quaternion."""
    w, x, y, z = q
    yaw_ = np.arctan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z))
    pitch = np.arcsin(np.clip(2 * (w*y - z*x), -1.0, 1.0))
    roll = np.arctan2(2 * (w*x + y*z), 1 - 2 * (x*x + y*y))
    return yaw_, pitch, roll


def yaw(q):
    """Heading — yaw about world z (rad) — from a wxyz quaternion."""
    w, x, y, z = q
    return np.arctan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z))


def wrap_pi(a):
    """Wrap an angle (rad) to [-π, π)."""
    return (a + np.pi) % (2 * np.pi) - np.pi


def wave_elevation(x, t, waves):
    """Deep-water planar wave elevation at along-track `x` and time `t` —
    the same sinusoid the Earth surface uses (deep-water dispersion
    ω = √(g·k)). `waves` is a `SeaWaves` (reads `.wavelength`,
    `.amplitude`)."""
    k = 2 * np.pi / waves.wavelength
    g0 = Earth.MU / Earth.R_EQ**2
    omega = k * np.sqrt(g0 * waves.wavelength / (2 * np.pi))
    return waves.amplitude * np.cos(k * x - omega * t)
