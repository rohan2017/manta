"""Module-type registry for the Mako builder — the server side of the
catalog tables in `web/mako/SPEC.md`, transcribed from the design document
`~/projects/MakoModules.pdf` (2026-07-07).

Every module is a segment of the ONE spine: `kind` is `"nose"` (front cap,
attaches aft only), `"rear"` (stern cap, attaches forward only) or `"spine"`
(attaches both ways). There are no side/top mount points any more — the
craft is a stack of ~20 cm-diameter cylinders. Per the document:

  * hull diameter 0.20 m everywhere (`HULL_R` = 0.10);
  * every module is NEUTRALLY buoyant — its displaced volume equals its
    mass / ρ — plus a small shared RESERVE (`BUOY_RESERVE`), so the whole
    craft creeps to the surface unpowered. Mass is DERIVED (fairing volume
    × a flooded-fraction `fill`), not a user option. Free-flooding modules
    (transverse thrusters, the stern modules' wetted tails) have `fill` < 1;
  * CoM sits slightly below CoB (1 cm) → slight passive roll/pitch
    stability;
  * drag comes from WETTED surfaces only: spine modules impart axial drag
    from their cylindrical skin alone (the fore/aft faces are concealed by
    stacking) and crossflow drag from the D·L profile; only nose/rear caps
    add face drag;
  * thrusters are reversible, ~250 N, and carry prop counter-torque.

Builders share one signature::

    build(craft, name, xc, opts) -> Contrib

with `name` = `"<type><spineIndex>"`, `xc` the module centre x and `opts`
the clamped options. The returned `Contrib` records the analytic
bookkeeping (mass, displacement, thruster wrench columns, drag tensors)
that `builder.py` turns into trim / feedforward / rate scales and
`meta.json` — nothing is re-derived from the parts.

Coordinates: x forward (nose at +x), y left, z up, water surface z = 0,
hull axis on z = 0.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from manta.fields import FluidField
from manta.ir.frames import WorldFrame
from manta.ir.types import Scalar
from manta.parts import (
    Barometer, ControlSurface, DragSurface, IMU, Mass, Output, PointBuoy,
    PositionSensor, Thruster, VelocitySensor,
)

# --- world constants (mirrored by builder.py's flat ocean) -------------------
CRAFT = "mako"
RHO_WATER = 1025.0                # seawater density, kg/m³
GRAVITY = 9.81                    # m/s²
HULL_R = 0.10                     # hull radius everywhere (m) — 20 cm dia
COM_DROP = 0.01                   # CoM this far below the buoy line (m)
# (halved 2026-07-17: 2 cm gave a righting moment that overpowered the
# fins/attitude control — the boat snapped level too aggressively)
# Reserve buoyancy: every module displaces this fraction MORE than its
# neutral volume, so an unpowered mako slowly creeps to the surface
# (~1.2 % of weight; terminal rise ~0.06 m/s against the hull's
# crossflow drag). Real AUVs are trimmed the same way as a dead-ship
# failsafe.
BUOY_RESERVE = 1.012

# Prop counter-torque per newton of thrust (N·m/N): an undersea prop's
# reaction torque about its own axis, signed by the spin direction. Paired
# thrusters (agility modules) counter-rotate so it cancels; a lone stern
# prop heels the boat a few degrees at full power — that's the physics.
K_CT = 0.015

# Drag surfaces get a small LINEAR term (A_1) alongside the quadratic one.
# Pure-quadratic drag has a zero Jacobian at rest, so every unactuated mode
# (roll, notably) linearizes to an undamped oscillator on the unit circle
# and the LQR's Riccati iteration cannot converge. A linear skin-friction
# floor — the quadratic law evaluated at V_LIN — keeps the physics
# essentially identical at cruise while making the at-rest linearization
# honestly damped.
V_LIN = 0.4                       # linear-drag reference speed (m/s)

_CD_CROSSFLOW = 1.1               # cylinder crossflow cd (lateral drag)
_CF_SKIN = 0.008                  # axial skin-friction coefficient
_CD_FACE = 0.25                   # exposed nose/stern face cd


def _module_mass(length: float, fill: float) -> float:
    """Neutral-buoyancy mass: fairing volume × flooded fraction × ρ."""
    return RHO_WATER * math.pi * HULL_R ** 2 * length * fill


def _body(craft, c: "Contrib", name: str, xc: float, length: float,
          fill: float, *, face_drag: bool = False) -> float:
    """The shared hull segment: derived neutral mass (CoM COM_DROP low), the
    matching buoy, and the wetted-surface drag pair. Returns the mass."""
    m = _module_mass(length, fill)
    # Roll MoI at 0.8·m·r² (shell + internals, not a solid cylinder) — the
    # roll axis of a 20 cm hull is feather-light either way and needs every
    # honest kg·m² it can get.
    i_ax = 0.8 * m * HULL_R ** 2
    i_tr = m * (3.0 * HULL_R ** 2 + length ** 2) / 12.0
    at_m = (xc, 0.0, -COM_DROP)
    craft.add(Mass(name, mass=m, moi=(i_ax, i_tr, i_tr), transform=at_m))
    c.masses.append((m, at_m))
    vol = m / RHO_WATER * BUOY_RESERVE
    craft.add(PointBuoy(f"{name}_buoy", volume=vol, transform=(xc, 0.0, 0.0)))
    c.buoys.append((vol, (xc, 0.0, 0.0)))

    # Wetted-surface drag: crossflow from the D·L profile, axial from the
    # cylindrical skin (+ the exposed face on nose/rear caps). Split FOUR
    # ways around the rim (z = ±0.09, y = ±0.09) so the linear V_LIN floor
    # damps roll as well as pitch/yaw — a bare spinning hull is otherwise
    # a near-undamped oscillator on a feather-light axis.
    k_x = 0.5 * (math.pi * 2.0 * HULL_R * length) * _CF_SKIN
    if face_drag:
        k_x += 0.5 * (math.pi * HULL_R ** 2) * _CD_FACE
    k_yz = 0.5 * (2.0 * HULL_R * length) * _CD_CROSSFLOW
    rim = 0.09
    for tag, off in (("u", (0.0, +rim)), ("d", (0.0, -rim)),
                     ("l", (+rim, 0.0)), ("r", (-rim, 0.0))):
        _drag(craft, c, f"{name}_drag{tag}", kx=k_x / 4, kyz=k_yz / 4,
              transform=(xc, off[0], off[1]))
    return m


def _drag(craft, c: "Contrib", name: str, *, kx: float, kyz: float,
          transform) -> None:
    """Anisotropic quadratic drag + the linear V_LIN floor (see above).
    `k` = ½·area·cd per axis; recorded on `c.drags` for the feedforward."""
    k = np.diag([kx, kyz, kyz])
    craft.add(DragSurface(name, force_tensors=[-k * V_LIN, -k],
                          transform=transform))
    c.drags.append({"k": (kx, kyz, kyz), "at": tuple(transform)})


def _thruster(craft, c: "Contrib", part: str, *, axis, at, spin: int,
              opts) -> None:
    """A reversible prop: unit force axis (throttle in newtons), counter-
    torque K_CT·spin about its own axis, noise from the option."""
    ax = np.asarray(axis, dtype=float)
    ax /= np.linalg.norm(ax)
    ct = K_CT * spin
    craft.add(Thruster(part, force=tuple(ax), torque=tuple(ct * ax),
                       force_noise_sigma=opts["noise"], transform=at))
    mt = opts["max_thrust"]
    c.thrusters.append({"part": part, "axis": tuple(ax), "at": tuple(at),
                        "ct": ct, "max_thrust": mt})
    c.inputs.append({"name": f"{part}.throttle", "min": -mt, "max": mt})


# --- the water-gated GPS (submarine.py's SurfaceGPS pattern) -----------------

class SurfaceGPS(PositionSensor):
    """`PositionSensor` that also reports whether its antenna is underwater.

    `wet` is a smooth 0..1 from the local fluid density (seawater → 1, air →
    0) — a submerged antenna gets no fix."""

    requires_fields = [FluidField]

    wet = Output()

    def update(self, ctx):
        out = super().update(ctx)
        rho = ctx.field(FluidField).value_at_sym(
            ctx.position[WorldFrame], ctx.t).density
        out.outputs["wet"] = Scalar(rho * (1.0 / RHO_WATER))
        out.rates["wet"] = self.rate
        return out


# The GPS antenna geometry is shared by every module that carries one
# (Brain, Fin Control, Rear Thrust): a short mast on the hull top.
GPS_ANT_Z = HULL_R + 0.06


def _gps(craft, part: str, x: float, opts) -> None:
    craft.add(SurfaceGPS(part, position_noise_sigma=opts["gps_noise"],
                         transform=(x, 0.0, GPS_ANT_Z)))


# --- catalog plumbing --------------------------------------------------------

@dataclass(frozen=True)
class Opt:
    """One clamped option: `[lo, hi]` range and the default."""
    lo: float
    hi: float
    default: float


@dataclass
class Contrib:
    """What one module contributed — the builder's analytic bookkeeping.

    masses    — (kg, (x, y, z)) per Mass part.
    buoys     — (m³, (x, y, z)) per PointBuoy.
    thrusters — {part, axis, at, ct, max_thrust}: one wrench column each
                (torque = r×axis + ct·axis — ct is the prop counter-torque).
    inputs    — {name (part-local, e.g. "p.throttle"), min, max} for meta.
    drags     — {k: (kx, ky, kz) ½·area·cd per axis, at}: feeds the client's
                rate feedforward (drag at the commanded rates).
    fins      — {input, area, x, normal ("y"|"z"), max_defl}: control
                surfaces — credited AT SPEED by the mixer/rate_scale (they
                have no authority at rest).
    """
    masses:    list = field(default_factory=list)
    buoys:     list = field(default_factory=list)
    thrusters: list = field(default_factory=list)
    inputs:    list = field(default_factory=list)
    drags:     list = field(default_factory=list)
    fins:      list = field(default_factory=list)

    def merge(self, other: "Contrib") -> "Contrib":
        for f in ("masses", "buoys", "thrusters", "inputs",
                  "drags", "fins"):
            getattr(self, f).extend(getattr(other, f))
        return self


@dataclass(frozen=True)
class ModuleType:
    """One catalog entry: where it may sit on the spine, its fixed length,
    flooded fraction (drives the derived neutral mass) and options."""
    kind:    str                      # "nose" | "spine" | "rear"
    length:  float                    # x extent (m), fixed per the document
    fill:    float                    # displaced fraction of fairing volume
    options: dict                     # option name -> Opt
    build:   object                   # build(craft, name, xc, opts)


# --- nose modules -------------------------------------------------------------

def _build_nose_dvl(craft, name, xc, opts) -> Contrib:
    """Nose cone with a downward-pointing DVL embedded in the underside."""
    c = Contrib()
    _body(craft, c, name, xc, 0.30, 0.85, face_drag=True)
    craft.add(VelocitySensor(f"{name}_dvl",
                             velocity_noise_sigma=opts["dvl_noise"],
                             transform=(xc + 0.03, 0.0, -0.08)))
    return c


def _build_nose_camera(craft, name, xc, opts) -> Contrib:
    """Nose cone with 2 stereo cameras looking forward. Structure-only for
    now — manta cameras need optical targets and the Mako swims alone."""
    c = Contrib()
    _body(craft, c, name, xc, 0.20, 0.85, face_drag=True)
    return c


def _build_nose_sonar(craft, name, xc, opts) -> Contrib:
    """Nose cone with a sonar array (sonar not in manta yet) — mass/drag/
    buoyancy only. The hanging transducer adds a little extra crossflow."""
    c = Contrib()
    _body(craft, c, name, xc, 0.50, 0.80, face_drag=True)
    _drag(craft, c, f"{name}_xdcr", kx=0.5 * 0.004 * 0.9,
          kyz=0.5 * 0.006 * 1.1, transform=(xc + 0.15, 0.0, -0.13))
    return c


# --- plain spine modules --------------------------------------------------------

def _build_brain(craft, name, xc, opts) -> Contrib:
    """Spine module with the flight computer: GPS antenna on top + a depth
    (pressure) sensor — the one sensor every real AUV computer carries."""
    c = Contrib()
    _body(craft, c, name, xc, 0.20, 1.0)
    _gps(craft, f"{name}_gps", xc, opts)
    craft.add(Barometer(f"{name}_depth",
                        pressure_noise_sigma=opts["depth_noise"],
                        transform=(xc, 0.0, 0.0)))
    return c


def _build_ui(craft, name, xc, opts) -> Contrib:
    """Spine module with a payload attachment point underneath (payloads
    arrive later) — structure only for now."""
    c = Contrib()
    _body(craft, c, name, xc, 0.15, 1.0)
    return c


def _build_battery(craft, name, xc, opts) -> Contrib:
    """Battery segment — no battery model in manta yet; mass/drag only."""
    c = Contrib()
    _body(craft, c, name, xc, 0.35, 1.0)
    return c


def _build_ins(craft, name, xc, opts) -> Contrib:
    """Spine module with an INS-grade IMU."""
    c = Contrib()
    _body(craft, c, name, xc, 0.30, 1.0)
    craft.add(IMU(f"{name}_imu", gyro_noise_sigma=opts["gyro_noise"],
                  accel_noise_sigma=opts["accel_noise"],
                  gyro_bias_sigma=opts["gyro_bias"],
                  transform=(xc, 0.0, 0.0)))
    return c


def _build_side_scan(craft, name, xc, opts) -> Contrib:
    """Long spine module with two side-scan sonars (sonar not in manta yet)
    — mass/drag/buoyancy only."""
    c = Contrib()
    _body(craft, c, name, xc, 0.55, 1.0)
    return c


# --- transverse thruster spine modules -------------------------------------------

def _build_thrust_vert(craft, name, xc, opts) -> Contrib:
    """Spine module with a vertical through-hull tunnel thruster. The
    tunnel floods → less displacement than the fairing."""
    c = Contrib()
    _body(craft, c, name, xc, 0.30, 0.70)
    _thruster(craft, c, name + "_prop", axis=(0.0, 0.0, 1.0),
              at=(xc, 0.0, 0.0), spin=+1, opts=opts)
    return c


def _build_thrust_horiz(craft, name, xc, opts) -> Contrib:
    """Spine module with a horizontal (sway) tunnel thruster."""
    c = Contrib()
    _body(craft, c, name, xc, 0.30, 0.70)
    _thruster(craft, c, name + "_prop", axis=(0.0, 1.0, 0.0),
              at=(xc, 0.0, 0.0), spin=+1, opts=opts)
    return c


_CANT = math.radians(25.0)

# Pod drag: each agility pod (duct + motor + strut) is a bluff body of
# ~0.008 m² frontal area, cd ≈ 1, sitting 0.165–0.19 m off the hull axis —
# far outside the 0.09 m rim surfaces. k = ½·A·cd, isotropic (the duct's
# through-flow relief roughly cancels its strut/housing extra). These are
# the dominant roll dampers on an otherwise feather-light axis: a craft
# with sponson pods should NOT roll like a bare cylinder.
POD_K = 0.004


def _build_agility(cant_sign: float):
    """Front/rear agility module: 4 pod thrusters on side sponsons, two
    vertical (±y, counter-rotating) and two surge thrusters canted 25° —
    inward for the front module (`cant_sign` = −1), outward for the rear
    (+1). As in the document, the canted pods sit at the front corners of
    the front module and the rear corners of the rear one (vertical pods
    staggered the other way) — matching web/mako/catalog.js's meshes."""
    def _build(craft, name, xc, opts) -> Contrib:
        c = Contrib()
        _body(craft, c, name, xc, 0.25, 0.80)
        canted_x = xc - cant_sign * 0.05
        vert_x = xc + cant_sign * 0.06
        for tag, side in (("l", +1.0), ("r", -1.0)):
            _thruster(craft, c, f"{name}_v{tag}", axis=(0.0, 0.0, 1.0),
                      at=(vert_x, side * 0.165, 0.0), spin=int(side),
                      opts=opts)
            fx = (math.cos(_CANT), cant_sign * side * math.sin(_CANT), 0.0)
            _thruster(craft, c, f"{name}_f{tag}", axis=fx,
                      at=(canted_x, side * 0.19, 0.0), spin=-int(side),
                      opts=opts)
            # the pods' own bluff-body drag, at the pod positions
            _drag(craft, c, f"{name}_podv{tag}", kx=POD_K, kyz=POD_K,
                  transform=(vert_x, side * 0.165, 0.0))
            _drag(craft, c, f"{name}_podf{tag}", kx=POD_K, kyz=POD_K,
                  transform=(canted_x, side * 0.19, 0.0))
        return c
    return _build


# --- rear (stern) modules ----------------------------------------------------------

def _build_rear_thrust(craft, name, xc, opts) -> Contrib:
    """Stern module: main prop in a duct + the shared GPS antenna."""
    c = Contrib()
    _body(craft, c, name, xc, 0.40, 0.70, face_drag=True)
    _thruster(craft, c, name + "_prop", axis=(1.0, 0.0, 0.0),
              at=(xc - 0.20, 0.0, 0.0), spin=+1, opts=opts)
    _gps(craft, f"{name}_gps", xc + 0.14, opts)
    return c


def _build_fin_control(craft, name, xc, opts) -> Contrib:
    """Stern module: main prop + four identical actuating fins in a +
    configuration (top/bottom rudders, left/right stern planes) + the
    shared GPS antenna."""
    c = Contrib()
    _body(craft, c, name, xc, 0.40, 0.65, face_drag=True)
    _thruster(craft, c, name + "_prop", axis=(1.0, 0.0, 0.0),
              at=(xc - 0.20, 0.0, 0.0), spin=+1, opts=opts)
    _gps(craft, f"{name}_gps", xc + 0.12, opts)

    lim = math.radians(opts["max_deflection"])
    area = opts["fin_area"]
    x_f = xc - 0.12
    # + tail, all four fins identical: the vertical pair (normal y) steers
    # yaw, the horizontal pair (normal z) steers pitch; each has its own
    # deflection input and the LQR mixes them.
    fins = (("t", (0.0, 0.0, +0.16), (0.0, 1.0, 0.0)),
            ("b", (0.0, 0.0, -0.16), (0.0, 1.0, 0.0)),
            ("l", (0.0, +0.16, 0.0), (0.0, 0.0, 1.0)),
            ("r", (0.0, -0.16, 0.0), (0.0, 0.0, 1.0)))
    for tag, off, normal in fins:
        part = f"{name}_fin{tag}"
        at = (x_f + off[0], off[1], off[2])
        craft.add(ControlSurface(part, area=area, chord=math.sqrt(area),
                                 chord_axis=(1.0, 0.0, 0.0),
                                 normal_axis=normal, max_deflection=lim,
                                 transform=at))
        c.inputs.append({"name": f"{part}.deflection_cmd",
                         "min": -lim, "max": lim})
        c.fins.append({"input": f"{part}.deflection_cmd",
                       "area": area, "x": x_f + off[0],
                       "normal": "y" if tag in ("t", "b") else "z",
                       # signed radial arm ⊥ to the lift direction — the
                       # roll lever (z offset for lift-y fins, y for lift-z)
                       "r_off": off[2] if tag in ("t", "b") else off[1],
                       "max_defl": lim})
        # Parasitic (interference) drag at the fin station, normal to the
        # blade. Beyond honesty, this matters dynamically: it damps roll
        # and yaw at the fin arms — the ControlSurface's own lift model
        # goes erratic past stall and provides no reliable damping there.
        k_fin = 0.5 * area * 1.2
        _drag(craft, c, f"{part}_pdrag",
              kx=0.5 * area * 0.02,
              kyz=k_fin, transform=at)
    return c


# --- the registry ------------------------------------------------------------------

_THRUST_OPTS = {"max_thrust": Opt(50, 500, 250), "noise": Opt(0, 10, 1)}
_GPS_OPT = {"gps_noise": Opt(0.02, 2, 0.1)}

CATALOG: dict[str, ModuleType] = {
    # noses
    "nose_dvl": ModuleType("nose", 0.30, 0.85, {
        "dvl_noise": Opt(0.005, 0.2, 0.02)}, _build_nose_dvl),
    "nose_camera": ModuleType("nose", 0.20, 0.85, {}, _build_nose_camera),
    "nose_sonar": ModuleType("nose", 0.50, 0.80, {}, _build_nose_sonar),
    # spine
    "brain": ModuleType("spine", 0.20, 1.0, {
        **_GPS_OPT, "depth_noise": Opt(100, 5000, 500)}, _build_brain),
    "ins": ModuleType("spine", 0.30, 1.0, {
        "gyro_noise": Opt(1e-4, 0.01, 5e-4),
        "gyro_bias": Opt(0, 1e-3, 2e-5),
        "accel_noise": Opt(0.005, 0.5, 0.02)}, _build_ins),
    "ui": ModuleType("spine", 0.15, 1.0, {}, _build_ui),
    "battery": ModuleType("spine", 0.35, 1.0, {}, _build_battery),
    "side_scan": ModuleType("spine", 0.55, 1.0, {}, _build_side_scan),
    "thrust_vert": ModuleType("spine", 0.30, 0.70, dict(_THRUST_OPTS),
                              _build_thrust_vert),
    "thrust_horiz": ModuleType("spine", 0.30, 0.70, dict(_THRUST_OPTS),
                               _build_thrust_horiz),
    "agility_front": ModuleType("spine", 0.25, 0.80, dict(_THRUST_OPTS),
                                _build_agility(cant_sign=-1.0)),
    "agility_rear": ModuleType("spine", 0.25, 0.80, dict(_THRUST_OPTS),
                               _build_agility(cant_sign=+1.0)),
    # sterns
    "rear_thrust": ModuleType("rear", 0.40, 0.70, {
        **_THRUST_OPTS, **_GPS_OPT}, _build_rear_thrust),
    "fin_control": ModuleType("rear", 0.40, 0.65, {
        **_THRUST_OPTS, **_GPS_OPT,
        "fin_area": Opt(0.005, 0.03, 0.011),
        "max_deflection": Opt(10, 40, 25)}, _build_fin_control),
}


def spine_length(type_name: str, opts: dict) -> float:
    """A module's x extent — fixed per type in this catalog."""
    return CATALOG[type_name].length


def clamp_options(type_name: str, raw: dict) -> tuple[dict, list[str]]:
    """Clamp `raw` options to the catalog ranges. Unknown keys are ignored,
    missing options take defaults, non-numeric values are reported as
    errors. Returns `(opts, errors)` with `opts` sorted by key (the
    canonical form the design hash runs on)."""
    schema = CATALOG[type_name].options
    errors: list[str] = []
    out: dict[str, float] = {}
    raw = raw if isinstance(raw, dict) else {}
    for key in sorted(schema):
        spec = schema[key]
        v = raw.get(key, spec.default)
        try:
            v = float(v)
            if not math.isfinite(v):
                raise ValueError
        except (TypeError, ValueError):
            errors.append(f"{type_name}.{key}: expected a number, "
                          f"got {v!r}")
            v = spec.default
        out[key] = round(min(max(v, spec.lo), spec.hi), 9)
    return out, errors
