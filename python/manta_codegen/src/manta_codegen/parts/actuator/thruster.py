"""Thruster — polynomial-in-throttle force/torque actuator (1st through 4th order)."""

from __future__ import annotations

from ..._format import cpp_float as _f
from ...core import NoiseChannel, PartDescriptor
from ...signal import scalar_in_signal, scalar_out_signal


class _ThrusterBase(PartDescriptor):
    """Common scaffolding for `Thruster1`..`Thruster4`. Each tick:
        F = Σ_k F_k · throttle^k
        τ = Σ_k τ_k · throttle^k
    where F_k and τ_k are user-supplied 3-vectors in part frame.
    """

    cpp_header  = "manta/parts/actuator/thruster.hpp"
    _order:  int = 0   # set by subclass
    _cpp_class_concrete:  str = ""
    _cpp_class_template_: str = ""

    signals = [
        scalar_out_signal("throttle",     "throttle"),
        scalar_in_signal ("set_throttle", "set_throttle"),
    ]

    # Mirror the commanded throttle from MFloat to Jet before predict so
    # the Jet world's force model matches what the value-side craft was
    # told to apply (via cross-world `connect()` or external Zenoh).
    actuator_state = [("set_throttle", "throttle")]

    def __init__(self,
                 name: str,
                 force_coefs:  list[tuple[float, float, float]],
                 torque_coefs: list[tuple[float, float, float]] | None = None,
                 force_noise_sigma: float = -1.0,
                 **kwargs) -> None:
        super().__init__(name=name, **kwargs)
        if len(force_coefs) != self._order:
            raise ValueError(f"{type(self).__name__} requires {self._order} force_coefs")
        if torque_coefs is None:
            torque_coefs = [(0.0, 0.0, 0.0)] * self._order
        if len(torque_coefs) != self._order:
            raise ValueError(f"{type(self).__name__} requires {self._order} torque_coefs")
        self.force_coefs       = [tuple(float(x) for x in v) for v in force_coefs]
        self.torque_coefs      = [tuple(float(x) for x in v) for v in torque_coefs]
        # σ < 0 (default) = no noise sampling, no EKF slot.
        # σ ≥ 0 = sample on sim path, register slot for auto-Q.
        self.force_noise_sigma = float(force_noise_sigma)

    def noise_channels(self) -> list[NoiseChannel]:
        return [NoiseChannel("force_noise", "white_3d", self.force_noise_sigma)]

    @property
    def cpp_class(self) -> str: return self._cpp_class_concrete
    @property
    def cpp_class_template(self) -> str: return self._cpp_class_template_

    def _vec_array_expr(self, coefs, scalar: str) -> str:
        vec = f"manta::geom::Vec3<manta::PartFrame, {scalar}>"
        elems = ", ".join(
            f"{vec}{{{scalar}({_f(v[0])}), {scalar}({_f(v[1])}), {scalar}({_f(v[2])})}}"
            for v in coefs)
        return f"std::array<{vec}, {self._order}>{{{elems}}}"

    def emit_constructor_args(self, scalar: str = "manta::MFloat") -> str:
        return (f'"{self.name}", '
                f'{self._vec_array_expr(self.force_coefs,  scalar)}, '
                f'{self._vec_array_expr(self.torque_coefs, scalar)}, '
                f'{_f(self.force_noise_sigma)}')

    def telemetry_fields(self) -> list[tuple[str, str]]:
        return [("throttle", "float")]

    def emit_telemetry_reads(self) -> list[tuple[str, str]]:
        return [("throttle", f"craft.{self.name}().throttle()")]


class Thruster1(_ThrusterBase):
    """1st-order linear thruster: F = F_1 · t, τ = τ_1 · t.

    Convenience constructor `Thruster1.linear(name, max_thrust, direction)`
    builds F_1 = direction · max_thrust, τ_1 = 0 — the common case.
    """
    _order = 1
    _cpp_class_concrete = "manta::parts::Thruster1"
    _cpp_class_template_ = "manta::parts::Thruster1T"

    @classmethod
    def linear(cls,
               name: str,
               max_thrust: float,
               direction: tuple[float, float, float] = (0.0, 0.0, 1.0),
               **kwargs) -> "Thruster1":
        d = tuple(float(x) for x in direction)
        F1 = (d[0] * float(max_thrust), d[1] * float(max_thrust), d[2] * float(max_thrust))
        return cls(name, [F1], None, **kwargs)


class Thruster2(_ThrusterBase):
    _order = 2
    _cpp_class_concrete = "manta::parts::Thruster2"
    _cpp_class_template_ = "manta::parts::Thruster2T"


class Thruster3(_ThrusterBase):
    _order = 3
    _cpp_class_concrete = "manta::parts::Thruster3"
    _cpp_class_template_ = "manta::parts::Thruster3T"


class Thruster4(_ThrusterBase):
    _order = 4
    _cpp_class_concrete = "manta::parts::Thruster4"
    _cpp_class_template_ = "manta::parts::Thruster4T"


class Thruster(Thruster1):
    """Backward-compat shim — `Thruster(name, max_thrust, direction)` is
    equivalent to `Thruster1.linear(name, max_thrust, direction)`. Most
    user code wants this one.
    """

    def __init__(self,
                 name: str,
                 max_thrust: float,
                 direction: tuple[float, float, float] = (0.0, 0.0, 1.0),
                 **kwargs) -> None:
        d = tuple(float(x) for x in direction)
        F1 = (d[0] * float(max_thrust), d[1] * float(max_thrust), d[2] * float(max_thrust))
        super().__init__(name, [F1], None, **kwargs)
        self.max_thrust = float(max_thrust)
        self.direction  = d

    def render(self, telemetry: dict, path: str) -> None:
        try:
            import rerun as rr
        except ImportError:
            return
        rr.log(path, rr.Boxes3D(half_sizes=[[0.04, 0.04, 0.06]],
                                colors=[[200, 100, 50]]))
        thr = float(telemetry.get("throttle", 0.0))
        if thr > 0:
            d = self.direction
            length = 0.5 * thr
            rr.log(f"{path}/plume",
                   rr.Arrows3D(origins=[[0, 0, 0]],
                               vectors=[[-d[0]*length, -d[1]*length, -d[2]*length]],
                               colors=[[255, 100, 50]]))
        rr.log(f"{path}/throttle", rr.Scalar(thr))
