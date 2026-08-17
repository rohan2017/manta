"""Distributed displacement-hull composite for surface craft.

``DisplacementHull`` is deliberately a composition of Manta's existing
``PointBuoy`` and ``DragSurface`` primitives, not another hydrodynamics
solver.  It fills an ellipsoidal design envelope with deterministic,
volume-weighted quadrature samples.  Every sample queries the local fluid, so
the wet subset changes naturally with draft, heel, trim, waves, and spatially
varying currents.  The normal wrench roll-up then turns those distributed
forces into hydrostatic righting moments and rotational damping.

The ellipsoid is a calibration envelope rather than collision geometry.  Its
``displacement_volume`` may differ from its geometric volume: measured
displacement is usually a better hydrostatic calibration than idealized hull
dimensions.  ``hydrostatic_offset`` shifts the whole sample cloud relative to
the hull frame, providing the corresponding centre-of-buoyancy calibration.

This is a low-speed displacement model.  It intentionally excludes planing,
slamming, wave radiation, added mass, mesh collision, and CFD interactions.
Use separate parts for any of those effects.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..aero.drag_surface import DragSurface
from ..base import CompositePart
from .point_buoy import PointBuoy


@dataclass(frozen=True)
class HullSample:
    """One immutable hydrostatic quadrature sample.

    ``offset`` is in the hull part frame, ``volume`` is m³, and ``areas`` is
    the sample's share of the x/y/z quadratic-drag reference areas in m².
    The shares sum exactly to the hull's configured displacement and areas.
    """

    offset: tuple[float, float, float]
    volume: float
    areas: tuple[float, float, float]


def _positive_triplet(value, *, name: str) -> tuple[float, float, float]:
    try:
        out = tuple(float(x) for x in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be three finite positive values") from exc
    if len(out) != 3 or any(not math.isfinite(x) or x <= 0.0 for x in out):
        raise ValueError(f"{name} must be three finite positive values, got {value!r}")
    return out


def _nonnegative_triplet(value, *, name: str) -> tuple[float, float, float]:
    try:
        out = tuple(float(x) for x in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be three finite non-negative values") from exc
    if len(out) != 3 or any(not math.isfinite(x) or x < 0.0 for x in out):
        raise ValueError(
            f"{name} must be three finite non-negative values, got {value!r}"
        )
    return out


def _resolution(value) -> tuple[int, int, int]:
    try:
        nx, nr, nt = tuple(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "sample_resolution must be (axial, radial, circumferential)"
        ) from exc
    if any(
        isinstance(x, bool) or not isinstance(x, (int, np.integer))
        for x in (nx, nr, nt)
    ):
        raise ValueError("sample_resolution entries must be integers")
    nx, nr, nt = int(nx), int(nr), int(nt)
    if nx < 2 or nr < 1 or nt < 4 or nt % 2:
        raise ValueError(
            "sample_resolution requires axial >= 2, radial >= 1, and an "
            "even circumferential count >= 4"
        )
    return nx, nr, nt


def _ellipsoid_samples(
    *,
    dimensions: tuple[float, float, float],
    displacement_volume: float,
    reference_areas: tuple[float, float, float],
    hydrostatic_offset: tuple[float, float, float],
    resolution: tuple[int, int, int],
) -> tuple[HullSample, ...]:
    """Gauss/product quadrature of a solid ellipsoid.

    ``x = a*u`` and each cross-section is integrated in ``q = r²`` and
    azimuth.  Its Jacobian is ``a*b*c*(1-u²)/2``.  Gauss-Legendre in
    ``u`` and ``q`` integrates the total ellipsoid volume exactly; the even,
    midpoint azimuth grid keeps every sample paired across the x-y, x-z, and
    y-z symmetry planes.
    """

    length, beam, height = dimensions
    a, b, c = 0.5 * length, 0.5 * beam, 0.5 * height
    nx, nr, nt = resolution
    ox, oy, oz = hydrostatic_offset

    u_nodes, u_weights = np.polynomial.legendre.leggauss(nx)
    q_raw, q_raw_weights = np.polynomial.legendre.leggauss(nr)
    q_nodes = 0.5 * (q_raw + 1.0)
    q_weights = 0.5 * q_raw_weights
    dtheta = 2.0 * math.pi / nt

    geometric_volume = 4.0 * math.pi * a * b * c / 3.0
    volume_scale = displacement_volume / geometric_volume
    samples: list[HullSample] = []
    for u, wu in zip(u_nodes, u_weights, strict=True):
        section_scale = math.sqrt(max(0.0, 1.0 - float(u) ** 2))
        for q, wq in zip(q_nodes, q_weights, strict=True):
            radius = math.sqrt(float(q))
            raw_volume = (
                a * b * c * (1.0 - float(u) ** 2) * 0.5 * float(wu) * float(wq) * dtheta
            )
            volume = raw_volume * volume_scale
            fraction = volume / displacement_volume
            areas = tuple(area * fraction for area in reference_areas)
            for k in range(nt):
                theta = (k + 0.5) * dtheta
                samples.append(
                    HullSample(
                        offset=(
                            ox + a * float(u),
                            oy + b * section_scale * radius * math.cos(theta),
                            oz + c * section_scale * radius * math.sin(theta),
                        ),
                        volume=volume,
                        areas=areas,
                    )
                )

    # Floating arithmetic from the product sum should not leak into a
    # hydrostatic calibration.  Apply the tiny common correction once and
    # recompute area shares from the corrected volume fractions.
    correction = displacement_volume / sum(s.volume for s in samples)
    return tuple(
        HullSample(
            offset=s.offset,
            volume=s.volume * correction,
            areas=tuple(
                area * (s.volume * correction / displacement_volume)
                for area in reference_areas
            ),
        )
        for s in samples
    )


class DisplacementHull(CompositePart):
    """Low-speed surface-piercing displacement hull.

    Args:
        dimensions:
            ``(length, beam, height)`` of the ellipsoidal hydrostatic sample
            envelope, metres, in the hull part frame.
        displacement_volume:
            Full-submersion displaced volume in m³.  Defaults to the exact
            volume ``pi/6 * length * beam * height`` of the envelope.  Set it
            from a measured displacement calibration when available.
        hydrostatic_offset:
            Translation of the sample cloud in the hull frame, metres.  This
            is the explicit centre-of-buoyancy calibration knob.
        drag_coefficients:
            Per-axis non-negative quadratic ``(Cx, Cy, Cz)``.  Body x is
            normally longitudinal, y lateral, and z vertical.
        reference_areas:
            Optional per-axis drag reference areas in m².  Defaults to the
            ellipsoid's projected frontal, lateral, and planform areas.
        sample_resolution:
            ``(axial, radial, circumferential)`` product-quadrature counts.
            The circumferential count must be even.  ``(5, 2, 8)`` gives 80
            buoy/drag pairs and is the practical default; convergence should
            be checked for the craft's draft and sea-state bandwidth.

    ``mount_offset`` and ``mount_orientation`` are the normal ``Part`` mount
    pose and apply to the entire generated sample cloud.
    """

    def __init__(
        self,
        name: str,
        *,
        dimensions: tuple[float, float, float],
        displacement_volume: float | None = None,
        hydrostatic_offset: tuple[float, float, float] = (0, 0, 0),
        drag_coefficients: tuple[float, float, float] = (0.2, 0.8, 1.0),
        reference_areas: tuple[float, float, float] | None = None,
        sample_resolution: tuple[int, int, int] = (5, 2, 8),
        **mount_overrides,
    ) -> None:
        dimensions = _positive_triplet(dimensions, name="dimensions")
        drag_coefficients = _nonnegative_triplet(
            drag_coefficients, name="drag_coefficients"
        )
        resolution = _resolution(sample_resolution)
        try:
            hydrostatic_offset = tuple(float(x) for x in hydrostatic_offset)
        except (TypeError, ValueError) as exc:
            raise ValueError("hydrostatic_offset must be three finite values") from exc
        if len(hydrostatic_offset) != 3 or any(
            not math.isfinite(x) for x in hydrostatic_offset
        ):
            raise ValueError(
                "hydrostatic_offset must be three finite values, got "
                f"{hydrostatic_offset!r}"
            )

        length, beam, height = dimensions
        geometric_volume = math.pi * length * beam * height / 6.0
        if displacement_volume is None:
            displacement_volume = geometric_volume
        displacement_volume = float(displacement_volume)
        if not math.isfinite(displacement_volume) or displacement_volume <= 0.0:
            raise ValueError(
                "displacement_volume must be finite and > 0, got "
                f"{displacement_volume!r}"
            )

        if reference_areas is None:
            reference_areas = (
                math.pi * beam * height / 4.0,
                math.pi * length * height / 4.0,
                math.pi * length * beam / 4.0,
            )
        else:
            reference_areas = _nonnegative_triplet(
                reference_areas, name="reference_areas"
            )

        super().__init__(name, **mount_overrides)
        self.dimensions = dimensions
        self.displacement_volume = displacement_volume
        self.geometric_volume = geometric_volume
        self.hydrostatic_offset = hydrostatic_offset
        self.drag_coefficients = drag_coefficients
        self.reference_areas = reference_areas
        self.sample_resolution = resolution
        self.samples = _ellipsoid_samples(
            dimensions=dimensions,
            displacement_volume=displacement_volume,
            reference_areas=reference_areas,
            hydrostatic_offset=hydrostatic_offset,
            resolution=resolution,
        )

        for index, sample in enumerate(self.samples):
            # Names include the composite name because names are globally
            # unique across a craft, not scoped to their parent subtree.
            self.add(
                PointBuoy(
                    f"{name}_buoy_{index}",
                    volume=sample.volume,
                    mount_offset=sample.offset,
                )
            )
            self.add(
                DragSurface(
                    f"{name}_drag_{index}",
                    # Directional coefficients are folded into each sample's
                    # quadratic tensor so each axis retains its own Cd.
                    force_tensors=[
                        np.zeros((3, 3)),
                        -0.5
                        * np.diag(
                            np.asarray(sample.areas) * np.asarray(drag_coefficients)
                        ),
                    ],
                    mount_offset=sample.offset,
                )
            )

    def displaced_volume_below(self, waterline_z: float) -> float:
        """Discrete displaced volume below a flat hull-frame waterline.

        This calibration helper uses a hard horizontal cut through sample
        centres.  Runtime physics does *not*: each child queries the world's
        smooth, possibly moving fluid boundary.  Use this helper only to
        compare resolutions or choose an initial calm-water draft.
        """

        waterline_z = float(waterline_z)
        if not math.isfinite(waterline_z):
            raise ValueError("waterline_z must be finite")
        return sum(s.volume for s in self.samples if s.offset[2] <= waterline_z)


__all__ = ["DisplacementHull", "HullSample"]
