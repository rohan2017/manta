#pragma once

// Naca00xx — symmetric NACA 4-digit airfoil aerodynamics, modeled as a
// rectangular planform discretized into N spanwise segments. Each segment
// is treated as an independent 2D airfoil section ("blade-element"
// approach — no induced-flow / wingtip vortex / downwash modeling, no
// chordwise pressure distribution beyond what the lift/drag coefficients
// already capture).
//
// Conventions (part frame):
//
//   +x  : chord direction, leading edge at +c/2 (the "front" of the
//         wing), trailing edge at −c/2. Forward flight in body +x means
//         the relative wind blows from +x toward −x in part frame,
//         hitting the leading edge first at α = 0.
//   +y  : span direction, the wing extends along ±y.
//   +z  : thickness / "up" — positive lift acts roughly along +z at α=0.
//
// The part origin sits at the geometric center of the wing (mid-chord,
// mid-span, mid-thickness). Move the wing by attaching it under a
// CompositePart or by `set_transform(...)`.
//
// Per-segment force model (textbook, e.g. Anderson §5):
//
//   v_rel  = (fluid velocity at sample) − (segment velocity)
//   α      = atan2(v_rel.z, -v_rel.x)      // angle of attack about the y-axis
//   V_2d   = sqrt(v_rel.x² + v_rel.z²)     // chord-plane speed (spanwise drag ignored)
//   q      = 0.5 · ρ · V_2d²               // dynamic pressure
//
//   Cl_α   = 5.7  rad⁻¹                    // ≈ 2π·0.91 — empirical for NACA 00xx
//   Cd0    = 0.006 + 0.005·(t/c)           // empirical zero-α profile drag
//   Cl(α)  = Cl_α · α                       (|α| ≤ α_stall)
//   Cd(α)  = Cd0 + 0.05·α²                  (|α| ≤ α_stall)
//
// Past stall (|α| > α_stall ≈ 0.26 rad / 15°) the model switches to a
// flat-plate Newtonian form:
//
//   Cl(α)  = 2·sin(α)·cos(α)
//   Cd(α)  = Cd0 + 2·sin²(α)
//
// Discontinuous at α_stall but small (Cl matches at α_stall to ~10%).
//
// Force per segment:
//
//   F_seg  = q · S_seg · (Cl · L̂  +  Cd · D̂)
//   D̂     = (v_rel.x, 0, v_rel.z) / V_2d                  (drag along wind in xz plane)
//   L̂     = (v_rel.z, 0, -v_rel.x) / V_2d                 (lift ⟂ wind, +z at α=0)
//   S_seg = c · (b / N)
//
// where c = chord, b = span. The segment force is applied at the
// quarter-chord (x = -c/4) of the segment in part frame — the
// aerodynamic-center convention for symmetric airfoils. The pitching
// moment at the quarter-chord vanishes for a symmetric section, so
// `apply_force_at` carries the full per-segment wrench correctly to
// the part origin.
//
// Required field: FluidField (provides local ρ and ambient velocity).

#include <cmath>
#include <utility>
#include <vector>

#include "../../core/craft.hpp"
#include "../../core/features.hpp"
#include "../../core/types.hpp"
#include "../../fields/fluid_field.hpp"
#include "../../geom/casts.hpp"

namespace manta::parts {

template <class Scalar = MFloat>
class Naca00xxT : public PartT<Scalar> {
public:
    Naca00xxT(std::string name,
              MFloat chord,                 // m, total chord length
              MFloat span,                  // m, total span (along part-frame y)
              MFloat thickness_ratio,       // t/c, e.g. 0.15 for NACA 0015
              int    n_sample_points = 8)
        : PartT<Scalar>(std::move(name)),
          chord_(chord),
          span_(span),
          thickness_ratio_(thickness_ratio),
          n_segments_(n_sample_points < 1 ? 1 : n_sample_points),
          // Empirical profile-drag baseline for NACA 00xx, smooth surface.
          Cd0_(MFloat(0.006) + MFloat(0.005) * thickness_ratio_) {}

    MFloat chord()           const noexcept { return chord_; }
    MFloat span()            const noexcept { return span_; }
    MFloat thickness_ratio() const noexcept { return thickness_ratio_; }
    int    n_segments()      const noexcept { return n_segments_; }

    // ---- Knobs (tweak if your operating regime needs a different fit). ----
    MFloat Cl_alpha()      const noexcept { return Cl_alpha_; }
    MFloat alpha_stall()   const noexcept { return alpha_stall_; }
    MFloat Cd_alpha2()     const noexcept { return Cd_alpha2_; }
    MFloat Cd0()           const noexcept { return Cd0_; }

    void set_Cl_alpha   (MFloat v) noexcept { Cl_alpha_    = v; }
    void set_alpha_stall(MFloat v) noexcept { alpha_stall_ = v; }
    void set_Cd_alpha2  (MFloat v) noexcept { Cd_alpha2_   = v; }
    void set_Cd0        (MFloat v) noexcept { Cd0_         = v; }

    // Last-tick scalars per segment, primarily for telemetry/tests.
    // segments_[i] = (alpha, V_2d, dynamic_pressure, Cl, Cd).
    struct SegmentDiag {
        MFloat alpha, V, q, Cl, Cd;
    };
    const std::vector<SegmentDiag>& segments() const noexcept { return seg_diag_; }

    MANTA_PART_REQUIRES_FIELD(MANTA_HAS_FLUID_FIELD,
        "Naca00xx requires a FluidField on the world (lift/drag are "
        "ρ-scaled). Register one with World.add_field(FluidField(...)), "
        "or remove this part from the craft.");

    void update() override {
        auto* fp = this->template field_or_null<fields::FluidField>();
        if (!fp) return;
        auto& fluid = *fp;

        // Per-segment area: chord × (span / N).
        const Scalar S_seg = Scalar(chord_) * Scalar(span_ / MFloat(n_segments_));
        const Scalar c_quarter = Scalar(chord_ * MFloat(0.25));    // quarter-chord forward of center (+x is LE side)

        seg_diag_.clear();
        seg_diag_.reserve(static_cast<std::size_t>(n_segments_));

        // Cache the (scene→part) rotation once for v_rel into part frame
        // and (part→scene) for the quarter-chord sample location.
        const auto q_scene_from_part_raw =
            this->template orientation<SceneFrame>().raw();
        const auto q_part_from_scene_raw = q_scene_from_part_raw.conjugate();
        const auto p_part_origin_scene_raw =
            this->template position<SceneFrame>().raw();
        const auto v_part_origin_scene_raw =
            this->template velocity<SceneFrame>().raw();
        const auto w_part_scene_raw =
            this->template angular_velocity<SceneFrame>().raw();

        for (int i = 0; i < n_segments_; ++i) {
            // Segment center along span: evenly spaced from -b/2 + Δ/2 to +b/2 - Δ/2.
            const MFloat y_local = (MFloat(i) + MFloat(0.5)) *
                                       (span_ / MFloat(n_segments_)) -
                                   MFloat(0.5) * span_;
            // Sample location in part frame: quarter-chord, segment-center span.
            Eigen::Matrix<Scalar, 3, 1> sample_part(c_quarter,
                                                    Scalar(y_local),
                                                    Scalar(0));

            // Sample location in scene frame. MUST materialize: `auto` on
            // an Eigen expression that mixes a quaternion-product temporary
            // with a stored matrix produces a dangling-reference expression
            // template — when later evaluated, it reads stack memory that
            // has been reused for unrelated locals (the fluid state below).
            Eigen::Matrix<Scalar, 3, 1> sample_scene_raw =
                q_scene_from_part_raw * sample_part + p_part_origin_scene_raw;
            Eigen::Matrix<MFloat, 3, 1> sample_scene_mfloat =
                sample_scene_raw.template cast<MFloat>();
            fields::FluidState fs = fluid.state_at(
                geom::Vec3<SceneFrame>::from_raw(sample_scene_mfloat));

            // Fluid velocity at sample, in scene frame.
            Eigen::Matrix<Scalar, 3, 1> v_fluid_scene =
                fs.velocity.raw().template cast<Scalar>();

            // Velocity of THIS segment (sample point) in scene frame:
            // v_origin + ω × r_sample. The kinematic cache only gives
            // velocities at the part origin; the segment is offset by
            // sample_scene_raw - p_origin = (rotated sample_part).
            Eigen::Matrix<Scalar, 3, 1> r_seg_scene = sample_scene_raw -
                                                      p_part_origin_scene_raw;
            Eigen::Matrix<Scalar, 3, 1> v_self_scene =
                v_part_origin_scene_raw + w_part_scene_raw.cross(r_seg_scene);

            // Relative wind in part frame.
            Eigen::Matrix<Scalar, 3, 1> v_rel_part =
                q_part_from_scene_raw * (v_fluid_scene - v_self_scene);

            const Scalar vx = v_rel_part(0);
            const Scalar vz = v_rel_part(2);
            const Scalar V2d_sq = vx * vx + vz * vz;
            if (V2d_sq < Scalar(1e-12)) {
                seg_diag_.push_back({MFloat(0), MFloat(0), MFloat(0),
                                     MFloat(0), MFloat(0)});
                continue;
            }
            const Scalar V2d = std::sqrt(V2d_sq);

            // α = atan2(vz, -vx): positive when wind tilts upward.
            const Scalar alpha = std::atan2(vz, -vx);

            // 2D lift/drag coefficients.
            Scalar Cl, Cd;
            section_coefficients(alpha, Cl, Cd);

            const Scalar rho   = Scalar(fs.density);
            const Scalar q_dyn = Scalar(0.5) * rho * V2d_sq;

            // Drag along the wind, lift 90° CW from wind (both in xz plane).
            // D̂ = (vx, 0, vz) / V2d   ;   L̂ = (vz, 0, -vx) / V2d
            const Scalar f_drag = q_dyn * S_seg * Cd;
            const Scalar f_lift = q_dyn * S_seg * Cl;

            Eigen::Matrix<Scalar, 3, 1> F_part(
                (f_drag * vx + f_lift * vz) / V2d,
                Scalar(0),
                (f_drag * vz - f_lift * vx) / V2d);

            // Apply at the quarter-chord of this segment in part frame.
            this->apply_force_at(
                geom::Vec3<PartFrame, Scalar>::from_raw(F_part),
                geom::Vec3<PartFrame, Scalar>::from_raw(sample_part));

            seg_diag_.push_back(SegmentDiag{
                static_cast<MFloat>(geom::strip_to_real(alpha)),
                static_cast<MFloat>(geom::strip_to_real(V2d)),
                static_cast<MFloat>(geom::strip_to_real(q_dyn)),
                static_cast<MFloat>(geom::strip_to_real(Cl)),
                static_cast<MFloat>(geom::strip_to_real(Cd)),
            });
        }
    }

private:
    // 2D-section coefficients for a NACA 00xx airfoil. Pre-stall: linear
    // lift slope + parabolic drag polar. Post-stall: flat-plate Newtonian.
    void section_coefficients(Scalar alpha, Scalar& Cl, Scalar& Cd) const noexcept {
        using std::abs; using std::sin; using std::cos;
        const Scalar abs_a = abs(alpha);
        const Scalar a_stall = Scalar(alpha_stall_);
        if (abs_a <= a_stall) {
            Cl = Scalar(Cl_alpha_)  * alpha;
            Cd = Scalar(Cd0_)       + Scalar(Cd_alpha2_) * alpha * alpha;
        } else {
            const Scalar sa = sin(alpha);
            const Scalar ca = cos(alpha);
            Cl = Scalar(2) * sa * ca;
            Cd = Scalar(Cd0_) + Scalar(2) * sa * sa;
        }
    }

    MFloat                       chord_;
    MFloat                       span_;
    MFloat                       thickness_ratio_;
    int                          n_segments_;
    // Coefficient knobs (defaults match NACA 0012/0015 in incompressible flow).
    MFloat                       Cl_alpha_    = MFloat(5.7);     // rad⁻¹
    MFloat                       alpha_stall_ = MFloat(0.26);    // ≈ 15°
    MFloat                       Cd_alpha2_   = MFloat(0.05);
    MFloat                       Cd0_;
    std::vector<SegmentDiag>     seg_diag_;
};

using Naca00xx = Naca00xxT<MFloat>;

}  // namespace manta::parts
