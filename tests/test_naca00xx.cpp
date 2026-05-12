// Naca00xx — 2-D-section blade-element airfoil. Checks:
//   * at α = 0 with non-zero head-on wind, drag dominates and is small
//     (Cd0); lift ≈ 0; force is anti-parallel to the airfoil's velocity.
//   * at α > 0 (nose pitched up while flying forward), lift is positive
//     and roughly linear in α, drag grows with α².
//   * sample-point aggregation: same total wrench whether the wing is
//     sampled at N=1 or N=8 in a uniform flow (per-segment summation
//     equals the analytic integrated answer).
//   * post-stall transition: Cl drops past α_stall.

#include <doctest/doctest.h>

#include <cmath>

#include "../include/manta/core/craft.hpp"
#include "../include/manta/core/scene.hpp"
#include "../include/manta/core/world.hpp"
#include "../include/manta/fields/fluid_field.hpp"
#include "../include/manta/parts/aero/naca00xx.hpp"
#include "../include/manta/parts/structure/mass.hpp"

using namespace manta;
using namespace manta::fields;
using namespace manta::parts;

namespace {

// Run one tick of a single-craft world holding the given airfoil at the
// given pitch (about +y). Returns the wrench accumulated at the part
// root just before integration.
struct OneTickResult {
    geom::Vec3<PartFrame> force;
    geom::Vec3<PartFrame> torque;
    std::vector<Naca00xx::SegmentDiag> segments;
};

OneTickResult run_airfoil_one_tick(MFloat chord, MFloat span,
                                   MFloat thickness_ratio,
                                   int n_segments,
                                   MFloat alpha_rad,
                                   MFloat wind_speed = MFloat(20),
                                   MFloat rho = MFloat(1.225)) {
    WorldT<MFloat> world;
    world.clock().set_dt(MFloat(0.001));
    FluidField fluid;
    // Uniform incompressible "air" — ρ kg/m³ — flowing in scene −x at
    // `wind_speed`. The airfoil's leading edge is at part-frame +x
    // (the "leading edge of the chord points in the direction of
    // travel" convention), so wind from scene −x meets the leading
    // edge first at α = 0.
    fluid.add(FluidDisturbance::uniform_incompressible(
        rho, FluidDisturbance::Vec{-wind_speed, MFloat(0), MFloat(0)}),
        PERSISTENT);
    world.register_field(fluid);

    CraftT<MFloat> craft("test");
    craft.root().template add<MassT<MFloat>>("body", MFloat(1.0));
    auto& wing = craft.root().template add<Naca00xxT<MFloat>>(
        "wing", chord, span, thickness_ratio, n_segments);
    craft.root().compute_params();

    // Pitch the craft nose-up by α. In a z-up right-handed frame,
    // "nose up" is a rotation by −α about +y (right-hand rule about +y
    // would otherwise carry +x toward −z, i.e. nose down).
    Eigen::Quaternion<MFloat> q_y;
    q_y = Eigen::AngleAxis<MFloat>(-alpha_rad,
                                   Eigen::Matrix<MFloat, 3, 1>::UnitY());
    InitialState init;
    init.orientation = geom::Ori<SceneFrame>{q_y};
    auto& scene = world.create_scene();
    scene.add_craft(craft, init);

    // One kinematic_and_aggregate to populate per-part caches and run
    // update() (which accumulates wrenches without integrating). The
    // wing's accum is drained into the root by aggregate_wrenches, so
    // the test pulls the rolled-up total from the root.
    world.kinematic_and_aggregate();

    OneTickResult out;
    out.force    = craft.root().net_wrench().force();
    out.torque   = craft.root().net_wrench().torque();
    out.segments = wing.segments();
    return out;
}

}  // namespace

TEST_CASE("Naca00xx: α=0, head-on flow → pure drag along the wind") {
    const MFloat chord = 1.0;
    const MFloat span  = 4.0;
    const auto r = run_airfoil_one_tick(chord, span, /*t/c=*/0.15, /*N=*/4,
                                        /*alpha=*/0.0, /*V=*/20.0, /*ρ=*/1.225);
    // V=20 m/s, ρ=1.225, S = c·b = 4 m². Cd0 = 0.006 + 0.005·0.15 = 0.00675.
    // F_drag magnitude = 0.5·1.225·400·4·Cd0 ≈ 6.6 N along the wind in
    // scene −x. At α=0 the body frame == scene frame so the force in
    // part frame is also −x.
    const MFloat magnitude = MFloat(0.5) * MFloat(1.225) * MFloat(400) * MFloat(4)
                            * (MFloat(0.006) + MFloat(0.005) * MFloat(0.15));
    CHECK(r.force.x() == doctest::Approx(-magnitude).epsilon(1e-3));
    CHECK(std::abs(r.force.z()) < MFloat(0.01));   // no lift at α=0
}

TEST_CASE("Naca00xx: positive α → positive lift, linear in α (pre-stall)") {
    const MFloat chord = 1.0;
    const MFloat span  = 4.0;
    const MFloat S     = chord * span;
    const MFloat V     = 20.0;
    const MFloat rho   = 1.225;
    const MFloat q_dyn = MFloat(0.5) * rho * V * V;

    // α = 0.1 rad (~5.7°), well below stall. Cl = 5.7 · 0.1 = 0.57.
    // F_lift = q · S · Cl = 0.5·1.225·400·4·0.57 ≈ 559 N.
    const MFloat alpha = 0.1;
    const auto r = run_airfoil_one_tick(chord, span, 0.15, 4, alpha, V, rho);
    const MFloat Cl_expect = MFloat(5.7) * alpha;
    const MFloat lift_expect = q_dyn * S * Cl_expect;

    // Relative wind in part frame: v_rel_part = R(-α)·(-V x̂) =
    // (-V·cos α, 0, V·sin α). vx<0, vz>0 ⇒ α_calc = +α (matches input).
    //   D̂ = (vx, 0, vz)/V2d = (-cos α, 0, sin α)
    //   L̂ = (vz, 0, -vx)/V2d = (sin α, 0, cos α)
    //   F = q·S·(Cd · D̂ + Cl · L̂)
    const MFloat Cd_expect = MFloat(0.00675) + MFloat(0.05) * alpha * alpha;
    const MFloat f_d = q_dyn * S * Cd_expect;
    const MFloat f_l = q_dyn * S * Cl_expect;
    const MFloat ca = std::cos(alpha), sa = std::sin(alpha);
    const MFloat fx_expect = -f_d * ca + f_l * sa;
    const MFloat fz_expect =  f_d * sa + f_l * ca;

    CHECK(r.force.x() == doctest::Approx(fx_expect).epsilon(2e-3));
    CHECK(r.force.z() == doctest::Approx(fz_expect).epsilon(2e-3));
    CHECK(std::abs(r.force.z()) > MFloat(0.5) * lift_expect); // sanity: real lift
}

TEST_CASE("Naca00xx: per-segment summation invariant under sample count") {
    // Uniform flow, no rotation → every segment is identical, so total
    // wrench should be independent of N.
    const auto r1 = run_airfoil_one_tick(1.0, 4.0, 0.15, /*N=*/1, /*α=*/0.1);
    const auto r8 = run_airfoil_one_tick(1.0, 4.0, 0.15, /*N=*/8, /*α=*/0.1);
    CHECK(r1.force.x() == doctest::Approx(r8.force.x()).epsilon(1e-9));
    CHECK(r1.force.y() == doctest::Approx(r8.force.y()).epsilon(1e-9));
    CHECK(r1.force.z() == doctest::Approx(r8.force.z()).epsilon(1e-9));
    CHECK(r1.segments.size() == 1);
    CHECK(r8.segments.size() == 8);
}

TEST_CASE("Naca00xx: past α_stall, lift coefficient drops") {
    const MFloat alpha_pre  = 0.20;     // ~11°, below stall (15°)
    const MFloat alpha_post = 0.40;     // ~23°, deep stall
    const auto r_pre  = run_airfoil_one_tick(1.0, 4.0, 0.15, 1, alpha_pre);
    const auto r_post = run_airfoil_one_tick(1.0, 4.0, 0.15, 1, alpha_post);
    // Pre-stall Cl: 5.7 · 0.20 = 1.14.
    // Post-stall Cl: 2·sin(0.40)·cos(0.40) = 2·0.3894·0.9211 ≈ 0.717.
    CHECK(std::abs(r_post.segments[0].Cl) <
          std::abs(r_pre.segments[0].Cl));
}
