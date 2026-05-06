// Cross-instance Disturbance replication via the tag/params/factory wire
// format. Wires two field instances in-process via tx/rx hooks (simulating
// two processes communicating through Zenoh). Verifies:
//   * Stock factories self-tag.
//   * The tx hook fires with the right (tag, params, lifetime).
//   * receive() rebuilds an equivalent Disturbance via the registry.
//   * state_at(pos) on the receiver matches state_at(pos) on the sender.
//   * The recursion guard prevents echo-feedback when receive() calls add().
//   * USER-tagged (custom-lambda) disturbances are NOT replicated.
//   * Unknown tags drop silently.
//   * User-defined factories registered at >= USER_BASE replicate end-to-end.

#include <cstring>
#include <doctest/doctest.h>

#include "../include/manta/fields/fluid_field.hpp"
#include "../include/manta/fields/gravity_field.hpp"
#include "../include/manta/fields/mag_field.hpp"

using namespace manta;
using namespace manta::geom;
using namespace manta::fields;

// Helper: in-process wire — sender's tx_hook directly calls receiver.receive.
template <class FieldT>
static void wire_one_way(FieldT& tx, FieldT& rx) {
    tx.set_tx_hook([&rx](std::uint16_t tag,
                         const manta::fields::Params& params,
                         int lifetime) {
        rx.receive(tag, params, lifetime);
    });
}

TEST_CASE("Sync GravityField: point_mass replicates and queries match") {
    GravityField a, b;
    wire_one_way(a, b);

    a.add(GravityField::Disturbance::point_mass(
              Vec3<SceneFrame>{Real(1e7f), Real(0), Real(0)}, Real(4.0e14f)),
          PERSISTENT);

    CHECK(a.disturbance_count() == 1u);
    CHECK(b.disturbance_count() == 1u);

    Vec3<SceneFrame> q{Real(2e7f), Real(0), Real(0)};
    auto ga = a.state_at(q);
    auto gb = b.state_at(q);
    CHECK(ga.x() == doctest::Approx(gb.x()).epsilon(1e-5));
    CHECK(ga.y() == doctest::Approx(gb.y()).epsilon(1e-5));
    CHECK(ga.z() == doctest::Approx(gb.z()).epsilon(1e-5));
}

TEST_CASE("Sync GravityField: uniform + point_mass_j2 replicate and superpose") {
    GravityField a, b;
    wire_one_way(a, b);

    a.add(GravityField::Disturbance::uniform(
              Vec3<SceneFrame>{Real(0), Real(0), Real(-1)}),
          PERSISTENT);
    a.add(GravityField::Disturbance::point_mass_j2(
              Vec3<SceneFrame>::zero(),
              Real(3.986e14f), Real(1.0826e-3f), Real(6.378e6f)),
          PERSISTENT);

    CHECK(b.disturbance_count() == 2u);

    Vec3<SceneFrame> q{Real(6.378e6f), Real(0), Real(0)};
    auto ga = a.state_at(q);
    auto gb = b.state_at(q);
    CHECK(ga.x() == doctest::Approx(gb.x()).epsilon(1e-3));
    CHECK(ga.z() == doctest::Approx(gb.z()).epsilon(1e-3));   // includes the -1 uniform offset
}

TEST_CASE("Sync GravityField: USER-tagged (custom lambda) is NOT replicated") {
    GravityField a, b;
    wire_one_way(a, b);

    GravityField::Disturbance custom;
    // tag stays at USER (0) because we didn't go through a stock factory.
    custom.delta_g = [](const Vec3<SceneFrame>&) noexcept {
        return Vec3<SceneFrame>{Real(0), Real(0), Real(-3.7f)};
    };
    a.add(custom, PERSISTENT);

    CHECK(a.disturbance_count() == 1u);
    CHECK(b.disturbance_count() == 0u);   // not replicated
}

TEST_CASE("Sync GravityField: recursion guard prevents echo on receive") {
    // Bidirectional wiring — without the guard, a stock-tagged disturbance
    // arriving on rx would re-fire the local tx, looping forever. Guard
    // ensures the receiving side's add() does NOT re-emit.
    GravityField a, b;
    int a_tx_count = 0, b_tx_count = 0;
    a.set_tx_hook([&](std::uint16_t tag, const manta::fields::Params& p, int lt) {
        ++a_tx_count;
        b.receive(tag, p, lt);
    });
    b.set_tx_hook([&](std::uint16_t tag, const manta::fields::Params& p, int lt) {
        ++b_tx_count;
        a.receive(tag, p, lt);
    });

    a.add(GravityField::Disturbance::point_mass(
              Vec3<SceneFrame>::zero(), Real(1e6f)),
          PERSISTENT);

    CHECK(a_tx_count == 1);   // fired once by user-side add
    CHECK(b_tx_count == 0);   // receive() suppressed re-broadcast
    CHECK(a.disturbance_count() == 1u);
    CHECK(b.disturbance_count() == 1u);
}

TEST_CASE("Sync MagField: dipole replicates") {
    MagField a, b;
    wire_one_way(a, b);

    a.add(MagField::Disturbance::dipole(
              Vec3<SceneFrame>::zero(),
              Vec3<SceneFrame>{Real(0), Real(0), Real(-7.94e22f)}),
          PERSISTENT);

    Vec3<SceneFrame> q{Real(0), Real(0), Real(1.0e7f)};
    auto ba = a.state_at(q);
    auto bb = b.state_at(q);
    CHECK(ba.z() == doctest::Approx(bb.z()).epsilon(1e-5));
}

TEST_CASE("Sync FluidField: uniform_incompressible + uniform_gas replicate") {
    FluidField a, b;
    wire_one_way(a, b);

    a.add(FluidField::Disturbance::uniform_incompressible(
              Real(1000.0f), Vec3<SceneFrame>{Real(0.5f), Real(0), Real(0)}),
          PERSISTENT);
    a.add(FluidField::Disturbance::uniform_gas(
              Real(287.0f), Real(288.15f), Real(101325.0f)),
          PERSISTENT);

    CHECK(b.disturbance_count() == 2u);
    auto sb = b.state_at(Vec3<SceneFrame>::zero());
    // Gas pool dominates when both are unbounded (no in_influence).
    CHECK(sb.R == doctest::Approx(287.0));
    CHECK(sb.density == doctest::Approx(1.225).epsilon(1e-2));
}

TEST_CASE("Sync GravityField: unknown tag drops silently on receive") {
    GravityField b;
    manta::fields::Params p{};
    bool ok = b.receive(/*tag=*/9999, p, /*lifetime=*/PERSISTENT);
    CHECK(ok == false);
    CHECK(b.disturbance_count() == 0u);
}

TEST_CASE("Sync GravityField: lifetime is preserved across the wire") {
    GravityField a, b;
    wire_one_way(a, b);

    // Finite-lifetime disturbance: 3 ticks.
    a.add(GravityField::Disturbance::uniform(
              Vec3<SceneFrame>{Real(0), Real(0), Real(-1)}),
          /*lifetime=*/3);

    CHECK(b.disturbance_count() == 1u);
    b.update(); CHECK(b.disturbance_count() == 1u);   // 3 → 2
    b.update(); CHECK(b.disturbance_count() == 1u);   // 2 → 1
    b.update(); CHECK(b.disturbance_count() == 0u);   // 1 → 0, dropped
}

// User-defined factory: a custom point-mass-with-cutoff that's only
// active within a finite radius. Demonstrates the >= USER_BASE registration
// path.
namespace {

struct CutoffParams { Real ox, oy, oz, mu, cutoff; };
constexpr std::uint16_t kCutoffTag = manta::fields::USER_BASE + 7;

GravityField::Disturbance make_cutoff(Vec3<SceneFrame> origin,
                                      Real mu, Real cutoff) {
    GravityField::Disturbance d;
    d.origin = origin;
    d.tag    = kCutoffTag;
    CutoffParams cp{Real(origin.x()), Real(origin.y()), Real(origin.z()), mu, cutoff};
    static_assert(sizeof(cp) <= manta::fields::kParamsBytes);
    std::memcpy(d.params.data(), &cp, sizeof(cp));
    d.delta_g = [mu](const Vec3<SceneFrame>& off) noexcept {
        Real r2 = off.raw().squaredNorm();
        if (r2 < Real(1e-12f)) return Vec3<SceneFrame>::zero();
        Real r = std::sqrt(r2);
        return Vec3<SceneFrame>::from_raw(off.raw() * (-mu / (r2 * r)));
    };
    Real cut2 = cutoff * cutoff;
    d.in_influence = [cut2](const Vec3<SceneFrame>& off) noexcept {
        return off.raw().squaredNorm() < cut2;
    };
    return d;
}

bool register_cutoff_factory_once = []() {
    GravityField::register_factory(kCutoffTag, [](const manta::fields::Params& p) {
        CutoffParams cp;
        std::memcpy(&cp, p.data(), sizeof(cp));
        return make_cutoff(Vec3<SceneFrame>{cp.ox, cp.oy, cp.oz}, cp.mu, cp.cutoff);
    });
    return true;
}();

}   // namespace

TEST_CASE("Sync GravityField: user-registered factory replicates end-to-end") {
    GravityField a, b;
    wire_one_way(a, b);

    a.add(make_cutoff(Vec3<SceneFrame>::zero(), Real(1e6f), Real(100.0f)),
          PERSISTENT);

    // Inside the cutoff: nonzero force on both sides.
    Vec3<SceneFrame> near{Real(50.0f), Real(0), Real(0)};
    auto ga = a.state_at(near);
    auto gb = b.state_at(near);
    CHECK(ga.x() == doctest::Approx(gb.x()).epsilon(1e-3));
    CHECK(ga.x() < Real(0));   // attractive

    // Outside the cutoff: in_influence is false → zero on both sides.
    Vec3<SceneFrame> far{Real(500.0f), Real(0), Real(0)};
    CHECK(a.state_at(far).raw().norm() == doctest::Approx(0.0).epsilon(1e-9));
    CHECK(b.state_at(far).raw().norm() == doctest::Approx(0.0).epsilon(1e-9));
}
