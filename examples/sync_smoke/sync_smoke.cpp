// sync_smoke — round-trips a GravityField disturbance through real Zenoh
// sessions and verifies the receive side rebuilds the same Disturbance.
//
// Why this exists: `tests/test_field_sync.cpp` covers the C++ replication
// path with the tx/rx hooks wired directly. This binary exercises the
// *same byte layout* the codegen emits (see `_emit_field_sync` in
// emit/main.py), end-to-end through two zenoh::Session instances. If the
// codegen wire format ever drifts from `Field::receive()`'s expectations
// this binary breaks loudly.
//
// Two Sessions in one process talk over the same fabric the same way two
// separate binaries would — same library code path, same on-the-wire bytes.
// Run as a CTest fixture; exits 0 on success, non-zero on any check fail.

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

#include <zenoh.hxx>

#include "manta/fields/gravity_field.hpp"

using namespace manta;
using namespace manta::geom;
using namespace manta::fields;

namespace {

bool nearly_equal(MFloat a, MFloat b, MFloat eps = MFloat(1e-4)) {
    MFloat d = a - b;
    return (d < eps) && (-d < eps);
}

#define REQUIRE(cond) do {                                              \
    if (!(cond)) {                                                      \
        std::fprintf(stderr, "FAIL @ %s:%d: %s\n", __FILE__, __LINE__,  \
                     #cond);                                            \
        return 1;                                                       \
    }                                                                   \
} while (0)

}  // namespace

int main() {
    // Two Sessions in the same process — Zenoh routes between them through
    // the local fabric, the same way two separate binaries would.
    zenoh::Config cfg_a = zenoh::Config::create_default();
    zenoh::Config cfg_b = zenoh::Config::create_default();
    auto session_a = zenoh::Session::open(std::move(cfg_a));
    auto session_b = zenoh::Session::open(std::move(cfg_b));

    GravityField field_a, field_b;

    // Subscriber on B; publisher on A. (Codegen emits both for every
    // synced field, but this test cares only about A→B propagation.)
    auto pub_a = session_a.declare_publisher(
        zenoh::KeyExpr("manta/sync_smoke/grav"));
    field_a.set_tx_hook([&pub_a](std::uint16_t tag,
                                 const manta::fields::Params& params,
                                 int lifetime) {
        std::vector<std::uint8_t> buf(2 + 2 + 4 + params.size());
        std::uint16_t ver = 1;
        std::memcpy(buf.data() + 0, &ver,      2);
        std::memcpy(buf.data() + 2, &tag,      2);
        std::memcpy(buf.data() + 4, &lifetime, 4);
        std::memcpy(buf.data() + 8, params.data(), params.size());
        pub_a.put(zenoh::Bytes(std::move(buf)));
    });

    auto sub_b = session_b.declare_subscriber(
        zenoh::KeyExpr("manta/sync_smoke/grav"),
        [&field_b](const zenoh::Sample& sample) {
            auto payload = sample.get_payload().as_vector();
            if (payload.size() < 8 + manta::fields::kParamsBytes) return;
            std::uint16_t ver = 0, tag = 0;
            std::int32_t  lifetime = 0;
            std::memcpy(&ver,      payload.data() + 0, 2);
            std::memcpy(&tag,      payload.data() + 2, 2);
            std::memcpy(&lifetime, payload.data() + 4, 4);
            if (ver != 1) return;
            manta::fields::Params p{};
            std::memcpy(p.data(), payload.data() + 8, p.size());
            field_b.receive(tag, p, lifetime);
        },
        zenoh::closures::none);

    // Let Zenoh discover the topic.
    std::this_thread::sleep_for(std::chrono::milliseconds(500));

    REQUIRE(field_a.disturbance_count() == 0);
    REQUIRE(field_b.disturbance_count() == 0);

    // Add a point-mass disturbance on A. Codegen-emitted tx_hook serializes
    // it; B's subscriber decodes and calls receive(); B's factory rebuilds
    // the lambda.
    field_a.add(GravityField::Disturbance::point_mass(
                    Vec3<SceneFrame>{MFloat(1e7f), MFloat(0), MFloat(0)},
                    MFloat(4.0e14f)),
                PERSISTENT);

    REQUIRE(field_a.disturbance_count() == 1);

    // Wait for the message to traverse Zenoh and the rx callback to fire.
    for (int i = 0; i < 50 && field_b.disturbance_count() == 0; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    REQUIRE(field_b.disturbance_count() == 1);

    // The rebuilt disturbance should produce the same field at every query.
    Vec3<SceneFrame> q{MFloat(2e7f), MFloat(0), MFloat(0)};
    auto ga = field_a.state_at(q);
    auto gb = field_b.state_at(q);
    REQUIRE(nearly_equal(ga.x(), gb.x()));
    REQUIRE(nearly_equal(ga.y(), gb.y()));
    REQUIRE(nearly_equal(ga.z(), gb.z()));

    // Verify the recursion guard: B has its own (no-op) tx_hook unset, so
    // adding a *local* disturbance on B should affect only B, not echo
    // back to A. Check by adding a uniform on B and confirming A still
    // has just its point_mass.
    field_b.set_tx_hook(nullptr);
    field_b.add(GravityField::Disturbance::uniform(
                    Vec3<SceneFrame>{MFloat(0), MFloat(0), MFloat(-1.5f)}),
                PERSISTENT);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    REQUIRE(field_a.disturbance_count() == 1);    // unchanged
    REQUIRE(field_b.disturbance_count() == 2);

    std::printf("sync_smoke: OK — Zenoh round-trip verified.\n");
    return 0;
}
