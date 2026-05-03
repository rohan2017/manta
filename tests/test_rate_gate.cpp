// Verifies the sim::RateGate helper and its end-to-end usage on the
// rate-capped sensor parts (IMU / DVL). Three things to lock in:
//   * Tick 0 fires unconditionally — consumers see freshness on the
//     very first sim step (so the EKF gets a measurement immediately).
//   * Subsequent fires happen at 1/rate_hz of accumulated sim time.
//   * `consume_fresh()` is one-shot — clears the bit on read so the EKF
//     doesn't double-count a single measurement.

#include <doctest/doctest.h>

#include "../include/manta/core/craft.hpp"
#include "../include/manta/core/scene.hpp"
#include "../include/manta/core/world.hpp"
#include "../include/manta/parts/sensor/dvl.hpp"
#include "../include/manta/parts/sensor/imu.hpp"
#include "../include/manta/sim/rate_gate.hpp"

using namespace manta;

TEST_CASE("RateGate: unrated (rate_hz=0) fires every tick") {
    sim::RateGate g;
    for (int i = 0; i < 5; ++i) CHECK(g.tick(0.001f));
}

TEST_CASE("RateGate: tick 0 always fires regardless of rate") {
    sim::RateGate g{Real(50)};   // 20 ms period
    CHECK(g.tick(0.001f) == true);    // tick 0
    CHECK(g.tick(0.001f) == false);   // tick 1
}

TEST_CASE("RateGate: 100 Hz on a 1 kHz sim fires every 10 ticks") {
    sim::RateGate g{Real(100)};   // 10 ms period
    int fires = 0;
    for (int i = 0; i < 50; ++i) {
        if (g.tick(Real(0.001f))) ++fires;
    }
    // 50 ticks at 1 kHz = 50 ms. Expected: tick 0 + tick 10 + 20 + 30 + 40 = 5.
    CHECK(fires == 5);
}

TEST_CASE("RateGate: variable dt still hits target rate") {
    sim::RateGate g{Real(20)};   // 50 ms period
    int fires = 0;
    // 200 ms total in 4 chunks of varying sizes. Should fire at t=0,
    // 50, 100, 150 — i.e. 4 times.
    Real dts[] = {Real(0.030f), Real(0.025f), Real(0.060f), Real(0.030f),
                  Real(0.020f), Real(0.020f), Real(0.015f)};
    for (Real dt : dts) {
        if (g.tick(dt)) ++fires;
    }
    CHECK(fires == 4);
}

TEST_CASE("IMU: default (no rate cap) is fresh after every update") {
    World w;
    w.clock().set_dt(0.001f);
    auto& s = w.create_scene();
    Craft c("imu_test");
    auto& imu = c.root().add<parts::IMU>("imu");
    c.root().compute_params();
    s.add_craft(c);

    CHECK(imu.fresh() == false);
    w.update();
    CHECK(imu.fresh() == true);
    CHECK(imu.consume_fresh() == true);
    CHECK(imu.fresh() == false);
    w.update();
    CHECK(imu.fresh() == true);
}

TEST_CASE("IMU: rate_hz=100 on 1kHz sim fires every 10 ticks") {
    World w;
    w.clock().set_dt(0.001f);
    auto& s = w.create_scene();
    Craft c("imu_rated");
    auto& imu = c.root().add<parts::IMU>("imu",
        parts::ImuNoiseParams{}, /*rate_hz=*/Real(100));
    c.root().compute_params();
    s.add_craft(c);

    int fires = 0;
    for (int i = 0; i < 50; ++i) {
        w.update();
        if (imu.consume_fresh()) ++fires;
    }
    CHECK(fires == 5);   // tick 0, 10, 20, 30, 40
}

TEST_CASE("DVL: rate_hz=50 on 1kHz sim fires every 20 ticks") {
    World w;
    w.clock().set_dt(0.001f);
    auto& s = w.create_scene();
    Craft c("dvl_rated");
    auto& dvl = c.root().add<parts::DVL>("dvl",
        parts::DvlNoiseParams{}, /*rate_hz=*/Real(50));
    c.root().compute_params();
    s.add_craft(c);

    int fires = 0;
    for (int i = 0; i < 100; ++i) {
        w.update();
        if (dvl.consume_fresh()) ++fires;
    }
    CHECK(fires == 5);   // tick 0, 20, 40, 60, 80
}

TEST_CASE("set_measurement bypasses the gate (always fresh)") {
    World w;
    w.clock().set_dt(0.001f);
    auto& s = w.create_scene();
    Craft c("imu_external");
    auto& imu = c.root().add<parts::IMU>("imu",
        parts::ImuNoiseParams{}, /*rate_hz=*/Real(10));   // very slow native rate
    c.root().compute_params();
    s.add_craft(c);

    w.update();              // first tick fires
    (void)imu.consume_fresh();
    w.update();              // gate would block this one
    CHECK(imu.fresh() == false);

    // External feed: should mark fresh regardless of rate cap.
    imu.set_measurement(geom::Vec3<PartFrame>{Real(0.1f), Real(0), Real(0)},
                        geom::Vec3<PartFrame>{Real(0), Real(0), Real(0.5f)});
    CHECK(imu.fresh() == true);
    CHECK(imu.consume_fresh() == true);
}
