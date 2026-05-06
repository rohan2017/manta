// wire_debug — pretty-prints field-sync disturbance messages off the wire.
//
// Field sync emits binary messages on `manta/<world>/field_<i>/disturbance`:
//
//     uint16  schema_version (=1)
//     uint16  tag
//     int32   lifetime
//     uint8[K] params           (K = field's kParamsBytes; currently 96 for all)
//
// This subscriber decodes each incoming sample, looks up the (field-type,
// tag) pair in a small dispatch table, and prints the corresponding POD
// param struct as named fields. Useful when debugging cross-process sync
// drift — `zenoh-cli sub --raw` shows you bytes; this binary shows you
// what the bytes mean.
//
// Usage:
//     wire_debug <field_type> <topic>
//   field_type: gravity | mag | fluid
//   topic:      e.g. "manta/ex1/field_0/disturbance"  or a wildcard
//                    expression like "manta/**/disturbance".

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <string_view>
#include <thread>

#include <zenoh.hxx>

#include "manta/fields/fluid_field.hpp"
#include "manta/fields/gravity_field.hpp"
#include "manta/fields/mag_field.hpp"

using namespace manta;
using namespace manta::fields;

namespace {

std::atomic<bool> g_run{true};
void on_signal(int) { g_run.store(false); }

void print_lifetime(int lifetime) {
    if (lifetime < 0) std::printf("PERSISTENT");
    else              std::printf("%d ticks", lifetime);
}

void print_gravity(std::uint16_t tag, const std::uint8_t* p) {
    using D = GravityField::Disturbance;
    switch (tag) {
        case gravity_tags::UNIFORM: {
            D::UniformParams up;  std::memcpy(&up, p, sizeof(up));
            std::printf("UNIFORM g=(%.4g, %.4g, %.4g)", up.gx, up.gy, up.gz);
            break;
        }
        case gravity_tags::POINT_MASS: {
            D::PointMassParams pm;  std::memcpy(&pm, p, sizeof(pm));
            std::printf("POINT_MASS origin=(%.4g, %.4g, %.4g) mu=%.6g",
                        pm.ox, pm.oy, pm.oz, pm.mu);
            break;
        }
        case gravity_tags::POINT_MASS_J2: {
            D::PointMassJ2Params pp;  std::memcpy(&pp, p, sizeof(pp));
            std::printf("POINT_MASS_J2 origin=(%.4g, %.4g, %.4g) mu=%.6g "
                        "j2=%.4g R_eq=%.4g axis=(%.4g, %.4g, %.4g)",
                        pp.ox, pp.oy, pp.oz, pp.mu, pp.j2, pp.eq_radius,
                        pp.ax, pp.ay, pp.az);
            break;
        }
        case manta::fields::USER_TAG:
            std::printf("USER (untagged, should not appear on wire)");
            break;
        default:
            std::printf("tag=%u (unknown — user-registered? expect >= %u)",
                        unsigned(tag), unsigned(manta::fields::USER_BASE));
            break;
    }
}

void print_mag(std::uint16_t tag, const std::uint8_t* p) {
    using D = MagField::Disturbance;
    switch (tag) {
        case mag_tags::UNIFORM: {
            D::UniformParams up;  std::memcpy(&up, p, sizeof(up));
            std::printf("UNIFORM B=(%.4g, %.4g, %.4g)", up.bx, up.by, up.bz);
            break;
        }
        case mag_tags::DIPOLE: {
            D::DipoleParams dp;  std::memcpy(&dp, p, sizeof(dp));
            std::printf("DIPOLE origin=(%.4g, %.4g, %.4g) moment=(%.4g, %.4g, %.4g)",
                        dp.ox, dp.oy, dp.oz, dp.mx, dp.my, dp.mz);
            break;
        }
        case manta::fields::USER_TAG:
            std::printf("USER (untagged, should not appear on wire)");
            break;
        default:
            std::printf("tag=%u (unknown — user-registered? expect >= %u)",
                        unsigned(tag), unsigned(manta::fields::USER_BASE));
            break;
    }
}

void print_fluid(std::uint16_t tag, const std::uint8_t* p) {
    using D = FluidField::Disturbance;
    switch (tag) {
        case fluid_tags::UNIFORM_INCOMPRESS: {
            D::UniformIncompressParams up;  std::memcpy(&up, p, sizeof(up));
            std::printf("UNIFORM_INCOMPRESS rho=%.4g v=(%.4g, %.4g, %.4g)",
                        up.density, up.vx, up.vy, up.vz);
            break;
        }
        case fluid_tags::UNIFORM_GAS: {
            D::UniformGasParams up;  std::memcpy(&up, p, sizeof(up));
            std::printf("UNIFORM_GAS R=%.4g T=%.4g p=%.4g v=(%.4g, %.4g, %.4g)",
                        up.R, up.temperature, up.pressure, up.vx, up.vy, up.vz);
            break;
        }
        case manta::fields::USER_TAG:
            std::printf("USER (untagged, should not appear on wire)");
            break;
        default:
            std::printf("tag=%u (unknown — user-registered? expect >= %u)",
                        unsigned(tag), unsigned(manta::fields::USER_BASE));
            break;
    }
}

void print_timestamp() {
    using namespace std::chrono;
    auto now = system_clock::now();
    auto t   = system_clock::to_time_t(now);
    auto ms  = duration_cast<milliseconds>(now.time_since_epoch()) % 1000;
    char buf[32];
    std::strftime(buf, sizeof(buf), "%H:%M:%S", std::localtime(&t));
    std::printf("[%s.%03d] ", buf, int(ms.count()));
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr,
            "usage: %s <field_type> <topic>\n"
            "  field_type: gravity | mag | fluid\n"
            "  topic:      e.g. manta/ex1/field_0/disturbance\n",
            argv[0]);
        return 2;
    }
    std::string_view field_type = argv[1];
    std::string      topic      = argv[2];

    void (*decode)(std::uint16_t, const std::uint8_t*) = nullptr;
    std::size_t expected_size = 0;
    if (field_type == "gravity") {
        decode = print_gravity;
        expected_size = 8 + manta::fields::kParamsBytes;
    } else if (field_type == "mag") {
        decode = print_mag;
        expected_size = 8 + manta::fields::kParamsBytes;
    } else if (field_type == "fluid") {
        decode = print_fluid;
        expected_size = 8 + manta::fields::kParamsBytes;
    } else {
        std::fprintf(stderr, "wire_debug: unknown field_type %.*s\n",
                     int(field_type.size()), field_type.data());
        return 2;
    }

    std::signal(SIGINT,  on_signal);
    std::signal(SIGTERM, on_signal);

    zenoh::Config cfg = zenoh::Config::create_default();
    auto session = zenoh::Session::open(std::move(cfg));

    auto sub = session.declare_subscriber(
        zenoh::KeyExpr(topic.c_str()),
        [&](const zenoh::Sample& s) {
            auto payload = s.get_payload().as_vector();
            if (payload.size() < expected_size) {
                print_timestamp();
                std::printf("%s  short payload (%zu < %zu)\n",
                            std::string(s.get_keyexpr().as_string_view()).c_str(),
                            payload.size(), expected_size);
                return;
            }
            std::uint16_t ver = 0, tag = 0;
            std::int32_t  lifetime = 0;
            std::memcpy(&ver,      payload.data() + 0, 2);
            std::memcpy(&tag,      payload.data() + 2, 2);
            std::memcpy(&lifetime, payload.data() + 4, 4);

            print_timestamp();
            std::printf("%s  ", std::string(s.get_keyexpr().as_string_view()).c_str());
            if (ver != 1) {
                std::printf("schema_version=%u (expected 1) — refusing to decode\n",
                            unsigned(ver));
                return;
            }
            std::printf("lifetime=");
            print_lifetime(lifetime);
            std::printf("  ");
            decode(tag, payload.data() + 8);
            std::printf("\n");
            std::fflush(stdout);
        },
        zenoh::closures::none);

    std::printf("wire_debug: listening on '%s' (%s field), Ctrl-C to exit.\n",
                topic.c_str(), std::string(field_type).c_str());
    while (g_run.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    return 0;
}
