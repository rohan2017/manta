#pragma once

#include <array>
#include <cstdint>
#include <cstring>
#include <functional>
#include <unordered_map>

#include "field.hpp"
#include "../core/frame.hpp"
#include "../core/types.hpp"
#include "../geom/vec3.hpp"

namespace manta::fields {

namespace fluid_tags {
    constexpr std::uint16_t UNIFORM_INCOMPRESS = 1;
    constexpr std::uint16_t UNIFORM_GAS        = 2;
}

// Snapshot of fluid state at a query point.
//
//   R is the specific gas constant in J/(kg·K). The sentinel value
//   `kIncompressibleR` (= -1) marks the fluid as weakly compressible
//   (water-like): density is a free variable independent of pressure
//   and temperature, so the four state channels add independently
//   with no correction.
//
//   For R != kIncompressibleR (gas), the state honors p = ρRT.
//   Disturbances contribute delta_temperature and delta_pressure;
//   density is *derived* from the summed p and T as ρ = p/(RT). Any
//   contribution from delta_density is ignored on gases. Velocity
//   always sums.
struct FluidState {
    static constexpr MFloat kIncompressibleR = MFloat(-1);

    MFloat                   R           = kIncompressibleR;
    MFloat                   temperature = MFloat(0);
    MFloat                   density     = MFloat(0);
    MFloat                   pressure    = MFloat(0);
    geom::Vec3<SceneFrame> velocity    = geom::Vec3<SceneFrame>::zero();

    bool is_incompressible() const noexcept { return R == kIncompressibleR; }
};

struct FluidDisturbance {
    using Vec       = geom::Vec3<SceneFrame>;
    using ScalarFn  = std::function<MFloat(const Vec& offset)>;
    using VecFn     = std::function<Vec(const Vec& offset)>;
    using Influence = std::function<bool(const Vec& offset)>;

    Vec           origin           = Vec::zero();
    MFloat          R                = FluidState::kIncompressibleR;
    ScalarFn      delta_temperature;
    ScalarFn      delta_density;
    ScalarFn      delta_pressure;
    VecFn         delta_velocity;
    Influence     in_influence;
    std::uint16_t tag              = USER_TAG;
    Params        params{};

    struct UniformIncompressParams { MFloat density, vx, vy, vz; };
    struct UniformGasParams        { MFloat R, temperature, pressure, vx, vy, vz; };

    // Uniform incompressible fluid.
    static FluidDisturbance uniform_incompressible(MFloat density,
                                                   Vec  velocity = Vec::zero()) {
        FluidDisturbance d;
        d.R   = FluidState::kIncompressibleR;
        d.tag = fluid_tags::UNIFORM_INCOMPRESS;
        UniformIncompressParams up{
            density, MFloat(velocity.x()), MFloat(velocity.y()), MFloat(velocity.z())};
        std::memcpy(d.params.data(), &up, sizeof(up));
        d.delta_density  = [density] (const Vec&) noexcept { return density; };
        d.delta_velocity = [velocity](const Vec&) noexcept { return velocity; };
        return d;
    }

    // Uniform gas (R, T, p, velocity all constant).
    static FluidDisturbance uniform_gas(MFloat R_,
                                        MFloat temperature,
                                        MFloat pressure,
                                        Vec  velocity = Vec::zero()) {
        FluidDisturbance d;
        d.R   = R_;
        d.tag = fluid_tags::UNIFORM_GAS;
        UniformGasParams up{R_, temperature, pressure,
            MFloat(velocity.x()), MFloat(velocity.y()), MFloat(velocity.z())};
        std::memcpy(d.params.data(), &up, sizeof(up));
        d.delta_temperature = [temperature](const Vec&) noexcept { return temperature; };
        d.delta_pressure    = [pressure]   (const Vec&) noexcept { return pressure;    };
        d.delta_velocity    = [velocity]   (const Vec&) noexcept { return velocity;    };
        return d;
    }
};

class FluidField : public DisturbanceField<FluidField, FluidDisturbance> {
public:
    using Vec         = FluidDisturbance::Vec;
    using Disturbance = FluidDisturbance;

    static_assert(sizeof(Disturbance::UniformIncompressParams) <= kParamsBytes);
    static_assert(sizeof(Disturbance::UniformGasParams)        <= kParamsBytes);

    FluidState state_at(const Vec& pos) const noexcept {
        FluidState gas, liq;
        bool gas_active = false, liq_active = false;
        MFloat gas_R = MFloat(0);

        for (const auto& e : entries_) {
            Vec off = Vec::from_raw(pos.raw() - e.d.origin.raw());
            if (e.d.in_influence && !e.d.in_influence(off)) continue;

            if (e.d.R == FluidState::kIncompressibleR) {
                liq_active = true;
                if (e.d.delta_density)
                    liq.density += e.d.delta_density(off);
                if (e.d.delta_velocity)
                    liq.velocity = geom::Vec3<SceneFrame>::from_raw(
                        liq.velocity.raw() + e.d.delta_velocity(off).raw());
            } else {
                if (!gas_active) gas_R = e.d.R;
                gas_active = true;
                if (e.d.delta_temperature)
                    gas.temperature += e.d.delta_temperature(off);
                if (e.d.delta_pressure)
                    gas.pressure += e.d.delta_pressure(off);
                if (e.d.delta_velocity)
                    gas.velocity = geom::Vec3<SceneFrame>::from_raw(
                        gas.velocity.raw() + e.d.delta_velocity(off).raw());
            }
        }

        if (gas_active) {
            gas.R = gas_R;
            gas.density = (gas.temperature > MFloat(0))
                        ? gas.pressure / (gas_R * gas.temperature)
                        : MFloat(0);
            return gas;
        }
        if (liq_active) { liq.R = FluidState::kIncompressibleR; return liq; }
        return FluidState{};
    }

    static std::unordered_map<std::uint16_t, DisturbanceFactory> stock_factories() {
        std::unordered_map<std::uint16_t, DisturbanceFactory> m;
        m[fluid_tags::UNIFORM_INCOMPRESS] = [](const Params& p) {
            Disturbance::UniformIncompressParams up;
            std::memcpy(&up, p.data(), sizeof(up));
            return Disturbance::uniform_incompressible(up.density,
                Vec{up.vx, up.vy, up.vz});
        };
        m[fluid_tags::UNIFORM_GAS] = [](const Params& p) {
            Disturbance::UniformGasParams up;
            std::memcpy(&up, p.data(), sizeof(up));
            return Disturbance::uniform_gas(up.R, up.temperature, up.pressure,
                Vec{up.vx, up.vy, up.vz});
        };
        return m;
    }
};

} // namespace manta::fields
