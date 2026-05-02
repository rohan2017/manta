#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <list>
#include <unordered_map>

#include "field.hpp"
#include "../core/frame.hpp"
#include "../core/types.hpp"
#include "../geom/vec3.hpp"

namespace manta::fields {

namespace fluid_tags {
    constexpr std::uint16_t USER                 = 0;
    constexpr std::uint16_t UNIFORM_INCOMPRESS   = 1;
    constexpr std::uint16_t UNIFORM_GAS          = 2;
    constexpr std::uint16_t USER_BASE            = 1024;
}

// Snapshot of fluid state at a query point.
//
//   R is the specific gas constant in J/(kg·K). Sentinel value -1 marks the
//   fluid as weakly compressible (water-like): density is treated as a free
//   variable independent of pressure and temperature, so the four state
//   channels are added independently with no correction.
//
//   For R != -1 (gas), the state honors p = ρRT. Disturbances contribute
//   delta_temperature and delta_pressure; density is *derived* from the
//   summed p and T as ρ = p/(RT). Any contribution from delta_density is
//   ignored on gases. Velocity always sums.
struct FluidState {
    Real                   R           = Real(-1);
    Real                   temperature = Real(0);
    Real                   density     = Real(0);
    Real                   pressure    = Real(0);
    geom::Vec3<SceneFrame> velocity    = geom::Vec3<SceneFrame>::zero();
};

// FluidField as a superposition of disturbances. Same tagged-replication
// scaffolding as GravityField — see that header for design notes.
class FluidField : public Field {
public:
    using Vec       = geom::Vec3<SceneFrame>;
    using ScalarFn  = std::function<Real(const Vec& offset)>;
    using VecFn     = std::function<Vec(const Vec& offset)>;
    using Influence = std::function<bool(const Vec& offset)>;

    static constexpr std::size_t kParamsBytes = 96;
    using Params = std::array<std::uint8_t, kParamsBytes>;

    struct Disturbance {
        Vec           origin           = Vec::zero();
        Real          R                = Real(-1);
        ScalarFn      delta_temperature;
        ScalarFn      delta_density;
        ScalarFn      delta_pressure;
        VecFn         delta_velocity;
        Influence     in_influence;
        std::uint16_t tag              = fluid_tags::USER;
        Params        params{};

        // Uniform incompressible fluid (R=-1).
        static Disturbance uniform_incompressible(Real density, Vec velocity = Vec::zero()) {
            Disturbance d;
            d.origin = Vec::zero();
            d.R      = Real(-1);
            d.tag    = fluid_tags::UNIFORM_INCOMPRESS;
            UniformIncompressParams up{
                density, Real(velocity.x()), Real(velocity.y()), Real(velocity.z())};
            std::memcpy(d.params.data(), &up, sizeof(up));
            d.delta_density  = [density] (const Vec&) noexcept { return density; };
            d.delta_velocity = [velocity](const Vec&) noexcept { return velocity; };
            return d;
        }

        // Uniform gas (R, T, p, velocity all constant).
        static Disturbance uniform_gas(Real R_,
                                       Real temperature,
                                       Real pressure,
                                       Vec  velocity = Vec::zero()) {
            Disturbance d;
            d.origin = Vec::zero();
            d.R      = R_;
            d.tag    = fluid_tags::UNIFORM_GAS;
            UniformGasParams up{R_, temperature, pressure,
                Real(velocity.x()), Real(velocity.y()), Real(velocity.z())};
            std::memcpy(d.params.data(), &up, sizeof(up));
            d.delta_temperature = [temperature](const Vec&) noexcept { return temperature; };
            d.delta_pressure    = [pressure]   (const Vec&) noexcept { return pressure;    };
            d.delta_velocity    = [velocity]   (const Vec&) noexcept { return velocity;    };
            return d;
        }

        struct UniformIncompressParams { Real density, vx, vy, vz; };
        struct UniformGasParams        { Real R, temperature, pressure, vx, vy, vz; };
    };

    static_assert(sizeof(Disturbance::UniformIncompressParams) <= kParamsBytes);
    static_assert(sizeof(Disturbance::UniformGasParams)        <= kParamsBytes);

private:
    struct Entry { Disturbance d; int lifetime; };

public:
    using Handle = std::list<Entry>::iterator;

    Handle add(Disturbance d, int lifetime = 1) {
        if (tx_hook_ && d.tag != fluid_tags::USER && !in_rx_context()) {
            tx_hook_(d.tag, d.params, lifetime);
        }
        entries_.push_back(Entry{std::move(d), lifetime});
        auto it = entries_.end();
        return --it;
    }

    void remove(Handle h) noexcept { entries_.erase(h); }

    FluidState state_at(const Vec& pos) const noexcept {
        FluidState gas, liq;
        bool gas_active = false, liq_active = false;
        Real gas_R = Real(0);

        for (const auto& e : entries_) {
            Vec off = Vec::from_raw(pos.raw() - e.d.origin.raw());
            if (e.d.in_influence && !e.d.in_influence(off)) continue;

            if (e.d.R == Real(-1)) {
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
            gas.density = (gas.temperature > Real(0))
                        ? gas.pressure / (gas_R * gas.temperature)
                        : Real(0);
            return gas;
        }
        if (liq_active) { liq.R = Real(-1); return liq; }
        return FluidState{};
    }

    void update() override { decay_disturbances(entries_); }

    std::size_t disturbance_count() const noexcept { return entries_.size(); }

    // ---- replication ----

    using TxHook = std::function<void(std::uint16_t, const Params&, int)>;
    void set_tx_hook(TxHook h) { tx_hook_ = std::move(h); }

    using DisturbanceFactory = std::function<Disturbance(const Params&)>;
    static void register_factory(std::uint16_t tag, DisturbanceFactory f) {
        factory_map()[tag] = std::move(f);
    }

    bool receive(std::uint16_t tag, const Params& params, int lifetime) {
        auto& m = factory_map();
        auto  it = m.find(tag);
        if (it == m.end()) return false;
        Disturbance d = it->second(params);
        in_rx_context() = true;
        add(std::move(d), lifetime);
        in_rx_context() = false;
        return true;
    }

private:
    static std::unordered_map<std::uint16_t, DisturbanceFactory>& factory_map() {
        static std::unordered_map<std::uint16_t, DisturbanceFactory> m =
            make_stock_factories();
        return m;
    }

    static std::unordered_map<std::uint16_t, DisturbanceFactory> make_stock_factories() {
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

    static bool& in_rx_context() {
        thread_local bool flag = false;
        return flag;
    }

    std::list<Entry>  entries_;
    TxHook            tx_hook_;
};

} // namespace manta::fields
