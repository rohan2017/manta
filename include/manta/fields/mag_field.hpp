#pragma once

#include <array>
#include <cmath>
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

namespace mag_tags {
    constexpr std::uint16_t USER      = 0;
    constexpr std::uint16_t UNIFORM   = 1;
    constexpr std::uint16_t DIPOLE    = 2;
    constexpr std::uint16_t USER_BASE = 1024;
}

// Magnetic field as a superposition of disturbances. Same tagged-replication
// scaffolding as GravityField — see that header for design notes.
class MagField : public Field {
public:
    using Vec       = geom::Vec3<SceneFrame>;
    using DeltaB    = std::function<Vec(const Vec& offset)>;
    using Influence = std::function<bool(const Vec& offset)>;

    static constexpr Real kMu0Over4Pi = Real(1.0e-7f); // T·m/A

    static constexpr std::size_t kParamsBytes = 96;
    using Params = std::array<std::uint8_t, kParamsBytes>;

    struct Disturbance {
        Vec           origin = Vec::zero();
        DeltaB        delta_b;
        Influence     in_influence;
        std::uint16_t tag    = mag_tags::USER;
        Params        params{};

        // Uniform B everywhere.
        static Disturbance uniform(Vec b) {
            Disturbance d;
            d.tag = mag_tags::UNIFORM;
            UniformParams up{Real(b.x()), Real(b.y()), Real(b.z())};
            std::memcpy(d.params.data(), &up, sizeof(up));
            d.delta_b = [b](const Vec&) noexcept { return b; };
            return d;
        }

        // Magnetic dipole at `origin` with moment vector `moment` (A·m²).
        static Disturbance dipole(Vec origin, Vec moment) {
            Disturbance d;
            d.origin = origin;
            d.tag    = mag_tags::DIPOLE;
            DipoleParams dp{
                Real(origin.x()), Real(origin.y()), Real(origin.z()),
                Real(moment.x()), Real(moment.y()), Real(moment.z()),
            };
            std::memcpy(d.params.data(), &dp, sizeof(dp));
            d.delta_b = [moment](const Vec& off) noexcept {
                const auto& r = off.raw();
                Real r2 = r.squaredNorm();
                if (r2 < Real(1e-12f)) return Vec::zero();
                Real r_norm = std::sqrt(r2);
                Real inv_r3 = Real(1) / (r2 * r_norm);
                Real m_dot_r_over_r2 = moment.raw().dot(r) / r2;
                Eigen::Matrix<Real, 3, 1> b =
                      kMu0Over4Pi * inv_r3
                      * (Real(3) * m_dot_r_over_r2 * r - moment.raw());
                return Vec::from_raw(b);
            };
            return d;
        }

        struct UniformParams { Real bx, by, bz; };
        struct DipoleParams  { Real ox, oy, oz, mx, my, mz; };
    };

    static_assert(sizeof(Disturbance::UniformParams) <= kParamsBytes);
    static_assert(sizeof(Disturbance::DipoleParams)  <= kParamsBytes);

private:
    struct Entry { Disturbance d; int lifetime; };

public:
    using Handle = std::list<Entry>::iterator;

    Handle add(Disturbance d, int lifetime = 1) {
        if (tx_hook_ && d.tag != mag_tags::USER && !in_rx_context()) {
            tx_hook_(d.tag, d.params, lifetime);
        }
        entries_.push_back(Entry{std::move(d), lifetime});
        auto it = entries_.end();
        return --it;
    }

    void remove(Handle h) noexcept { entries_.erase(h); }

    Vec state_at(const Vec& pos) const noexcept {
        Eigen::Matrix<Real, 3, 1> sum = Eigen::Matrix<Real, 3, 1>::Zero();
        for (const auto& e : entries_) {
            Vec off = Vec::from_raw(pos.raw() - e.d.origin.raw());
            if (e.d.in_influence && !e.d.in_influence(off)) continue;
            if (!e.d.delta_b) continue;
            sum += e.d.delta_b(off).raw();
        }
        return Vec::from_raw(sum);
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
        m[mag_tags::UNIFORM] = [](const Params& p) {
            Disturbance::UniformParams up;
            std::memcpy(&up, p.data(), sizeof(up));
            return Disturbance::uniform(Vec{up.bx, up.by, up.bz});
        };
        m[mag_tags::DIPOLE] = [](const Params& p) {
            Disturbance::DipoleParams dp;
            std::memcpy(&dp, p.data(), sizeof(dp));
            return Disturbance::dipole(Vec{dp.ox, dp.oy, dp.oz},
                                        Vec{dp.mx, dp.my, dp.mz});
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
