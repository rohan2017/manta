#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <unordered_map>

#include "field.hpp"
#include "../core/frame.hpp"
#include "../core/types.hpp"
#include "../geom/vec3.hpp"

namespace manta::fields {

namespace mag_tags {
    constexpr std::uint16_t UNIFORM   = 1;
    constexpr std::uint16_t DIPOLE    = 2;
}

struct MagDisturbance {
    using Vec       = geom::Vec3<SceneFrame>;
    using DeltaB    = std::function<Vec(const Vec& offset)>;
    using Influence = std::function<bool(const Vec& offset)>;

    Vec           origin = Vec::zero();
    DeltaB        delta_b;
    Influence     in_influence;
    std::uint16_t tag    = USER_TAG;
    Params        params{};

    struct UniformParams { MFloat bx, by, bz; };
    struct DipoleParams  { MFloat ox, oy, oz, mx, my, mz; };

    static MagDisturbance uniform(Vec b) {
        MagDisturbance d;
        d.tag = mag_tags::UNIFORM;
        UniformParams up{MFloat(b.x()), MFloat(b.y()), MFloat(b.z())};
        std::memcpy(d.params.data(), &up, sizeof(up));
        d.delta_b = [b](const Vec&) noexcept { return b; };
        return d;
    }

    static MagDisturbance dipole(Vec origin, Vec moment) {
        MagDisturbance d;
        d.origin = origin;
        d.tag    = mag_tags::DIPOLE;
        DipoleParams dp{
            MFloat(origin.x()), MFloat(origin.y()), MFloat(origin.z()),
            MFloat(moment.x()), MFloat(moment.y()), MFloat(moment.z()),
        };
        std::memcpy(d.params.data(), &dp, sizeof(dp));
        constexpr MFloat kMu0Over4Pi = MFloat(1.0e-7f); // T·m/A
        d.delta_b = [moment](const Vec& off) noexcept {
            const auto& r = off.raw();
            MFloat r2 = r.squaredNorm();
            if (r2 < MFloat(1e-12f)) return Vec::zero();
            MFloat r_norm = std::sqrt(r2);
            MFloat inv_r3 = MFloat(1) / (r2 * r_norm);
            MFloat m_dot_r_over_r2 = moment.raw().dot(r) / r2;
            Eigen::Matrix<MFloat, 3, 1> b =
                  kMu0Over4Pi * inv_r3
                  * (MFloat(3) * m_dot_r_over_r2 * r - moment.raw());
            return Vec::from_raw(b);
        };
        return d;
    }
};

class MagField : public DisturbanceField<MagField, MagDisturbance> {
public:
    using Vec         = MagDisturbance::Vec;
    using Disturbance = MagDisturbance;

    static_assert(sizeof(Disturbance::UniformParams) <= kParamsBytes);
    static_assert(sizeof(Disturbance::DipoleParams)  <= kParamsBytes);

    Vec state_at(const Vec& pos) const noexcept {
        Eigen::Matrix<MFloat, 3, 1> sum = Eigen::Matrix<MFloat, 3, 1>::Zero();
        for (const auto& e : entries_) {
            Vec off = Vec::from_raw(pos.raw() - e.d.origin.raw());
            if (e.d.in_influence && !e.d.in_influence(off)) continue;
            if (!e.d.delta_b) continue;
            sum += e.d.delta_b(off).raw();
        }
        return Vec::from_raw(sum);
    }

    static std::unordered_map<std::uint16_t, DisturbanceFactory> stock_factories() {
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
};

} // namespace manta::fields
