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

// Tag IDs identifying disturbance kinds for cross-process replication. The
// Field's factory registry maps `tag → (params blob → Disturbance)` so a
// receiving process can rebuild the same Disturbance lambda from the wire
// bytes. Stock IDs are reserved at the bottom; user-defined factories
// should pick values >= USER_BASE.
namespace gravity_tags {
    constexpr std::uint16_t USER          = 0;     // untagged, local-only
    constexpr std::uint16_t UNIFORM       = 1;
    constexpr std::uint16_t POINT_MASS    = 2;
    constexpr std::uint16_t POINT_MASS_J2 = 3;
    constexpr std::uint16_t USER_BASE     = 1024;
}

// Gravity as a superposition of disturbances. Each Disturbance is a
// (tag, origin, lambda, params) tuple where the lambda returns the
// gravitational acceleration contribution at a given offset and the
// (tag, params) pair lets the disturbance be re-bound on a different
// process via the factory registry.
class GravityField : public Field {
public:
    using Vec       = geom::Vec3<SceneFrame>;
    using DeltaG    = std::function<Vec(const Vec& offset)>;
    using Influence = std::function<bool(const Vec& offset)>;

    // Fixed-size params blob big enough for the largest stock kind
    // (point_mass_j2 = 9 Reals = 36 B with float, 72 B with double). The
    // pad lets future kinds grow without bumping the wire schema.
    static constexpr std::size_t kParamsBytes = 96;
    using Params = std::array<std::uint8_t, kParamsBytes>;

    struct Disturbance {
        Vec           origin = Vec::zero();
        DeltaG        delta_g;
        Influence     in_influence;

        // Replication metadata. `tag == USER` (0) means local-only — the
        // tx hook skips it. Stock factories below set tag and params.
        std::uint16_t tag    = gravity_tags::USER;
        Params        params{};

        // Uniform gravity everywhere (e.g. flat-earth -9.81 ẑ).
        static Disturbance uniform(Vec g) {
            Disturbance d;
            d.origin  = Vec::zero();
            d.tag     = gravity_tags::UNIFORM;
            UniformParams up{Real(g.x()), Real(g.y()), Real(g.z())};
            std::memcpy(d.params.data(), &up, sizeof(up));
            d.delta_g = [g](const Vec&) noexcept { return g; };
            return d;
        }

        // Inverse-square gravity from a point mass at `origin`. `mu` is GM.
        static Disturbance point_mass(Vec origin, Real mu) {
            Disturbance d;
            d.origin  = origin;
            d.tag     = gravity_tags::POINT_MASS;
            PointMassParams pm{Real(origin.x()), Real(origin.y()), Real(origin.z()), mu};
            std::memcpy(d.params.data(), &pm, sizeof(pm));
            d.delta_g = [mu](const Vec& off) noexcept {
                Real r2 = off.raw().squaredNorm();
                if (r2 < Real(1e-12f)) return Vec::zero();
                Real r  = std::sqrt(r2);
                return Vec::from_raw(off.raw() * (-mu / (r2 * r)));
            };
            return d;
        }

        // Point mass with J2 oblateness.
        static Disturbance point_mass_j2(Vec  origin,
                                         Real mu,
                                         Real j2_coeff,
                                         Real equatorial_radius,
                                         Vec  polar_axis = Vec{Real(0), Real(0), Real(1)}) {
            Disturbance d;
            d.origin = origin;
            d.tag    = gravity_tags::POINT_MASS_J2;
            PointMassJ2Params pp{
                Real(origin.x()), Real(origin.y()), Real(origin.z()),
                mu, j2_coeff, equatorial_radius,
                Real(polar_axis.x()), Real(polar_axis.y()), Real(polar_axis.z()),
            };
            std::memcpy(d.params.data(), &pp, sizeof(pp));
            d.delta_g = [mu, j2_coeff, equatorial_radius, polar_axis](const Vec& off) noexcept {
                const auto& r = off.raw();
                Real r2 = r.squaredNorm();
                if (r2 < Real(1e-12f)) return Vec::zero();
                Real r_norm = std::sqrt(r2);
                Real inv_r3 = Real(1) / (r2 * r_norm);
                Eigen::Matrix<Real, 3, 1> g = r * (-mu * inv_r3);
                Real inv_r2 = Real(1) / r2;
                Real z_p    = r.dot(polar_axis.raw());
                Real z_p2   = z_p * z_p;
                Real factor = Real(1.5) * j2_coeff
                            * (equatorial_radius * equatorial_radius) * inv_r2;
                Real five_z2_over_r2 = Real(5) * z_p2 * inv_r2;
                Eigen::Matrix<Real, 3, 1> r_eq = r - polar_axis.raw() * z_p;
                g += r_eq * (-mu * inv_r3 * factor * (five_z2_over_r2 - Real(1)))
                   + polar_axis.raw() * (-mu * inv_r3 * factor * z_p
                                         * (five_z2_over_r2 - Real(3)));
                return Vec::from_raw(g);
            };
            return d;
        }

        // POD param structs — `Real`-typed for portability between processes
        // that share the same `Real` build flag. Trivially copyable; layout
        // is the wire format. `static_assert`s below pin layout-sensitive
        // assumptions so a sneaky struct change won't silently desync.
        struct UniformParams       { Real gx, gy, gz; };
        struct PointMassParams     { Real ox, oy, oz, mu; };
        struct PointMassJ2Params   { Real ox, oy, oz, mu, j2, eq_radius, ax, ay, az; };
    };

    static_assert(sizeof(Disturbance::UniformParams)     <= kParamsBytes);
    static_assert(sizeof(Disturbance::PointMassParams)   <= kParamsBytes);
    static_assert(sizeof(Disturbance::PointMassJ2Params) <= kParamsBytes);

private:
    struct Entry { Disturbance d; int lifetime; };

public:
    using Handle = std::list<Entry>::iterator;

    GravityField() = default;

    explicit GravityField(Vec uniform_g) {
        add(Disturbance::uniform(uniform_g), PERSISTENT);
    }

    // ---- core API ----

    Handle add(Disturbance d, int lifetime = 1) {
        // Outbound replication: stock-tagged disturbances added via the
        // user-facing path (i.e. NOT from receive()) are streamed through
        // the tx hook. The recursion guard prevents echo when `add` is
        // called from inside `receive`.
        if (tx_hook_ && d.tag != gravity_tags::USER && !in_rx_context()) {
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
            if (!e.d.delta_g) continue;
            sum += e.d.delta_g(off).raw();
        }
        return Vec::from_raw(sum);
    }

    void update() override { decay_disturbances(entries_); }

    std::size_t disturbance_count() const noexcept { return entries_.size(); }

    // ---- replication: tx/rx hooks + factory registry ----

    using TxHook = std::function<void(std::uint16_t tag,
                                      const Params& params,
                                      int            lifetime)>;
    void set_tx_hook(TxHook h) { tx_hook_ = std::move(h); }

    using DisturbanceFactory =
        std::function<Disturbance(const Params& params)>;

    // Register a factory for a custom tag (>= USER_BASE). Stock tags are
    // populated by the static `factory_map()` initializer; user code
    // should call this during process startup before any sync rx happens.
    static void register_factory(std::uint16_t tag, DisturbanceFactory f) {
        factory_map()[tag] = std::move(f);
    }

    // Apply an incoming disturbance from rx. Looks up `tag` in the
    // factory registry, builds a Disturbance, and adds it with the
    // recursion guard set so the local tx hook does NOT re-broadcast.
    // Returns false if `tag` is unknown (the disturbance is silently
    // dropped — the receiver process simply doesn't model that kind).
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
        m[gravity_tags::UNIFORM] = [](const Params& p) {
            Disturbance::UniformParams up;
            std::memcpy(&up, p.data(), sizeof(up));
            return Disturbance::uniform(Vec{up.gx, up.gy, up.gz});
        };
        m[gravity_tags::POINT_MASS] = [](const Params& p) {
            Disturbance::PointMassParams pm;
            std::memcpy(&pm, p.data(), sizeof(pm));
            return Disturbance::point_mass(Vec{pm.ox, pm.oy, pm.oz}, pm.mu);
        };
        m[gravity_tags::POINT_MASS_J2] = [](const Params& p) {
            Disturbance::PointMassJ2Params pp;
            std::memcpy(&pp, p.data(), sizeof(pp));
            return Disturbance::point_mass_j2(
                Vec{pp.ox, pp.oy, pp.oz},
                pp.mu, pp.j2, pp.eq_radius,
                Vec{pp.ax, pp.ay, pp.az});
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
