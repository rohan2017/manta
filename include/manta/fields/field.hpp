#pragma once

#include <list>

namespace manta::fields {

// Abstract base for all field models. Fields are shared, additive, physical
// media — gravity, fluid, magnetic — that crafts and parts query and may
// also contribute to (via Disturbance objects).
//
// Each concrete field class defines a `Disturbance` nested type with the
// per-physics shape (lambdas + origin + parameters) and provides:
//
//   Handle add(Disturbance, int lifetime = 1)
//   void   remove(Handle)
//   StateT state_at(query_position)
//
// `state_at` sums the contributions of all live disturbances, applying any
// physics-specific aggregation (e.g. FluidField's gas-state correction).
//
// Disturbances have a `lifetime` in ticks. Default 1 — added disturbances
// disappear at the end of the next `update()`. Setting lifetime negative
// (or to the named `PERSISTENT` sentinel) makes a disturbance permanent;
// other positive values give a finite N-tick lifetime (e.g. an explosion
// that lasts a few frames).
class Field {
public:
    virtual ~Field() = default;

    // Called once per tick by World after all craft updates complete.
    // Concrete fields decrement disturbance lifetimes here and drop expired
    // entries; the helper `decay_disturbances` does this in one line.
    virtual void update() {}
};

// Lifetime sentinel for "never expire." Any negative value is treated as
// permanent; this is the canonical name.
inline constexpr int PERSISTENT = -1;

// Lifetime helper used by each Field's update(). Decrements the lifetime
// of every entry whose lifetime is positive; entries that hit zero are
// erased. PERSISTENT (negative) entries are untouched.
//
// `Entry` must have an `int lifetime` member.
template <typename Entry>
inline void decay_disturbances(std::list<Entry>& es) noexcept {
    for (auto it = es.begin(); it != es.end(); ) {
        if (it->lifetime > 0 && --it->lifetime == 0) it = es.erase(it);
        else                                          ++it;
    }
}

} // namespace manta::fields
