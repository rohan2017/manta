#pragma once

#include <array>
#include <cstdint>
#include <functional>
#include <list>
#include <mutex>
#include <unordered_map>

namespace manta::fields {

// Abstract base for all field models. Fields are shared, additive, physical
// media — gravity, fluid, magnetic — that crafts and parts query and may
// also contribute to (via Disturbance objects).
//
// Each concrete field is a `DisturbanceField<Self, Disturbance>` (below) that
// inherits add/remove/lifecycle/replication scaffolding and supplies its own
// `state_at(query_position)` aggregator and Disturbance factory map.
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
    virtual void update() {}
};

// Lifetime sentinel for "never expire."
inline constexpr int PERSISTENT = -1;

// Stock-tag namespace shared by every disturbance-based field. tag == 0 marks
// a disturbance as "local-only" — the tx hook skips it. Concrete fields
// reserve tag values 1..1023 for stock kinds (uniform / point_mass / dipole /
// ...) and reserve tag values >= USER_BASE for user-registered factories.
inline constexpr std::uint16_t USER_TAG = 0;
inline constexpr std::uint16_t USER_BASE = 1024;

// Fixed-size params blob big enough for the largest stock disturbance kind.
// Trivially copyable POD; layout is the wire format. `static_assert`s in
// each derived class pin layout-sensitive assumptions.
inline constexpr std::size_t kParamsBytes = 96;
using Params = std::array<std::uint8_t, kParamsBytes>;

// RAII guard for the rx-recursion flag. Construction sets the flag; the
// destructor clears it on every path including exceptions thrown from
// receive().
class ScopedRxFlag {
public:
    explicit ScopedRxFlag(bool& flag) noexcept : flag_(flag) { flag_ = true; }
    ~ScopedRxFlag() { flag_ = false; }
    ScopedRxFlag(const ScopedRxFlag&) = delete;
    ScopedRxFlag& operator=(const ScopedRxFlag&) = delete;
private:
    bool& flag_;
};

// CRTP base for fields modeled as a superposition of typed disturbances.
//
// Derived classes provide:
//   * a `Disturbance` type;
//   * a `state_at(...)` aggregator method;
//   * a `static auto stock_factories()` returning a tag→factory map for
//     replication.
//
// Disturbance handles are stable, generation-checked tokens — `Handle{0}`
// is the never-valid sentinel, and remove(h) is idempotent: calling
// `remove` on an already-expired or already-removed handle is a no-op.
template <class Derived, class Disturbance>
class DisturbanceField : public Field {
public:
    using DisturbanceFactory = std::function<Disturbance(const Params&)>;
    using TxHook             = std::function<void(std::uint16_t tag,
                                                  const Params&  params,
                                                  int            lifetime)>;

    // Opaque, never-zero on success. Use the default-constructed Handle{}
    // to mean "no handle" / "already removed" — `remove` ignores it.
    struct Handle {
        std::uint64_t id = 0;
        explicit operator bool() const noexcept { return id != 0; }
    };

    // Add a disturbance. Stock-tagged disturbances (tag != USER_TAG) added
    // outside an rx context are streamed through the tx hook for cross-
    // process replication.
    Handle add(Disturbance d, int lifetime = 1) {
        if (tx_hook_ && d.tag != USER_TAG && !in_rx_context_) {
            tx_hook_(d.tag, d.params, lifetime);
        }
        const std::uint64_t id = next_id_++;
        entries_.push_back(Entry{std::move(d), lifetime, id});
        return Handle{id};
    }

    // Remove the disturbance owning Handle h. Idempotent — silently no-ops
    // when h is the default sentinel or refers to an already-removed entry.
    void remove(Handle h) noexcept {
        if (h.id == 0) return;
        for (auto it = entries_.begin(); it != entries_.end(); ++it) {
            if (it->id == h.id) { entries_.erase(it); return; }
        }
    }

    // Tick-end lifecycle: decrement positive lifetimes, drop expired.
    void update() override {
        for (auto it = entries_.begin(); it != entries_.end(); ) {
            if (it->lifetime > 0 && --it->lifetime == 0) it = entries_.erase(it);
            else                                          ++it;
        }
    }

    std::size_t disturbance_count() const noexcept { return entries_.size(); }

    // ---- replication ----

    void set_tx_hook(TxHook h) { tx_hook_ = std::move(h); }

    // Register a custom factory for a user-defined tag (>= USER_BASE).
    // Stock tags are pre-populated by `Derived::stock_factories()`. This
    // mutates a shared static map — guard with the mutex so concurrent
    // setup-time registrations and live receive() reads don't race.
    static void register_factory(std::uint16_t tag, DisturbanceFactory f) {
        std::lock_guard<std::mutex> lock(factory_mutex());
        factory_map()[tag] = std::move(f);
    }

    // Apply an incoming disturbance from rx. Returns false when `tag` is
    // unknown (the disturbance is silently dropped — the receiver doesn't
    // model that kind). The recursion guard is RAII so an exception
    // thrown from `add` (e.g. bad_alloc) cannot leave the flag stuck.
    bool receive(std::uint16_t tag, const Params& params, int lifetime) {
        DisturbanceFactory factory;
        {
            std::lock_guard<std::mutex> lock(factory_mutex());
            auto& m = factory_map();
            auto it = m.find(tag);
            if (it == m.end()) return false;
            factory = it->second;
        }
        Disturbance d = factory(params);
        ScopedRxFlag guard(in_rx_context_);
        add(std::move(d), lifetime);
        return true;
    }

protected:
    struct Entry { Disturbance d; int lifetime; std::uint64_t id; };
    std::list<Entry> entries_;

private:
    static std::mutex& factory_mutex() {
        static std::mutex m;
        return m;
    }
    static std::unordered_map<std::uint16_t, DisturbanceFactory>& factory_map() {
        static std::unordered_map<std::uint16_t, DisturbanceFactory> m =
            Derived::stock_factories();
        return m;
    }

    std::uint64_t next_id_     = 1;
    bool          in_rx_context_ = false;
    TxHook        tx_hook_;
};

} // namespace manta::fields
