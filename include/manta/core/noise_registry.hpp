#pragma once

// NoiseRegistry — collects per-part white-noise sources at EKF bind time
// and assembles the diagonal of the noise-input covariance Σ each tick.
//
// Used by `manta::estimation::EKF` to implement automatic Q assembly.
// At bind, the EKF walks the Jet-side part tree calling
// `PartT::register_noise(reg)` on each part. Parts hand their
// `Noise<WhiteGaussian>` instances to the registry, which assigns each
// one a contiguous slot block (3 slots for a 3-axis Vec noise, 1 slot
// for a scalar noise).
//
// Each tick during predict, the EKF queries `sigma_diag(dt, ...)` to
// build the per-input covariance diagonal `diag(σᵢ²·dt)`. The Jet
// world's noise inputs propagate through the dynamics as additional
// Jet columns; the EKF reads back the noise-input gain matrix L from
// those Jet columns and assembles
//
//     Q = L · diag(σᵢ²·dt) · Lᵀ
//
// State-dependent σ is supported: the registry calls `source->sigma()`
// fresh each tick, so a part that mutated `set_sigma()` from its
// `update()` produces the correct Q on the next predict.

#include <vector>

#include "noise.hpp"

namespace manta {

class NoiseRegistry {
public:
    // Register a 3-axis white-noise source. Allocates 3 contiguous slots
    // and stores the slot start on the source. Returns the slot start
    // (also written into the source via set_slot).
    int register_white_3d(Noise<WhiteGaussian>& source) {
        const int start = next_slot_;
        source.set_slot(start);
        entries_.push_back(Entry{&source, /*dim=*/3, start});
        next_slot_ += 3;
        return start;
    }

    // Register a scalar white-noise source. 1 slot.
    int register_white_1d(Noise<WhiteGaussian>& source) {
        const int start = next_slot_;
        source.set_slot(start);
        entries_.push_back(Entry{&source, /*dim=*/1, start});
        next_slot_ += 1;
        return start;
    }

    int num_slots() const noexcept { return next_slot_; }
    std::size_t source_count() const noexcept { return entries_.size(); }

    // Shift every registered source's external slot by `offset`. Used by
    // the EKF wrapper after bind to convert registry-local slots (0..n)
    // — which the Noise objects originally held — into global Jet-column
    // indices (kNoiseColStart..kNoiseColStart+n) that the `Vec3 + Noise`
    // operator writes into.
    //
    // The registry's *internal* slot_start bookkeeping stays unshifted
    // (the registry's tables are 0-indexed from registry-local zero), so
    // sigma_squared_diag's output is sized to num_slots() and indexable
    // via local slot indices.
    void apply_slot_offset(int offset) {
        for (auto& e : entries_) {
            e.source->set_slot(e.source->slot() + offset);
        }
    }

    // Per-tick variance of the unit noise inputs at each slot, into
    // `out` (resized to num_slots()).
    //
    // For Noise<WhiteGaussian>: the σ scaling is already inside the
    // noise-input gain L (the operator+ injects `Jet(0)·σ`), so the
    // input itself is unit-variance and this returns 1.0 per slot.
    // The resulting auto-Q is L·diag(1)·Lᵀ = L·Lᵀ — which works out to
    // L_eff·diag(σ²)·L_effᵀ where L_eff = L/σ is the unit-input gain,
    // i.e. the standard Kalman discrete Q.
    //
    // RandomWalk biases (Phase E) will return σ²·dt per slot instead;
    // this method is the per-noise-policy hook for that.
    //
    // dt is unused for WhiteGaussian (per-tick discrete noise has σ as
    // its per-tick stddev directly) but accepted for consistency with
    // future RandomWalk slots.
    void input_variance_diag(double /*dt*/, std::vector<double>& out) const {
        out.assign(static_cast<std::size_t>(next_slot_), 1.0);
    }

    // Clear all slots — flips every registered source back to "no slot."
    // Lets a filter rebind cleanly.
    void clear() {
        for (auto& e : entries_) e.source->clear_slot();
        entries_.clear();
        next_slot_ = 0;
    }

private:
    struct Entry {
        Noise<WhiteGaussian>* source;
        int                   dim;
        int                   slot_start;
    };

    std::vector<Entry> entries_;
    int                next_slot_ = 0;
};

} // namespace manta
