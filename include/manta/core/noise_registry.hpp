#pragma once

// NoiseRegistry — collects per-part white-noise sources at filter bind
// time and assigns each one a contiguous slot block in the EKF's Jet
// noise-input range (3 slots for a Vec3 noise, 1 slot for a scalar).
//
// Used by `manta::estimation::EKFGeneric`'s `finalize_bindings()`. The
// EKF walks the Jet-side part tree calling `PartT::register_noise(reg)`;
// each part hands its `Noise<WhiteGaussian>` instances over and the
// registry stamps the slot index back onto the source.
//
// RW biases are tracked as explicit StateSpec slices in EKFGeneric (not
// via this registry); `register_random_walk` is therefore a no-op that
// just clears the source's slots so its `operator+` injects no Jets.

#include <vector>

#include "noise.hpp"

namespace manta {

template <class Scalar> class PartT;

class NoiseRegistry {
public:
    // Register a 3-axis white-noise source. Allocates 3 contiguous slots
    // and stores the slot start on the source.
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

    // RW biases are tracked as StateSpec slices, not via the registry.
    // Clear the source's slots so its Jet operator+ becomes a pass-
    // through (no bias-state Jet injection).
    int register_random_walk(NoiseRandomWalkBase& source) {
        source.clear_slots();
        return kNoNoiseSlot;
    }

    int         num_slots()   const noexcept { return next_slot_; }
    std::size_t source_count() const noexcept { return entries_.size(); }

    // Promote each registered source's slot index by the EKF's global
    // noise-input column offset (= tangent_dim, where the Jet's noise
    // columns start).
    void apply_slot_offsets(int noise_input_offset) {
        for (auto& e : entries_) {
            e.source->set_slot(e.source->slot() + noise_input_offset);
        }
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

// Recursively register noise sources from a part subtree into the
// registry. Used at filter finalize time to walk the bound craft's root
// part. Defined here so both estimator wrappers can use it without
// including each other's headers.
template <class Scalar>
inline void walk_register_noise(PartT<Scalar>& part, NoiseRegistry& reg) {
    part.register_noise(reg);
    auto* kids = part.children();
    if (!kids) return;
    for (auto& child : *kids) walk_register_noise(*child, reg);
}

} // namespace manta
