#pragma once

#include <atomic>

#include "frame.hpp"

// Inline implementation of mint_frame_id. Pulled into a separate header so the
// atomic counter has a single home — include this exactly once in a TU that
// owns the storage (we do this from a stub TU compiled into examples/tests).

namespace manta {

inline FrameId mint_frame_id() noexcept {
#if defined(MANTAPILOT_DEBUG_FRAMES)
    static std::atomic<std::uint64_t> counter{1};
    return FrameId{counter.fetch_add(1, std::memory_order_relaxed)};
#else
    return FrameId{};
#endif
}

} // namespace manta
