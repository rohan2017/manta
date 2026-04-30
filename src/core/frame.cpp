#include <atomic>
#include "manta/core/frame.hpp"

namespace manta {

FrameId mint_frame_id() noexcept {
#if defined(MANTA_DEBUG_FRAMES)
    static std::atomic<std::uint64_t> counter{1};
    return FrameId{counter.fetch_add(1, std::memory_order_relaxed)};
#else
    return {};
#endif
}

} // namespace manta
