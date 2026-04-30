#pragma once

#include <cstdint>

// Frame typing.
//
// Frame KIND is checked at compile time via tag types. Mixing kinds (e.g.
// Vec3<WorldFrame> + Vec3<CraftFrame>) is a compile error.
//
// Frame IDENTITY is checked at runtime in debug builds via FrameId, and is
// zero-cost in release. Two parts on the same craft each have a distinct
// FrameId; mixing their PartFrame vectors triggers a debug assertion.

namespace manta {

enum class FrameKind : std::uint8_t {
    World,
    Planet,
    Scene,
    Craft,
    Part,
};

template <FrameKind K>
struct frame_tag {
    static constexpr FrameKind kind = K;
};

using WorldFrame  = frame_tag<FrameKind::World>;
using PlanetFrame = frame_tag<FrameKind::Planet>;
using SceneFrame  = frame_tag<FrameKind::Scene>;
using CraftFrame  = frame_tag<FrameKind::Craft>;
using PartFrame   = frame_tag<FrameKind::Part>;

// ParentFrame is a sentinel used in part-relative transforms — it stands in
// for "the frame of this part's parent part," resolved at composition time.
struct ParentFrame { static constexpr FrameKind kind = FrameKind::Part; };

// Runtime identity of a frame instance. Only meaningful in debug builds; in
// release, FrameId is empty and zero-sized.
class FrameId {
public:
#if defined(MANTA_DEBUG_FRAMES)
    constexpr FrameId() noexcept                      = default;
    constexpr explicit FrameId(std::uint64_t id) noexcept : id_(id) {}
    constexpr std::uint64_t value() const noexcept    { return id_; }
    constexpr bool operator==(const FrameId&) const noexcept = default;
private:
    std::uint64_t id_ = 0;  // 0 = unspecified / wildcard
#else
    constexpr FrameId() noexcept                                  = default;
    constexpr explicit FrameId(std::uint64_t /*id*/) noexcept     {}
    constexpr std::uint64_t value() const noexcept                { return 0; }
    constexpr bool operator==(const FrameId&) const noexcept      { return true; }
#endif
};

// Mints a new globally-unique FrameId. Called by Part construction. In release
// builds this is a no-op returning an empty FrameId.
FrameId mint_frame_id() noexcept;

} // namespace manta
