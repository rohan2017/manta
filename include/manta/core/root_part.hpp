#pragma once

#include "composite_part.hpp"

namespace manta {

// Top of the part tree. Represents the craft's geometric center / body frame.
// Owned by Craft; has no parent part.
class RootPart final : public CompositePart {
public:
    RootPart() noexcept : CompositePart("root") {}
};

} // namespace manta
