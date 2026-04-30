#pragma once

namespace manta::fields {

// Abstract base for all field models. Fields are shared physical media
// (gravity, EM, fluid) that crafts interact with. Subclasses define
// query and contribution APIs appropriate to the field.
class Field {
public:
    virtual ~Field() = default;

    // Called once per tick by World after all craft updates complete.
    virtual void update() {}
};

} // namespace manta::fields
