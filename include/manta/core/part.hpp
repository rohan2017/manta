#pragma once

#include <string>
#include <typeinfo>
#include <type_traits>
#include "frame.hpp"
#include "types.hpp"
#include "wrench.hpp"
#include "../geom/kinematic_link.hpp"
#include "../geom/mat3.hpp"
#include "../geom/ori.hpp"
#include "../geom/static_link.hpp"
#include "../geom/vec3.hpp"

namespace manta {

template <class Scalar> class CompositePartT;
template <class Scalar> class ArticulatedPartT;
template <class Scalar> class CraftT;

namespace fields { class Field; }

// Base class for all simulation components. Subclasses implement update() to
// compute forces, read sensors, etc. The framework calls update() every tick
// after populating the kinematic cache (craft_to_part_).
//
// Templated on Scalar so the same class hierarchy can host both the sim
// path (Scalar = Real, typically float) and the estimator path
// (Scalar = ceres::Jet<double, N>) — the same Craft can be evaluated by a
// Scene (sim) or by an EKF (estimator) depending on which scalar it's
// instantiated with. Most user code uses the `Part` alias below, which is
// `PartT<Real>` — existing non-templated Part subclasses bind to this
// instantiation and need no changes.
template <class Scalar = Real>
class PartT {
public:
    explicit PartT(std::string name) noexcept : name_(std::move(name)) {}
    virtual ~PartT() = default;

    // Subclasses implement this to sense the environment and apply wrenches.
    virtual void update() = 0;

    // Physical parameters. Set from the subclass constructor or user code.
    // CompositePart overrides these as no-ops; use compute_params() there.
    virtual void set_mass(Scalar m) noexcept                           { mass_ = m; }
    virtual void set_moi(const geom::Mat3<PartFrame, PartFrame, Scalar>& moi) noexcept { moi_ = moi; }
    virtual void set_com(const geom::Vec3<PartFrame, Scalar>& com) noexcept { com_ = com; }
    void set_transform(const geom::StaticLink<ParentFrame, PartFrame, Scalar>& tf) noexcept {
        transform_ = geom::StaticLink<PartFrame, PartFrame, Scalar>{
            geom::Vec3<PartFrame, Scalar>::from_raw(tf.position().raw()),
            geom::Ori<PartFrame, Scalar>{tf.orientation().raw()},
        };
    }

    Scalar                                              get_mass()      const noexcept { return mass_; }
    const geom::Mat3<PartFrame, PartFrame, Scalar>&                get_moi()       const noexcept { return moi_; }
    const geom::Vec3<PartFrame, Scalar>&                get_com()       const noexcept { return com_; }
    const geom::StaticLink<PartFrame, PartFrame, Scalar>& get_transform() const noexcept { return transform_; }

    // Wrench application — accumulates within a single tick. Multiple calls add.
    void apply_force_at(const geom::Vec3<PartFrame, Scalar>& force,
                        const geom::Vec3<PartFrame, Scalar>& point =
                            geom::Vec3<PartFrame, Scalar>::zero()) noexcept {
        wrench_accum_ += Wrench<PartFrame, Scalar>::from_force_at(force, point);
    }
    void apply_torque(const geom::Vec3<PartFrame, Scalar>& torque) noexcept {
        wrench_accum_ += Wrench<PartFrame, Scalar>::from_torque(torque);
    }
    void apply_wrench(const Wrench<PartFrame, Scalar>& w) noexcept {
        wrench_accum_ += w;
    }

    // Kinematic queries — read from the cache filled by the kinematic pass.
    // Convention: `quantity<F>()` returns the kinematic quantity of the part
    // *relative to F* (its origin and axes). `<PartFrame>` is therefore
    // trivially zero/identity. For the body-frame *absolute* quantities a
    // sensor on the part would read, use the `_body()` accessors.
    template<typename F> geom::Vec3<F, Scalar> position()              const noexcept;
    template<typename F> geom::Vec3<F, Scalar> velocity()              const noexcept;
    template<typename F> geom::Vec3<F, Scalar> acceleration()          const noexcept;
    template<typename F> geom::Vec3<F, Scalar> angular_velocity()      const noexcept;
    template<typename F> geom::Vec3<F, Scalar> angular_acceleration()  const noexcept;
    template<typename F> geom::Ori<F, Scalar>  orientation()           const noexcept;

    geom::Vec3<PartFrame, Scalar> velocity_body()             const noexcept;
    geom::Vec3<PartFrame, Scalar> acceleration_body()         const noexcept;
    geom::Vec3<PartFrame, Scalar> angular_velocity_body()     const noexcept;
    geom::Vec3<PartFrame, Scalar> angular_acceleration_body() const noexcept;

    // Field access — delegates to craft().field<FieldT>().
    template<typename FieldT>
    FieldT& field() const {
        return *static_cast<FieldT*>(field_ptr(typeid(FieldT)));
    }

    const std::string& name()    const noexcept { return name_; }
    PartT*             parent()       noexcept { return parent_; }
    const PartT*       parent() const noexcept { return parent_; }

    // Craft accessors. For the sim path the craft pointer is set when the
    // Part is added to a CraftT<Real>; for the estimator path (Jet), it
    // points to the EKF's `CraftT<Jet>` instance.
    CraftT<Scalar>&       craft();
    const CraftT<Scalar>& craft() const;

    fields::Field* field_ptr(const std::type_info& ti) const;

    const Wrench<PartFrame, Scalar>& net_wrench() const noexcept { return wrench_accum_; }

    const geom::KinematicLink<SceneFrame, PartFrame, Scalar>& scene_to_part() const noexcept {
        return scene_to_part_;
    }
    const geom::KinematicLink<CraftFrame, PartFrame, Scalar>& craft_to_part() const noexcept {
        return craft_to_part_;
    }

protected:
    CraftT<Scalar>* craft_ = nullptr;

    Scalar                              mass_ = Scalar(0);
    geom::Mat3<PartFrame, PartFrame, Scalar>       moi_  = geom::Mat3<PartFrame, PartFrame, Scalar>::zero();
    geom::Vec3<PartFrame, Scalar>       com_;

private:
    template <class S> friend class CompositePartT;
    template <class S> friend class ArticulatedPartT;
    template <class S> friend class CraftT;

    Wrench<PartFrame, Scalar> drain_wrench() noexcept {
        auto w = wrench_accum_;
        wrench_accum_ = Wrench<PartFrame, Scalar>::zero();
        return w;
    }

    std::string name_;
    geom::StaticLink<PartFrame, PartFrame, Scalar> transform_ =
        geom::StaticLink<PartFrame, PartFrame, Scalar>::identity();

    PartT* parent_ = nullptr;

    geom::KinematicLink<CraftFrame, PartFrame, Scalar> craft_to_part_;
    geom::KinematicLink<SceneFrame, PartFrame, Scalar> scene_to_part_;

    Wrench<PartFrame, Scalar> wrench_accum_;
};

// Backwards-compat alias: existing user code writing `class Foo : public Part`
// binds to PartT<Real> automatically. The Scalar=Real instantiation is the
// only one used by the sim path; estimator-side uses (Scalar = Jet<...>)
// activate when the Craft class is templated in a future step.
using Part = PartT<Real>;

// --- template implementations ---

template <class Scalar>
template <typename F>
geom::Vec3<F, Scalar> PartT<Scalar>::position() const noexcept {
    static_assert(std::is_same_v<F, SceneFrame> ||
                  std::is_same_v<F, CraftFrame> || std::is_same_v<F, PartFrame>,
                  "PartT::position<F>: unsupported frame");
    if constexpr (std::is_same_v<F, SceneFrame>) {
        return scene_to_part_.position();
    } else if constexpr (std::is_same_v<F, CraftFrame>) {
        return craft_to_part_.position();
    } else {
        return geom::Vec3<PartFrame, Scalar>::zero();
    }
}

template <class Scalar>
template <typename F>
geom::Vec3<F, Scalar> PartT<Scalar>::velocity() const noexcept {
    static_assert(std::is_same_v<F, SceneFrame> ||
                  std::is_same_v<F, CraftFrame> || std::is_same_v<F, PartFrame>,
                  "PartT::velocity<F>: unsupported frame");
    if constexpr (std::is_same_v<F, SceneFrame>) {
        return scene_to_part_.vel_linear();
    } else if constexpr (std::is_same_v<F, CraftFrame>) {
        return craft_to_part_.vel_linear();
    } else {
        return geom::Vec3<PartFrame, Scalar>::zero();
    }
}

template <class Scalar>
template <typename F>
geom::Vec3<F, Scalar> PartT<Scalar>::angular_velocity() const noexcept {
    static_assert(std::is_same_v<F, SceneFrame> ||
                  std::is_same_v<F, CraftFrame> || std::is_same_v<F, PartFrame>,
                  "PartT::angular_velocity<F>: unsupported frame");
    if constexpr (std::is_same_v<F, SceneFrame>) {
        return scene_to_part_.rotate(scene_to_part_.vel_angular());
    } else if constexpr (std::is_same_v<F, CraftFrame>) {
        return craft_to_part_.rotate(craft_to_part_.vel_angular());
    } else {
        return geom::Vec3<PartFrame, Scalar>::zero();
    }
}

template <class Scalar>
template <typename F>
geom::Ori<F, Scalar> PartT<Scalar>::orientation() const noexcept {
    static_assert(std::is_same_v<F, SceneFrame> ||
                  std::is_same_v<F, CraftFrame> || std::is_same_v<F, PartFrame>,
                  "PartT::orientation<F>: unsupported frame");
    if constexpr (std::is_same_v<F, SceneFrame>) {
        return scene_to_part_.orientation();
    } else if constexpr (std::is_same_v<F, CraftFrame>) {
        return craft_to_part_.orientation();
    } else {
        return geom::Ori<PartFrame, Scalar>::identity();
    }
}

template <class Scalar>
template <typename F>
geom::Vec3<F, Scalar> PartT<Scalar>::acceleration() const noexcept {
    static_assert(std::is_same_v<F, SceneFrame> || std::is_same_v<F, PartFrame>,
                  "PartT::acceleration<F>: unsupported frame");
    if constexpr (std::is_same_v<F, SceneFrame>) {
        return scene_to_part_.acc_linear();
    } else {
        return geom::Vec3<PartFrame, Scalar>::zero();
    }
}

template <class Scalar>
template <typename F>
geom::Vec3<F, Scalar> PartT<Scalar>::angular_acceleration() const noexcept {
    static_assert(std::is_same_v<F, SceneFrame> || std::is_same_v<F, PartFrame>,
                  "PartT::angular_acceleration<F>: unsupported frame");
    if constexpr (std::is_same_v<F, SceneFrame>) {
        return scene_to_part_.rotate(scene_to_part_.acc_angular());
    } else {
        return geom::Vec3<PartFrame, Scalar>::zero();
    }
}

template <class Scalar>
geom::Vec3<PartFrame, Scalar> PartT<Scalar>::velocity_body() const noexcept {
    return scene_to_part_.rotate_inverse(scene_to_part_.vel_linear());
}

template <class Scalar>
geom::Vec3<PartFrame, Scalar> PartT<Scalar>::acceleration_body() const noexcept {
    return scene_to_part_.rotate_inverse(scene_to_part_.acc_linear());
}

template <class Scalar>
geom::Vec3<PartFrame, Scalar> PartT<Scalar>::angular_velocity_body() const noexcept {
    return scene_to_part_.vel_angular();
}

template <class Scalar>
geom::Vec3<PartFrame, Scalar> PartT<Scalar>::angular_acceleration_body() const noexcept {
    return scene_to_part_.acc_angular();
}

} // namespace manta
