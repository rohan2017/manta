#pragma once

#include <memory>
#include <vector>
#include "part.hpp"

namespace manta {

template <class Scalar> class CraftT;


// A part that owns child parts. Structural grouping node. Its own update() is
// a no-op; the framework drives children separately. Wrench aggregation
// collects child wrenches, transforms them into this frame, and sums them.
//
// Templated on Scalar to match PartT. The non-templated `CompositePart` alias
// (= CompositePartT<Real>) is what existing user code uses.
template <class Scalar = Real>
class CompositePartT : public PartT<Scalar> {
public:
    explicit CompositePartT(std::string name = "composite") noexcept
        : PartT<Scalar>(std::move(name)) {}

    // No-op update. Children are driven by the Craft's sense+force pass.
    void update() override {}

    // Construct a child of type ChildT in-place, wire it to this composite,
    // and return a reference to it. Propagates craft_ if it is already set.
    template <typename ChildT, typename... Args>
    ChildT& add(Args&&... args) {
        auto ptr = std::make_unique<ChildT>(std::forward<Args>(args)...);
        ChildT& ref = *ptr;
        ref.parent_ = this;
        propagate_craft(ref, this->craft_);
        children_.push_back(std::move(ptr));
        return ref;
    }

    // Collect child wrenches (after their update()) into wrench_accum_.
    // ArticulatedPart overrides to call resolve() instead.
    virtual void aggregate_wrenches() {
        for (auto& child_ptr : children_) {
            PartT<Scalar>& child = *child_ptr;

            // Recurse first so the child's subtree is aggregated before we drain it.
            if (auto* cp = dynamic_cast<CompositePartT<Scalar>*>(&child)) {
                cp->aggregate_wrenches();
            }

            Wrench<PartFrame, Scalar> cw = child.drain_wrench();

            // Transform child wrench into this (parent) frame.
            using V = geom::Vec3<PartFrame, Scalar>;
            V force_p  = child.transform_.rotate(cw.force());
            V torque_p = child.transform_.rotate(cw.torque())
                       + child.transform_.position().cross(force_p);

            this->wrench_accum_ += Wrench<PartFrame, Scalar>{force_p, torque_p};
        }
    }

    // Recompute mass / MOI / COM from children (lazy; called before integration).
    virtual void compute_params() {
        using EigenV = Eigen::Matrix<Scalar, 3, 1>;
        using EigenM = Eigen::Matrix<Scalar, 3, 3>;

        Scalar total_mass = Scalar(0);
        EigenV com_sum    = EigenV::Zero();

        // Pass 1: recurse, accumulate mass and mass-weighted COM positions.
        for (auto& child_ptr : children_) {
            PartT<Scalar>& child = *child_ptr;
            if (auto* cp = dynamic_cast<CompositePartT<Scalar>*>(&child)) {
                cp->compute_params();
            }

            Scalar m = child.get_mass();
            total_mass += m;

            auto com_in_parent = child.get_transform().apply_position(child.get_com());
            com_sum += m * com_in_parent.raw();
        }

        this->mass_ = total_mass;
        this->com_  = (total_mass > Scalar(0))
                    ? geom::Vec3<PartFrame, Scalar>::from_raw(com_sum / total_mass)
                    : geom::Vec3<PartFrame, Scalar>::zero();

        // Pass 2: accumulate MOI at composite COM via parallel-axis theorem.
        EigenV com_composite = this->com_.raw();
        EigenM I_total = EigenM::Zero();

        for (auto& child_ptr : children_) {
            PartT<Scalar>& child = *child_ptr;
            Scalar m = child.get_mass();
            if (m <= Scalar(0)) continue;

            EigenM R_cp = child.get_transform().orientation().raw().toRotationMatrix();
            EigenM I_rotated = R_cp * child.get_moi().raw() * R_cp.transpose();

            EigenV r = child.get_transform().apply_position(child.get_com()).raw() - com_composite;
            EigenM I_steiner = m * (r.squaredNorm() * EigenM::Identity() - r * r.transpose());

            I_total += I_rotated + I_steiner;
        }

        this->moi_ = geom::Mat3<PartFrame, PartFrame, Scalar>{I_total};
    }

    // No-op overrides: composite mass/MOI/COM come from children.
    void set_mass(Scalar) noexcept override {}
    void set_moi(const geom::Mat3<PartFrame, PartFrame, Scalar>&) noexcept override {}
    void set_com(const geom::Vec3<PartFrame, Scalar>&) noexcept override {}

    std::size_t child_count() const noexcept { return children_.size(); }

protected:
    std::vector<std::unique_ptr<PartT<Scalar>>> children_;

private:
    template <class S> friend class CraftT;

    // Recursively propagate a Craft pointer to a subtree.
    static void propagate_craft(PartT<Scalar>& root, CraftT<Scalar>* craft) {
        root.craft_ = craft;
        auto* cp = dynamic_cast<CompositePartT<Scalar>*>(&root);
        if (!cp) return;
        for (auto& child : cp->children_) {
            propagate_craft(*child, craft);
        }
    }
};

// Backwards-compat alias.
using CompositePart = CompositePartT<Real>;

} // namespace manta
