#pragma once

#include <cmath>
#include <type_traits>

#include "../core/frame.hpp"
#include "../core/types.hpp"
#include "ori.hpp"
#include "static_link.hpp"
#include "vec3.hpp"

namespace manta::geom {

// Map an axis-angle vector aa = θ·n̂ to its unit quaternion exp(aa/2),
// stored in Hamilton (w, x, y, z) order matching Eigen::Quaternion's
// constructor. Autodiff-safe at the origin via a Taylor-series fallback
// (exact match for value and first derivatives at θ=0). The naive form
// `axis = aa/|aa|` is undefined at zero; this form expresses the result
// as `(cos(θ/2), sin(θ/2)/θ · aa)`, with the scalar `sin(θ/2)/θ → 1/2`
// substituted at the singular point.
//
// Same algorithm as Ceres' `AngleAxisToQuaternion`; reimplemented here
// to keep `<ceres/rotation.h>` out of the manta header chain (its DCHECK
// usage clashes with doctest's `CHECK_NE` macro in test TUs).
template <typename T>
inline Eigen::Quaternion<T> angle_axis_to_quat(
        const Eigen::Matrix<T, 3, 1>& aa) noexcept {
    using std::cos;
    using std::sin;
    using std::sqrt;
    const T a0 = aa(0), a1 = aa(1), a2 = aa(2);
    const T theta_sq = a0 * a0 + a1 * a1 + a2 * a2;
    // The branch is value-conditional, but the two arms agree on value AND
    // first derivative at θ = 0, so autodiff is correct on either side.
    if (theta_sq > T(0)) {
        const T theta      = sqrt(theta_sq);
        const T half_theta = theta * T(0.5);
        const T k          = sin(half_theta) / theta;       // → 0.5 as θ → 0
        return Eigen::Quaternion<T>(cos(half_theta), a0 * k, a1 * k, a2 * k);
    }
    // Taylor at the origin: cos(θ/2)→1, sin(θ/2)/θ→0.5.
    const T half(0.5);
    return Eigen::Quaternion<T>(T(1), a0 * half, a1 * half, a2 * half);
}

// A transform between two MOVING frames. Carries position, orientation,
// linear and angular velocity, and linear and angular acceleration.
//
// Conventions (matching the design plan):
//   position    : To-origin expressed in From
//   orientation : To-frame attitude with components in From
//   vel_linear  : velocity of To-origin relative to From, in From frame
//   vel_angular : angular velocity of To relative to From, in To frame
//   acc_linear  : in From frame
//   acc_angular : in To frame
template <typename From, typename To, typename Scalar = MFloat>
class KinematicLink {
public:
    using PosF = Vec3<From, Scalar>;
    using PosT = Vec3<To,   Scalar>;
    using Att  = Ori<From,  Scalar>;

    constexpr KinematicLink() noexcept = default;

    KinematicLink(PosF position, Att orientation,
                  PosF vel_linear, PosT vel_angular,
                  PosF acc_linear = PosF::zero(),
                  PosT acc_angular = PosT::zero()) noexcept
        : position_(position), orientation_(orientation),
          vel_linear_(vel_linear),  vel_angular_(vel_angular),
          acc_linear_(acc_linear),  acc_angular_(acc_angular) {}

    static KinematicLink identity() noexcept {
        return KinematicLink{PosF::zero(), Att::identity(),
                             PosF::zero(), PosT::zero(),
                             PosF::zero(), PosT::zero()};
    }

    // Promote a static link to a kinematic one with zero motion.
    static KinematicLink from_static(const StaticLink<From, To, Scalar>& s) noexcept {
        return KinematicLink{s.position(), s.orientation(),
                             PosF::zero(), PosT::zero(),
                             PosF::zero(), PosT::zero()};
    }

    const PosF& position()             const noexcept { return position_; }
    const Att&  orientation()          const noexcept { return orientation_; }
    const PosF& vel_linear()           const noexcept { return vel_linear_; }
    const PosT& vel_angular()          const noexcept { return vel_angular_; }
    const PosF& acc_linear()           const noexcept { return acc_linear_; }
    const PosT& acc_angular()          const noexcept { return acc_angular_; }

    void set_position(const PosF& p)     noexcept { position_    = p; }
    void set_orientation(const Att& q)   noexcept { orientation_ = q; }
    void set_vel_linear(const PosF& v)   noexcept { vel_linear_  = v; }
    void set_vel_angular(const PosT& w)  noexcept { vel_angular_ = w; }
    void set_acc_linear(const PosF& a)   noexcept { acc_linear_  = a; }
    void set_acc_angular(const PosT& al) noexcept { acc_angular_ = al; }

    // Reinterpret the To type tag while keeping raw data unchanged. Used when
    // two frame tags denote the same physical frame (e.g. root PartFrame ≡ CraftFrame).
    template<typename NewTo>
    KinematicLink<From, NewTo, Scalar> reinterpret_to_frame() const noexcept {
        return KinematicLink<From, NewTo, Scalar>{
            Vec3<From,   Scalar>::from_raw(position_.raw()),
            Ori<From,    Scalar>{orientation_.raw()},
            Vec3<From,   Scalar>::from_raw(vel_linear_.raw()),
            Vec3<NewTo,  Scalar>::from_raw(vel_angular_.raw()),
            Vec3<From,   Scalar>::from_raw(acc_linear_.raw()),
            Vec3<NewTo,  Scalar>::from_raw(acc_angular_.raw()),
        };
    }

    // Transform a stationary point from To-frame to From-frame.
    PosF apply_position(const PosT& p_to) const noexcept {
        return PosF::from_raw(orientation_.raw() * p_to.raw() + position_.raw());
    }

    // Pure rotation of a vector from To to From (no position offset).
    PosF rotate(const PosT& v_to) const noexcept {
        return PosF::from_raw(orientation_.raw() * v_to.raw());
    }
    PosT rotate_inverse(const PosF& v_from) const noexcept {
        return PosT::from_raw(orientation_.raw().conjugate() * v_from.raw());
    }

    // Velocity of a point that is stationary in To, expressed in From.
    // = v_linear + omega x (R * p_to)
    PosF apply_velocity_of_static_point(const PosT& point_in_To) const noexcept {
        auto p_from   = orientation_.raw() * point_in_To.raw();
        auto omega_F  = orientation_.raw() * vel_angular_.raw();
        auto v_from   = vel_linear_.raw() + omega_F.cross(p_from);
        return PosF::from_raw(v_from);
    }

    // Composition: KL<A,B> * KL<B,C> -> KL<A,C>.
    // Implements the chain rule for position, velocity, and acceleration.
    template <typename C>
    KinematicLink<From, C, Scalar> operator*(const KinematicLink<To, C, Scalar>& bc) const noexcept {
        using EigenV = Eigen::Matrix<Scalar, 3, 1>;

        // Position and orientation compose like static transforms.
        auto    q_AC       = orientation_.raw() * bc.orientation().raw();
        EigenV  p_BC_in_A  = orientation_.raw() * bc.position().raw();
        EigenV  p_AC       = p_BC_in_A + position_.raw();

        // Linear velocity in A: v_AB + omega_AB x (R_AB * p_BC) + R_AB * v_BC
        EigenV  omega_AB_A = orientation_.raw() * vel_angular_.raw();
        EigenV  v_BC_in_A  = orientation_.raw() * bc.vel_linear().raw();
        EigenV  v_AC_A     = vel_linear_.raw() + omega_AB_A.cross(p_BC_in_A) + v_BC_in_A;

        // Angular velocity in C: omega_AC^C = R_CB * omega_AB^B + omega_BC^C.
        // omega_AB in B-frame is just vel_angular_ (already in To=B by convention).
        EigenV  omega_AB_C = bc.orientation().raw().conjugate() * vel_angular_.raw();
        EigenV  omega_AC_C = omega_AB_C + bc.vel_angular().raw();

        // Linear acceleration of C-origin in A: full Coriolis/centripetal:
        //   a_AC^A = a_AB^A + alpha_AB^A x r + omega x (omega x r)
        //          + 2*omega x v_BC^A + a_BC^A
        // where r = R_AB * p_BC^B = p_BC_in_A.
        EigenV  alpha_AB_A = orientation_.raw() * acc_angular_.raw();
        EigenV  a_BC_in_A  = orientation_.raw() * bc.acc_linear().raw();
        EigenV  a_AC_A     = acc_linear_.raw()
                           + alpha_AB_A.cross(p_BC_in_A)
                           + omega_AB_A.cross(omega_AB_A.cross(p_BC_in_A))
                           + Scalar(2) * omega_AB_A.cross(v_BC_in_A)
                           + a_BC_in_A;

        // Angular acceleration in C: differentiating omega_AC^C gives
        //   alpha_AC^C = alpha_AB^C + alpha_BC^C - omega_BC^C x omega_AB^C
        // (the last term comes from dR_CB/dt acting on omega_AB^B).
        EigenV  alpha_AB_C = bc.orientation().raw().conjugate() * acc_angular_.raw();
        EigenV  omega_BC_C = bc.vel_angular().raw();
        EigenV  alpha_AC_C = alpha_AB_C + bc.acc_angular().raw()
                           - omega_BC_C.cross(omega_AB_C);

        return KinematicLink<From, C, Scalar>{
            Vec3<From, Scalar>::from_raw(p_AC),
            Ori<From,  Scalar>{q_AC},
            Vec3<From, Scalar>::from_raw(v_AC_A),
            Vec3<C,    Scalar>::from_raw(omega_AC_C),
            Vec3<From, Scalar>::from_raw(a_AC_A),
            Vec3<C,    Scalar>::from_raw(alpha_AC_C),
        };
    }

    // Mixed composition with a StaticLink on the right.
    template <typename C>
    KinematicLink<From, C, Scalar> operator*(const StaticLink<To, C, Scalar>& bc) const noexcept {
        return *this * KinematicLink<To, C, Scalar>::from_static(bc);
    }

    // Inverse of a moving transform.
    //
    // Given KL<A,B> with R_AB, p_AB (in A), v_AB (in A), omega_AB (in B),
    // the inverse KL<B,A> has:
    //   R_BA       = R_AB^T
    //   p_BA       = -R_BA * p_AB                            (in B)
    //   omega_BA   = -omega_AB^A                             (in A; the new
    //                                                         dest frame)
    //   v_BA       = -R_BA * (v_AB - omega_AB^A x p_AB)      (in B)
    //
    // Derivation: position of A-origin in B is p_BA^B = -R_BA * p_AB^A. Time-
    // differentiating with the kinematic transport theorem and using
    // dR_BA/dt = -S(omega_AB^B) * R_BA gives the formula above.
    //
    // Accelerations are intentionally zeroed here. Inverting accelerations
    // across moving frames adds Coriolis/centripetal coupling we don't need
    // anywhere yet. Replace when a use case appears.
    KinematicLink<To, From, Scalar> inverse() const noexcept {
        using EigenV = Eigen::Matrix<Scalar, 3, 1>;
        auto   R_AB       = orientation_.raw().toRotationMatrix();
        auto   R_BA       = R_AB.transpose();
        auto   p_BA       = -(R_BA * position_.raw());
        EigenV omega_AB_A = R_AB * vel_angular_.raw();
        EigenV v_BA       = -(R_BA * (vel_linear_.raw() - omega_AB_A.cross(position_.raw())));
        auto   omega_BA_A = -omega_AB_A;

        return KinematicLink<To, From, Scalar>{
            Vec3<To,   Scalar>::from_raw(p_BA),
            Ori<To,    Scalar>{typename Ori<To, Scalar>::QuatT{R_BA}},
            Vec3<To,   Scalar>::from_raw(v_BA),
            Vec3<From, Scalar>::from_raw(omega_BA_A),
            Vec3<To,   Scalar>::zero(),
            Vec3<From, Scalar>::zero(),
        };
    }

    // RK4 integration of the link's pose under its current velocities and
    // accelerations. Updates position, orientation, and velocities in place.
    void update(Scalar dt) noexcept;

private:
    PosF position_;
    Att  orientation_ = Att::identity();
    PosF vel_linear_;
    PosT vel_angular_;
    PosF acc_linear_;
    PosT acc_angular_;
};

// Single-evaluation, 2nd-order, symplectic-flavored integrator.
// Per tick: acceleration is held constant over [t, t+dt] (the dynamics pass
// evaluated it once at the start of the tick), and:
//   p(t+dt) = p + v*dt + 0.5*a*dt^2     (= midpoint-velocity update; exact for constant a)
//   v(t+dt) = v + a*dt                  (explicit Euler)
//   q(t+dt) = q ⊗ exp((ω + 0.5*α*dt) * dt)   (midpoint angular velocity, exponential map)
//   ω(t+dt) = ω + α*dt
//
// NOT RK4 (despite an earlier comment). RK4 would re-evaluate the dynamics
// function 4× per tick, quadrupling the autodiff cost on the EKF predict hot
// path for 4th-order accuracy that's invisible at typical robotics timesteps
// (1–10 ms). RK4 is also non-symplectic, so it dissipates/accumulates energy
// over long horizons. Single-eval midpoint-velocity is the right trade for
// auto-diffable craft models used in EKFs: cheap autodiff, stable orbits at
// 2nd order, no implicit solve.
//
// Upgrade path if/when long-horizon orbital accuracy is needed: Velocity
// Verlet (2 dynamics evals per tick) — true symplectic 2nd-order. Drop in
// here behind a compile-time switch.
//
// Quaternion exp at ω=0: the (angle, axis = ω/|ω|) form has a 0/0 axis at
// zero, so the previous implementation took a value-conditional branch to
// identity — which silently zeroed out the q↔ω coupling in the predict
// Jacobian whenever a craft was at rest. This version uses
// `angle_axis_to_quat` (Taylor-safe at the origin) so first derivatives
// match the analytical Lie generator dt/2·e_i at ω=0.
template <typename From, typename To, typename Scalar>
inline void KinematicLink<From, To, Scalar>::update(Scalar dt) noexcept {
    using EigenV = Eigen::Matrix<Scalar, 3, 1>;
    using QuatT  = typename Att::QuatT;

    auto v0 = vel_linear_.raw();
    auto a0 = acc_linear_.raw();
    auto w0 = vel_angular_.raw();
    auto al = acc_angular_.raw();

    auto p_new = position_.raw() + v0 * dt + Scalar(0.5) * a0 * dt * dt;
    auto v_new = v0 + a0 * dt;

    // Midpoint angular velocity → axis-angle delta over [t, t+dt].
    EigenV  aa = (w0 + Scalar(0.5) * al * dt) * dt;
    QuatT   dq = angle_axis_to_quat<Scalar>(aa);

    // q ⊗ dq is unit by construction. Trailing renormalize is a guard
    // against numerical drift across long sim runs; at unit q its
    // Jacobian is the tangent projection (I − qq^T), which is the right
    // thing when feeding back into a 4-DOF state vector.
    QuatT q_new = orientation_.raw() * dq;
    q_new.normalize();

    auto w_new = w0 + al * dt;

    position_    = PosF::from_raw(p_new, position_.id());
    orientation_ = Att{q_new, orientation_.id()};
    vel_linear_  = PosF::from_raw(v_new, vel_linear_.id());
    vel_angular_ = PosT::from_raw(w_new, vel_angular_.id());
}

} // namespace manta::geom
