#pragma once

// Measurement — a typed handle to a sensor's h(x) cache + R source.
// =================================================================
//
// A `Measurement` is the contract a sensor part exposes to the EKF/UKF.
// It bundles three things, behind a non-templated façade:
//
//   * `h(x)` storage — a Vec<Scalar, Dim> the part populates in its
//     `update()`. The Scalar varies (double on the value world,
//     ceres::Jet on the Jet shadow); the Measurement holds a void*
//     that the function-pointer reads below interpret correctly.
//   * `R` σ source — a pointer to the part's noise-σ field, read live
//     each update so state-dependent noise (e.g. temperature-modulated
//     IMU σ) propagates into R automatically.
//   * Freshness flag — a pointer to a bool that the part's `update()`
//     sets/clears via its rate gate.
//
// Construction is via `make_measurement<Scalar, Dim>` which stamps the
// Scalar-aware function pointers onto the storage. The Measurement
// itself is *not* templated on Scalar — both `IMUT<double>::accel` and
// `IMUT<Jet>::accel` are `Measurement` instances of the same type, but
// their function pointers know which Scalar to interpret the storage
// as.
//
// Sensor parts expose their Measurements as public members
// (`imu.accel`, `imu.gyro`) and register them in PartT's non-virtual
// measurements_ vector during construction. The EKF resolves Jet-side
// counterparts at bind time by walking the Jet world's part tree and
// matching by name.

#include <Eigen/Core>
#include <ceres/jet.h>
#include <string>
#include <type_traits>

#include "../core/noise.hpp"   // is_ceres_jet_v

namespace manta {

class Measurement {
public:
    // ---- Public surface ----
    int          dim          = 0;
    std::string  name;
    void*        h_storage    = nullptr;   // → owner's Vec<Scalar, Dim>
    const float* sigma_storage = nullptr;
    const bool*  fresh_storage = nullptr;

    // Type-erased reads, stamped at construction with the owner's Scalar.
    void (*read_a_)(const void* h_storage, double* out, int dim) = nullptr;
    void (*read_v_)(const void* h_storage, double* out_row,
                    int row, int jet_width) = nullptr;

    Measurement() noexcept = default;
    Measurement(int dim_,
                std::string name_,
                void* h_storage_,
                const float* sigma_storage_,
                const bool* fresh_storage_,
                void (*read_a)(const void*, double*, int),
                void (*read_v)(const void*, double*, int, int)) noexcept
        : dim(dim_),
          name(std::move(name_)),
          h_storage(h_storage_),
          sigma_storage(sigma_storage_),
          fresh_storage(fresh_storage_),
          read_a_(read_a),
          read_v_(read_v) {}

    // R diagonal: σ from the live owner field. State-dependent σ flows
    // through automatically because we always read it at update time.
    float r_sigma() const noexcept { return sigma_storage ? *sigma_storage : 0.0f; }

    // Freshness: true on ticks where the owner's update() refreshed
    // the h(x) cache. The EKF reads non-destructively; the owner's
    // next update() resets the flag.
    bool fresh() const noexcept { return fresh_storage ? *fresh_storage : true; }

    // Read h(x).a into a double buffer of length `dim`.
    void read_value(double* out) const noexcept {
        if (read_a_) read_a_(h_storage, out, dim);
    }

    // Read row `row` of H (= ∂h/∂tangent) into a double buffer of length
    // `jet_width`. The owner's storage must be Jet-typed for this to
    // do anything; on value-typed storage it's a no-op.
    void read_jacobian_row(int row, double* out, int jet_width) const noexcept {
        if (read_v_) read_v_(h_storage, out, row, jet_width);
    }

    bool is_jet() const noexcept { return read_v_ != nullptr; }
};

// ---------------------------------------------------------------------
// Factory: stamp the Scalar-aware function pointers.
// ---------------------------------------------------------------------
template <class Scalar, int Dim>
inline Measurement make_measurement(std::string name,
                                     Eigen::Matrix<Scalar, Dim, 1>* h_storage,
                                     const float* sigma_storage,
                                     const bool*  fresh_storage) noexcept {
    auto read_a = +[](const void* storage, double* out, int dim) {
        const auto* v =
            static_cast<const Eigen::Matrix<Scalar, Dim, 1>*>(storage);
        for (int i = 0; i < dim; ++i) {
            if constexpr (std::is_floating_point_v<Scalar>) {
                out[i] = static_cast<double>((*v)(i));
            } else if constexpr (is_ceres_jet_v<Scalar>) {
                out[i] = (*v)(i).a;
            }
        }
    };

    using read_v_t = void (*)(const void*, double*, int, int);
    read_v_t read_v = nullptr;
    if constexpr (is_ceres_jet_v<Scalar>) {
        read_v = +[](const void* storage, double* out_row,
                     int row, int jet_width) {
            const auto* v =
                static_cast<const Eigen::Matrix<Scalar, Dim, 1>*>(storage);
            for (int j = 0; j < jet_width; ++j) {
                out_row[j] = (*v)(row).v[j];
            }
        };
    }

    return Measurement{
        Dim,
        std::move(name),
        static_cast<void*>(h_storage),
        sigma_storage,
        fresh_storage,
        read_a,
        read_v,
    };
}

// MeasurementHandle<Dim> — Measurement tagged with a compile-time dim.
// =====================================================================
//
// Lets `reading_from(handle)` and `ekf.measure(handle, reading)` deduce
// Dim from the field type instead of forcing the user to spell `<3>` at
// every call site. Storage and Scalar erasure are unchanged — Dim is
// the only piece liftable to compile time without breaking double/Jet
// polymorphism. Parts hold their public measurement fields as
// `MeasurementHandle<3>` etc.; everything that works on a
// `Measurement*` upcasts transparently.
template <int Dim>
class MeasurementHandle : public Measurement {
public:
    static constexpr int kDim = Dim;
    MeasurementHandle() noexcept = default;
    MeasurementHandle(Measurement&& m) noexcept : Measurement(std::move(m)) {}
    MeasurementHandle(const Measurement& m) : Measurement(m) {}
    MeasurementHandle& operator=(Measurement&& m) noexcept {
        Measurement::operator=(std::move(m));
        return *this;
    }
};

} // namespace manta
