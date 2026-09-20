#pragma once
// Shared CUDA support for extended descriptor translation units.
#include "mdescriptor/cuda/extended_descriptors.hpp"
#include "mdescriptor/cuda/descriptor_dispatch.hpp"
#include "mdescriptor/cuda/error.hpp"

#include "mdescriptor/detail/mbtr.hpp"
#include "mdescriptor/detail/rotational_bispectrum.hpp"
#include "mdescriptor/matrix.hpp"
#include "mdescriptor/neighbor.hpp"
#include "local_spherical_common.hpp"
#include "rotational_math.hpp"

#include <cuda_runtime.h>

#include <cfloat>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace mdescriptor::cuda {
namespace {

namespace mbtr = mdescriptor::detail::mbtr;

using I32 = std::int32_t;
using I64 = std::int64_t;
using F64Array = py::array_t<double, py::array::c_style | py::array::forcecast>;

constexpr double kPi = 3.141592653589793238462643383279502884;
template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    ~DeviceBuffer() noexcept { clear(); }

    T* get() const noexcept { return data_; }

    void clear() noexcept {
        if (data_ != nullptr) (void)cudaFree(data_);
        data_ = nullptr;
    }

    void allocate(std::size_t count, const char* operation) {
        if (count == 0) return;
        check_cuda(
            cudaMalloc(reinterpret_cast<void**>(&data_), count * sizeof(T)), operation);
    }

    void upload(const T* source, std::size_t count, cudaStream_t stream, const char* operation) {
        allocate(count, operation);
        if (count == 0) return;
        check_cuda(
            cudaMemcpyAsync(data_, source, count * sizeof(T), cudaMemcpyHostToDevice, stream),
            operation);
    }

private:
    T* data_ = nullptr;
};

py::array download_output_with_gil_release(
    CudaExecutionContext& context,
    std::size_t count,
    I64 rows,
    I64 columns) {
    py::array_t<double> result({
        static_cast<py::ssize_t>(rows), static_cast<py::ssize_t>(columns)});
    double* destination = result.mutable_data();
    if (count != 0) {
        // The NumPy array owns the host storage.  Keep Python object access
        // before releasing the GIL, and do the device copy/synchronization
        // without holding it.
        py::gil_scoped_release release;
        context.download_output_into(destination, count);
    }
    return result;
}
template <typename Value>
Value option(const py::dict& options, const char* key, Value fallback) {
    const py::str name(key);
    if (!options.contains(name) || options[name].is_none()) return fallback;
    return py::cast<Value>(options[name]);
}
std::vector<I32> species_option(const py::dict& options) {
    const py::str key("species");
    if (!options.contains(key) || options[key].is_none()) return {};
    return py::cast<std::vector<I32>>(options[key]);
}

std::int64_t feature_count_option(const py::dict& options, std::int64_t fallback) {
    const py::str key("_cuda_feature_count");
    if (!options.contains(key) || options[key].is_none()) return fallback;
    const auto value = py::cast<std::int64_t>(options[key]);
    return value > 0 ? value : fallback;
}

py::list labels_option(const py::dict& options, const std::string& name, std::int64_t width) {
    const py::str key("_cuda_labels");
    if (options.contains(key) && !options[key].is_none()) {
        const py::list configured = py::list(options[key]);
        if (py::len(configured) == width) return configured;
    }
    py::list labels;
    for (std::int64_t index = 0; index < width; ++index) {
        labels.append(name + ":" + std::to_string(index));
    }
    return labels;
}

py::dict metadata(const py::dict& options, const std::string& name) {
    py::dict result;
    result["descriptor"] = name;
    result["backend"] = "mdescriptor-cuda";
    py::dict execution;
    execution["device"] = "cuda";
    const py::str key("execution");
    if (options.contains(key) && !options[key].is_none()) {
        const py::dict configured = py::cast<py::dict>(options[key]);
        execution["num_threads"] = configured.contains("num_threads")
            ? configured["num_threads"] : py::none();
    } else {
        execution["num_threads"] = py::none();
    }
    result["execution"] = execution;
    return result;
}

py::array values_array(
    const std::vector<double>& values,
    std::int64_t rows,
    std::int64_t columns) {
    py::array_t<double> result({
        static_cast<py::ssize_t>(rows), static_cast<py::ssize_t>(columns)});
    if (!values.empty()) std::copy(values.begin(), values.end(), result.mutable_data());
    return result;
}

py::array i64_array(const std::vector<I64>& values) {
    py::array_t<I64> result(values.size());
    if (!values.empty()) std::copy(values.begin(), values.end(), result.mutable_data());
    return result;
}

// Zero a double output buffer on the context stream; the error string is
// per descriptor.
inline void zeroed_output(
    CudaExecutionContext& context,
    double* output,
    std::size_t size,
    const char* operation) {
    check_cuda(
        cudaMemsetAsync(output, 0, size * sizeof(double), context.stream()),
        operation);
}

inline std::vector<I64> host_row_offsets(const detail::StructureBatchView& host_batch) {
    return std::vector<I64>(
        host_batch.offsets, host_batch.offsets + host_batch.structures + 1);
}

py::dict atom_result(
    const py::array& values,
    I64 columns,
    const std::string& name,
    const py::dict& options,
    bool per_system,
    const std::vector<I64>& offsets) {
    py::dict result;
    result["values"] = values;
    result["level"] = per_system ? "structure" : "atom";
    if (!per_system) result["row_offsets"] = i64_array(offsets);
    result["labels"] = labels_option(options, name, columns);
    result["metadata"] = metadata(options, name);
    return result;
}
__device__ int species_index(I32 number, const I32* species, int count);

template <int MaxAngular>
__device__ void harmonic_values(const double* vector, double* output, int requested);
py::dict child_dict(const py::dict& options, const char* key) {
    const py::str name(key);
    if (!options.contains(name) || options[name].is_none()) return py::dict();
    try {
        return py::cast<py::dict>(options[name]);
    } catch (const py::cast_error&) {
        throw std::invalid_argument(std::string(key) + " must be an object");
    }
}

std::vector<double> vector_child(const py::dict& object, const char* key) {
    const py::str name(key);
    if (!object.contains(name) || object[name].is_none()) return {};
    if (py::isinstance<py::array>(object[name])) {
        const auto values = F64Array::ensure(object[name]);
        if (!values) {
            throw std::invalid_argument(std::string(key) + " must be a numeric array");
        }
        return std::vector<double>(
            values.data(), values.data() + static_cast<std::size_t>(values.size()));
    }
    try {
        return py::cast<std::vector<double>>(object[name]);
    } catch (const py::cast_error&) {
        throw std::invalid_argument(std::string(key) + " must be an array of numbers");
    }
}
std::vector<double> numeric_vector_option(
    const py::dict& options, const char* key, double fallback, std::size_t count) {
    const py::str name(key);
    if (!options.contains(name) || options[name].is_none()) {
        return std::vector<double>(count, fallback);
    }
    try {
        if (py::isinstance<py::sequence>(options[name])
            && !py::isinstance<py::str>(options[name])) {
            auto values = py::cast<std::vector<double>>(options[name]);
            if (values.size() != count) {
                throw std::invalid_argument(std::string(key) + " must have one value per species");
            }
            return values;
        }
        return std::vector<double>(count, py::cast<double>(options[name]));
    } catch (const py::cast_error&) {
        throw std::invalid_argument(std::string(key) + " must be a number or numeric array");
    }
}

std::vector<double> species_dictionary_values(
    const py::object& object,
    const std::vector<I32>& species,
    double fallback) {
    std::vector<double> result(species.size(), fallback);
    if (object.is_none()) return result;
    const py::dict values = py::cast<py::dict>(object);
    static const char* const symbols[] = {
        "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
        "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
        "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
        "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
        "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
        "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
        "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
        "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
        "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
        "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
        "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
        "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
    };
    for (std::size_t index = 0; index < species.size(); ++index) {
        const py::int_ number(species[index]);
        if (values.contains(number)) {
            result[index] = py::cast<double>(values[number]);
            continue;
        }
        const py::str text(std::to_string(species[index]));
        if (values.contains(text)) {
            result[index] = py::cast<double>(values[text]);
            continue;
        }
        if (species[index] >= 1 && species[index] <= 118) {
            const py::str symbol(symbols[species[index] - 1]);
            if (values.contains(symbol)) result[index] = py::cast<double>(values[symbol]);
        }
    }
    return result;
}

__device__ int species_index(I32 number, const I32* species, int count) {
    for (int index = 0; index < count; ++index) {
        if (species[index] == number) return index;
    }
    return -1;
}
template <int MaxAngular>
__device__ void harmonic_values(const double* vector, double* output, int requested) {
    const int max_angular = requested;
    double legendre[(MaxAngular + 1) * (MaxAngular + 2) / 2]{};
    const double norm = sqrt(vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2]);
    double direction[3] = {vector[0], vector[1], vector[2]};
    if (norm < 1e-6) {
        direction[0] = 0.0;
        direction[1] = 0.0;
        direction[2] = 1.0;
    } else {
        direction[0] /= norm;
        direction[1] /= norm;
        direction[2] /= norm;
    }
    auto legendre_index = [](int angular, int m) {
        return m + angular * (angular + 1) / 2;
    };
    constexpr double sqrt_1_over_2pi = 0.398942280401432677939946059934;
    constexpr double sqrt_3 = 1.732050807568877293527446341505872;
    constexpr double sqrt_3_over_2 = 1.224744871391589049098642;
    const double xy = hypot(direction[0], direction[1]);
    const double cos_theta = direction[2];
    const double sin_theta = xy;
    legendre[legendre_index(0, 0)] = sqrt_1_over_2pi;
    double value = -sqrt_3_over_2 * sin_theta * sqrt_1_over_2pi;
    if (max_angular > 0) {
        legendre[legendre_index(1, 0)] = cos_theta * sqrt_3 * sqrt_1_over_2pi;
        legendre[legendre_index(1, 1)] = value;
        for (int angular = 2; angular <= max_angular; ++angular) {
            for (int m = 0; m < angular - 1; ++m) {
                const double ls = static_cast<double>(angular * angular);
                const double lm1s = static_cast<double>((angular - 1) * (angular - 1));
                const double ms = static_cast<double>(m * m);
                const double a = sqrt((4.0 * ls - 1.0) / (ls - ms));
                const double b = -sqrt((lm1s - ms) / (4.0 * lm1s - 1.0));
                legendre[legendre_index(angular, m)] = a * (
                    cos_theta * legendre[legendre_index(angular - 1, m)]
                    + b * legendre[legendre_index(angular - 2, m)]);
            }
            legendre[legendre_index(angular, angular - 1)] = cos_theta
                * sqrt(2.0 * angular + 1.0) * value;
            value *= -sqrt(1.0 + 0.5 / angular) * sin_theta;
            legendre[legendre_index(angular, angular)] = value;
        }
    }
    for (int angular = 0; angular <= max_angular; ++angular) {
        output[angular * angular + angular] =
            legendre[legendre_index(angular, 0)] / 1.414213562373095048801688724209698079;
    }
    const double cos_phi = xy > DBL_EPSILON ? direction[0] / xy : 1.0;
    const double sin_phi = xy > DBL_EPSILON ? direction[1] / xy : 0.0;
    double cos_previous = 1.0;
    double sin_previous = 0.0;
    double cos_current = -cos_phi;
    double sin_current = sin_phi;
    const double minus_two_cos = -2.0 * cos_phi;
    for (int m = 1; m <= max_angular; ++m) {
        const double sin_m = minus_two_cos * sin_previous - sin_current;
        const double cos_m = minus_two_cos * cos_previous - cos_current;
        sin_current = sin_previous;
        sin_previous = sin_m;
        cos_current = cos_previous;
        cos_previous = cos_m;
        for (int angular = m; angular <= max_angular; ++angular) {
            output[angular * angular + angular + m] =
                legendre[legendre_index(angular, m)] * cos_m;
            output[angular * angular + angular - m] =
                legendre[legendre_index(angular, m)] * sin_m;
        }
    }
}

__device__ double positive_hypergeometric(double a, double b, double x) {
    if (x > 30.0) {
        double sum = 1.0;
        double term = 1.0;
        for (int index = 1; index <= 30; ++index) {
            term = -term * (b - a + index - 1.0) * (a - index) / (x * index);
            sum += term;
        }
        return sum;
    }
    double sum = 1.0;
    double term = 1.0;
    for (int index = 1; index <= 500; ++index) {
        term *= (a + index - 1.0) * x / ((b + index - 1.0) * index);
        sum += term;
        if (fabs(term) <= fabs(sum) * 2e-15) break;
    }
    return sum;
}
__device__ double cell_volume_device(const double* cell) {
    return mdescriptor::detail::mbtr::cell_volume(cell);
}
__device__ int pair_channel_device(int first, int second, int species_count) {
    return mdescriptor::detail::mbtr::pair_channel(first, second, species_count);
}

I64 payload_or_option_feature_count(
    const py::dict& options, I64 fallback, const std::string& name) {
    const I64 value = feature_count_option(options, fallback);
    if (value <= 0) throw std::invalid_argument(name + " has no CUDA feature layout");
    return value;
}

std::vector<I32> batch_species(const detail::StructureBatchView& batch) {
    std::vector<I32> result;
    for (I64 atom = 0; atom < batch.atoms; ++atom) {
        if (std::find(result.begin(), result.end(), batch.numbers[atom]) == result.end()) {
            result.push_back(batch.numbers[atom]);
        }
    }
    std::sort(result.begin(), result.end());
    return result;
}
using DeviceComplex = mdescriptor::detail::rotational::Complex;
using mdescriptor::detail::rotational::complex_conjugate;
using mdescriptor::detail::rotational::complex_scale;
// Complex spherical harmonics for the SO3 (Stride = 9) and ACE (Stride = 21)
// paths.  Stride sizes the local Legendre scratch so the small SO3
// instantiation keeps its register footprint.
template <int Stride>
__device__ void complex_spherical_harmonics_device(
    const double* vector,
    int max_angular,
    DeviceComplex* output) {
    double legendre[Stride * Stride]{};
    const double radius = sqrt(
        vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2]);
    if (radius <= 1e-14) {
        output[0] = {0.5 / sqrt(kPi), 0.0};
        return;
    }
    const double cos_theta = vector[2] / radius;
    const double sin_theta = hypot(vector[0], vector[1]) / radius;
    legendre[0] = 1.0;
    for (int m = 1; m <= max_angular; ++m) {
        legendre[m * Stride + m] = -(2.0 * m - 1.0) * sin_theta
            * legendre[(m - 1) * Stride + (m - 1)];
    }
    for (int m = 0; m < max_angular; ++m) {
        legendre[(m + 1) * Stride + m] = (2.0 * m + 1.0) * cos_theta
            * legendre[m * Stride + m];
        for (int angular = m + 2; angular <= max_angular; ++angular) {
            legendre[angular * Stride + m] = (
                (2.0 * angular - 1.0) * cos_theta
                    * legendre[(angular - 1) * Stride + m]
                - (angular + m - 1.0) * legendre[(angular - 2) * Stride + m])
                / (angular - m);
        }
    }
    const double phi = atan2(vector[1], vector[0]);
    const double cos_phi = cos(phi);
    const double sin_phi = sin(phi);
    double cos_m = 1.0;
    double sin_m = 0.0;
    for (int m = 0; m <= max_angular; ++m) {
        if (m > 0) {
            const double next_cos = cos_m * cos_phi - sin_m * sin_phi;
            const double next_sin = sin_m * cos_phi + cos_m * sin_phi;
            cos_m = next_cos;
            sin_m = next_sin;
        }
        for (int angular = m; angular <= max_angular; ++angular) {
            const double normalization = sqrt(
                (2.0 * angular + 1.0) / (4.0 * kPi)
                * tgamma(static_cast<double>(angular - m) + 1.0)
                / tgamma(static_cast<double>(angular + m) + 1.0));
            const double scale = normalization * legendre[angular * Stride + m];
            const DeviceComplex positive = {scale * cos_m, scale * sin_m};
            output[angular * angular + angular + m] = positive;
            if (m > 0) {
                output[angular * angular + angular - m] =
                    m % 2 == 0 ? complex_conjugate(positive)
                               : complex_scale(complex_conjugate(positive), -1.0);
            }
        }
    }
}

// Close the anonymous namespace so includer code starts inside
// namespace mdescriptor::cuda (which stays open for the translation unit).
} // namespace
