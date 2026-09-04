#pragma once
// Shared CUDA support and SOAP/local kernels for extended descriptor TUs.
#include "mdescriptor/cuda/extended_descriptors.hpp"
#include "mdescriptor/cuda/descriptor_dispatch.hpp"
#include "mdescriptor/cuda/error.hpp"

#include "mdescriptor/detail/mbtr.hpp"
#include "mdescriptor/detail/rotational_bispectrum.hpp"
#include "mdescriptor/matrix.hpp"
#include "mdescriptor/neighbor.hpp"
#include "local_spherical_common.hpp"

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
constexpr int kMatrixKindSine = static_cast<int>(::mdescriptor::MatrixKind::Sine);
constexpr int kMatrixKindEwald = static_cast<int>(::mdescriptor::MatrixKind::Ewald);
constexpr int kMatrixKindCoulomb = static_cast<int>(::mdescriptor::MatrixKind::Coulomb);
constexpr int kMatrixPermutationNone = static_cast<int>(::mdescriptor::MatrixPermutation::None);
constexpr int kMatrixPermutationSortedL2 = static_cast<int>(::mdescriptor::MatrixPermutation::SortedL2);
constexpr int kMatrixPermutationEigenspectrum = static_cast<int>(::mdescriptor::MatrixPermutation::Eigenspectrum);
constexpr int kRotationalUCapacity = static_cast<int>(
    mdescriptor::detail::rotational::u_total_size(10));

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

template <typename T>
std::vector<T> download(
    const T* source,
    std::size_t count,
    CudaExecutionContext& context,
    const char* operation) {
    std::vector<T> result(count);
    if (count != 0) {
        // CUDA work is asynchronous.  Let another Python thread cancel the
        // public ComputeControl while this stream waits for the device.
        py::gil_scoped_release release;
        check_cuda(
            cudaMemcpyAsync(
                result.data(), source, count * sizeof(T), cudaMemcpyDeviceToHost,
                context.stream()),
            operation);
        context.synchronize();
    }
    return result;
}

std::vector<double> download_output_with_gil_release(
    CudaExecutionContext& context,
    std::size_t count) {
    py::gil_scoped_release release;
    return context.download_output(count);
}

std::vector<double> inverse_symmetric_sqrt_host(
    const std::vector<double>& matrix,
    int size) {
    std::vector<double> values = matrix;
    std::vector<double> vectors(static_cast<std::size_t>(size * size), 0.0);
    for (int index = 0; index < size; ++index) {
        vectors[static_cast<std::size_t>(index * size + index)] = 1.0;
    }
    for (int iteration = 0; iteration < 100 * size * size; ++iteration) {
        int p = 0;
        int q = size > 1 ? 1 : 0;
        double largest = 0.0;
        for (int row = 0; row < size; ++row) {
            for (int column = row + 1; column < size; ++column) {
                const double candidate = std::abs(
                    values[static_cast<std::size_t>(row * size + column)]);
                if (candidate > largest) {
                    largest = candidate;
                    p = row;
                    q = column;
                }
            }
        }
        if (largest < 1e-15) break;
        const double angle = 0.5 * std::atan2(
            2.0 * values[static_cast<std::size_t>(p * size + q)],
            values[static_cast<std::size_t>(q * size + q)]
                - values[static_cast<std::size_t>(p * size + p)]);
        const double cosine = std::cos(angle);
        const double sine = std::sin(angle);
        for (int row = 0; row < size; ++row) {
            const double row_p = values[static_cast<std::size_t>(row * size + p)];
            const double row_q = values[static_cast<std::size_t>(row * size + q)];
            values[static_cast<std::size_t>(row * size + p)] = cosine * row_p - sine * row_q;
            values[static_cast<std::size_t>(row * size + q)] = sine * row_p + cosine * row_q;
        }
        for (int column = 0; column < size; ++column) {
            const double column_p = values[static_cast<std::size_t>(p * size + column)];
            const double column_q = values[static_cast<std::size_t>(q * size + column)];
            values[static_cast<std::size_t>(p * size + column)] = cosine * column_p - sine * column_q;
            values[static_cast<std::size_t>(q * size + column)] = sine * column_p + cosine * column_q;
        }
        for (int row = 0; row < size; ++row) {
            const double row_p = vectors[static_cast<std::size_t>(row * size + p)];
            const double row_q = vectors[static_cast<std::size_t>(row * size + q)];
            vectors[static_cast<std::size_t>(row * size + p)] = cosine * row_p - sine * row_q;
            vectors[static_cast<std::size_t>(row * size + q)] = sine * row_p + cosine * row_q;
        }
    }
    std::vector<double> result(static_cast<std::size_t>(size * size), 0.0);
    for (int row = 0; row < size; ++row) {
        for (int column = 0; column < size; ++column) {
            for (int eigen = 0; eigen < size; ++eigen) {
                const double eigenvalue = values[static_cast<std::size_t>(eigen * size + eigen)];
                if (!(eigenvalue > 0.0) || !std::isfinite(eigenvalue)) {
                    throw std::invalid_argument("SO3 radial overlap matrix is not positive definite");
                }
                result[static_cast<std::size_t>(row * size + column)] +=
                    vectors[static_cast<std::size_t>(row * size + eigen)]
                    * vectors[static_cast<std::size_t>(column * size + eigen)]
                    / std::sqrt(eigenvalue);
            }
        }
    }
    return result;
}

std::vector<double> so3_basis_host(
    int nmax,
    int lmax,
    double cutoff,
    double alpha,
    int* quadrature_count) {
    std::vector<double> overlap(static_cast<std::size_t>(nmax * nmax), 0.0);
    for (int first = 1; first <= nmax; ++first) {
        for (int second = 1; second <= nmax; ++second) {
            overlap[static_cast<std::size_t>((first - 1) * nmax + second - 1)] = std::sqrt(
                (2.0 * first + 5.0) * (2.0 * first + 6.0) * (2.0 * first + 7.0)
                * (2.0 * second + 5.0) * (2.0 * second + 6.0) * (2.0 * second + 7.0))
                / ((5.0 + first + second) * (6.0 + first + second)
                    * (7.0 + first + second));
        }
    }
    const auto inverse_sqrt = inverse_symmetric_sqrt_host(overlap, nmax);
    const int quadrature = (nmax + lmax + 1) * 10;
    if (quadrature_count != nullptr) *quadrature_count = quadrature;
    std::vector<double> basis(static_cast<std::size_t>(nmax * quadrature), 0.0);
    for (int q_index = 0; q_index < quadrature; ++q_index) {
        const double x = std::cos(
            (2.0 * (q_index + 1) - 1.0) * kPi / (2.0 * quadrature));
        const double radius = cutoff * 0.5 * (x + 1.0);
        const double weight = (kPi / quadrature) * cutoff * 0.5;
        const double common = radius * radius * std::exp(-alpha * radius * radius)
            * std::sqrt(std::max(0.0, 1.0 - x * x)) * weight;
        for (int radial = 0; radial < nmax; ++radial) {
            double value = 0.0;
            for (int exponent = 1; exponent <= nmax; ++exponent) {
                const double phi = std::pow(cutoff - radius, exponent + 2.0) / std::sqrt(
                    2.0 * std::pow(cutoff, 2.0 * exponent + 7.0)
                    / ((2.0 * exponent + 5.0) * (2.0 * exponent + 6.0)
                        * (2.0 * exponent + 7.0)));
                value += inverse_sqrt[static_cast<std::size_t>(radial * nmax + exponent - 1)]
                    * phi;
            }
            basis[static_cast<std::size_t>(radial * quadrature + q_index)] = value * common;
        }
    }
    return basis;
}

template <typename Value>
Value option(const py::dict& options, const char* key, Value fallback) {
    const py::str name(key);
    if (!options.contains(name) || options[name].is_none()) return fallback;
    return py::cast<Value>(options[name]);
}

struct RotationalCudaOptions {
    int kind = 0;
    int nmax = 3;
    int lmax = 3;
    int twojmax = 3;
    int diagonal = 3;
    double cutoff = 3.5;
    double alpha = 2.0;
    double rfac0 = 1.0;
    double rmin0 = 0.0;
    double rcutfac = 1.0;
    bool weight_on = false;
    bool normalize_u = false;
};

RotationalCudaOptions rotational_options(
    const std::string& name, const py::dict& options) {
    RotationalCudaOptions result;
    if (name == "SO3") {
        result.kind = 0;
        result.nmax = option(options, "nmax", 3);
    } else if (name == "SO4") {
        result.kind = 1;
        result.nmax = 1;
        result.rfac0 = option(options, "rfac0", 1.0);
    } else if (name == "SNAP") {
        result.kind = 2;
        result.nmax = 1;
        result.rfac0 = option(options, "rfac0", 0.99363);
    } else if (name == "LBispectrum") {
        result.kind = 3;
        result.nmax = 1;
        result.rfac0 = option(options, "rfac0", 0.99363);
    } else {
        throw std::invalid_argument("unknown CUDA rotational descriptor: " + name);
    }
    result.lmax = option(options, "lmax", 3);
    result.twojmax = option(options, "twojmax", 3);
    result.diagonal = option(options, "diagonal", 3);
    result.cutoff = option(options, "rcut", 3.5);
    result.alpha = option(options, "alpha", 2.0);
    result.rmin0 = option(options, "rmin0", 0.0);
    result.rcutfac = option(options, "rcutfac", 1.0);
    result.weight_on = option(options, "weight_on", false);
    result.normalize_u = option(options, "normalize_U", false);
    return result;
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

py::dict atom_result(
    const std::vector<double>& values,
    I64 rows,
    I64 columns,
    const std::string& name,
    const py::dict& options,
    bool per_system,
    const std::vector<I64>& offsets) {
    py::dict result;
    result["values"] = values_array(values, rows, columns);
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

std::vector<std::vector<double>> nested_payload_vectors(
    const py::dict& payload,
    const char* key) {
    const py::str name(key);
    if (!payload.contains(name) || payload[name].is_none()) return {};
    py::sequence sequence;
    try {
        sequence = py::cast<py::sequence>(payload[name]);
    } catch (const py::cast_error&) {
        throw std::invalid_argument(std::string(key) + " must be a sequence of numeric arrays");
    }
    std::vector<std::vector<double>> result;
    result.reserve(static_cast<std::size_t>(sequence.size()));
    for (py::ssize_t index = 0; index < sequence.size(); ++index) {
        const auto values = F64Array::ensure(sequence[index]);
        if (!values || values.ndim() != 1) {
            throw std::invalid_argument(std::string(key) + " must contain one-dimensional arrays");
        }
        result.emplace_back(
            values.data(), values.data() + static_cast<std::size_t>(values.shape(0)));
    }
    return result;
}

std::vector<double> numeric_values_option(
    const py::dict& options, const char* key, double fallback) {
    const py::str name(key);
    if (!options.contains(name) || options[name].is_none()) return {fallback};
    if (py::isinstance<py::array>(options[name])) {
        const auto values = F64Array::ensure(options[name]);
        if (!values || values.ndim() != 1) {
            throw std::invalid_argument(std::string(key) + " must be a one-dimensional array");
        }
        return std::vector<double>(
            values.data(), values.data() + static_cast<std::size_t>(values.shape(0)));
    }
    try {
        if (py::isinstance<py::sequence>(options[name])
            && !py::isinstance<py::str>(options[name])) {
            return py::cast<std::vector<double>>(options[name]);
        }
        return {py::cast<double>(options[name])};
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
    // The Python adapters accept both atomic numbers and chemical symbols as
    // dictionary keys.  Keep this small table local to the CUDA boundary so
    // the device kernels still receive compact numeric arrays.
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

std::vector<I32> integer_vector_option(
    const py::dict& options, const char* key, I32 fallback, std::size_t count) {
    const py::str name(key);
    if (!options.contains(name) || options[name].is_none()) {
        return std::vector<I32>(count, fallback);
    }
    try {
        if (py::isinstance<py::sequence>(options[name])
            && !py::isinstance<py::str>(options[name])) {
            auto values = py::cast<std::vector<I32>>(options[name]);
            if (values.size() != count) {
                throw std::invalid_argument(std::string(key) + " must have one value per species");
            }
            return values;
        }
        return std::vector<I32>(count, py::cast<I32>(options[name]));
    } catch (const py::cast_error&) {
        throw std::invalid_argument(std::string(key) + " must be an integer or integer array");
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

__device__ double soap_weight_device(
    int function,
    double r0,
    double c,
    double d,
    double m,
    double threshold,
    double w0,
    bool has_w0,
    bool exact_self,
    double distance,
    double species_weight) {
    double value = 1.0;
    const double ratio = r0 > 0.0 ? distance / r0 : 0.0;
    if (function == 1) {
        value = distance > r0 ? 0.0
            : c * pow(fmax(0.0, 1.0 + 2.0 * ratio * ratio * ratio
                - 3.0 * ratio * ratio), m);
    } else if (function == 2) {
        value = c / (d + pow(fmax(ratio, 1e-30), m));
    } else if (function == 3) {
        value = c / (d + exp(-ratio));
    }
    if (exact_self && has_w0) value = w0;
    return value * species_weight;
}

__device__ double soap_polynomial_flir(
    double distance,
    double radial_coordinate,
    int angular,
    double sigma) {
    const double eta = 1.0 / (2.0 * sigma * sigma);
    const double radial2 = radial_coordinate * radial_coordinate;
    if (distance <= 1e-14) {
        return angular == 0 ? exp(-eta * radial2) : 0.0;
    }
    const double denominator = eta * distance * radial_coordinate;
    if (fabs(denominator) <= 1e-30) return 0.0;
    const double prefactor = 0.25 / denominator;
    const double minus = exp(-eta * (radial_coordinate - distance)
        * (radial_coordinate - distance));
    const double plus = exp(-eta * (radial_coordinate + distance)
        * (radial_coordinate + distance));
    double previous = prefactor * (minus - plus);
    if (angular == 0) return previous;
    double current = prefactor * (minus + plus - 2.0 * previous);
    if (angular == 1) return current;
    for (int degree = 2; degree <= angular; ++degree) {
        const double next = fmax(0.0, previous - prefactor
            * (4.0 * degree - 2.0) * current);
        previous = current;
        current = next;
    }
    return current;
}

__device__ double soap_gto_radial(
    double distance2,
    int angular,
    int radial,
    int radial_count,
    double sigma,
    const double* alphas,
    const double* betas) {
    const double eta = 1.0 / (2.0 * sigma * sigma);
    const double eta_power = pow(eta, angular);
    const double pi_sqrt_pi = kPi * sqrt(kPi);
    double result = 0.0;
    for (int raw = 0; raw < radial_count; ++raw) {
        const int index = angular * radial_count + raw;
        const double alpha = alphas[index];
        const double denominator = alpha + eta;
        const double prefactor = eta_power * pow(denominator, -angular - 1.5)
            * exp(-alpha * eta / denominator * distance2);
        // The CPU basis stores beta[n, k], while the raw loop here is over k.
        result += betas[(angular * radial_count + radial) * radial_count + raw] * prefactor;
    }
    return pi_sqrt_pi * result;
}

__global__ void soap_coefficients_kernel(
    const I32* numbers,
    const I64* graph_offsets,
    const I32* graph_atoms,
    const I32* graph_shifts,
    const double* graph_displacements,
    const double* graph_distance2,
    const I32* species,
    int species_count,
    int coefficient_types,
    int radial_count,
    int max_angular,
    int radial_basis,
    double cutoff,
    double graph_cutoff,
    double sigma,
    const double* alphas,
    const double* betas,
    const double* radial_grid,
    int radial_grid_count,
    const double* radial_weights,
    const double* radial_values,
    int weighting_function,
    double weighting_r0,
    double weighting_c,
    double weighting_d,
    double weighting_m,
    double weighting_threshold,
    double weighting_w0,
    bool weighting_has_w0,
    const double* species_weights,
    I64 atoms,
    double* coefficients) {
    const I64 center = static_cast<I64>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (center >= atoms) return;
    const int harmonic_count = (max_angular + 1) * (max_angular + 1);
    const I64 coefficient_size = static_cast<I64>(coefficient_types)
        * radial_count * harmonic_count;
    double* target = coefficients + center * coefficient_size;
    for (I64 index = 0; index < coefficient_size; ++index) target[index] = 0.0;
    const I64 begin = graph_offsets[center];
    const I64 end = graph_offsets[center + 1];
    for (I64 edge = begin; edge < end; ++edge) {
        const I32 atom = graph_atoms[edge];
        const int type = species_index(numbers[atom], species, species_count);
        if (type < 0) continue;
        const double distance2 = fmax(0.0, graph_distance2[edge]);
        const double distance = sqrt(distance2);
        if (distance >= graph_cutoff) continue;
        const bool exact_self = atom == center
            && graph_shifts[edge * 3] == 0
            && graph_shifts[edge * 3 + 1] == 0
            && graph_shifts[edge * 3 + 2] == 0;
        const double weight = soap_weight_device(
            weighting_function, weighting_r0, weighting_c, weighting_d, weighting_m,
            weighting_threshold, weighting_w0, weighting_has_w0, exact_self,
            distance, species_weights == nullptr ? 1.0 : species_weights[type]);
        if (weight == 0.0) continue;
        double harmonics[441]{};
        harmonic_values<20>(graph_displacements + edge * 3, harmonics, max_angular);
        for (int angular = 0; angular <= max_angular; ++angular) {
            const double radius_power = pow(distance, angular);
            for (int radial = 0; radial < radial_count; ++radial) {
                const int destination_type = coefficient_types == 1 ? 0 : type;
                double* destination = target + (
                    destination_type * radial_count + radial) * harmonic_count
                    + angular * angular;
                double radial_value = 0.0;
                if (radial_basis == 0) {
                    radial_value = soap_gto_radial(
                        distance2, angular, radial, radial_count, sigma, alphas, betas);
                } else {
                    for (int q = 0; q < radial_grid_count; ++q) {
                        radial_value += radial_weights[q] * radial_grid[q] * radial_grid[q]
                            * soap_polynomial_flir(
                                distance, radial_grid[q], angular, sigma)
                            * radial_values[radial * radial_grid_count + q];
                    }
                    radial_value *= 4.0 * kPi;
                }
                for (int m = -angular; m <= angular; ++m) {
                    const double angular_factor = radial_basis == 0
                        ? radius_power : 1.0;
                    destination[angular + m] += weight * radial_value * angular_factor
                        * harmonics[angular * angular + angular + m];
                }
            }
        }
        // mu2 combines species densities in one coefficient block; the loop
        // above already accumulates every neighbor into that block.
    }
}

__device__ double soap_coefficient_at(
    const double* coefficients,
    int type,
    int radial,
    int harmonic,
    int radial_count,
    int harmonic_count) {
    return coefficients[(type * radial_count + radial) * harmonic_count + harmonic];
}

__device__ double soap_power_feature(
    const double* coefficients,
    int feature,
    int features,
    int species_count,
    int coefficient_types,
    int radial_count,
    int max_angular,
    int compression) {
    (void)features;
    const int harmonic_count = (max_angular + 1) * (max_angular + 1);
    int first = 0;
    int second = 0;
    int angular = 0;
    int n1 = 0;
    int n2 = 0;
    int remainder = feature;
    if (compression == 1) {
        for (angular = 0; angular <= max_angular; ++angular) {
            const int block = radial_count * (radial_count + 1) / 2;
            if (remainder < block) break;
            remainder -= block;
        }
        for (n1 = 0; n1 < radial_count; ++n1) {
            const int block = radial_count - n1;
            if (remainder < block) { n2 = n1 + remainder; break; }
            remainder -= block;
        }
    } else if (compression == 2) {
        const int per_type = (max_angular + 1) * radial_count * radial_count;
        first = feature / per_type;
        remainder = feature % per_type;
        angular = remainder / (radial_count * radial_count);
        remainder %= radial_count * radial_count;
        n1 = remainder / radial_count;
        n2 = remainder % radial_count;
        second = -1; // mu1nu1 uses a sum over the second species below.
    } else if (compression == 3) {
        const int per_type = (max_angular + 1) * radial_count * (radial_count + 1) / 2;
        first = feature / per_type;
        remainder = feature % per_type;
        for (angular = 0; angular <= max_angular; ++angular) {
            const int block = radial_count * (radial_count + 1) / 2;
            if (remainder < block) break;
            remainder -= block;
        }
        for (n1 = 0; n1 < radial_count; ++n1) {
            const int block = radial_count - n1;
            if (remainder < block) { n2 = n1 + remainder; break; }
            remainder -= block;
        }
        second = first;
    } else {
        // The uncompressed CPU layout uses triangular radial blocks for
        // same-species pairs and rectangular blocks for cross-species pairs.
        // Walk those variable-sized blocks before decoding l,n1,n2.
        bool decoded = false;
        for (first = 0; first < species_count; ++first) {
            for (second = first; second < species_count; ++second) {
                const int radial_pairs = first == second
                    ? radial_count * (radial_count + 1) / 2
                    : radial_count * radial_count;
                const int block = (max_angular + 1) * radial_pairs;
                if (remainder < block) {
                    angular = remainder / radial_pairs;
                    remainder %= radial_pairs;
                    if (first == second) {
                        for (n1 = 0; n1 < radial_count; ++n1) {
                            const int count = radial_count - n1;
                            if (remainder < count) {
                                n2 = n1 + remainder;
                                break;
                            }
                            remainder -= count;
                        }
                    } else {
                        n1 = remainder / radial_count;
                        n2 = remainder % radial_count;
                    }
                    decoded = true;
                    break;
                }
                remainder -= block;
            }
            if (decoded) break;
        }
    }
    double sum = 0.0;
    const int first_type = coefficient_types == 1 ? 0 : first;
    if (compression == 2) {
        for (int type = 0; type < species_count; ++type) {
            for (int m = -angular; m <= angular; ++m) {
                sum += soap_coefficient_at(
                    coefficients, first_type, n1, angular * angular + angular + m,
                    radial_count, harmonic_count)
                    * soap_coefficient_at(
                        coefficients, 0, n2, angular * angular + angular + m,
                        radial_count, harmonic_count);
            }
        }
        // The second factor above is the combined density for a mu1nu1
        // representation.  It is materialized below by the caller in the
        // species-summed coefficient buffer; this branch is replaced there.
        return sum * kPi * sqrt(8.0 / (2.0 * angular + 1.0));
    }
    for (int m = -angular; m <= angular; ++m) {
        sum += soap_coefficient_at(
            coefficients, first_type, n1, angular * angular + angular + m,
            radial_count, harmonic_count)
            * soap_coefficient_at(
                coefficients, coefficient_types == 1 ? 0 : second, n2,
                angular * angular + angular + m, radial_count, harmonic_count);
    }
    return kPi * sqrt(8.0 / (2.0 * angular + 1.0)) * sum;
}

__global__ void soap_power_kernel(
    const double* coefficients,
    I64 rows,
    I64 coefficient_stride,
    int features,
    int species_count,
    int coefficient_types,
    int radial_count,
    int max_angular,
    int compression,
    double* output) {
    const I64 row = static_cast<I64>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    const double* source = coefficients + row * coefficient_stride;
    double* target = output + row * features;
    if (compression != 2) {
        for (int feature = 0; feature < features; ++feature) {
            target[feature] = soap_power_feature(
                source, feature, features, species_count, coefficient_types,
                radial_count, max_angular, compression);
        }
        return;
    }
    // Build the species-summed density once per row for mu1nu1.  The temporary
    // block is kept in registers/local memory so no host-side reduction is
    // introduced into the CUDA path.
    const int harmonic_count = (max_angular + 1) * (max_angular + 1);
    const int sum_count = radial_count * harmonic_count;
    double summed[32 * 441]{};
    if (sum_count > static_cast<int>(sizeof(summed) / sizeof(double))) return;
    for (int type = 0; type < species_count; ++type) {
        const double* block = source + type * radial_count * harmonic_count;
        for (int index = 0; index < sum_count; ++index) summed[index] += block[index];
    }
    for (int feature = 0; feature < features; ++feature) {
        int remainder = feature;
        const int per_type = (max_angular + 1) * radial_count * radial_count;
        const int first = feature / per_type;
        remainder %= per_type;
        int angular = remainder / (radial_count * radial_count);
        remainder %= radial_count * radial_count;
        const int n1 = remainder / radial_count;
        const int n2 = remainder % radial_count;
        double value = 0.0;
        for (int m = -angular; m <= angular; ++m) {
            const int harmonic = angular * angular + angular + m;
            value += soap_coefficient_at(
                source, first, n1, harmonic, radial_count, harmonic_count)
                * summed[n2 * harmonic_count + harmonic];
        }
        target[feature] = kPi * sqrt(8.0 / (2.0 * angular + 1.0)) * value;
    }
}

__global__ void soap_average_coefficients_kernel(
    const I64* offsets,
    I64 structures,
    I64 coefficient_stride,
    const double* atom_coefficients,
    double* structure_coefficients) {
    const I64 structure = static_cast<I64>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (structure >= structures) return;
    const I64 begin = offsets[structure];
    const I64 end = offsets[structure + 1];
    double* target = structure_coefficients + structure * coefficient_stride;
    for (I64 index = 0; index < coefficient_stride; ++index) target[index] = 0.0;
    if (end <= begin) return;
    for (I64 atom = begin; atom < end; ++atom) {
        const double* source = atom_coefficients + atom * coefficient_stride;
        for (I64 index = 0; index < coefficient_stride; ++index) target[index] += source[index];
    }
    const double scale = 1.0 / static_cast<double>(end - begin);
    for (I64 index = 0; index < coefficient_stride; ++index) target[index] *= scale;
}

__global__ void soap_average_power_kernel(
    const I64* offsets,
    I64 structures,
    int features,
    const double* atom_power,
    double* output) {
    const I64 structure = static_cast<I64>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (structure >= structures) return;
    const I64 begin = offsets[structure];
    const I64 end = offsets[structure + 1];
    double* target = output + structure * features;
    for (int feature = 0; feature < features; ++feature) target[feature] = 0.0;
    if (end <= begin) return;
    for (I64 atom = begin; atom < end; ++atom) {
        const double* source = atom_power + atom * features;
        for (int feature = 0; feature < features; ++feature) target[feature] += source[feature];
    }
    const double scale = 1.0 / static_cast<double>(end - begin);
    for (int feature = 0; feature < features; ++feature) target[feature] *= scale;
}

py::dict compute_soap_descriptor(
    CudaExecutionContext& context,
    DeviceBatch& batch,
    DeviceNeighborGraph& graph,
    const detail::StructureBatchView& host_batch,
    const py::dict& options) {
    (void)graph;
    const auto species = species_option(options);
    if (species.empty()) throw std::invalid_argument("SOAP species must not be empty");
    const int radial_count = option(options, "n_max", 8);
    const int max_angular = option(options, "l_max", 6);
    const double cutoff = option(options, "r_cut", 6.0);
    const double sigma = option(options, "sigma", 1.0);
    if (radial_count < 1 || radial_count > 32 || max_angular < 0 || max_angular > 20
        || cutoff <= 0.0 || sigma <= 0.0) {
        throw std::invalid_argument("invalid CUDA SOAP parameters");
    }
    const std::string radial_name = option(options, "rbf", std::string("gto"));
    const int radial_basis = radial_name == "gto" ? 0 : radial_name == "polynomial" ? 1 : -1;
    if (radial_basis < 0) throw std::invalid_argument("unsupported CUDA SOAP radial basis");
    const py::dict compression_object = child_dict(options, "compression");
    const std::string compression_name = option(
        compression_object, "mode", std::string("off"));
    const int compression = compression_name == "off" ? 0
        : compression_name == "mu2" ? 1
        : compression_name == "mu1nu1" ? 2
        : compression_name == "crossover" ? 3 : -1;
    if (compression < 0) throw std::invalid_argument("unsupported CUDA SOAP compression");
    const std::string average = option(options, "average", std::string("inner"));
    if (average != "off" && average != "inner" && average != "outer") {
        throw std::invalid_argument("unsupported CUDA SOAP average mode");
    }
    const int species_count = static_cast<int>(species.size());
    const int coefficient_types = compression == 1 ? 1 : species_count;
    I64 computed_features = 0;
    if (compression == 1) {
        computed_features = static_cast<I64>(radial_count) * (radial_count + 1) / 2
            * (max_angular + 1);
    } else if (compression == 2) {
        computed_features = static_cast<I64>(species_count) * radial_count * radial_count
            * (max_angular + 1);
    } else if (compression == 3) {
        computed_features = static_cast<I64>(species_count) * radial_count
            * (radial_count + 1) / 2 * (max_angular + 1);
    } else {
        computed_features = static_cast<I64>(species_count) * (species_count + 1) / 2
            * radial_count * radial_count * (max_angular + 1);
        // Same-species radial pairs are triangular, while cross-species pairs
        // retain the full rectangular radial block.
        computed_features = 0;
        for (int first = 0; first < species_count; ++first) {
            for (int second = first; second < species_count; ++second) {
                computed_features += static_cast<I64>(
                    first == second ? radial_count * (radial_count + 1) / 2
                                    : radial_count * radial_count) * (max_angular + 1);
            }
        }
    }
    const I64 features = feature_count_option(options, computed_features);
    if (features != computed_features) {
        throw std::invalid_argument("CUDA SOAP feature count does not match its layout");
    }
    const py::dict payload = child_dict(options, "_cuda_payload");
    const py::dict radial_payload = child_dict(payload, "radial_basis");
    auto alphas = vector_child(radial_payload, "alphas");
    auto betas = vector_child(radial_payload, "betas");
    auto radial_grid = vector_child(radial_payload, "radial_grid");
    auto radial_weights = vector_child(radial_payload, "radial_weights");
    auto radial_values = vector_child(radial_payload, "radial_values");
    if (radial_basis == 0 && (alphas.size() != static_cast<std::size_t>((max_angular + 1) * radial_count)
        || betas.size() != static_cast<std::size_t>((max_angular + 1) * radial_count * radial_count))) {
        throw std::invalid_argument("CUDA SOAP GTO payload has an invalid shape");
    }
    if (radial_basis == 1 && (radial_grid.size() < 2
        || radial_weights.size() != radial_grid.size()
        || radial_values.size() != radial_grid.size() * static_cast<std::size_t>(radial_count))) {
        throw std::invalid_argument("CUDA SOAP polynomial payload has an invalid shape");
    }
    const py::dict weighting = child_dict(options, "weighting");
    const std::string weighting_name = option(weighting, "function", std::string());
    const int weighting_function = weighting_name == "" ? 0
        : weighting_name == "poly" ? 1
        : weighting_name == "pow" ? 2
        : weighting_name == "exp" ? 3 : -1;
    if (weighting_function < 0) throw std::invalid_argument("unsupported CUDA SOAP weighting");
    const py::object species_weight_object = compression_object.contains("species_weighting")
        ? compression_object["species_weighting"] : py::none();
    const auto species_weights = species_dictionary_values(species_weight_object, species, 1.0);
    const double padding = radial_basis == 0 ? sigma * sqrt(-2.0 * log(1e-3)) : 0.0;
    const double graph_cutoff = cutoff + padding;
    graph.build_dpa(context, batch, host_batch, graph_cutoff, true, false, true);
    DeviceBuffer<I32> d_species;
    DeviceBuffer<double> d_species_weights;
    DeviceBuffer<double> d_alphas;
    DeviceBuffer<double> d_betas;
    DeviceBuffer<double> d_grid;
    DeviceBuffer<double> d_radial_weights;
    DeviceBuffer<double> d_values;
    d_species.upload(species.data(), species.size(), context.stream(), "could not upload SOAP species");
    d_species_weights.upload(species_weights.data(), species_weights.size(), context.stream(), "could not upload SOAP species weights");
    d_alphas.upload(alphas.data(), alphas.size(), context.stream(), "could not upload SOAP GTO alphas");
    d_betas.upload(betas.data(), betas.size(), context.stream(), "could not upload SOAP GTO betas");
    d_grid.upload(radial_grid.data(), radial_grid.size(), context.stream(), "could not upload SOAP polynomial grid");
    d_radial_weights.upload(
        radial_weights.data(), radial_weights.size(), context.stream(),
        "could not upload SOAP polynomial quadrature weights");
    d_values.upload(radial_values.data(), radial_values.size(), context.stream(), "could not upload SOAP polynomial basis");
    const int harmonic_count = (max_angular + 1) * (max_angular + 1);
    const I64 coefficient_stride = static_cast<I64>(coefficient_types) * radial_count * harmonic_count;
    const std::size_t coefficient_size = static_cast<std::size_t>(batch.atoms())
        * static_cast<std::size_t>(coefficient_stride);
    const bool inner = average == "inner";
    const bool outer = average == "outer";
    const I64 rows = inner || outer ? batch.structures() : batch.atoms();
    const std::size_t output_size = static_cast<std::size_t>(rows)
        * static_cast<std::size_t>(features);
    double* output = context.output_buffer(output_size);
    const std::size_t power_size = outer ? static_cast<std::size_t>(batch.atoms())
        * static_cast<std::size_t>(features) : 0U;
    const std::size_t average_size = inner ? static_cast<std::size_t>(batch.structures())
        * static_cast<std::size_t>(coefficient_stride) : 0U;
    const std::size_t workspace_size = coefficient_size + power_size + average_size;
    auto* workspace = static_cast<double*>(context.workspace_buffer(
        workspace_size * sizeof(double)));
    double* coefficients = workspace;
    double* atom_power = outer ? coefficients + coefficient_size : nullptr;
    double* structure_coefficients = inner
        ? coefficients + coefficient_size + power_size : nullptr;
    if (output_size > 0) {
        check_cuda(cudaMemsetAsync(output, 0, output_size * sizeof(double), context.stream()),
            "could not clear CUDA SOAP output");
    }
    if (batch.atoms() > 0) {
        constexpr unsigned block_size = 64;
        soap_coefficients_kernel<<<static_cast<unsigned>((batch.atoms() + block_size - 1) / block_size),
            block_size, 0, context.stream()>>>(
            batch.numbers(), graph.offsets(), graph.atoms(), graph.shifts(), graph.displacements(),
            graph.distance2(), d_species.get(), species_count, coefficient_types, radial_count,
            max_angular, radial_basis, cutoff, graph_cutoff, sigma,
            d_alphas.get(), d_betas.get(), d_grid.get(),
            static_cast<int>(radial_grid.size()), d_radial_weights.get(), d_values.get(), weighting_function,
            option(weighting, "r0", 1.0), option(weighting, "c", 1.0),
            option(weighting, "d", 0.0), option(weighting, "m", 1.0),
            option(weighting, "threshold", 1e-2), option(weighting, "w0", 1.0),
            weighting.contains("w0"), d_species_weights.get(), batch.atoms(), coefficients);
        check_cuda(cudaGetLastError(), "CUDA SOAP coefficient kernel launch failed");
    }
    constexpr unsigned block_size = 64;
    if (inner) {
        if (batch.structures() > 0) {
            soap_average_coefficients_kernel<<<static_cast<unsigned>((batch.structures() + block_size - 1) / block_size),
                block_size, 0, context.stream()>>>(
                batch.offsets(), batch.structures(), coefficient_stride, coefficients,
                structure_coefficients);
            check_cuda(cudaGetLastError(), "CUDA SOAP coefficient average launch failed");
            soap_power_kernel<<<static_cast<unsigned>((batch.structures() + block_size - 1) / block_size),
                block_size, 0, context.stream()>>>(
                structure_coefficients, batch.structures(), coefficient_stride,
                static_cast<int>(features), species_count, coefficient_types, radial_count,
                max_angular, compression, output);
            check_cuda(cudaGetLastError(), "CUDA SOAP inner power kernel launch failed");
        }
    } else {
        if (batch.atoms() > 0) {
            soap_power_kernel<<<static_cast<unsigned>((batch.atoms() + block_size - 1) / block_size),
                block_size, 0, context.stream()>>>(
                coefficients, batch.atoms(), coefficient_stride, static_cast<int>(features),
                species_count, coefficient_types, radial_count, max_angular, compression,
                outer ? atom_power : output);
            check_cuda(cudaGetLastError(), "CUDA SOAP power kernel launch failed");
        }
        if (outer && batch.structures() > 0) {
            soap_average_power_kernel<<<static_cast<unsigned>((batch.structures() + block_size - 1) / block_size),
                block_size, 0, context.stream()>>>(
                batch.offsets(), batch.structures(), static_cast<int>(features), atom_power, output);
            check_cuda(cudaGetLastError(), "CUDA SOAP outer power average launch failed");
        }
    }
    const auto values = download_output_with_gil_release(context, output_size);
    py::dict result;
    result["values"] = values_array(values, rows, features);
    result["level"] = inner || outer ? "structure" : "atom";
    if (!inner && !outer) result["row_offsets"] = i64_array(
        std::vector<I64>(host_batch.offsets, host_batch.offsets + host_batch.structures + 1));
    result["labels"] = labels_option(options, "SOAP", features);
    result["metadata"] = metadata(options, "SOAP");
    return result;
}

__device__ int species_index(I32 number, const I32* species, int count) {
    for (int index = 0; index < count; ++index) {
        if (species[index] == number) return index;
    }
    return -1;
}

__global__ void atomic_composition_kernel(
    const I32* numbers,
    const I64* offsets,
    I64 structures,
    I64 atoms,
    const I32* species,
    int species_count,
    bool per_system,
    double* output) {
    const I64 row = static_cast<I64>(blockIdx.x) * blockDim.x + threadIdx.x;
    const I64 rows = per_system ? structures : atoms;
    if (row >= rows) return;
    if (per_system) {
        for (I64 atom = offsets[row]; atom < offsets[row + 1]; ++atom) {
            const int type = species_index(numbers[atom], species, species_count);
            if (type >= 0) output[row * species_count + type] += 1.0;
        }
        return;
    }
    const int type = species_index(numbers[row], species, species_count);
    if (type >= 0) output[row * species_count + type] = 1.0;
}

__global__ void sorted_distances_kernel(
    const I32* numbers,
    const I64* graph_offsets,
    const I32* graph_atoms,
    const double* graph_distance2,
    const I32* species,
    int species_count,
    int max_neighbors,
    bool separate,
    double cutoff,
    I64 atoms,
    double* output) {
    const I64 center = static_cast<I64>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (center >= atoms) return;
    const I64 begin = graph_offsets[center];
    const I64 end = graph_offsets[center + 1];
    if (separate) {
        for (int wanted = 0; wanted < species_count; ++wanted) {
            int count = 0;
            const I64 base = center * static_cast<I64>(species_count * max_neighbors)
                + static_cast<I64>(wanted * max_neighbors);
            for (I64 edge = begin; edge < end && count < max_neighbors; ++edge) {
                if (species_index(numbers[graph_atoms[edge]], species, species_count) != wanted) {
                    continue;
                }
                output[base + count++] = sqrt(fmax(0.0, graph_distance2[edge]));
            }
            if (count > 0) {
                for (int index = count; index < max_neighbors; ++index) {
                    output[base + index] = cutoff;
                }
            }
        }
        return;
    }
    const I64 base = center * max_neighbors;
    int count = 0;
    for (I64 edge = begin; edge < end && count < max_neighbors; ++edge) {
        output[base + count++] = sqrt(fmax(0.0, graph_distance2[edge]));
    }
    if (count > 0) {
        for (int index = count; index < max_neighbors; ++index) output[base + index] = cutoff;
    }
}

// The following device helpers mirror the radial/harmonic path used by the
// existing CUDA local descriptors.  Keeping the pair path here avoids a host
// materialization of edge features while preserving the CPU layout.
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

__device__ double radial_value(
    double distance,
    int angular,
    int target_radial,
    int radial_count,
    double density_width,
    const double* gto_constants,
    const double* gamma_a,
    const double* gamma_b,
    const double* orthonormalization) {
    const double density_width2 = density_width * density_width;
    const double density_constant = 1.0 / (2.0 * density_width2);
    const double global_factor = pow(kPi / density_width2, 0.75);
    const double c_r = density_constant * distance;
    const double factor = global_factor * exp(-distance * c_r) * pow(c_r, angular);
    double value = 0.0;
    for (int raw_index = 0; raw_index < radial_count; ++raw_index) {
        const double gto_constant = gto_constants[angular * radial_count + raw_index];
        const double z = c_r * c_r / (density_constant + gto_constant);
        const double a = 0.5 * (raw_index + angular + 3.0);
        const double b = angular + 1.5;
        double raw;
        if (z > 30.0) {
            const double logarithm = log(global_factor) - distance * c_r
                + static_cast<double>(angular) * log(c_r)
                - a * log(density_constant + gto_constant) + z + (a - b) * log(z);
            raw = exp(logarithm) * positive_hypergeometric(a, b, z);
        } else {
            raw = gamma_a[angular * radial_count + raw_index] / gamma_b[angular]
                * positive_hypergeometric(a, b, z)
                * pow(density_constant + gto_constant, -a) * factor;
        }
        value += raw * orthonormalization[
            (angular * radial_count + raw_index) * radial_count + target_radial];
    }
    return value;
}

__device__ double smooth_cutoff(double distance, double cutoff) {
    if (distance >= cutoff) return 0.0;
    const double width = fmin(0.5, cutoff);
    if (distance <= cutoff - width) return 1.0;
    return 0.5 * (1.0 + cos(kPi * (distance - cutoff + width) / width));
}

__device__ I64 center_for_edge(const I64* offsets, I64 atoms, I64 edge) {
    I64 left = 0;
    I64 right = atoms;
    while (left + 1 < right) {
        const I64 middle = left + (right - left) / 2;
        if (offsets[middle] <= edge) left = middle;
        else right = middle;
    }
    return left;
}

__device__ bool inverse3_device(const double* matrix, double* inverse) {
    const double determinant = matrix[0] * (matrix[4] * matrix[8] - matrix[5] * matrix[7])
        - matrix[1] * (matrix[3] * matrix[8] - matrix[5] * matrix[6])
        + matrix[2] * (matrix[3] * matrix[7] - matrix[4] * matrix[6]);
    if (!isfinite(determinant) || fabs(determinant) <= 1e-12) return false;
    const double scale = 1.0 / determinant;
    inverse[0] = (matrix[4] * matrix[8] - matrix[5] * matrix[7]) * scale;
    inverse[1] = (matrix[2] * matrix[7] - matrix[1] * matrix[8]) * scale;
    inverse[2] = (matrix[1] * matrix[5] - matrix[2] * matrix[4]) * scale;
    inverse[3] = (matrix[5] * matrix[6] - matrix[3] * matrix[8]) * scale;
    inverse[4] = (matrix[0] * matrix[8] - matrix[2] * matrix[6]) * scale;
    inverse[5] = (matrix[2] * matrix[3] - matrix[0] * matrix[5]) * scale;
    inverse[6] = (matrix[3] * matrix[7] - matrix[4] * matrix[6]) * scale;
    inverse[7] = (matrix[1] * matrix[6] - matrix[0] * matrix[7]) * scale;
    inverse[8] = (matrix[0] * matrix[4] - matrix[1] * matrix[3]) * scale;
    return true;
}

__device__ double cell_volume_device(const double* cell) {
    return mdescriptor::detail::mbtr::cell_volume(cell);
}

__device__ void fractional_device(
    const double* inverse, double x, double y, double z,
    double& fx, double& fy, double& fz) {
    fx = x * inverse[0] + y * inverse[3] + z * inverse[6];
    fy = x * inverse[1] + y * inverse[4] + z * inverse[7];
    fz = x * inverse[2] + y * inverse[5] + z * inverse[8];
}

__device__ void cartesian_from_fractional(
    const double* cell, double fx, double fy, double fz,
    double& x, double& y, double& z) {
    x = fx * cell[0] + fy * cell[3] + fz * cell[6];
    y = fx * cell[1] + fy * cell[4] + fz * cell[7];
    z = fx * cell[2] + fy * cell[5] + fz * cell[8];
}

__device__ double sine_matrix_off_diagonal(
    const double* cell, const double* inverse,
    double dx, double dy, double dz) {
    double fx = 0.0;
    double fy = 0.0;
    double fz = 0.0;
    fractional_device(inverse, dx, dy, dz, fx, fy, fz);
    const double sx = sin(kPi * fx);
    const double sy = sin(kPi * fy);
    const double sz = sin(kPi * fz);
    const double tx = sx * sx * cell[0] + sy * sy * cell[3] + sz * sz * cell[6];
    const double ty = sx * sx * cell[1] + sy * sy * cell[4] + sz * sz * cell[7];
    const double tz = sx * sx * cell[2] + sy * sy * cell[5] + sz * sz * cell[8];
    return sqrt(tx * tx + ty * ty + tz * tz);
}

__device__ double ewald_off_diagonal(
    const I32* numbers,
    const double* positions,
    const double* wrapped_positions,
    const double* cell,
    const double* inverse,
    const double* reciprocal_vectors,
    int first,
    int second,
    double alpha,
    double r_cut,
    double g_cut,
    double volume,
    double inverse_norm_x,
    double inverse_norm_y,
    double inverse_norm_z,
    int gx,
    int gy,
    int gz) {
    const double xi = positions[first * 3 + 0];
    const double yi = positions[first * 3 + 1];
    const double zi = positions[first * 3 + 2];
    const double xj = positions[second * 3 + 0];
    const double yj = positions[second * 3 + 1];
    const double zj = positions[second * 3 + 2];
    double fi_x = 0.0;
    double fi_y = 0.0;
    double fi_z = 0.0;
    fractional_device(inverse, xi, yi, zi, fi_x, fi_y, fi_z);
    fi_x -= floor(fi_x);
    fi_y -= floor(fi_y);
    fi_z -= floor(fi_z);
    const double wj_x = wrapped_positions[second * 3 + 0];
    const double wj_y = wrapped_positions[second * 3 + 1];
    const double wj_z = wrapped_positions[second * 3 + 2];
    double real = 0.0;
    for (int sx = static_cast<int>(floor(fi_x - r_cut * inverse_norm_x));
         sx < static_cast<int>(ceil(fi_x + r_cut * inverse_norm_x)); ++sx) {
        for (int sy = static_cast<int>(floor(fi_y - r_cut * inverse_norm_y));
             sy < static_cast<int>(ceil(fi_y + r_cut * inverse_norm_y)); ++sy) {
            for (int sz = static_cast<int>(floor(fi_z - r_cut * inverse_norm_z));
                 sz < static_cast<int>(ceil(fi_z + r_cut * inverse_norm_z)); ++sz) {
                const double tx = wj_x - xi + sx * cell[0] + sy * cell[3] + sz * cell[6];
                const double ty = wj_y - yi + sx * cell[1] + sy * cell[4] + sz * cell[7];
                const double tz = wj_z - zi + sx * cell[2] + sy * cell[5] + sz * cell[8];
                const double distance2 = tx * tx + ty * ty + tz * tz;
                if (distance2 > 1e-16 && distance2 <= r_cut * r_cut) {
                    real += erfc(alpha * sqrt(distance2)) / sqrt(distance2);
                }
            }
        }
    }
    const double reciprocal_x = reciprocal_vectors[0];
    const double reciprocal_y = reciprocal_vectors[1];
    const double reciprocal_z = reciprocal_vectors[2];
    const double reciprocal2_x = reciprocal_vectors[3];
    const double reciprocal2_y = reciprocal_vectors[4];
    const double reciprocal2_z = reciprocal_vectors[5];
    const double reciprocal3_x = reciprocal_vectors[6];
    const double reciprocal3_y = reciprocal_vectors[7];
    const double reciprocal3_z = reciprocal_vectors[8];
    double reciprocal = 0.0;
    for (int gx_i = -gx; gx_i <= gx; ++gx_i) {
        for (int gy_i = -gy; gy_i <= gy; ++gy_i) {
            for (int gz_i = -gz; gz_i <= gz; ++gz_i) {
                const double vx = gx_i * reciprocal_x + gy_i * reciprocal2_x + gz_i * reciprocal3_x;
                const double vy = gx_i * reciprocal_y + gy_i * reciprocal2_y + gz_i * reciprocal3_y;
                const double vz = gx_i * reciprocal_z + gy_i * reciprocal2_z + gz_i * reciprocal3_z;
                const double length2 = vx * vx + vy * vy + vz * vz;
                if (length2 <= 1e-24 || length2 > g_cut * g_cut) continue;
                const double phase_i = vx * xi + vy * yi + vz * zi;
                const double phase_j = vx * xj + vy * yj + vz * zj;
                reciprocal += cos(phase_j - phase_i - 0.25 * kPi)
                    * exp(-length2 / (4.0 * alpha * alpha)) / length2;
            }
        }
    }
    const double zi_number = numbers[first];
    const double zj_number = numbers[second];
    const double scale = zi_number * zj_number;
    return scale * (real + reciprocal * (4.0 * kPi / volume) * sqrt(2.0));
}

__device__ void eigenvalues_symmetric_device(
    double* matrix,
    int size,
    int stride,
    double* output,
    int output_size) {
    if (size <= 0) {
        for (int index = 0; index < output_size; ++index) output[index] = 0.0;
        return;
    }

    // Householder reduction followed by implicit-shift QL is O(n^3).  The
    // former CUDA path used a maximum-pivot Jacobi loop whose repeated O(n^2)
    // pivot scans made the practical cost approach O(n^4) for eigenspectra.
    // Keep the matrix in the existing n_atoms_max-strided workspace so this
    // change does not add a second matrix allocation or alter the output ABI.
    double diagonal[256]{};
    double off_diagonal[256]{};
    for (int row = size - 1; row > 0; --row) {
        const int last = row - 1;
        double scale = 0.0;
        for (int column = 0; column <= last; ++column) {
            scale += fabs(matrix[row * stride + column]);
        }
        if (scale == 0.0) {
            off_diagonal[row] = matrix[row * stride + last];
            continue;
        }

        double squared_norm = 0.0;
        for (int column = 0; column <= last; ++column) {
            double& value = matrix[row * stride + column];
            value /= scale;
            squared_norm += value * value;
        }
        const double first = matrix[row * stride + last];
        double reflector = sqrt(squared_norm);
        if (first > 0.0) reflector = -reflector;
        off_diagonal[row] = scale * reflector;
        squared_norm -= first * reflector;
        matrix[row * stride + last] = first - reflector;

        double projection = 0.0;
        for (int column = 0; column <= last; ++column) {
            double value = 0.0;
            for (int index = 0; index <= column; ++index) {
                value += matrix[column * stride + index]
                    * matrix[row * stride + index];
            }
            for (int index = column + 1; index <= last; ++index) {
                value += matrix[index * stride + column]
                    * matrix[row * stride + index];
            }
            off_diagonal[column] = value / squared_norm;
            projection += off_diagonal[column] * matrix[row * stride + column];
        }

        const double correction = projection / (squared_norm + squared_norm);
        for (int column = 0; column <= last; ++column) {
            const double row_value = matrix[row * stride + column];
            off_diagonal[column] -= correction * row_value;
            const double column_value = off_diagonal[column];
            for (int index = 0; index <= column; ++index) {
                matrix[column * stride + index] -= row_value
                    * off_diagonal[index]
                    + column_value * matrix[row * stride + index];
            }
        }
    }
    for (int index = 0; index < size; ++index) {
        diagonal[index] = matrix[index * stride + index];
    }
    off_diagonal[0] = 0.0;
    for (int index = 1; index < size; ++index) {
        off_diagonal[index - 1] = off_diagonal[index];
    }
    off_diagonal[size - 1] = 0.0;

    for (int lower = 0; lower < size; ++lower) {
        int upper;
        int iteration = 0;
        do {
            for (upper = lower; upper < size - 1; ++upper) {
                const double scale = fabs(diagonal[upper]) + fabs(diagonal[upper + 1]);
                if (fabs(off_diagonal[upper]) + scale == scale) break;
            }
            if (upper == lower) break;
            if (++iteration > 100) break;

            double shift = (diagonal[lower + 1] - diagonal[lower])
                / (2.0 * off_diagonal[lower]);
            double radius = hypot(shift, 1.0);
            shift = diagonal[upper] - diagonal[lower]
                + off_diagonal[lower]
                    / (shift + copysign(radius, shift));

            double sine = 1.0;
            double cosine = 1.0;
            double carry = 0.0;
            for (int index = upper - 1; index >= lower; --index) {
                const double first = sine * off_diagonal[index];
                const double second = cosine * off_diagonal[index];
                double ratio;
                if (fabs(first) >= fabs(shift)) {
                    cosine = shift / first;
                    radius = hypot(cosine, 1.0);
                    off_diagonal[index + 1] = first * radius;
                    sine = 1.0 / radius;
                    cosine *= sine;
                } else {
                    sine = first / shift;
                    radius = hypot(sine, 1.0);
                    off_diagonal[index + 1] = shift * radius;
                    cosine = 1.0 / radius;
                    sine *= cosine;
                }
                const double gap = diagonal[index + 1] - carry;
                ratio = (diagonal[index] - gap) * sine + 2.0 * cosine * second;
                carry = sine * ratio;
                diagonal[index + 1] = gap + carry;
                shift = cosine * ratio - second;
            }
            diagonal[lower] -= carry;
            off_diagonal[lower] = shift;
            off_diagonal[upper] = 0.0;
        } while (upper != lower);
    }

    for (int index = 0; index < size; ++index) output[index] = diagonal[index];
    for (int index = 1; index < size; ++index) {
        int current = index;
        while (current > 0 && fabs(output[current]) > fabs(output[current - 1])) {
            const double saved = output[current];
            output[current] = output[current - 1];
            output[current - 1] = saved;
            --current;
        }
    }
    for (int index = size; index < output_size; ++index) output[index] = 0.0;
}

__global__ void matrix_kernel(
    const I32* numbers,
    const double* positions,
    const double* cells,
    const I64* offsets,
    I64 structures,
    int n_atoms_max,
    I64 workspace_stride,
    int kind,
    int permutation,
    double exponent,
    double accuracy,
    double w,
    double r_cut_option,
    double g_cut_option,
    double a_option,
    double* matrices,
    double* output) {
    const I64 structure = static_cast<I64>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (structure >= structures) return;
    const I64 begin = offsets[structure];
    const I64 end = offsets[structure + 1];
    const int count = static_cast<int>(end - begin);
    double* matrix = matrices + structure * workspace_stride;
    double* wrapped_positions = matrix
        + static_cast<I64>(n_atoms_max) * n_atoms_max;
    double* row = output + structure * (permutation == kMatrixPermutationEigenspectrum
        ? n_atoms_max : n_atoms_max * n_atoms_max);

    if (count <= 0) {
        const I64 columns = permutation == kMatrixPermutationEigenspectrum
            ? n_atoms_max : static_cast<I64>(n_atoms_max) * n_atoms_max;
        for (I64 index = 0; index < columns; ++index) {
            row[index] = 0.0;
        }
        return;
    }

    const double* cell = cells + structure * 9;
    double inverse[9]{};
    const bool inverse_valid = kind == kMatrixKindCoulomb || inverse3_device(cell, inverse);
    double volume = 0.0;
    double alpha = 0.0;
    double r_cut = 0.0;
    double g_cut = 0.0;
    double inverse_norm_x = 0.0;
    double inverse_norm_y = 0.0;
    double inverse_norm_z = 0.0;
    double reciprocal_vectors[9]{};
    int gx = 0;
    int gy = 0;
    int gz = 0;
    if (kind == kMatrixKindEwald && inverse_valid) {
        volume = cell_volume_device(cell);
        alpha = a_option > 0.0
            ? a_option : pow(static_cast<double>(count) * w / (volume * volume), 1.0 / 6.0) * sqrt(kPi);
        const double factor = sqrt(-log(accuracy));
        r_cut = r_cut_option > 0.0 ? r_cut_option : factor / alpha;
        g_cut = g_cut_option > 0.0 ? g_cut_option : 2.0 * alpha * factor;
        inverse_norm_x = sqrt(inverse[0] * inverse[0] + inverse[3] * inverse[3] + inverse[6] * inverse[6]);
        inverse_norm_y = sqrt(inverse[1] * inverse[1] + inverse[4] * inverse[4] + inverse[7] * inverse[7]);
        inverse_norm_z = sqrt(inverse[2] * inverse[2] + inverse[5] * inverse[5] + inverse[8] * inverse[8]);
        reciprocal_vectors[0] = 2.0 * kPi * inverse[0];
        reciprocal_vectors[1] = 2.0 * kPi * inverse[3];
        reciprocal_vectors[2] = 2.0 * kPi * inverse[6];
        reciprocal_vectors[3] = 2.0 * kPi * inverse[1];
        reciprocal_vectors[4] = 2.0 * kPi * inverse[4];
        reciprocal_vectors[5] = 2.0 * kPi * inverse[7];
        reciprocal_vectors[6] = 2.0 * kPi * inverse[2];
        reciprocal_vectors[7] = 2.0 * kPi * inverse[5];
        reciprocal_vectors[8] = 2.0 * kPi * inverse[8];
        gx = static_cast<int>(ceil(g_cut / sqrt(
            reciprocal_vectors[0] * reciprocal_vectors[0]
            + reciprocal_vectors[1] * reciprocal_vectors[1]
            + reciprocal_vectors[2] * reciprocal_vectors[2]))) + 1;
        gy = static_cast<int>(ceil(g_cut / sqrt(
            reciprocal_vectors[3] * reciprocal_vectors[3]
            + reciprocal_vectors[4] * reciprocal_vectors[4]
            + reciprocal_vectors[5] * reciprocal_vectors[5]))) + 1;
        gz = static_cast<int>(ceil(g_cut / sqrt(
            reciprocal_vectors[6] * reciprocal_vectors[6]
            + reciprocal_vectors[7] * reciprocal_vectors[7]
            + reciprocal_vectors[8] * reciprocal_vectors[8]))) + 1;
        for (int atom = 0; atom < count; ++atom) {
            double fractional_x = 0.0;
            double fractional_y = 0.0;
            double fractional_z = 0.0;
            const double* position = positions + (begin + atom) * 3;
            fractional_device(
                inverse, position[0], position[1], position[2],
                fractional_x, fractional_y, fractional_z);
            fractional_x -= floor(fractional_x);
            fractional_y -= floor(fractional_y);
            fractional_z -= floor(fractional_z);
            cartesian_from_fractional(
                cell, fractional_x, fractional_y, fractional_z,
                wrapped_positions[atom * 3 + 0],
                wrapped_positions[atom * 3 + 1],
                wrapped_positions[atom * 3 + 2]);
        }
    }
    // Only the rows that can be read below need clearing.  The old kernel
    // cleared the full n_atoms_max square even when structures were smaller.
    for (int i = 0; i < count; ++i) {
        for (int j = 0; j < n_atoms_max; ++j) matrix[i * n_atoms_max + j] = 0.0;
    }
    for (int i = 0; i < count; ++i) {
        const double zi = static_cast<double>(numbers[begin + i]);
        for (int j = 0; j < count; ++j) {
            const double zj = static_cast<double>(numbers[begin + j]);
            double value = 0.0;
            if (i == j && kind != kMatrixKindEwald) {
                value = 0.5 * pow(zi, exponent);
            } else if (kind == kMatrixKindCoulomb) {
                const double dx = positions[(begin + i) * 3 + 0] - positions[(begin + j) * 3 + 0];
                const double dy = positions[(begin + i) * 3 + 1] - positions[(begin + j) * 3 + 1];
                const double dz = positions[(begin + i) * 3 + 2] - positions[(begin + j) * 3 + 2];
                value = zi * zj / sqrt(dx * dx + dy * dy + dz * dz);
            } else if (kind == kMatrixKindSine) {
                if (inverse_valid) {
                    const double dx = positions[(begin + i) * 3 + 0] - positions[(begin + j) * 3 + 0];
                    const double dy = positions[(begin + i) * 3 + 1] - positions[(begin + j) * 3 + 1];
                    const double dz = positions[(begin + i) * 3 + 2] - positions[(begin + j) * 3 + 2];
                    const double denominator = sine_matrix_off_diagonal(
                        cell, inverse, dx, dy, dz);
                    value = denominator > 1e-14 ? zi * zj / denominator : 0.0;
                }
            } else if (kind == kMatrixKindEwald && inverse_valid) {
                value = ewald_off_diagonal(
                    numbers + begin, positions + begin * 3, wrapped_positions,
                    cell, inverse, reciprocal_vectors, i, j, alpha, r_cut, g_cut,
                    volume, inverse_norm_x, inverse_norm_y, inverse_norm_z,
                    gx, gy, gz);
                // The CPU Ewald implementation applies the half self term and
                // the neutralizing-background correction after accumulating
                // both real- and reciprocal-space contributions.  Applying
                // only 0.5 * Z^p on the diagonal drops the periodic images,
                // reciprocal sum, and self energy, and is especially visible
                // for the default ``permutation=none`` matrix.
                if (i == j) {
                    value = 0.5 * value
                        - alpha / sqrt(kPi) * zi * zi;
                }
                value += -kPi / (2.0 * volume * alpha * alpha) * 2.0 * zi * zj;
                if (i == j) {
                    value -= -kPi / (2.0 * volume * alpha * alpha) * zi * zj;
                }
            }
            matrix[i * n_atoms_max + j] = value;
        }
    }
    if (permutation == kMatrixPermutationEigenspectrum) {
        eigenvalues_symmetric_device(matrix, count, n_atoms_max, row, n_atoms_max);
        return;
    }
    if (permutation == kMatrixPermutationNone) {
        for (int i = 0; i < count; ++i) {
            for (int j = 0; j < count; ++j) {
                row[i * n_atoms_max + j] = matrix[i * n_atoms_max + j];
            }
            for (int j = count; j < n_atoms_max; ++j) {
                row[i * n_atoms_max + j] = 0.0;
            }
        }
        for (int i = count; i < n_atoms_max; ++i) {
            for (int j = 0; j < n_atoms_max; ++j) {
                row[i * n_atoms_max + j] = 0.0;
            }
        }
        return;
    }
    int order[256];
    double norms[256];
    double maximum_norm_squared = 1.0;
    for (int i = 0; i < count; ++i) {
        order[i] = i;
        double norm2 = 0.0;
        const int grouped_end = count & ~3;
        for (int j = 0; j < grouped_end; j += 4) {
            norm2 += matrix[i * n_atoms_max + j] * matrix[i * n_atoms_max + j]
                + matrix[i * n_atoms_max + j + 1] * matrix[i * n_atoms_max + j + 1]
                + matrix[i * n_atoms_max + j + 2] * matrix[i * n_atoms_max + j + 2]
                + matrix[i * n_atoms_max + j + 3] * matrix[i * n_atoms_max + j + 3];
        }
        for (int j = grouped_end; j < count; ++j) {
            norm2 += matrix[i * n_atoms_max + j] * matrix[i * n_atoms_max + j];
        }
        norms[i] = norm2;
        maximum_norm_squared = max(maximum_norm_squared, norm2);
    }
    if (permutation == kMatrixPermutationSortedL2) {
        for (int i = 1; i < count; ++i) {
            int current = i;
            while (current > 0 && norms[current] > norms[current - 1]) {
                const double norm_saved = norms[current]; norms[current] = norms[current - 1]; norms[current - 1] = norm_saved;
                const int index_saved = order[current]; order[current] = order[current - 1]; order[current - 1] = index_saved;
                --current;
            }
        }
        const double tie_tolerance = 4.0 * DBL_EPSILON * maximum_norm_squared;
        for (int group_begin = 0; group_begin < count;) {
            int group_end = group_begin + 1;
            while (group_end < count
                && norms[group_end - 1] - norms[group_end] <= tie_tolerance) {
                ++group_end;
            }
            for (int i = group_begin + 1; i < group_end; ++i) {
                int current = i;
                while (current > group_begin && order[current] < order[current - 1]) {
                    const int index_saved = order[current];
                    order[current] = order[current - 1];
                    order[current - 1] = index_saved;
                    --current;
                }
            }
            group_begin = group_end;
        }
    }
    for (int i = 0; i < n_atoms_max; ++i) {
        for (int j = 0; j < n_atoms_max; ++j) {
            row[i * n_atoms_max + j] = i < count && j < count
                ? matrix[order[i] * n_atoms_max + order[j]] : 0.0;
        }
    }
}

template <int MaxAngular>
__global__ void spherical_pair_kernel(
    const I64* graph_offsets,
    const I32* graph_atoms,
    const I32* graph_shifts,
    const double* graph_displacements,
    const double* graph_distance2,
    I64 atoms,
    double cutoff,
    double density_width,
    int radial_count,
    int max_angular,
    const double* gto_constants,
    const double* gamma_a,
    const double* gamma_b,
    const double* orthonormalization,
    double* records,
    double* output) {
    const I64 edge = static_cast<I64>(blockIdx.x) * blockDim.x + threadIdx.x;
    const I64 total = graph_offsets[atoms];
    if (edge >= total) return;
    const I64 center = center_for_edge(graph_offsets, atoms, edge);
    const I32 atom = graph_atoms[edge];
    records[edge * 5 + 0] = static_cast<double>(center);
    records[edge * 5 + 1] = static_cast<double>(atom);
    records[edge * 5 + 2] = static_cast<double>(graph_shifts[edge * 3 + 0]);
    records[edge * 5 + 3] = static_cast<double>(graph_shifts[edge * 3 + 1]);
    records[edge * 5 + 4] = static_cast<double>(graph_shifts[edge * 3 + 2]);
    const double distance = sqrt(fmax(0.0, graph_distance2[edge]));
    const double scale = smooth_cutoff(distance, cutoff);
    const I64 feature_count = static_cast<I64>((max_angular + 1) * (max_angular + 1) * radial_count);
    double harmonics[(MaxAngular + 1) * (MaxAngular + 1)]{};
    if (scale != 0.0) harmonic_values<MaxAngular>(graph_displacements + edge * 3, harmonics, max_angular);
    double* row = output + edge * feature_count;
    for (int angular = 0; angular <= max_angular; ++angular) {
        for (int m = -angular; m <= angular; ++m) {
            const double harmonic = scale == 0.0 ? 0.0 : harmonics[angular * angular + angular + m];
            const I64 base = static_cast<I64>(angular * angular + angular + m) * radial_count;
            for (int radial = 0; radial < radial_count; ++radial) {
                row[base + radial] = scale * harmonic * radial_value(
                    distance, angular, radial, radial_count, density_width,
                    gto_constants, gamma_a, gamma_b, orthonormalization);
            }
        }
    }
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

__device__ double smooth_radial_device(double distance, double cutoff) {
    if (distance >= cutoff) return 0.0;
    return 0.5 * (1.0 + cos(kPi * distance / cutoff));
}

__device__ double legendre_device(int degree, double x) {
    if (degree == 0) return 1.0;
    if (degree == 1) return x;
    double previous = 1.0;
    double current = x;
    for (int l = 2; l <= degree; ++l) {
        const double next = ((2.0 * l - 1.0) * x * current - (l - 1.0) * previous) / l;
        previous = current;
        current = next;
    }
    return current;
}

using DeviceComplex = mdescriptor::detail::rotational::Complex;
using mdescriptor::detail::rotational::complex_add;
using mdescriptor::detail::rotational::complex_conjugate;
using mdescriptor::detail::rotational::complex_multiply;
using mdescriptor::detail::rotational::complex_scale;

__device__ double factorial_for_so3_device(int value) {
    return value < 0 ? 0.0 : tgamma(static_cast<double>(value) + 1.0);
}

__device__ void modified_spherical_bessel_device(
    double x,
    int max_angular,
    double* result) {
    const double absolute = fabs(x);
    if (absolute < 1.0) {
        const double square = absolute * absolute;
        for (int angular = 0; angular <= max_angular; ++angular) {
            double term = 0.0;
            if (angular == 0) {
                term = 1.0;
            } else if (absolute > 0.0) {
                term = exp(
                    angular * log(absolute) + 0.5 * log(kPi)
                    - (angular + 1.0) * log(2.0)
                    - lgamma(angular + 1.5));
            }
            double sum = term;
            for (int index = 0; index < 80; ++index) {
                term *= square / (4.0 * (index + 1.0) * (index + angular + 1.5));
                sum += term;
                if (fabs(term) <= fabs(sum) * 1e-16) break;
            }
            result[angular] = sum;
        }
        return;
    }
    result[0] = sinh(absolute) / absolute;
    if (max_angular == 0) return;
    result[1] = (absolute * cosh(absolute) - sinh(absolute))
        / (absolute * absolute);
    for (int angular = 1; angular < max_angular; ++angular) {
        result[angular + 1] = result[angular - 1]
            - (2.0 * angular + 1.0) / absolute * result[angular];
    }
}

__device__ void complex_spherical_harmonics_device(
    const double* vector,
    int max_angular,
    DeviceComplex* output) {
    constexpr int stride = 9;
    double legendre[stride * stride]{};
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
        legendre[m * stride + m] = -(2.0 * m - 1.0) * sin_theta
            * legendre[(m - 1) * stride + (m - 1)];
    }
    for (int m = 0; m < max_angular; ++m) {
        legendre[(m + 1) * stride + m] = (2.0 * m + 1.0) * cos_theta
            * legendre[m * stride + m];
        for (int angular = m + 2; angular <= max_angular; ++angular) {
            legendre[angular * stride + m] = (
                (2.0 * angular - 1.0) * cos_theta
                    * legendre[(angular - 1) * stride + m]
                - (angular + m - 1.0) * legendre[(angular - 2) * stride + m])
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
                * factorial_for_so3_device(angular - m)
                / factorial_for_so3_device(angular + m));
            const double scale = normalization * legendre[angular * stride + m];
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

__global__ void so3_kernel(
    const I32* numbers,
    const I64* graph_offsets,
    const I32* graph_atoms,
    const double* graph_displacements,
    const double* graph_distance2,
    int nmax,
    int lmax,
    double cutoff,
    double alpha,
    bool weight_on,
    int quadrature_count,
    const double* basis,
    I64 features,
    I64 atoms,
    double* output) {
    const I64 center = static_cast<I64>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (center >= atoms) return;
    DeviceComplex coefficients[8 * 9 * 17]{};
    const I64 begin = graph_offsets[center];
    const I64 end = graph_offsets[center + 1];
    for (I64 edge = begin; edge < end; ++edge) {
        const double radius = sqrt(fmax(0.0, graph_distance2[edge]));
        if (radius <= 0.0 || radius >= cutoff) continue;
        const I32 atom = graph_atoms[edge];
        DeviceComplex harmonics[81]{};
        complex_spherical_harmonics_device(
            graph_displacements + edge * 3, lmax, harmonics);
        const double cutoff_value = 0.5 * (cos(kPi * radius / cutoff) + 1.0);
        const double sign = weight_on && numbers[atom] != numbers[center] ? -1.0 : 1.0;
        const double atom_weight = sign * static_cast<double>(numbers[atom])
            * 4.0 * kPi * exp(-alpha * radius * radius) * cutoff_value;
        for (int radial = 0; radial < nmax; ++radial) {
            for (int angular = 0; angular <= lmax; ++angular) {
                double radial_value = 0.0;
                for (int q_index = 0; q_index < quadrature_count; ++q_index) {
                    const double x = cos(
                        (2.0 * (q_index + 1) - 1.0) * kPi
                        / (2.0 * quadrature_count));
                    const double q = cutoff * 0.5 * (x + 1.0);
                    double bessel[9]{};
                    modified_spherical_bessel_device(
                        2.0 * alpha * radius * q, lmax, bessel);
                    radial_value += basis[radial * quadrature_count + q_index]
                        * bessel[angular];
                }
                const double angular_normalization = sqrt(
                    2.0 * sqrt(2.0) * kPi / sqrt(2.0 * angular + 1.0));
                const I64 base = static_cast<I64>(
                    radial * (lmax + 1) + angular) * (2 * lmax + 1);
                for (int m = -angular; m <= angular; ++m) {
                    coefficients[base + lmax + m] = complex_add(
                        coefficients[base + lmax + m], complex_scale(
                            harmonics[angular * angular + angular + m],
                            atom_weight * radial_value * angular_normalization));
                }
            }
        }
    }
    double* target = output + center * features;
    I64 offset = 0;
    for (int first = 0; first < nmax; ++first) {
        for (int second = 0; second <= first; ++second) {
            for (int angular = 0; angular <= lmax; ++angular) {
                double value = 0.0;
                for (int m = -angular; m <= angular; ++m) {
                    const DeviceComplex left = coefficients[
                        (second * (lmax + 1) + angular) * (2 * lmax + 1)
                            + lmax + m];
                    const DeviceComplex right = coefficients[
                        (first * (lmax + 1) + angular) * (2 * lmax + 1)
                            + lmax + m];
                    value += left.real * right.real + left.imag * right.imag;
                }
                if (offset < features) target[offset] = value;
                ++offset;
            }
        }
    }
}

__device__ int rotational_u_offset(int order, int angular) {
    const int offset = static_cast<int>(
        mdescriptor::detail::rotational::u_block_offset(angular));
    (void)order;
    return offset;
}

__device__ int rotational_u_size(int order) {
    return static_cast<int>(mdescriptor::detail::rotational::u_total_size(order));
}

__device__ void hyperspherical_u_device(
    const double* vector,
    int order,
    double cutoff,
    double rmin0,
    double rfac0,
    DeviceComplex* output) {
    mdescriptor::detail::rotational::hyperspherical_u(
        vector[0], vector[1], vector[2], order, cutoff, rfac0, rmin0, output);
}

__device__ double bispectrum_component_device(
    const DeviceComplex* total,
    int component,
    const I64* z_inner_offsets,
    const I64* inner_term_offsets,
    const double* inner_outer_coefficients,
    const I64* term_first_indices,
    const I64* term_second_indices,
    const double* term_coefficients,
    const I64* projection_offsets,
    const I64* projection_u_indices,
    const I64* projection_z_indices,
    const double* projection_scales) {
    DeviceComplex bispectrum{};
    for (I64 projection = projection_offsets[component];
         projection < projection_offsets[component + 1]; ++projection) {
        DeviceComplex z{};
        const I64 z_index = projection_z_indices[projection];
        for (I64 inner = z_inner_offsets[z_index];
             inner < z_inner_offsets[z_index + 1]; ++inner) {
            DeviceComplex value{};
            for (I64 term = inner_term_offsets[inner];
                 term < inner_term_offsets[inner + 1]; ++term) {
                value = complex_add(value, complex_multiply(
                    complex_scale(
                        total[term_first_indices[term]], term_coefficients[term]),
                    total[term_second_indices[term]]));
            }
            z = complex_add(z, complex_scale(
                value, inner_outer_coefficients[inner]));
        }
        bispectrum = complex_add(bispectrum, complex_scale(
            complex_multiply(
                complex_conjugate(total[projection_u_indices[projection]]), z),
            projection_scales[projection]));
    }
    return 2.0 * bispectrum.real;
}

__global__ void rotational_kernel(
    const I32* numbers,
    const I64* graph_offsets,
    const I32* graph_atoms,
    const double* graph_displacements,
    const double* graph_distance2,
    const double* neighbor_weights,
    const double* neighbor_radii,
    const I64* bispectrum_z_inner_offsets,
    const I64* bispectrum_inner_term_offsets,
    const double* bispectrum_inner_outer_coefficients,
    const I64* bispectrum_term_first_indices,
    const I64* bispectrum_term_second_indices,
    const double* bispectrum_term_coefficients,
    const I64* bispectrum_projection_offsets,
    const I64* bispectrum_projection_u_indices,
    const I64* bispectrum_projection_z_indices,
    const double* bispectrum_projection_scales,
    int kind,
    int nmax,
    int lmax,
    int twojmax,
    double cutoff,
    double rfac0,
    double rmin0,
    double rcutfac,
    bool normalize_u,
    int features,
    I64 atoms,
    double* output) {
    const I64 center = static_cast<I64>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (center >= atoms) return;
    double* target = output + center * static_cast<I64>(features);
    const I64 begin = graph_offsets[center];
    const I64 end = graph_offsets[center + 1];
    int expansion = kind == 3 ? max(0, twojmax) : max(0, 2 * lmax);
    if (kind != 0) {
        // SO4, SNAP and L-Bispectrum use the same hyperspherical U and
        // Clebsch--Gordan contraction as the CPU reference.  Keeping the
        // complete per-center contraction in one CUDA thread avoids any
        // order-dependent atomic reductions.
        if (expansion > 10) return;
        DeviceComplex total[kRotationalUCapacity]{};
        DeviceComplex values[kRotationalUCapacity];
        const double center_weight = kind == 1
            ? static_cast<double>(numbers[center]) : 1.0;
        const int total_size = rotational_u_size(expansion);
        for (int angular = 0; angular <= expansion; ++angular) {
            const int base = rotational_u_offset(expansion, angular);
            for (int m = 0; m <= angular; ++m) {
                total[base + m * (angular + 1) + m] = {center_weight, 0.0};
            }
        }
        for (I64 first = begin; first < end; ++first) {
            const double radius = sqrt(fmax(0.0, graph_distance2[first]));
            if (radius <= mdescriptor::detail::rotational::kBispectrumMinimumRadius) continue;
            const I32 first_atom = graph_atoms[first];
            const double neighbor_cutoff = neighbor_radii == nullptr ? cutoff
                : (neighbor_radii[center] + neighbor_radii[first_atom]) * rcutfac;
            if (radius > neighbor_cutoff || neighbor_cutoff <= rmin0) continue;
            const double* vector = graph_displacements + first * 3;
            hyperspherical_u_device(
                vector, expansion, neighbor_cutoff, rmin0, rfac0, values);
            const double cutoff_value =
                mdescriptor::detail::rotational::bispectrum_cutoff(
                    radius, neighbor_cutoff, rmin0);
            const double neighbor_weight = kind == 1
                ? static_cast<double>(numbers[first_atom])
                : (neighbor_weights == nullptr ? 1.0 : neighbor_weights[first_atom]);
            for (int index = 0; index < total_size; ++index) {
                total[index] = complex_add(total[index], complex_scale(
                    values[index], cutoff_value * neighbor_weight));
            }
        }
        if (normalize_u) {
            for (int angular = 0; angular <= expansion; ++angular) {
                const double scale = 4.0 * kPi / sqrt(angular + 1.0);
                const int base = rotational_u_offset(expansion, angular);
                for (int mb = 0; mb <= angular; ++mb) {
                    for (int ma = 0; ma <= angular; ++ma) {
                        total[base + mb * (angular + 1) + ma] = complex_scale(
                            total[base + mb * (angular + 1) + ma], scale);
                    }
                }
            }
        }
        for (int feature = 0; feature < features; ++feature) {
            target[feature] = bispectrum_component_device(
                total, feature, bispectrum_z_inner_offsets,
                bispectrum_inner_term_offsets,
                bispectrum_inner_outer_coefficients,
                bispectrum_term_first_indices,
                bispectrum_term_second_indices,
                bispectrum_term_coefficients,
                bispectrum_projection_offsets,
                bispectrum_projection_u_indices,
                bispectrum_projection_z_indices,
                bispectrum_projection_scales);
        }
        return;
    }
    for (int feature = 0; feature < features; ++feature) {
        int wanted_l = 0;
        int n1 = 0;
        int n2 = 0;
        if (kind == 0) {
            int remainder = feature;
            for (n1 = 0; n1 < nmax; ++n1) {
                const int block = (n1 + 1) * (lmax + 1);
                if (remainder < block) break;
                remainder -= block;
            }
            n2 = remainder / (lmax + 1);
            wanted_l = remainder % (lmax + 1);
        } else {
            // Components are generated in the same lexicographic order as the
            // CPU rotational descriptor.  A compact modulo representation keeps
            // the kernel independent of a host-side component table.
            const int component_count = max(1, features);
            const int component = feature % component_count;
            const int order = max(0, expansion);
            wanted_l = order == 0 ? 0 : component % (order + 1);
            n1 = order == 0 ? 0 : (component / (order + 1)) % (order + 1);
            n2 = order == 0 ? 0 : (component / ((order + 1) * (order + 1))) % (order + 1);
        }
        double value = 0.0;
        for (I64 first = begin; first < end; ++first) {
            const double first_distance = sqrt(fmax(0.0, graph_distance2[first]));
            if (first_distance <= 1e-12) continue;
            const I32 first_atom = graph_atoms[first];
            const double first_cutoff = neighbor_radii == nullptr ? cutoff
                : (neighbor_radii[center] + neighbor_radii[first_atom]) * rcutfac;
            if (first_distance > first_cutoff) continue;
            const double first_weight = neighbor_weights == nullptr
                ? (kind == 1 ? static_cast<double>(numbers[first_atom]) : 1.0)
                : neighbor_weights[first_atom];
            const double first_radial = smooth_radial_device(first_distance, first_cutoff)
                * pow(fmax(0.0, (first_distance - rmin0) / fmax(first_cutoff - rmin0, 1e-12)), n1 + 1)
                * first_weight;
            for (I64 second = first; second < end; ++second) {
                const double second_distance = sqrt(fmax(0.0, graph_distance2[second]));
                if (second_distance <= 1e-12) continue;
                const I32 second_atom = graph_atoms[second];
                const double second_cutoff = neighbor_radii == nullptr ? cutoff
                    : (neighbor_radii[center] + neighbor_radii[second_atom]) * rcutfac;
                if (second_distance > second_cutoff) continue;
                const double second_weight = neighbor_weights == nullptr
                    ? (kind == 1 ? static_cast<double>(numbers[second_atom]) : 1.0)
                    : neighbor_weights[second_atom];
                const double second_radial = smooth_radial_device(second_distance, second_cutoff)
                    * pow(fmax(0.0, (second_distance - rmin0) / fmax(second_cutoff - rmin0, 1e-12)), n2 + 1)
                    * second_weight;
                const double* first_vector = graph_displacements + first * 3;
                const double* second_vector = graph_displacements + second * 3;
                const double denominator = first_distance * second_distance;
                const double cosine = denominator > 0.0
                    ? fmin(1.0, fmax(-1.0, (first_vector[0] * second_vector[0]
                        + first_vector[1] * second_vector[1]
                        + first_vector[2] * second_vector[2]) / denominator)) : 1.0;
                value += first_radial * second_radial * legendre_device(wanted_l, cosine);
            }
        }
        if (normalize_u) value *= 4.0 * kPi / sqrt(wanted_l + 1.0);
        target[feature] = value;
    }
    (void)rfac0;
}

__device__ void mtp4_radial_basis_device(
    double r_sq,
    int kind,
    int basis_size,
    double min_dist,
    double max_dist,
    double max_dist_sq,
    double max_dist_sq_minus_eps,
    double exp_ratio,
    double zeroth,
    const double* recursive,
    const double* vdw_damped_params,
    int self_species,
    int neighbor_species,
    double* output) {
    if (kind == 3) {
        const double r_sq_3 = r_sq * r_sq * r_sq;
        const double self_radius = vdw_damped_params[2 + self_species];
        const double neighbor_radius = vdw_damped_params[2 + neighbor_species];
        const double damp = vdw_damped_params[0] * (self_radius + neighbor_radius)
            + vdw_damped_params[1];
        const double damp_sq = damp * damp;
        const double damp_6 = damp_sq * damp_sq * damp_sq;
        output[0] = 100.0 / (r_sq_3 + damp_6);
        if (basis_size > 1) {
            output[1] = 100.0 / (r_sq_3 * r_sq + damp_6 * damp_sq);
        }
        return;
    }
    if (kind == 2) {
        const double min_dist_sq = min_dist * min_dist;
        if (r_sq <= min_dist_sq * 1.02) {
            for (int index = 0; index < basis_size; ++index) output[index] = 0.0;
            output[0] = 2.5;
            return;
        }
        if (r_sq >= max_dist_sq) {
            for (int index = 0; index < basis_size; ++index) output[index] = 0.0;
            return;
        }
        const double x_sq = min_dist_sq / r_sq;
        const double my_exp = exp(1.0 / (x_sq - 1.0));
        const double mult = x_sq * x_sq * x_sq * my_exp;
        output[0] = 2.5 * pow(1.0 - 2.71828182845904524 * my_exp, 3.0);
        if (basis_size == 1) return;
        output[1] = 102.295067549833082 * mult;
        double previous = 0.0;
        for (int index = 1; index < basis_size - 1; ++index) {
            output[index + 1] = recursive[index * 3]
                * ((x_sq + recursive[index * 3 + 1]) * output[index]
                    + recursive[index * 3 + 2] * previous);
            previous = output[index];
        }
        return;
    }
    if (kind == 1) {
        if (r_sq >= max_dist_sq) {
            for (int index = 0; index < basis_size; ++index) output[index] = 0.0;
            return;
        }
        const double radius = sqrt(r_sq);
        const double ksi = (2.0 * radius - (min_dist + max_dist)) / (max_dist - min_dist);
        const double edge = radius - max_dist;
        output[0] = edge * edge;
        if (basis_size > 1) output[1] = ksi * edge * edge;
        for (int index = 2; index < basis_size; ++index) {
            output[index] = 2.0 * ksi * output[index - 1] - output[index - 2];
        }
        return;
    }
    if (r_sq >= max_dist_sq_minus_eps) {
        for (int index = 0; index < basis_size; ++index) output[index] = 0.0;
        return;
    }
    const double x_sq = r_sq / max_dist_sq;
    const double mult = exp(exp_ratio / (1.0 - x_sq));
    output[0] = zeroth * mult;
    double previous = 0.0;
    for (int index = 0; index < basis_size - 1; ++index) {
        output[index + 1] = recursive[index * 3]
            * ((x_sq + recursive[index * 3 + 1]) * output[index]
                + recursive[index * 3 + 2] * previous);
        previous = output[index];
    }
}

__device__ void ace_spherical_harmonics(
    const double* vector, int max_angular, DeviceComplex* output) {
    constexpr int stride = 21;
    for (int index = 0; index < (max_angular + 1) * (max_angular + 1); ++index) {
        output[index] = {0.0, 0.0};
    }
    const double radius = sqrt(
        vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2]);
    if (radius <= 1e-14) {
        output[0] = {0.5 / sqrt(kPi), 0.0};
        return;
    }
    const double cos_theta = fmax(-1.0, fmin(1.0, vector[2] / radius));
    const double sin_theta = hypot(vector[0], vector[1]) / radius;
    double legendre[stride * stride]{};
    legendre[0] = 1.0;
    for (int m = 1; m <= max_angular; ++m) {
        legendre[m * stride + m] = -(2.0 * m - 1.0) * sin_theta
            * legendre[(m - 1) * stride + (m - 1)];
    }
    for (int m = 0; m < max_angular; ++m) {
        legendre[(m + 1) * stride + m] = (2.0 * m + 1.0) * cos_theta
            * legendre[m * stride + m];
        for (int angular = m + 2; angular <= max_angular; ++angular) {
            legendre[angular * stride + m] = (
                (2.0 * angular - 1.0) * cos_theta
                    * legendre[(angular - 1) * stride + m]
                - (angular + m - 1.0) * legendre[(angular - 2) * stride + m])
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
            const double scale = normalization * legendre[angular * stride + m];
            const DeviceComplex positive = {scale * cos_m, scale * sin_m};
            output[angular * angular + angular + m] = positive;
            if (m > 0) {
                output[angular * angular + angular - m] = m % 2 == 0
                    ? complex_conjugate(positive)
                    : complex_scale(complex_conjugate(positive), -1.0);
            }
        }
    }
}

__device__ void ace_radial_values(
    double distance,
    double transform_a,
    double transform_p,
    double transform_r0,
    double t_left,
    double t_right,
    int p_left,
    int p_right,
    const double* radial_a,
    const double* radial_b,
    const double* radial_c,
    int radial_count,
    double* result) {
    for (int index = 0; index < radial_count; ++index) result[index] = 0.0;
    const double t = pow((transform_a + transform_r0)
        / (transform_a + distance), transform_p);
    if ((p_left > 0 && t < t_left) || (p_right > 0 && t > t_right)) return;
    const double envelope = pow(t - t_left, p_left) * pow(t - t_right, p_right);
    result[0] = radial_a[0] * envelope;
    if (radial_count == 1) return;
    result[1] = (radial_a[1] * t + radial_b[1]) * result[0];
    for (int n = 2; n < radial_count; ++n) {
        result[n] = (radial_a[n] * t + radial_b[n]) * result[n - 1]
            + radial_c[n] * result[n - 2];
    }
}

__device__ double chebyshev_device(int order, double x) {
    if (order <= 0) return 1.0;
    double previous = 1.0;
    double current = x;
    for (int index = 2; index <= order; ++index) {
        const double next = 2.0 * x * current - previous;
        previous = current;
        current = next;
    }
    return current;
}
