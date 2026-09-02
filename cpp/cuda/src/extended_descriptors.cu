#include "mdescriptor/cuda/extended_descriptors.hpp"

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

void check_cuda(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) return;
    if (status == cudaErrorMemoryAllocation) throw CudaOutOfMemory(operation);
    if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver
        || status == cudaErrorSystemDriverMismatch) {
        throw CudaUnavailable(operation);
    }
    throw std::runtime_error(operation);
}

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
        check_cuda(
            cudaMemcpyAsync(
                result.data(), source, count * sizeof(T), cudaMemcpyDeviceToHost,
                context.stream()),
            operation);
        context.synchronize();
    }
    return result;
}

bool inverse_host3(const double* matrix, double* inverse) {
    const double determinant = matrix[0] * (matrix[4] * matrix[8] - matrix[5] * matrix[7])
        - matrix[1] * (matrix[3] * matrix[8] - matrix[5] * matrix[6])
        + matrix[2] * (matrix[3] * matrix[7] - matrix[4] * matrix[6]);
    if (!std::isfinite(determinant) || std::abs(determinant) <= 1e-12) return false;
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

std::vector<std::size_t> cpu_pair_order(
    const detail::StructureBatchView& batch,
    double cutoff,
    const std::vector<double>& records) {
    const std::size_t edges = records.size() / 5U;
    std::vector<std::size_t> order(edges);
    if (edges == 0) return order;

    struct Entry {
        I64 center = 0;
        I64 cell = 0;
        I64 extended = 0;
        std::size_t edge = 0;
    };
    std::vector<Entry> entries;
    entries.reserve(edges);
    for (std::size_t edge = 0; edge < edges; ++edge) {
        const I64 center = static_cast<I64>(records[edge * 5U + 0]);
        if (center < 0 || center >= batch.atoms) {
            throw std::runtime_error("CUDA pair output contains an invalid center index");
        }
        I64 structure = 0;
        while (structure + 1 < batch.structures
            && batch.offsets[structure + 1] <= center) {
            ++structure;
        }
        const I64 begin = batch.offsets[structure];
        const I64 end = batch.offsets[structure + 1];
        const I64 atom = static_cast<I64>(records[edge * 5U + 1]);
        if (atom < begin || atom >= end) {
            throw std::runtime_error("CUDA pair output contains an invalid neighbor index");
        }

        const bool periodic = batch.pbc[structure * 3 + 0] == 1
            && batch.pbc[structure * 3 + 1] == 1
            && batch.pbc[structure * 3 + 2] == 1;
        const I64 atom_count = end - begin;
        std::array<int, 3> bounds{0, 0, 0};
        double inverse[9]{};
        if (periodic) {
            const double* cell = batch.cells + structure * 9;
            if (!inverse_host3(cell, inverse)) {
                throw std::invalid_argument("cannot order CUDA pair output for a singular cell");
            }
            for (int axis = 0; axis < 3; ++axis) {
                const double x = inverse[axis];
                const double y = inverse[3 + axis];
                const double z = inverse[6 + axis];
                bounds[static_cast<std::size_t>(axis)] = static_cast<int>(std::floor(
                    cutoff * std::sqrt(x * x + y * y + z * z) + 1.0));
            }
        }

        double minimum[3]{};
        double maximum[3]{};
        bool first_image = true;
        const I64 image_count = periodic
            ? static_cast<I64>(2 * bounds[0] + 1)
                * static_cast<I64>(2 * bounds[1] + 1)
                * static_cast<I64>(2 * bounds[2] + 1)
            : 1;
        const double* cell = batch.cells + structure * 9;
        for (int sx = -bounds[0]; sx <= bounds[0]; ++sx) {
            for (int sy = -bounds[1]; sy <= bounds[1]; ++sy) {
                for (int sz = -bounds[2]; sz <= bounds[2]; ++sz) {
                    if (!periodic && (sx != 0 || sy != 0 || sz != 0)) continue;
                    const double shift_x = sx * cell[0] + sy * cell[3] + sz * cell[6];
                    const double shift_y = sx * cell[1] + sy * cell[4] + sz * cell[7];
                    const double shift_z = sx * cell[2] + sy * cell[5] + sz * cell[8];
                    for (I64 source = begin; source < end; ++source) {
                        const double* position = batch.positions + source * 3;
                        const double values[3] = {
                            position[0] + shift_x, position[1] + shift_y, position[2] + shift_z,
                        };
                        if (first_image) {
                            for (int axis = 0; axis < 3; ++axis) {
                                minimum[axis] = values[axis];
                                maximum[axis] = values[axis];
                            }
                            first_image = false;
                        } else {
                            for (int axis = 0; axis < 3; ++axis) {
                                minimum[axis] = std::min(minimum[axis], values[axis]);
                                maximum[axis] = std::max(maximum[axis], values[axis]);
                            }
                        }
                    }
                }
            }
        }
        if (first_image) {
            throw std::runtime_error("cannot order CUDA pair output for an empty structure");
        }
        for (int axis = 0; axis < 3; ++axis) {
            minimum[axis] -= 1e-10;
            maximum[axis] += 1e-10;
        }
        std::array<int, 3> dimensions{1, 1, 1};
        double spacing[3]{};
        for (int axis = 0; axis < 3; ++axis) {
            const double range = maximum[axis] - minimum[axis];
            dimensions[static_cast<std::size_t>(axis)] = std::max(
                1, static_cast<int>(range / cutoff));
            spacing[axis] = std::max(
                cutoff, range / dimensions[static_cast<std::size_t>(axis)]);
        }
        const int sx = static_cast<int>(records[edge * 5U + 2]);
        const int sy = static_cast<int>(records[edge * 5U + 3]);
        const int sz = static_cast<int>(records[edge * 5U + 4]);
        const double* position = batch.positions + atom * 3;
        const double image[3] = {
            position[0] + sx * cell[0] + sy * cell[3] + sz * cell[6],
            position[1] + sx * cell[1] + sy * cell[4] + sz * cell[7],
            position[2] + sx * cell[2] + sy * cell[5] + sz * cell[8],
        };
        int coordinates[3]{};
        for (int axis = 0; axis < 3; ++axis) {
            const int coordinate = static_cast<int>((image[axis] - minimum[axis]) / spacing[axis]);
            coordinates[axis] = std::max(
                0, std::min(dimensions[static_cast<std::size_t>(axis)] - 1, coordinate));
        }
        const I64 cell_index = coordinates[0]
            + static_cast<I64>(dimensions[0]) * (coordinates[1]
                + static_cast<I64>(dimensions[1]) * coordinates[2]);
        I64 extended_index = atom - begin;
        if (periodic) {
            const I64 y_extent = 2 * static_cast<I64>(bounds[1]) + 1;
            const I64 z_extent = 2 * static_cast<I64>(bounds[2]) + 1;
            const I64 image_index = (static_cast<I64>(sx) + bounds[0]) * y_extent * z_extent
                + (static_cast<I64>(sy) + bounds[1]) * z_extent
                + static_cast<I64>(sz) + bounds[2];
            extended_index = image_index * atom_count + (atom - begin);
            (void)image_count;
        }
        entries.push_back({center, cell_index, extended_index, edge});
    }
    std::stable_sort(entries.begin(), entries.end(), [](const Entry& left, const Entry& right) {
        if (left.center != right.center) return left.center < right.center;
        if (left.cell != right.cell) return left.cell < right.cell;
        return left.extended < right.extended;
    });
    for (std::size_t index = 0; index < entries.size(); ++index) {
        order[index] = entries[index].edge;
    }
    return order;
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
    const auto values = context.download_output(output_size);
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

template <int MaxAngular>
void launch_spherical_pair(
    int requested,
    cudaStream_t stream,
    const I64* graph_offsets,
    const I32* graph_atoms,
    const I32* graph_shifts,
    const double* graph_displacements,
    const double* graph_distance2,
    I64 atoms,
    double cutoff,
    double density_width,
    int radial_count,
    const double* gto_constants,
    const double* gamma_a,
    const double* gamma_b,
    const double* orthonormalization,
    double* records,
    double* output) {
    if (requested == MaxAngular) {
        const I64 total_hint = static_cast<I64>(atoms > 0 ? 1 : 0);
        (void)total_hint;
        // The kernel reads the final graph offset, so its launch count is
        // supplied by the caller and does not require a host graph copy.
        return;
    }
    if constexpr (MaxAngular < 31) {
        launch_spherical_pair<MaxAngular + 1>(
            requested, stream, graph_offsets, graph_atoms, graph_shifts,
            graph_displacements, graph_distance2, atoms, cutoff, density_width,
            radial_count, gto_constants, gamma_a, gamma_b, orthonormalization,
            records, output);
    } else {
        throw std::invalid_argument("CUDA pair descriptor max_angular is too large");
    }
}

py::dict compute_matrix_descriptor(
    CudaExecutionContext& context,
    DeviceBatch& batch,
    const detail::StructureBatchView& host_batch,
    int kind,
    const std::string& name,
    const py::dict& options) {
    int n_atoms_max = option(options, "n_atoms_max", 0);
    if (n_atoms_max <= 0) {
        for (I64 structure = 0; structure < host_batch.structures; ++structure) {
            n_atoms_max = std::max(
                n_atoms_max,
                static_cast<int>(host_batch.offsets[structure + 1] - host_batch.offsets[structure]));
        }
    }
    if (n_atoms_max <= 0 || n_atoms_max > 256) {
        throw std::invalid_argument(
            "CUDA matrix descriptors require 1 <= n_atoms_max <= 256");
    }
    for (I64 structure = 0; structure < host_batch.structures; ++structure) {
        const I64 count = host_batch.offsets[structure + 1] - host_batch.offsets[structure];
        if (count <= 0) {
            throw std::invalid_argument("matrix descriptors do not accept empty structures");
        }
        if (count > n_atoms_max) {
            throw std::invalid_argument("structure exceeds n_atoms_max");
        }
    }
    const std::string permutation_name = option(options, "permutation", std::string("sorted_l2"));
    const int permutation = permutation_name == "none" ? kMatrixPermutationNone
        : permutation_name == "sorted_l2" ? kMatrixPermutationSortedL2 : kMatrixPermutationEigenspectrum;
    if (permutation_name != "none" && permutation_name != "sorted_l2"
        && permutation_name != "eigenspectrum") {
        throw std::invalid_argument("invalid CUDA matrix permutation");
    }
    const I64 columns = permutation == kMatrixPermutationEigenspectrum
        ? n_atoms_max : static_cast<I64>(n_atoms_max) * n_atoms_max;
    const std::size_t output_size = static_cast<std::size_t>(host_batch.structures)
        * static_cast<std::size_t>(columns);
    const std::size_t matrix_stride = static_cast<std::size_t>(n_atoms_max) * n_atoms_max
        + (kind == kMatrixKindEwald ? static_cast<std::size_t>(3 * n_atoms_max) : 0);
    const std::size_t matrix_size = static_cast<std::size_t>(host_batch.structures)
        * matrix_stride;
    double* output = context.output_buffer(output_size);
    auto* matrices = static_cast<double*>(context.workspace_buffer(
        matrix_size * sizeof(double)));
    const double exponent = option(options, "exponent", 2.4);
    const double accuracy = option(options, "accuracy", 1e-5);
    const double weight = option(options, "w", 1.0);
    const double r_cut = option(options, "r_cut", 0.0);
    const double g_cut = option(options, "g_cut", 0.0);
    const double split = option(options, "a", 0.0);
    constexpr unsigned block_size = 64;
    const auto blocks = static_cast<unsigned>(
        (host_batch.structures + block_size - 1) / block_size);
    if (host_batch.structures > 0) {
        matrix_kernel<<<blocks, block_size, 0, context.stream()>>>(
            batch.numbers(), batch.positions(), batch.cells(), batch.offsets(),
            host_batch.structures, n_atoms_max, static_cast<I64>(matrix_stride), kind, permutation, exponent,
            accuracy, weight, r_cut, g_cut, split, matrices, output);
        check_cuda(cudaGetLastError(), "CUDA matrix kernel launch failed");
    }
    const auto values = context.download_output(output_size);
    py::dict result;
    result["values"] = values_array(values, host_batch.structures, columns);
    result["level"] = "structure";
    result["labels"] = labels_option(options, name, columns);
    result["metadata"] = metadata(options, name);
    return result;
}

__device__ int pair_channel_device(int first, int second, int species_count) {
    return mdescriptor::detail::mbtr::pair_channel(first, second, species_count);
}

__global__ void acsf_kernel(
    const I32* numbers,
    const I64* graph_offsets,
    const I32* graph_atoms,
    const double* graph_displacements,
    const double* graph_distance2,
    const I32* species,
    int species_count,
    double r_cut,
    const double* g2,
    int n_g2,
    const double* g3,
    int n_g3,
    const double* g4,
    int n_g4,
    const double* g5,
    int n_g5,
    I64 atoms,
    double* output) {
    const I64 center = static_cast<I64>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (center >= atoms) return;
    const int types = species_count;
    const int per_type = 1 + n_g2 + n_g3;
    const int angular_offset = types * per_type;
    double* values = output + center * (
        angular_offset + (n_g4 + n_g5) * types * (types + 1) / 2);
    const I64 begin = graph_offsets[center];
    const I64 end = graph_offsets[center + 1];
    for (I64 edge = begin; edge < end; ++edge) {
        const I32 atom = graph_atoms[edge];
        const int type = species_index(numbers[atom], species, species_count);
        if (type < 0) continue;
        const double distance = sqrt(fmax(0.0, graph_distance2[edge]));
        const double fc = 0.5 * (cos(kPi * distance / r_cut) + 1.0);
        I64 offset = static_cast<I64>(type) * per_type;
        values[offset++] += fc;
        for (int p = 0; p < n_g2; ++p) {
            const double eta = g2[p * 2 + 0];
            const double rs = g2[p * 2 + 1];
            values[offset++] += exp(-eta * (distance - rs) * (distance - rs)) * fc;
        }
        for (int p = 0; p < n_g3; ++p) {
            values[offset++] += cos(g3[p] * distance) * fc;
        }
    }
    for (I64 first = begin; first < end; ++first) {
        const I32 first_atom = graph_atoms[first];
        const int first_type = species_index(numbers[first_atom], species, species_count);
        if (first_type < 0) continue;
        const double first_distance2 = graph_distance2[first];
        const double first_distance = sqrt(fmax(0.0, first_distance2));
        const double first_cutoff = 0.5 * (cos(kPi * first_distance / r_cut) + 1.0);
        const double* first_vector = graph_displacements + first * 3;
        for (I64 second = begin; second < first; ++second) {
            const I32 second_atom = graph_atoms[second];
            const int second_type = species_index(numbers[second_atom], species, species_count);
            if (second_type < 0) continue;
            const double second_distance2 = graph_distance2[second];
            const double second_distance = sqrt(fmax(0.0, second_distance2));
            const double second_cutoff = 0.5 * (cos(kPi * second_distance / r_cut) + 1.0);
            const double* second_vector = graph_displacements + second * 3;
            const double dx = first_vector[0] - second_vector[0];
            const double dy = first_vector[1] - second_vector[1];
            const double dz = first_vector[2] - second_vector[2];
            const double third_distance2 = dx * dx + dy * dy + dz * dz;
            const double third_distance = sqrt(third_distance2);
            const double cosine = first_distance > 0.0 && second_distance > 0.0
                ? (first_vector[0] * second_vector[0]
                    + first_vector[1] * second_vector[1]
                    + first_vector[2] * second_vector[2])
                    / (first_distance * second_distance) : 0.0;
            const double clamped_cosine = fmin(1.0, fmax(-1.0, cosine));
            const int channel = pair_channel_device(first_type, second_type, species_count);
            const I64 base = angular_offset
                + static_cast<I64>(channel) * (n_g4 + n_g5);
            const double fc4 = first_cutoff * second_cutoff
                * (third_distance <= r_cut
                    ? 0.5 * (cos(kPi * third_distance / r_cut) + 1.0) : 0.0);
            const double fc5 = first_cutoff * second_cutoff;
            const double distance_sum = first_distance2 + second_distance2 + third_distance2;
            for (int p = 0; p < n_g4; ++p) {
                const double eta = g4[p * 3 + 0];
                const double zeta = g4[p * 3 + 1];
                const double lambda = g4[p * 3 + 2];
                const double angular = pow(
                    fmax(0.0, 0.5 * (1.0 + lambda * clamped_cosine)), zeta);
                const double radial = third_distance <= r_cut
                    ? exp(-eta * distance_sum) : 0.0;
                values[base + p] += 2.0 * angular * radial * fc4;
            }
            for (int p = 0; p < n_g5; ++p) {
                const double eta = g5[p * 3 + 0];
                const double zeta = g5[p * 3 + 1];
                const double lambda = g5[p * 3 + 2];
                values[base + n_g4 + p] += 2.0
                    * pow(fmax(0.0, 0.5 * (1.0 + lambda * clamped_cosine)), zeta)
                    * exp(-eta * (first_distance2 + second_distance2)) * fc5;
            }
        }
    }
}

std::vector<double> option_rows(
    const py::dict& options,
    const char* key,
    int columns) {
    std::vector<double> result;
    const py::str name(key);
    if (!options.contains(name) || options[name].is_none()) return result;
    const py::object value = options[name];
    if (py::isinstance<py::dict>(value)) {
        const py::dict object = py::cast<py::dict>(value);
        auto values = [&](const char* field) {
            const py::str field_name(field);
            if (!object.contains(field_name) || object[field_name].is_none()) {
                return std::vector<double>{};
            }
            return py::cast<std::vector<double>>(object[field_name]);
        };
        const auto first = values("eta");
        if (columns == 2) {
            auto second = values("Rs");
            if (second.empty()) second = values("rs");
            for (double left : first) for (double right : second) {
                result.push_back(left); result.push_back(right);
            }
        } else {
            const auto second = values("zeta");
            auto third = values("lambda");
            if (third.empty()) third = values("lambdas");
            for (double left : first) for (double middle : second) for (double right : third) {
                result.push_back(left); result.push_back(middle); result.push_back(right);
            }
        }
        return result;
    }
    try {
        const py::sequence rows = py::cast<py::sequence>(value);
        for (const py::handle item : rows) {
            const py::sequence row = py::cast<py::sequence>(item);
            if (py::len(row) != columns) {
                throw std::invalid_argument(std::string(key) + " has an invalid row width");
            }
            for (const py::handle component : row) result.push_back(py::cast<double>(component));
        }
    } catch (const py::cast_error&) {
        throw std::invalid_argument(std::string(key) + " must be an array or object");
    }
    return result;
}

std::vector<double> option_values(const py::dict& options, const char* key) {
    const py::str name(key);
    if (!options.contains(name) || options[name].is_none()) return {};
    try {
        return py::cast<std::vector<double>>(options[name]);
    } catch (const py::cast_error&) {
        throw std::invalid_argument(std::string(key) + " must be an array");
    }
}

py::dict compute_acsf_descriptor(
    CudaExecutionContext& context,
    DeviceBatch& batch,
    DeviceNeighborGraph& graph,
    const detail::StructureBatchView& host_batch,
    const py::dict& options) {
    const auto species = species_option(options);
    if (species.empty()) throw std::invalid_argument("ACSF species must not be empty");
    const double cutoff = option(options, "r_cut", 6.0);
    const auto g2 = option_rows(options, "g2_params", 2);
    const auto g3 = option_values(options, "g3_params");
    const auto g4 = option_rows(options, "g4_params", 3);
    const auto g5 = option_rows(options, "g5_params", 3);
    const int n_g2 = static_cast<int>(g2.size() / 2);
    const int n_g3 = static_cast<int>(g3.size());
    const int n_g4 = static_cast<int>(g4.size() / 3);
    const int n_g5 = static_cast<int>(g5.size() / 3);
    const I64 columns = static_cast<I64>(1 + n_g2 + n_g3) * species.size()
        + static_cast<I64>(n_g4 + n_g5) * species.size() * (species.size() + 1) / 2;
    graph.build_dpa(context, batch, host_batch, cutoff, true, false, false);
    const std::size_t size = static_cast<std::size_t>(batch.atoms())
        * static_cast<std::size_t>(columns);
    double* output = context.output_buffer(size);
    check_cuda(cudaMemsetAsync(output, 0, size * sizeof(double), context.stream()),
        "could not clear CUDA ACSF output");
    DeviceBuffer<I32> d_species;
    DeviceBuffer<double> d_g2;
    DeviceBuffer<double> d_g3;
    DeviceBuffer<double> d_g4;
    DeviceBuffer<double> d_g5;
    d_species.upload(species.data(), species.size(), context.stream(), "could not upload ACSF species");
    d_g2.upload(g2.data(), g2.size(), context.stream(), "could not upload ACSF G2 parameters");
    d_g3.upload(g3.data(), g3.size(), context.stream(), "could not upload ACSF G3 parameters");
    d_g4.upload(g4.data(), g4.size(), context.stream(), "could not upload ACSF G4 parameters");
    d_g5.upload(g5.data(), g5.size(), context.stream(), "could not upload ACSF G5 parameters");
    if (batch.atoms() > 0) {
        constexpr unsigned block_size = 64;
        acsf_kernel<<<static_cast<unsigned>((batch.atoms() + block_size - 1) / block_size),
            block_size, 0, context.stream()>>>(
                batch.numbers(), graph.offsets(), graph.atoms(), graph.displacements(),
                graph.distance2(), d_species.get(), static_cast<int>(species.size()), cutoff,
                d_g2.get(), n_g2, d_g3.get(), n_g3, d_g4.get(), n_g4, d_g5.get(), n_g5,
                batch.atoms(), output);
        check_cuda(cudaGetLastError(), "CUDA ACSF kernel launch failed");
    }
    const auto values = context.download_output(size);
    return atom_result(values, batch.atoms(), columns, "ACSF", options, false,
        std::vector<I64>(host_batch.offsets, host_batch.offsets + host_batch.structures + 1));
}

__device__ void add_histogram_device(
    double* target,
    double value,
    double weight,
    double grid_min,
    double grid_max,
    double grid_sigma,
    int grid_n,
    bool normalize) {
    mdescriptor::detail::mbtr::add_histogram(
        target, value, weight, grid_min, grid_max, grid_sigma, grid_n, normalize);
}

__device__ double mbtr_weight_device(
    int weighting,
    double scale,
    double threshold,
    double r_cut,
    double sharpness,
    double first,
    double second,
    double third) {
    return mdescriptor::detail::mbtr::weight(
        weighting, scale, threshold, r_cut, sharpness, first, second, third);
}

__device__ void normalize_mbtr_device(
    double* values,
    I64 features,
    int normalization,
    int atom_count,
    const int* species_counts,
    int species_count,
    double volume,
    int geometry,
    int grid_n,
    bool local) {
    if (normalization == mdescriptor::detail::mbtr::kNormalizationL2) {
        mdescriptor::detail::mbtr::normalize_l2(values, features);
    } else if (normalization == mdescriptor::detail::mbtr::kNormalizationNAtoms) {
        mdescriptor::detail::mbtr::normalize_n_atoms(values, features, atom_count);
    } else if (normalization == mdescriptor::detail::mbtr::kNormalizationValleOganov) {
        mdescriptor::detail::mbtr::normalize_valle_oganov(
            values, volume, species_counts, species_count, geometry, grid_n, local);
    }
}

__global__ void mbtr_kernel(
    const I32* numbers,
    const I32* atom_types,
    const double* cells,
    const I64* offsets,
    const I64* graph_offsets,
    const I32* graph_atoms,
    const I32* graph_shifts,
    const double* graph_displacements,
    const double* graph_distance2,
    int species_count,
    int geometry,
    int weighting,
    int normalization,
    double grid_min,
    double grid_max,
    double grid_sigma,
    int grid_n,
    bool normalize_gaussians,
    double scale,
    double threshold,
    double r_cut,
    double sharpness,
    bool local,
    I64 structures,
    I64 atoms,
    I64 features,
    double* output) {
    const I64 row = static_cast<I64>(blockIdx.x) * blockDim.x + threadIdx.x;
    const I64 rows = local ? atoms : structures;
    if (row >= rows) return;
    double* target = output + row * features;
    I64 begin = 0;
    I64 end = 0;
    I64 structure = row;
    if (local) {
        begin = row;
        end = row + 1;
        structure = 0;
        while (structure + 1 < structures && offsets[structure + 1] <= row) ++structure;
    } else {
        begin = offsets[row];
        end = offsets[row + 1];
    }
    const int atom_count = static_cast<int>(offsets[structure + 1] - offsets[structure]);
    int species_counts[64]{};
    const bool needs_species_counts = normalization
        == mdescriptor::detail::mbtr::kNormalizationValleOganov
        && !local && geometry != mbtr::kGeometryAtomicNumber;
    if (needs_species_counts && species_count <= 64) {
        for (I64 atom = offsets[structure]; atom < offsets[structure + 1]; ++atom) {
            const int type = atom_types[atom];
            if (type >= 0) ++species_counts[type];
        }
    }
    if (geometry == mbtr::kGeometryAtomicNumber) {
        if (local) {
            const int type = atom_types[row];
            if (type >= 0) add_histogram_device(
                target + type * grid_n, static_cast<double>(numbers[row]), 1.0,
                grid_min, grid_max, grid_sigma, grid_n, normalize_gaussians);
        } else {
            for (I64 atom = offsets[row]; atom < offsets[row + 1]; ++atom) {
                const int type = atom_types[atom];
                if (type >= 0) add_histogram_device(
                    target + type * grid_n, static_cast<double>(numbers[atom]), 1.0,
                    grid_min, grid_max, grid_sigma, grid_n, normalize_gaussians);
            }
        }
        normalize_mbtr_device(
            target, features, normalization, atom_count, species_counts,
            species_count, 0.0, geometry, grid_n, local);
        return;
    }
    if (local) {
        const I64 center = row;
        const I64 graph_begin = graph_offsets[center];
        const I64 graph_end = graph_offsets[center + 1];
        if (geometry == mbtr::kGeometryDistance
            || geometry == mbtr::kGeometryInverseDistance) {
            for (I64 edge = graph_begin; edge < graph_end; ++edge) {
                const I32 atom = graph_atoms[edge];
                const double distance = sqrt(fmax(0.0, graph_distance2[edge]));
                if (distance <= 1e-12
                    || (atom == center && graph_shifts[edge * 3] == 0
                        && graph_shifts[edge * 3 + 1] == 0
                        && graph_shifts[edge * 3 + 2] == 0)) continue;
                const int type = atom_types[atom];
                if (type < 0) continue;
                const double value = geometry == mbtr::kGeometryDistance
                    ? distance : 1.0 / distance;
                add_histogram_device(
                    target + (type + 1) * grid_n, value,
                    mbtr_weight_device(weighting, scale, threshold, r_cut, sharpness,
                        distance, 0.0, 0.0), grid_min, grid_max, grid_sigma,
                    grid_n, normalize_gaussians);
            }
        } else {
            const int element_count = species_count + 1;
            const int reserved = element_count * (element_count + 1) / 2;
            for (I64 first = graph_begin; first < graph_end; ++first) {
                const double first_distance = sqrt(fmax(0.0, graph_distance2[first]));
                if (first_distance <= 1e-12) continue;
                for (I64 second = graph_begin; second < first; ++second) {
                    const double second_distance = sqrt(fmax(0.0, graph_distance2[second]));
                    if (second_distance <= 1e-12) continue;
                    const double dx = graph_displacements[first * 3] - graph_displacements[second * 3];
                    const double dy = graph_displacements[first * 3 + 1] - graph_displacements[second * 3 + 1];
                    const double dz = graph_displacements[first * 3 + 2] - graph_displacements[second * 3 + 2];
                    const double third_distance = sqrt(dx * dx + dy * dy + dz * dz);
                    const double cosine = fmin(1.0, fmax(-1.0,
                        (first_distance * first_distance + second_distance * second_distance
                            - third_distance * third_distance)
                            / (2.0 * first_distance * second_distance)));
                    const double value = geometry == mbtr::kGeometryCosine ? cosine
                        : acos(cosine) * 180.0 / kPi;
                    const int first_type = atom_types[graph_atoms[first]] + 1;
                    const int second_type = atom_types[graph_atoms[second]] + 1;
                    if (first_type <= 0 || second_type <= 0) continue;
                    const double weight = mbtr_weight_device(
                        weighting, scale, threshold, r_cut, sharpness,
                        first_distance, second_distance, third_distance);
                    add_histogram_device(
                        target + pair_channel_device(first_type, second_type, element_count) * grid_n,
                        value, weight, grid_min, grid_max, grid_sigma,
                        grid_n, normalize_gaussians);
                    add_histogram_device(
                        target + (reserved + (first_type - 1) * element_count + second_type) * grid_n,
                        (geometry == mbtr::kGeometryCosine ? cosine
                            : acos(fmin(1.0, fmax(-1.0,
                                (first_distance * first_distance + third_distance * third_distance
                                    - second_distance * second_distance)
                                    / (2.0 * first_distance * third_distance))))
                                * 180.0 / kPi),
                        weight, grid_min, grid_max, grid_sigma, grid_n, normalize_gaussians);
                    add_histogram_device(
                        target + (reserved + (second_type - 1) * element_count + first_type) * grid_n,
                        (geometry == mbtr::kGeometryCosine ? cosine
                            : acos(fmin(1.0, fmax(-1.0,
                                (second_distance * second_distance + third_distance * third_distance
                                    - first_distance * first_distance)
                                    / (2.0 * second_distance * third_distance))))
                                * 180.0 / kPi),
                        weight, grid_min, grid_max, grid_sigma, grid_n, normalize_gaussians);
                }
            }
        }
        normalize_mbtr_device(
            target, features, normalization, atom_count, species_counts,
            species_count, 0.0, geometry, grid_n, local);
        return;
    }
    const int pair_count = species_count * (species_count + 1) / 2;
    // A non-local MBTR row is one structure, so all of its centers contribute
    // to the same target histogram.  The local branch above intentionally
    // handles one center per CUDA thread.
    for (I64 center = begin; center < end; ++center) {
        const int center_type = atom_types[center];
        if (center_type < 0) continue;
        const I64 graph_begin = graph_offsets[center];
        const I64 graph_end = graph_offsets[center + 1];
        if (geometry == mbtr::kGeometryDistance
            || geometry == mbtr::kGeometryInverseDistance) {
            for (I64 first = graph_begin; first < graph_end; ++first) {
                const I32 first_atom = graph_atoms[first];
                const double distance = sqrt(fmax(0.0, graph_distance2[first]));
                if (distance <= 1e-12) continue;
                const bool periodic = graph_shifts[first * 3] != 0
                    || graph_shifts[first * 3 + 1] != 0
                    || graph_shifts[first * 3 + 2] != 0;
                if (!periodic && first_atom < center) continue;
                const int first_type = atom_types[first_atom];
                if (first_type < 0) continue;
                const double value = geometry == mbtr::kGeometryDistance
                    ? distance : 1.0 / distance;
                const double pair_weight = mbtr_weight_device(
                    weighting, scale, threshold, r_cut, sharpness, distance, 0.0, 0.0)
                    * (periodic ? 0.5 : 1.0);
                add_histogram_device(
                    target + pair_channel_device(center_type, first_type, species_count) * grid_n,
                    value, pair_weight, grid_min, grid_max, grid_sigma,
                    grid_n, normalize_gaussians);
            }
        } else {
            for (I64 first = graph_begin; first < graph_end; ++first) {
                const double first_distance = sqrt(fmax(0.0, graph_distance2[first]));
                if (first_distance <= 1e-12) continue;
                for (I64 second = graph_begin; second < first; ++second) {
                    const double second_distance = sqrt(fmax(0.0, graph_distance2[second]));
                    if (second_distance <= 1e-12) continue;
                    const double dx = graph_displacements[first * 3] - graph_displacements[second * 3];
                    const double dy = graph_displacements[first * 3 + 1] - graph_displacements[second * 3 + 1];
                    const double dz = graph_displacements[first * 3 + 2] - graph_displacements[second * 3 + 2];
                    const double third_distance = sqrt(dx * dx + dy * dy + dz * dz);
                    const double cosine = fmin(1.0, fmax(-1.0,
                        (first_distance * first_distance + second_distance * second_distance
                            - third_distance * third_distance)
                            / (2.0 * first_distance * second_distance)));
                    const double value = geometry == mbtr::kGeometryCosine ? cosine
                        : acos(cosine) * 180.0 / kPi;
                    const int first_type = atom_types[graph_atoms[first]];
                    const int second_type = atom_types[graph_atoms[second]];
                    if (first_type < 0 || second_type < 0) continue;
                    const double weight = mbtr_weight_device(
                        weighting, scale, threshold, r_cut, sharpness,
                        first_distance, second_distance, third_distance);
                    const int channel = center_type * pair_count
                        + pair_channel_device(first_type, second_type, species_count);
                    add_histogram_device(
                        target + channel * grid_n, value, weight, grid_min, grid_max,
                        grid_sigma, grid_n, normalize_gaussians);
                }
            }
        }
    }
    const double* cell = cells + structure * 9;
    const double volume = cell_volume_device(cell);
    normalize_mbtr_device(
        target, features, normalization, atom_count, species_counts,
        species_count, volume, geometry, grid_n, false);
}

double nested_number(
    const py::dict& object, const char* key, double fallback) {
    const py::str name(key);
    if (!object.contains(name) || object[name].is_none()) return fallback;
    return py::cast<double>(object[name]);
}

std::string nested_string(
    const py::dict& object, const char* key, const std::string& fallback) {
    const py::str name(key);
    if (!object.contains(name) || object[name].is_none()) return fallback;
    return py::cast<std::string>(object[name]);
}

py::dict nested_dict_option(
    const py::dict& options, const char* key) {
    const py::str name(key);
    if (!options.contains(name) || options[name].is_none()) return py::dict();
    try {
        return py::cast<py::dict>(options[name]);
    } catch (const py::cast_error&) {
        throw std::invalid_argument(std::string(key) + " must be an object");
    }
}

py::dict mbtr_config_option(const py::dict& options) {
    const py::str payload_key("_cuda_payload");
    const py::str config_key("mbtr_config");
    if (!options.contains(payload_key) || options[payload_key].is_none()) {
        return py::dict();
    }
    py::dict payload;
    try {
        payload = py::cast<py::dict>(options[payload_key]);
    } catch (const py::cast_error&) {
        throw std::invalid_argument("_cuda_payload must be an object");
    }
    if (!payload.contains(config_key) || payload[config_key].is_none()) {
        return py::dict();
    }
    try {
        return py::cast<py::dict>(payload[config_key]);
    } catch (const py::cast_error&) {
        throw std::invalid_argument("_cuda_payload.mbtr_config must be an object");
    }
}

py::dict compute_mbtr_descriptor(
    CudaExecutionContext& context,
    DeviceBatch& batch,
    DeviceNeighborGraph& graph,
    const detail::StructureBatchView& host_batch,
    const std::string& name,
    const py::dict& options) {
    const py::dict canonical = mbtr_config_option(options);
    const bool has_canonical = py::len(canonical) != 0;
    const auto species = has_canonical ? species_option(canonical) : species_option(options);
    if (species.empty()) throw std::invalid_argument(name + " species must not be empty");
    int geometry = mbtr::kGeometryDistance;
    int weighting = mbtr::kWeightingUnity;
    int normalization = mbtr::kNormalizationNone;
    double grid_min = 0.0;
    double grid_max = 6.0;
    double grid_sigma = 0.1;
    int grid_n = 50;
    double scale = 0.5;
    double threshold = 1e-3;
    double r_cut = 0.0;
    double sharpness = 2.0;
    bool normalize_gaussians = option(options, "normalize_gaussians", true);
    bool local = name == "LMBTR";
    if (has_canonical) {
        geometry = option(canonical, "geometry", mbtr::kGeometryDistance);
        weighting = option(canonical, "weighting", mbtr::kWeightingUnity);
        normalization = option(canonical, "normalization", mbtr::kNormalizationNone);
        grid_min = option(canonical, "grid_min", 0.0);
        grid_max = option(canonical, "grid_max", 6.0);
        grid_sigma = option(canonical, "grid_sigma", 0.1);
        grid_n = option(canonical, "grid_n", 50);
        normalize_gaussians = option(canonical, "normalize_gaussians", true);
        scale = option(canonical, "scale", 0.5);
        threshold = option(canonical, "threshold", 1e-3);
        r_cut = option(canonical, "r_cut", 0.0);
        sharpness = option(canonical, "sharpness", 2.0);
        local = option(canonical, "local", local);
    } else if (name == "ValleOganov") {
        const std::string function = option(options, "function", std::string("distance"));
        geometry = function == "angle" ? mbtr::kGeometryAngle : mbtr::kGeometryDistance;
        grid_n = option(options, "n", 50);
        grid_sigma = option(options, "sigma", 0.1);
        r_cut = option(options, "r_cut", 6.0);
        grid_min = 0.0;
        grid_max = geometry == mbtr::kGeometryAngle ? 180.0 : r_cut;
        weighting = geometry == mbtr::kGeometryAngle
            ? mbtr::kWeightingSmoothCutoff : mbtr::kWeightingInverseSquare;
        sharpness = 2.0;
        normalization = mbtr::kNormalizationValleOganov;
        const std::string normalization_name = option(
            options, "normalization", std::string("valle_oganov"));
        if (normalization_name == "none") normalization = mbtr::kNormalizationNone;
        else if (normalization_name == "l2") normalization = mbtr::kNormalizationL2;
        else if (normalization_name == "n_atoms") normalization = mbtr::kNormalizationNAtoms;
    } else {
        const py::dict geometry_object = nested_dict_option(options, "geometry");
        const py::dict grid_object = nested_dict_option(options, "grid");
        const py::dict weighting_object = nested_dict_option(options, "weighting");
        const std::string geometry_name = nested_string(
            geometry_object, "function", "distance");
        if (geometry_name == "atomic_number") geometry = mbtr::kGeometryAtomicNumber;
        else if (geometry_name == "distance") geometry = mbtr::kGeometryDistance;
        else if (geometry_name == "inverse_distance") geometry = mbtr::kGeometryInverseDistance;
        else if (geometry_name == "angle") geometry = mbtr::kGeometryAngle;
        else if (geometry_name == "cosine") geometry = mbtr::kGeometryCosine;
        else throw std::invalid_argument("unsupported CUDA MBTR geometry");
        const std::string weighting_name = nested_string(
            weighting_object, "function", "unity");
        if (weighting_name == "unity" || weighting_name == "none") weighting = mbtr::kWeightingUnity;
        else if (weighting_name == "exp") weighting = mbtr::kWeightingExponential;
        else if (weighting_name == "inverse_square") weighting = mbtr::kWeightingInverseSquare;
        else if (weighting_name == "smooth_cutoff") weighting = mbtr::kWeightingSmoothCutoff;
        else throw std::invalid_argument("unsupported CUDA MBTR weighting");
        grid_min = nested_number(grid_object, "min", 0.0);
        grid_max = nested_number(grid_object, "max", 6.0);
        grid_sigma = nested_number(grid_object, "sigma", 0.1);
        grid_n = static_cast<int>(nested_number(grid_object, "n", 50));
        scale = nested_number(weighting_object, "scale", 0.5);
        threshold = nested_number(weighting_object, "threshold", 1e-3);
        const double default_cutoff = weighting == mbtr::kWeightingUnity ? 0.0 : grid_max;
        r_cut = nested_number(weighting_object, "r_cut", default_cutoff);
        sharpness = nested_number(weighting_object, "sharpness", 2.0);
        const std::string normalization_name = option(
            options, "normalization", std::string("none"));
        if (normalization_name == "l2") normalization = mbtr::kNormalizationL2;
        else if (normalization_name == "n_atoms") normalization = mbtr::kNormalizationNAtoms;
        else if (normalization_name == "valle_oganov") normalization = mbtr::kNormalizationValleOganov;
    }
    if (geometry != mbtr::kGeometryAtomicNumber && r_cut <= 0.0) {
        r_cut = grid_max;
    }
    if (weighting == mbtr::kWeightingExponential && scale > 0.0) {
        const double multiplier = geometry == mbtr::kGeometryAngle
            || geometry == mbtr::kGeometryCosine ? 0.5 : 1.0;
        r_cut = std::max(r_cut, multiplier * -log(threshold) / scale);
    }
    const bool distance_geometry = geometry == mbtr::kGeometryDistance
        || geometry == mbtr::kGeometryInverseDistance;
    const int species_count = static_cast<int>(species.size());
    const int pair_count = species_count * (species_count + 1) / 2;
    const I64 channels = geometry == mbtr::kGeometryAtomicNumber ? species_count
        : local ? (distance_geometry ? species_count + 1
            : (species_count + 1) * (3 * (species_count + 1) - 1) / 2)
        : distance_geometry ? pair_count : species_count * pair_count;
    const I64 features = channels * grid_n;
    const I64 rows = local ? batch.atoms() : batch.structures();
    const std::size_t size = static_cast<std::size_t>(rows)
        * static_cast<std::size_t>(features);
    double* output = context.output_buffer(size);
    check_cuda(cudaMemsetAsync(output, 0, size * sizeof(double), context.stream()),
        "could not clear CUDA MBTR output");
    // Resolve element channels once on the host.  The previous kernel did a
    // linear species-table search for every edge of every angle, which made
    // the lookup part of the hottest inner loop.
    std::vector<I32> atom_types(static_cast<std::size_t>(batch.atoms()), -1);
    for (I64 atom = 0; atom < batch.atoms(); ++atom) {
        const auto found = std::find(species.begin(), species.end(), host_batch.numbers[atom]);
        if (found != species.end()) {
            atom_types[static_cast<std::size_t>(atom)] = static_cast<I32>(
                std::distance(species.begin(), found));
        }
    }
    DeviceBuffer<I32> d_atom_types;
    d_atom_types.upload(
        atom_types.data(), atom_types.size(), context.stream(),
        "could not upload MBTR atom types");
    if (geometry != mbtr::kGeometryAtomicNumber) {
        graph.build_dpa(context, batch, host_batch, r_cut, true, false, false);
    }
    if (rows > 0) {
        constexpr unsigned block_size = 64;
        mbtr_kernel<<<static_cast<unsigned>((rows + block_size - 1) / block_size),
            block_size, 0, context.stream()>>>(
            batch.numbers(), d_atom_types.get(), batch.cells(), batch.offsets(), graph.offsets(), graph.atoms(),
            graph.shifts(), graph.displacements(), graph.distance2(),
            species_count, geometry, weighting, normalization, grid_min, grid_max,
            grid_sigma, grid_n, normalize_gaussians, scale, threshold, r_cut, sharpness,
            local, batch.structures(), batch.atoms(), features, output);
        check_cuda(cudaGetLastError(), "CUDA MBTR kernel launch failed");
    }
    const auto values = context.download_output(size);
    if (local) {
        return atom_result(values, rows, features, name, options, false,
            std::vector<I64>(host_batch.offsets, host_batch.offsets + host_batch.structures + 1));
    }
    py::dict result;
    result["values"] = values_array(values, rows, features);
    result["level"] = "structure";
    result["labels"] = labels_option(options, name, features);
    result["metadata"] = metadata(options, name);
    return result;
}

template <int MaxAngular>
void launch_spherical_pair_exact(
    int requested,
    cudaStream_t stream,
    const I64* graph_offsets,
    const I32* graph_atoms,
    const I32* graph_shifts,
    const double* graph_displacements,
    const double* graph_distance2,
    I64 atoms,
    I64 edges,
    double cutoff,
    double density_width,
    int radial_count,
    const double* gto_constants,
    const double* gamma_a,
    const double* gamma_b,
    const double* orthonormalization,
    double* records,
    double* output) {
    if (requested == MaxAngular) {
        constexpr unsigned block_size = 128;
        const auto blocks = static_cast<unsigned>((edges + block_size - 1) / block_size);
        spherical_pair_kernel<MaxAngular><<<blocks, block_size, 0, stream>>>(
            graph_offsets, graph_atoms, graph_shifts, graph_displacements,
            graph_distance2, atoms, cutoff, density_width, radial_count,
            requested, gto_constants, gamma_a, gamma_b, orthonormalization,
            records, output);
        return;
    }
    if constexpr (MaxAngular < 31) {
        launch_spherical_pair_exact<MaxAngular + 1>(
            requested, stream, graph_offsets, graph_atoms, graph_shifts,
            graph_displacements, graph_distance2, atoms, edges, cutoff,
            density_width, radial_count, gto_constants, gamma_a, gamma_b,
            orthonormalization, records, output);
    } else {
        throw std::invalid_argument("CUDA pair descriptor max_angular is too large");
    }
}

py::dict compute_atomic_composition(
    CudaExecutionContext& context,
    DeviceBatch& batch,
    const detail::StructureBatchView& host_batch,
    const std::vector<I32>& species,
    bool per_system,
    const std::string& name,
    const py::dict& options) {
    if (species.empty()) throw std::invalid_argument("species must not be empty");
    const I64 rows = per_system ? batch.structures() : batch.atoms();
    const I64 columns = static_cast<I64>(species.size());
    const std::size_t size = static_cast<std::size_t>(rows) * static_cast<std::size_t>(columns);
    double* output = context.output_buffer(size);
    check_cuda(cudaMemsetAsync(output, 0, size * sizeof(double), context.stream()),
        "could not clear CUDA atomic composition output");
    DeviceBuffer<I32> device_species;
    device_species.upload(species.data(), species.size(), context.stream(),
        "could not upload CUDA composition species");
    constexpr unsigned block_size = 128;
    const auto blocks = static_cast<unsigned>((rows + block_size - 1) / block_size);
    atomic_composition_kernel<<<blocks, block_size, 0, context.stream()>>>(
        batch.numbers(), batch.offsets(), batch.structures(), batch.atoms(),
        device_species.get(), static_cast<int>(species.size()), per_system, output);
    check_cuda(cudaGetLastError(), "CUDA atomic composition kernel launch failed");
    const auto values = context.download_output(size);
    std::vector<I64> offsets;
    if (!per_system) {
        offsets.assign(
            host_batch.offsets,
            host_batch.offsets + static_cast<std::size_t>(host_batch.structures + 1));
    }
    return atom_result(values, rows, columns, name, options, per_system, offsets);
}

py::dict compute_sorted_distances(
    CudaExecutionContext& context,
    DeviceBatch& batch,
    DeviceNeighborGraph& graph,
    const detail::StructureBatchView& host_batch,
    const std::vector<I32>& species,
    double cutoff,
    int max_neighbors,
    bool separate,
    const std::string& name,
    const py::dict& options) {
    if (species.empty()) throw std::invalid_argument("species must not be empty");
    if (max_neighbors <= 0) throw std::invalid_argument("max_neighbors must be positive");
    graph.build_dpa(context, batch, host_batch, cutoff, true, false, false);
    const I64 columns = separate
        ? static_cast<I64>(species.size()) * max_neighbors
        : static_cast<I64>(max_neighbors);
    const std::size_t size = static_cast<std::size_t>(batch.atoms())
        * static_cast<std::size_t>(columns);
    double* output = context.output_buffer(size);
    check_cuda(cudaMemsetAsync(output, 0, size * sizeof(double), context.stream()),
        "could not clear CUDA sorted distance output");
    DeviceBuffer<I32> device_species;
    device_species.upload(species.data(), species.size(), context.stream(),
        "could not upload CUDA sorted distance species");
    constexpr unsigned block_size = 128;
    const auto blocks = static_cast<unsigned>((batch.atoms() + block_size - 1) / block_size);
    sorted_distances_kernel<<<blocks, block_size, 0, context.stream()>>>(
        batch.numbers(), graph.offsets(), graph.atoms(), graph.distance2(),
        device_species.get(), static_cast<int>(species.size()), max_neighbors,
        separate, cutoff, batch.atoms(), output);
    check_cuda(cudaGetLastError(), "CUDA sorted distance kernel launch failed");
    const auto values = context.download_output(size);
    return atom_result(values, batch.atoms(), columns, name, options, false,
        std::vector<I64>(host_batch.offsets,
            host_batch.offsets + host_batch.structures + 1));
}

py::dict compute_spherical_pair(
    CudaExecutionContext& context,
    DeviceBatch& batch,
    DeviceNeighborGraph& graph,
    const detail::StructureBatchView& host_batch,
    double cutoff,
    double density_width,
    int max_radial,
    int max_angular,
    const std::string& name,
    const py::dict& options) {
    if (max_radial < 0 || max_angular < 0 || max_angular > 31) {
        throw std::invalid_argument("invalid CUDA spherical pair orders");
    }
    if (density_width <= 0.0 || cutoff <= 0.0) {
        throw std::invalid_argument("invalid CUDA spherical pair parameters");
    }
    // Keep the input coordinate convention for the public pair identifiers;
    // unlike reduced descriptors, pair samples expose the integer image shift.
    graph.build_dpa(context, batch, host_batch, cutoff, true, false, true, false);
    const I64 edges = static_cast<I64>(graph.pairs());
    const int radial_count = max_radial + 1;
    const I64 columns = static_cast<I64>((max_angular + 1) * (max_angular + 1) * radial_count);
    const std::size_t output_size = static_cast<std::size_t>(edges)
        * static_cast<std::size_t>(columns);
    double* output = context.output_buffer(output_size);

    std::vector<double> gto_constants;
    std::vector<double> gamma_a;
    std::vector<double> gamma_b;
    std::vector<double> orthonormalization;
    gto_constants.reserve(static_cast<std::size_t>(max_angular + 1) * radial_count);
    gamma_a.reserve(static_cast<std::size_t>(max_angular + 1) * radial_count);
    gamma_b.reserve(static_cast<std::size_t>(max_angular + 1));
    orthonormalization.reserve(
        static_cast<std::size_t>(max_angular + 1) * radial_count * radial_count);
    for (int angular = 0; angular <= max_angular; ++angular) {
        const detail::GtoRadialBasis basis(radial_count, cutoff, angular);
        gto_constants.insert(gto_constants.end(), basis.gto_constants.begin(), basis.gto_constants.end());
        gamma_a.insert(gamma_a.end(), basis.gamma_a.begin(), basis.gamma_a.end());
        gamma_b.push_back(basis.gamma_b);
        for (const auto& row : basis.orthonormalization) {
            orthonormalization.insert(orthonormalization.end(), row.begin(), row.end());
        }
    }
    const std::size_t records_size = static_cast<std::size_t>(edges) * 5U;
    DeviceBuffer<double> records;
    records.allocate(records_size, "could not allocate CUDA pair records");
    DeviceBuffer<double> device_gto;
    DeviceBuffer<double> device_gamma_a;
    DeviceBuffer<double> device_gamma_b;
    DeviceBuffer<double> device_orthonormalization;
    device_gto.upload(gto_constants.data(), gto_constants.size(), context.stream(),
        "could not upload CUDA pair radial constants");
    device_gamma_a.upload(gamma_a.data(), gamma_a.size(), context.stream(),
        "could not upload CUDA pair radial gamma values");
    device_gamma_b.upload(gamma_b.data(), gamma_b.size(), context.stream(),
        "could not upload CUDA pair radial denominators");
    device_orthonormalization.upload(
        orthonormalization.data(), orthonormalization.size(), context.stream(),
        "could not upload CUDA pair radial orthonormalization");
    if (edges > 0) {
        launch_spherical_pair_exact<0>(
            max_angular, context.stream(), graph.offsets(), graph.atoms(), graph.shifts(),
            graph.displacements(), graph.distance2(), batch.atoms(), edges, cutoff,
            density_width, radial_count, device_gto.get(), device_gamma_a.get(),
            device_gamma_b.get(), device_orthonormalization.get(), records.get(), output);
        check_cuda(cudaGetLastError(), "CUDA spherical pair kernel launch failed");
    }
    auto values = context.download_output(output_size);
    auto records_host = download(
        records.get(), records_size, context, "could not download CUDA pair records");
    // DeviceNeighborGraph deliberately uses a distance-stable order for the
    // descriptors that reduce over neighbors.  SphericalExpansionByPair is a
    // public pair table, however, and its CPU contract exposes the canonical
    // cell-list query order.  Reorder only the already-computed edge rows
    // using the same image/grid key; no pair feature is evaluated on the host.
    const auto pair_order = cpu_pair_order(host_batch, cutoff, records_host);
    if (!pair_order.empty()) {
        std::vector<double> ordered_values(values.size(), 0.0);
        std::vector<double> ordered_records(records_host.size(), 0.0);
        for (std::size_t target = 0; target < pair_order.size(); ++target) {
            const std::size_t source = pair_order[target];
            std::copy_n(
                values.data() + source * static_cast<std::size_t>(columns),
                static_cast<std::size_t>(columns),
                ordered_values.data() + target * static_cast<std::size_t>(columns));
            std::copy_n(
                records_host.data() + source * 5U, 5U,
                ordered_records.data() + target * 5U);
        }
        values.swap(ordered_values);
        records_host.swap(ordered_records);
    }
    const auto atom_offsets = download(
        graph.offsets(), static_cast<std::size_t>(batch.atoms()) + 1,
        context, "could not download CUDA pair graph offsets");
    std::vector<I64> row_offsets;
    row_offsets.reserve(static_cast<std::size_t>(batch.structures()) + 1U);
    row_offsets.push_back(0);
    for (I64 structure = 0; structure < batch.structures(); ++structure) {
        row_offsets.push_back(atom_offsets[static_cast<std::size_t>(
            host_batch.offsets[structure + 1])]);
    }
    py::dict result;
    result["values"] = values_array(values, edges, columns);
    result["level"] = "pair";
    result["row_offsets"] = i64_array(row_offsets);
    py::array_t<double> pair_records({
        static_cast<py::ssize_t>(edges), static_cast<py::ssize_t>(5)});
    if (!records_host.empty()) {
        std::copy(records_host.begin(), records_host.end(), pair_records.mutable_data());
    }
    result["pair_records"] = pair_records;
    result["labels"] = labels_option(options, name, columns);
    result["metadata"] = metadata(options, name);
    return result;
}

__device__ double integer_power_device(double value, int exponent) {
    double result = 1.0;
    for (int index = 0; index < exponent; ++index) result *= value;
    return result;
}

__global__ void ead_kernel(
    const I32* numbers,
    const I64* graph_offsets,
    const I32* graph_atoms,
    const double* graph_displacements,
    const double* graph_distance2,
    double cutoff,
    const double* eta,
    int n_eta,
    const double* centers,
    int n_centers,
    int max_degree,
    I64 atoms,
    double* output) {
    const I64 center = static_cast<I64>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (center >= atoms) return;
    const I64 begin = graph_offsets[center];
    const I64 end = graph_offsets[center + 1];
    const int radial_count = n_eta * n_centers;
    double* target = output + center * static_cast<I64>((max_degree + 1) * radial_count);
    for (int degree = 0; degree <= max_degree; ++degree) {
        for (int eta_index = 0; eta_index < n_eta; ++eta_index) {
            for (int center_index = 0; center_index < n_centers; ++center_index) {
                double squared_sum = 0.0;
                for (int power_degree = 0; power_degree <= degree; ++power_degree) {
                    for (int lx = 0; lx <= power_degree; ++lx) {
                        for (int ly = 0; ly <= power_degree - lx; ++ly) {
                            const int lz = power_degree - lx - ly;
                            double term = 0.0;
                            for (I64 edge = begin; edge < end; ++edge) {
                                const double distance2 = fmax(0.0, graph_distance2[edge]);
                                const double distance = sqrt(distance2);
                                if (distance <= 1e-12 || distance >= cutoff) continue;
                                const double smooth = 0.5 * (1.0 + cos(kPi * distance / cutoff));
                                const double delta = distance - centers[center_index];
                                const double gaussian = exp(-eta[eta_index] * delta * delta);
                                const double* vector = graph_displacements + edge * 3;
                                const double monomial = integer_power_device(vector[0], lx)
                                    * integer_power_device(vector[1], ly)
                                    * integer_power_device(vector[2], lz)
                                    / sqrt(tgamma(lx + 1.0) * tgamma(ly + 1.0) * tgamma(lz + 1.0));
                                term += monomial * gaussian * static_cast<double>(
                                    numbers[graph_atoms[edge]]) * smooth;
                            }
                            squared_sum += term * term;
                        }
                    }
                }
                target[degree * radial_count + eta_index * n_centers + center_index]
                    = tgamma(degree + 1.0) * squared_sum
                    / pow(cutoff, 2.0 * degree);
            }
        }
    }
}

I64 payload_or_option_feature_count(
    const py::dict& options, I64 fallback, const std::string& name) {
    const I64 value = feature_count_option(options, fallback);
    if (value <= 0) throw std::invalid_argument(name + " has no CUDA feature layout");
    return value;
}

py::dict compute_ead_descriptor(
    CudaExecutionContext& context,
    DeviceBatch& batch,
    DeviceNeighborGraph& graph,
    const detail::StructureBatchView& host_batch,
    const py::dict& options) {
    const py::dict parameters = child_dict(options, "parameters");
    const int max_degree = option(parameters, "L", 3);
    const auto eta = vector_child(parameters, "eta");
    const auto centers = vector_child(parameters, "Rs");
    const double cutoff = option(options, "Rc", 6.0);
    if (max_degree < 0 || max_degree > 8 || cutoff <= 0.0
        || eta.empty() || centers.empty()) {
        throw std::invalid_argument("invalid CUDA EAD parameters");
    }
    const I64 computed = static_cast<I64>(max_degree + 1) * eta.size() * centers.size();
    const I64 features = payload_or_option_feature_count(options, computed, "EAD");
    if (features != computed) throw std::invalid_argument("CUDA EAD feature count mismatch");
    graph.build_dpa(context, batch, host_batch, cutoff, true, false, false);
    DeviceBuffer<double> d_eta;
    DeviceBuffer<double> d_centers;
    d_eta.upload(eta.data(), eta.size(), context.stream(), "could not upload EAD eta");
    d_centers.upload(centers.data(), centers.size(), context.stream(), "could not upload EAD centers");
    const std::size_t size = static_cast<std::size_t>(batch.atoms())
        * static_cast<std::size_t>(features);
    double* output = context.output_buffer(size);
    if (size > 0) {
        check_cuda(cudaMemsetAsync(output, 0, size * sizeof(double), context.stream()),
            "could not clear CUDA EAD output");
        constexpr unsigned block_size = 64;
        ead_kernel<<<static_cast<unsigned>((batch.atoms() + block_size - 1) / block_size),
            block_size, 0, context.stream()>>>(
            batch.numbers(), graph.offsets(), graph.atoms(), graph.displacements(),
            graph.distance2(), cutoff, d_eta.get(), static_cast<int>(eta.size()),
            d_centers.get(), static_cast<int>(centers.size()), max_degree,
            batch.atoms(), output);
        check_cuda(cudaGetLastError(), "CUDA EAD kernel launch failed");
    }
    const auto values = context.download_output(size);
    return atom_result(values, batch.atoms(), features, "EAD", options, false,
        std::vector<I64>(host_batch.offsets, host_batch.offsets + host_batch.structures + 1));
}

__device__ double lode_fourier_device(double k_norm, double sigma, int exponent) {
    if (k_norm <= 1e-12) return 0.0;
    const double sigma2 = sigma * sigma;
    const double x = 0.5 * k_norm * k_norm * sigma2;
    if (exponent == 1) return 4.0 * kPi * exp(-x) / (k_norm * k_norm);
    const double p_eff = 3.0 - exponent;
    const double factor = pow(kPi, 1.5) * pow(2.0 * sigma2, 0.5 * p_eff)
        / tgamma(0.5 * exponent);
    const double root_x = sqrt(x);
    double value = 0.0;
    if (exponent == 2) {
        value = sqrt(kPi / x) * erfc(root_x);
    } else if (exponent == 3) {
        double term = 1.0;
        double series = 0.0;
        if (x < 1.0) {
            for (int index = 1; index < 200; ++index) {
                term *= -x;
                const double add = term / (index * index);
                series += add;
                if (fabs(add) < 2e-16 * fmax(1.0, fabs(series))) break;
            }
            value = -0.5772156649015329 - log(x) - series;
        } else {
            series = 1.0;
            for (int index = 1; index < 100; ++index) {
                term *= -static_cast<double>(index) / x;
                series += term;
                if (fabs(term) > fabs(series)) break;
            }
            value = exp(-x) * series / x;
        }
    } else if (exponent == 4) {
        value = 2.0 * (exp(-x) - sqrt(kPi * x) * erfc(root_x));
    } else if (exponent == 5) {
        double e1 = 0.0;
        double term = 1.0;
        double series = 0.0;
        if (x < 1.0) {
            for (int index = 1; index < 200; ++index) {
                term *= -x;
                const double add = term / (index * index);
                series += add;
                if (fabs(add) < 2e-16 * fmax(1.0, fabs(series))) break;
            }
            e1 = -0.5772156649015329 - log(x) - series;
        } else {
            series = 1.0;
            for (int index = 1; index < 100; ++index) {
                term *= -static_cast<double>(index) / x;
                series += term;
                if (fabs(term) > fabs(series)) break;
            }
            e1 = exp(-x) * series / x;
        }
        value = exp(-x) - x * e1;
    } else if (exponent == 6) {
        double term = 1.0;
        double series = 0.0;
        if (x < 1.0) {
            for (int index = 1; index < 200; ++index) {
                term *= -x;
                const double add = term / (index * index);
                series += add;
                if (fabs(add) < 2e-16 * fmax(1.0, fabs(series))) break;
            }
            series = -0.5772156649015329 - log(x) - series;
        } else {
            series = 1.0;
            for (int index = 1; index < 100; ++index) {
                term *= -static_cast<double>(index) / x;
                series += term;
                if (fabs(term) > fabs(series)) break;
            }
            series = exp(-x) * series / x;
        }
        value = ((2.0 - 4.0 * x) * exp(-x)
            + 4.0 * sqrt(kPi) * pow(x, 1.5) * erfc(root_x)) / 3.0;
    } else if (exponent == 7) {
        double term = 1.0;
        double series = 0.0;
        if (x < 1.0) {
            for (int index = 1; index < 200; ++index) {
                term *= -x;
                const double add = term / (index * index);
                series += add;
                if (fabs(add) < 2e-16 * fmax(1.0, fabs(series))) break;
            }
            series = -0.5772156649015329 - log(x) - series;
        } else {
            series = 1.0;
            for (int index = 1; index < 100; ++index) {
                term *= -static_cast<double>(index) / x;
                series += term;
                if (fabs(term) > fabs(series)) break;
            }
            series = exp(-x) * series / x;
        }
        value = (1.0 - x) * exp(-x) / 2.0 + x * x * series / 2.0;
    } else if (exponent == 8) {
        value = -2.0 / 15.0 * ((-3.0 + 2.0 * x - 4.0 * x * x) * exp(-x)
            + 4.0 * sqrt(kPi) * pow(x, 2.5) * erfc(root_x));
    } else if (exponent == 9) {
        double term = 1.0;
        double series = 0.0;
        if (x < 1.0) {
            for (int index = 1; index < 200; ++index) {
                term *= -x;
                const double add = term / (index * index);
                series += add;
                if (fabs(add) < 2e-16 * fmax(1.0, fabs(series))) break;
            }
            series = -0.5772156649015329 - log(x) - series;
        } else {
            series = 1.0;
            for (int index = 1; index < 100; ++index) {
                term *= -static_cast<double>(index) / x;
                series += term;
                if (fabs(term) > fabs(series)) break;
            }
            series = exp(-x) * series / x;
        }
        value = (x * x - x + 2.0) * exp(-x) / 6.0
            - x * x * x * series / 6.0;
    }
    return factor * value;
}

__device__ double lode_hyp1f1_device(double a, double b, double x) {
    if (x >= 0.0) return positive_hypergeometric(a, b, x);
    const double magnitude = -x;
    const double difference = b - a;
    const double rounded = round(difference);
    if (difference <= 0.0 && fabs(difference - rounded) <= 1e-12) {
        const int degree = static_cast<int>(-rounded);
        double polynomial = 1.0;
        double term = 1.0;
        for (int index = 1; index <= degree; ++index) {
            term *= (-degree + index - 1.0) * magnitude
                / ((b + index - 1.0) * index);
            polynomial += term;
        }
        return exp(-magnitude) * polynomial;
    }
    if (magnitude > 8.0 && difference < 0.0) {
        // Kummer's transformation turns the alternating negative series into
        // a positive-argument series with a non-positive first parameter.
        double term = 1.0;
        double sum = 1.0;
        for (int index = 1; index <= 500; ++index) {
            term *= (difference + index - 1.0) * magnitude
                / ((b + index - 1.0) * index);
            sum += term;
            if (fabs(term) <= fabs(sum) * 1e-15) break;
        }
        return exp(-magnitude) * sum;
    }
    double term = 1.0;
    double sum = 1.0;
    for (int index = 1; index <= 500; ++index) {
        term *= (a + index - 1.0) * x
            / ((b + index - 1.0) * index);
        sum += term;
        if (fabs(term) <= fabs(sum) * 2e-15) break;
    }
    return sum;
}

__device__ double lode_radial_value_device(
    double k_norm,
    int angular,
    int target_radial,
    int radial_count,
    const double* widths,
    const double* lode_prefactors,
    const double* gamma_a,
    const double* gamma_b,
    const double* orthonormalization) {
    double value = 0.0;
    for (int raw = 0; raw < radial_count; ++raw) {
        const double sigma = widths[raw];
        const double k_sigma_sqrt2 = k_norm * sigma / sqrt(2.0);
        const double angular_factor = pow(k_sigma_sqrt2, angular);
        const double factor = sqrt(kPi) / sqrt(2.0)
            * lode_prefactors[raw] * angular_factor;
        const double z = -0.5 * k_norm * k_norm * sigma * sigma;
        const double a = 0.5 * (raw + angular + 3.0);
        const double b = angular + 1.5;
        const double raw_value = gamma_a[angular * radial_count + raw]
            / gamma_b[angular] * lode_hyp1f1_device(a, b, z) * factor;
        value += raw_value * orthonormalization[
            (angular * radial_count + raw) * radial_count + target_radial];
    }
    return value;
}

__global__ void lode_exact_kernel(
    const I32* numbers,
    const double* positions,
    const I64* offsets,
    const I32* species,
    int species_count,
    int radial_count,
    int max_angular,
    double density_width,
    int exponent,
    const I64* k_offsets,
    const double* k_vectors,
    const double* k_norms,
    const double* widths,
    const double* lode_prefactors,
    const double* gamma_a,
    const double* gamma_b,
    const double* orthonormalization,
    const double* cells,
    I64 structures,
    I64 atoms,
    I64 features,
    double* output) {
    const I64 center = static_cast<I64>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (center >= atoms) return;
    I64 structure = 0;
    while (structure + 1 < structures && offsets[structure + 1] <= center) ++structure;
    const int center_type = species_index(numbers[center], species, species_count);
    if (center_type < 0) return;
    const I64 begin = offsets[structure];
    const I64 end = offsets[structure + 1];
    const I64 k_begin = k_offsets[structure];
    const I64 k_end = k_offsets[structure + 1];
    const double volume = cell_volume_device(cells + structure * 9);
    if (volume <= 0.0) return;
    const I64 angular_block = static_cast<I64>(max_angular + 1)
        * static_cast<I64>(max_angular + 1);
    const I64 channel_stride = angular_block * radial_count;
    double* target = output + center * features
        + static_cast<I64>(center_type) * species_count * channel_stride;
    const double global_factor = 4.0 * kPi / volume;
    const double center_x = positions[center * 3 + 0];
    const double center_y = positions[center * 3 + 1];
    const double center_z = positions[center * 3 + 2];
    for (int neighbor_type = 0; neighbor_type < species_count; ++neighbor_type) {
        for (I64 k_index = k_begin; k_index < k_end; ++k_index) {
            const double kx = k_vectors[k_index * 3 + 0];
            const double ky = k_vectors[k_index * 3 + 1];
            const double kz = k_vectors[k_index * 3 + 2];
            const double k_norm = k_norms[k_index];
            const double center_phase = kx * center_x + ky * center_y + kz * center_z;
            const double center_cos = cos(center_phase);
            const double center_sin = sin(center_phase);
            double sum_cos = 0.0;
            double sum_sin = 0.0;
            for (I64 atom = begin; atom < end; ++atom) {
                if (species_index(numbers[atom], species, species_count) != neighbor_type) {
                    continue;
                }
                const double phase = kx * positions[atom * 3 + 0]
                    + ky * positions[atom * 3 + 1]
                    + kz * positions[atom * 3 + 2];
                sum_cos += cos(phase);
                sum_sin += sin(phase);
            }
            const double density = global_factor * lode_fourier_device(
                k_norm, density_width, exponent);
            const double even_weight = density * 2.0 * (
                center_cos * sum_cos + center_sin * sum_sin);
            const double odd_weight = density * 2.0 * (
                center_sin * sum_cos - center_cos * sum_sin);
            double direction[3] = {kx / k_norm, ky / k_norm, kz / k_norm};
            double harmonics[441]{};
            harmonic_values<20>(direction, harmonics, max_angular);
            for (int angular = 0; angular <= max_angular; ++angular) {
                const double phase = angular % 4 == 0 ? 1.0
                    : angular % 4 == 1 ? -1.0
                    : angular % 4 == 2 ? -1.0 : 1.0;
                const double selected_weight = angular % 2 == 0
                    ? even_weight : odd_weight;
                const I64 base = static_cast<I64>(angular * angular) * radial_count;
                for (int m = -angular; m <= angular; ++m) {
                    const double harmonic = harmonics[angular * angular + angular + m];
                    for (int radial = 0; radial < radial_count; ++radial) {
                        target[static_cast<I64>(neighbor_type) * channel_stride
                            + base + static_cast<I64>(angular + m) * radial_count + radial]
                            += phase * selected_weight * harmonic
                            * lode_radial_value_device(
                                k_norm, angular, radial, radial_count, widths,
                                lode_prefactors, gamma_a, gamma_b,
                                orthonormalization);
                    }
                }
            }
        }
    }
}

__device__ double lode_reciprocal_factor(
    const double* positions,
    I64 structure_begin,
    I64 structure_end,
    I64 center,
    I64 atom,
    const double* inverse,
    double k_cutoff,
    double sigma,
    int exponent) {
    const double b0x = 2.0 * kPi * inverse[0];
    const double b0y = 2.0 * kPi * inverse[3];
    const double b0z = 2.0 * kPi * inverse[6];
    const double b1x = 2.0 * kPi * inverse[1];
    const double b1y = 2.0 * kPi * inverse[4];
    const double b1z = 2.0 * kPi * inverse[7];
    const double b2x = 2.0 * kPi * inverse[2];
    const double b2y = 2.0 * kPi * inverse[5];
    const double b2z = 2.0 * kPi * inverse[8];
    const double min_norm = fmin(sqrt(b0x * b0x + b0y * b0y + b0z * b0z),
        fmin(sqrt(b1x * b1x + b1y * b1y + b1z * b1z),
            sqrt(b2x * b2x + b2y * b2y + b2z * b2z)));
    const int bound = min_norm > 1e-12
        ? min(16, max(1, static_cast<int>(ceil(k_cutoff / min_norm)) + 1)) : 16;
    const double cx = positions[center * 3 + 0];
    const double cy = positions[center * 3 + 1];
    const double cz = positions[center * 3 + 2];
    const double ax = positions[atom * 3 + 0];
    const double ay = positions[atom * 3 + 1];
    const double az = positions[atom * 3 + 2];
    double result = 0.0;
    for (int i = -bound; i <= bound; ++i) {
        for (int j = -bound; j <= bound; ++j) {
            for (int k = -bound; k <= bound; ++k) {
                const double vx = i * b0x + j * b1x + k * b2x;
                const double vy = i * b0y + j * b1y + k * b2y;
                const double vz = i * b0z + j * b1z + k * b2z;
                const double norm2 = vx * vx + vy * vy + vz * vz;
                if (norm2 <= 1e-24 || norm2 >= k_cutoff * k_cutoff) continue;
                const double density = lode_fourier_device(
                    sqrt(norm2), sigma, exponent);
                const double phase = vx * (ax - cx) + vy * (ay - cy) + vz * (az - cz);
                result += density * cos(phase);
            }
        }
    }
    (void)structure_begin;
    (void)structure_end;
    return result;
}

__global__ void lode_kernel(
    const I32* numbers,
    const I64* offsets,
    const I64* graph_offsets,
    const I32* graph_atoms,
    const double* graph_displacements,
    const double* graph_distance2,
    const I32* species,
    int species_count,
    double density_width,
    double cutoff,
    double radial_radius,
    double k_cutoff,
    int exponent,
    int radial_count,
    int max_angular,
    I64 structures,
    I64 atoms,
    I64 features,
    const double* cells,
    double* output) {
    const I64 center = static_cast<I64>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (center >= atoms) return;
    I64 structure = 0;
    while (structure + 1 < structures && offsets[structure + 1] <= center) ++structure;
    double inverse[9]{};
    if (!inverse3_device(cells + structure * 9, inverse)) return;
    const int center_type = species_index(numbers[center], species, species_count);
    if (center_type < 0) return;
    const int angular_count = max_angular + 1;
    const int angular_block = angular_count * angular_count;
    const I64 center_stride = static_cast<I64>(species_count) * radial_count * angular_block;
    double* target = output + center * features + static_cast<I64>(center_type) * center_stride;
    const I64 begin = graph_offsets[center];
    const I64 end = graph_offsets[center + 1];
    const I64 structure_begin = offsets[structure];
    const I64 structure_end = offsets[structure + 1];
    for (I64 edge = begin; edge < end; ++edge) {
        const double distance = sqrt(fmax(0.0, graph_distance2[edge]));
        if (distance <= 1e-12 || distance >= cutoff) continue;
        const int neighbor_type = species_index(
            numbers[graph_atoms[edge]], species, species_count);
        if (neighbor_type < 0) continue;
        const double short_cutoff = 0.5 * (1.0 + cos(kPi * distance / cutoff));
            const double* vector = graph_displacements + edge * 3;
        double harmonics[441]{};
        harmonic_values<20>(vector, harmonics, max_angular);
        for (int angular = 0; angular <= max_angular; ++angular) {
            for (int m = -angular; m <= angular; ++m) {
                for (int radial = 0; radial < radial_count; ++radial) {
                    const double radial_value = short_cutoff
                        * exp(-(radial + 1.0) * distance * distance
                            / (radial_radius * radial_radius));
                    const I64 index = static_cast<I64>(neighbor_type) * radial_count * angular_block
                        + static_cast<I64>(angular * angular + angular + m) * radial_count + radial;
                    target[index] += radial_value * harmonics[angular * angular + angular + m];
                }
            }
        }
    }
}

// Position-aware LODE kernel.  Keeping the reciprocal sum in a separate
// kernel makes the device pointer contract explicit and avoids any host
// reciprocal-space materialization.
__global__ void lode_kernel_with_positions(
    const I32* numbers,
    const double* positions,
    const I64* offsets,
    const I64* graph_offsets,
    const I32* graph_atoms,
    const double* graph_displacements,
    const double* graph_distance2,
    const I32* species,
    int species_count,
    double density_width,
    double cutoff,
    double radial_radius,
    double k_cutoff,
    int exponent,
    int radial_count,
    int max_angular,
    I64 structures,
    I64 atoms,
    I64 features,
    const double* cells,
    double* output) {
    const I64 center = static_cast<I64>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (center >= atoms) return;
    I64 structure = 0;
    while (structure + 1 < structures && offsets[structure + 1] <= center) ++structure;
    double inverse[9]{};
    if (!inverse3_device(cells + structure * 9, inverse)) return;
    const int center_type = species_index(numbers[center], species, species_count);
    if (center_type < 0) return;
    const int angular_count = max_angular + 1;
    const int angular_block = angular_count * angular_count;
    const I64 center_stride = static_cast<I64>(species_count) * radial_count * angular_block;
    double* target = output + center * features + static_cast<I64>(center_type) * center_stride;
    const I64 begin = graph_offsets[center];
    const I64 end = graph_offsets[center + 1];
    for (I64 edge = begin; edge < end; ++edge) {
        const double distance = sqrt(fmax(0.0, graph_distance2[edge]));
        if (distance <= 1e-12 || distance >= cutoff) continue;
        const int neighbor_type = species_index(numbers[graph_atoms[edge]], species, species_count);
        if (neighbor_type < 0) continue;
        const double short_cutoff = 0.5 * (1.0 + cos(kPi * distance / cutoff));
        const double long_factor = lode_reciprocal_factor(
            positions, offsets[structure], offsets[structure + 1], center,
            graph_atoms[edge], inverse, k_cutoff, density_width, exponent);
        const double* vector = graph_displacements + edge * 3;
        double harmonics[441]{};
        harmonic_values<20>(vector, harmonics, max_angular);
        for (int angular = 0; angular <= max_angular; ++angular) {
            for (int m = -angular; m <= angular; ++m) {
                for (int radial = 0; radial < radial_count; ++radial) {
                    const double short_value = exp(-(radial + 1.0) * distance * distance
                        / (radial_radius * radial_radius));
                    const double radial_value = short_cutoff * short_value
                        * (1.0 + long_factor / (1.0 + radial));
                    const I64 index = static_cast<I64>(neighbor_type) * radial_count * angular_block
                        + static_cast<I64>(angular * angular + angular + m) * radial_count + radial;
                    target[index] += radial_value * harmonics[angular * angular + angular + m];
                }
            }
        }
    }
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

py::dict compute_lode_descriptor(
    CudaExecutionContext& context,
    DeviceBatch& batch,
    DeviceNeighborGraph& graph,
    const detail::StructureBatchView& host_batch,
    const py::dict& options) {
    (void)graph;
    const auto species = species_option(options);
    if (species.empty()) throw std::invalid_argument("LODE species must not be empty");
    const double cutoff = option(options, "cutoff", 6.0);
    const double sigma = option(options, "density_width", 0.3);
    const int max_radial = option(options, "max_radial", 6);
    const int max_angular = option(options, "max_angular", 4);
    const double k_cutoff = option(options, "k_cutoff", 2.5);
    const int exponent = option(options, "exponent", 1);
    const double radial_radius = option(options, "radial_radius", cutoff);
    if (cutoff <= 0.0 || sigma <= 0.0 || max_radial < 0 || max_angular < 0
        || max_angular > 20 || k_cutoff <= 0.0 || exponent < 1 || exponent > 9
        || radial_radius <= 0.0) {
        throw std::invalid_argument("invalid CUDA LODE parameters");
    }
    const int radial_count = max_radial + 1;
    const int angular_block = (max_angular + 1) * (max_angular + 1);
    const I64 computed = static_cast<I64>(species.size()) * species.size()
        * radial_count * angular_block;
    const I64 features = payload_or_option_feature_count(
        options, computed, "LodeSphericalExpansion");
    if (features != computed) throw std::invalid_argument("CUDA LODE feature count mismatch");

    // The LODE density is defined in reciprocal space.  Build only the small
    // per-structure k-vector and radial-basis metadata on the host; phases,
    // species sums, angular projection, and the final coefficients stay in
    // the CUDA kernel.
    std::vector<I64> k_offsets(static_cast<std::size_t>(host_batch.structures) + 1U, 0);
    std::vector<double> k_vectors;
    std::vector<double> k_norms;
    for (I64 structure = 0; structure < host_batch.structures; ++structure) {
        const auto vectors = detail::make_k_vectors(
            host_batch.cells + structure * 9, k_cutoff);
        if (vectors.empty()) {
            throw std::invalid_argument(
                "no LODE reciprocal vectors for the current cell and k_cutoff");
        }
        for (const auto& vector : vectors) {
            k_vectors.push_back(vector.vector[0]);
            k_vectors.push_back(vector.vector[1]);
            k_vectors.push_back(vector.vector[2]);
            k_norms.push_back(vector.norm);
        }
        k_offsets[static_cast<std::size_t>(structure + 1)] =
            static_cast<I64>(k_norms.size());
    }
    std::vector<double> widths(static_cast<std::size_t>(radial_count), 0.0);
    std::vector<double> lode_prefactors(static_cast<std::size_t>(radial_count), 0.0);
    std::vector<double> gamma_a(static_cast<std::size_t>(max_angular + 1) * radial_count, 0.0);
    std::vector<double> gamma_b(static_cast<std::size_t>(max_angular + 1), 0.0);
    std::vector<double> orthonormalization(
        static_cast<std::size_t>(max_angular + 1) * radial_count * radial_count, 0.0);
    for (int angular = 0; angular <= max_angular; ++angular) {
        const detail::GtoRadialBasis basis(radial_count, radial_radius, angular);
        if (angular == 0) {
            widths = basis.widths;
            lode_prefactors = basis.lode_prefactors;
        }
        std::copy(
            basis.gamma_a.begin(), basis.gamma_a.end(),
            gamma_a.begin() + static_cast<std::size_t>(angular) * radial_count);
        gamma_b[static_cast<std::size_t>(angular)] = basis.gamma_b;
        for (int raw = 0; raw < radial_count; ++raw) {
            for (int target = 0; target < radial_count; ++target) {
                orthonormalization[
                    (static_cast<std::size_t>(angular) * radial_count + raw) * radial_count
                        + target] = basis.orthonormalization[static_cast<std::size_t>(raw)]
                            [static_cast<std::size_t>(target)];
            }
        }
    }
    DeviceBuffer<I32> d_species;
    DeviceBuffer<I64> d_k_offsets;
    DeviceBuffer<double> d_k_vectors;
    DeviceBuffer<double> d_k_norms;
    DeviceBuffer<double> d_widths;
    DeviceBuffer<double> d_lode_prefactors;
    DeviceBuffer<double> d_gamma_a;
    DeviceBuffer<double> d_gamma_b;
    DeviceBuffer<double> d_orthonormalization;
    d_species.upload(species.data(), species.size(), context.stream(), "could not upload LODE species");
    d_k_offsets.upload(k_offsets.data(), k_offsets.size(), context.stream(), "could not upload LODE k offsets");
    d_k_vectors.upload(k_vectors.data(), k_vectors.size(), context.stream(), "could not upload LODE k vectors");
    d_k_norms.upload(k_norms.data(), k_norms.size(), context.stream(), "could not upload LODE k norms");
    d_widths.upload(widths.data(), widths.size(), context.stream(), "could not upload LODE radial widths");
    d_lode_prefactors.upload(
        lode_prefactors.data(), lode_prefactors.size(), context.stream(),
        "could not upload LODE radial prefactors");
    d_gamma_a.upload(gamma_a.data(), gamma_a.size(), context.stream(), "could not upload LODE gamma values");
    d_gamma_b.upload(gamma_b.data(), gamma_b.size(), context.stream(), "could not upload LODE gamma denominators");
    d_orthonormalization.upload(
        orthonormalization.data(), orthonormalization.size(), context.stream(),
        "could not upload LODE radial orthonormalization");
    const std::size_t size = static_cast<std::size_t>(batch.atoms()) * static_cast<std::size_t>(features);
    double* output = context.output_buffer(size);
    if (size > 0) {
        check_cuda(cudaMemsetAsync(output, 0, size * sizeof(double), context.stream()),
            "could not clear CUDA LODE output");
        constexpr unsigned block_size = 64;
        lode_exact_kernel<<<static_cast<unsigned>((batch.atoms() + block_size - 1) / block_size),
            block_size, 0, context.stream()>>>(
            batch.numbers(), batch.positions(), batch.offsets(), d_species.get(),
            static_cast<int>(species.size()), radial_count, max_angular, sigma, exponent,
            d_k_offsets.get(), d_k_vectors.get(), d_k_norms.get(), d_widths.get(),
            d_lode_prefactors.get(), d_gamma_a.get(), d_gamma_b.get(),
            d_orthonormalization.get(), batch.cells(), batch.structures(), batch.atoms(),
            features, output);
        check_cuda(cudaGetLastError(), "CUDA LODE kernel launch failed");
    }
    const auto values = context.download_output(size);
    return atom_result(values, batch.atoms(), features, "LodeSphericalExpansion", options, false,
        std::vector<I64>(host_batch.offsets, host_batch.offsets + host_batch.structures + 1));
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

py::dict compute_rotational_descriptor(
    CudaExecutionContext& context,
    DeviceBatch& batch,
    DeviceNeighborGraph& graph,
    const detail::StructureBatchView& host_batch,
    const std::string& name,
    const py::dict& options,
    RotationalPlanCache* rotational_plan_cache) {
    const auto species = batch_species(host_batch);
    if (species.empty()) throw std::invalid_argument(name + " requires at least one atom");
    const RotationalCudaOptions config = rotational_options(name, options);
    const int kind = config.kind;
    if (!std::isfinite(config.cutoff) || !std::isfinite(config.alpha)
        || !std::isfinite(config.rfac0) || !std::isfinite(config.rmin0)
        || !std::isfinite(config.rcutfac)
        || config.cutoff <= 0.0 || config.alpha <= 0.0 || config.rfac0 <= 0.0
        || config.rmin0 < 0.0 || config.rcutfac <= 0.0
        || config.lmax < 0 || config.nmax < 1
        || config.diagonal < 0 || config.diagonal > 3
        || (kind != 0 && config.twojmax < 0)) {
        throw std::invalid_argument("invalid CUDA rotational descriptor parameters");
    }
    if (kind == 0 && (config.nmax > 8 || config.lmax > 8)) {
        throw std::invalid_argument(
            "CUDA SO3 supports nmax and lmax up to 8 in the current kernel");
    }
    I64 computed = 0;
    detail::rotational::BispectrumPlan bispectrum_plan;
    detail::rotational::FlattenedBispectrumPlan flattened_plan;
    RotationalPlanDeviceView rotational_plan;
    if (kind == 0) {
        computed = static_cast<I64>(config.lmax + 1) * config.nmax * (config.nmax + 1) / 2;
    } else {
        const int order = mdescriptor::detail::rotational::expansion_order(
            kind, config.lmax, config.twojmax);
        if (order > 10) {
            throw std::invalid_argument(
                name + " CUDA expansion order is limited to 10");
        }
        if (rotational_plan_cache != nullptr) {
            rotational_plan = rotational_plan_cache->prepare(
                context, order, config.diagonal, name == "LBispectrum");
            computed = rotational_plan.features;
        } else {
            bispectrum_plan = detail::rotational::make_bispectrum_plan(
                order, config.diagonal, name == "LBispectrum");
            flattened_plan = detail::rotational::flatten(bispectrum_plan);
            computed = static_cast<I64>(bispectrum_plan.components.size());
        }
    }
    const I64 features = payload_or_option_feature_count(options, computed, name);
    if (features != computed) throw std::invalid_argument(name + " CUDA feature count mismatch");
    const py::str weights_key("weights");
    const bool custom_weights = options.contains(weights_key)
        && !options[weights_key].is_none();
    std::vector<double> weights;
    std::vector<double> radii;
    // SO4 uses atomic numbers directly in the rotational kernel.  For SNAP
    // and L-Bispectrum a null pointer is the common all-ones fast path.
    if (kind != 1 && custom_weights) {
        weights.assign(static_cast<std::size_t>(host_batch.atoms), 1.0);
        const auto configured = species_dictionary_values(options[weights_key], species, 1.0);
        for (I64 atom = 0; atom < host_batch.atoms; ++atom) {
            const auto found = std::find(species.begin(), species.end(), host_batch.numbers[atom]);
            if (found != species.end()) weights[static_cast<std::size_t>(atom)]
                = configured[static_cast<std::size_t>(found - species.begin())];
        }
    }
    const py::str radii_key("element_radii");
    if (name == "LBispectrum" && options.contains(radii_key) && !options[radii_key].is_none()) {
        const auto configured = species_dictionary_values(options[radii_key], species, config.cutoff * 0.5);
        radii.resize(static_cast<std::size_t>(host_batch.atoms), config.cutoff * 0.5);
        for (I64 atom = 0; atom < host_batch.atoms; ++atom) {
            const auto found = std::find(species.begin(), species.end(), host_batch.numbers[atom]);
            if (found != species.end()) radii[static_cast<std::size_t>(atom)]
                = configured[static_cast<std::size_t>(found - species.begin())];
        }
    }
    const double graph_cutoff = radii.empty() ? config.cutoff
        : 2.0 * *std::max_element(radii.begin(), radii.end()) * config.rcutfac;
    graph.build_dpa(
        context, batch, host_batch, graph_cutoff, true, false,
        mdescriptor::detail::rotational::kBispectrumIncludeExactSelf);
    DeviceBuffer<double> d_weights;
    DeviceBuffer<double> d_radii;
    d_weights.upload(weights.data(), weights.size(), context.stream(), "could not upload rotational weights");
    d_radii.upload(radii.data(), radii.size(), context.stream(), "could not upload rotational radii");
    DeviceBuffer<I64> d_bispectrum_z_inner_offsets;
    DeviceBuffer<I64> d_bispectrum_inner_term_offsets;
    DeviceBuffer<double> d_bispectrum_inner_outer_coefficients;
    DeviceBuffer<I64> d_bispectrum_term_first_indices;
    DeviceBuffer<I64> d_bispectrum_term_second_indices;
    DeviceBuffer<double> d_bispectrum_term_coefficients;
    DeviceBuffer<I64> d_bispectrum_projection_offsets;
    DeviceBuffer<I64> d_bispectrum_projection_u_indices;
    DeviceBuffer<I64> d_bispectrum_projection_z_indices;
    DeviceBuffer<double> d_bispectrum_projection_scales;
    if (kind != 0 && rotational_plan_cache == nullptr) {
        d_bispectrum_z_inner_offsets.upload(
            flattened_plan.z_inner_offsets.data(), flattened_plan.z_inner_offsets.size(),
            context.stream(), "could not upload CUDA bispectrum Z offsets");
        d_bispectrum_inner_term_offsets.upload(
            flattened_plan.inner_term_offsets.data(), flattened_plan.inner_term_offsets.size(),
            context.stream(), "could not upload CUDA bispectrum inner offsets");
        d_bispectrum_inner_outer_coefficients.upload(
            flattened_plan.inner_outer_coefficients.data(),
            flattened_plan.inner_outer_coefficients.size(), context.stream(),
            "could not upload CUDA bispectrum outer coefficients");
        d_bispectrum_term_first_indices.upload(
            flattened_plan.term_first_indices.data(), flattened_plan.term_first_indices.size(),
            context.stream(), "could not upload CUDA bispectrum first indices");
        d_bispectrum_term_second_indices.upload(
            flattened_plan.term_second_indices.data(), flattened_plan.term_second_indices.size(),
            context.stream(), "could not upload CUDA bispectrum second indices");
        d_bispectrum_term_coefficients.upload(
            flattened_plan.term_coefficients.data(), flattened_plan.term_coefficients.size(),
            context.stream(), "could not upload CUDA bispectrum CG coefficients");
        d_bispectrum_projection_offsets.upload(
            flattened_plan.projection_offsets.data(), flattened_plan.projection_offsets.size(),
            context.stream(), "could not upload CUDA bispectrum projection offsets");
        d_bispectrum_projection_u_indices.upload(
            flattened_plan.projection_u_indices.data(), flattened_plan.projection_u_indices.size(),
            context.stream(), "could not upload CUDA bispectrum projection U indices");
        d_bispectrum_projection_z_indices.upload(
            flattened_plan.projection_z_indices.data(), flattened_plan.projection_z_indices.size(),
            context.stream(), "could not upload CUDA bispectrum projection Z indices");
        d_bispectrum_projection_scales.upload(
            flattened_plan.projection_scales.data(), flattened_plan.projection_scales.size(),
            context.stream(), "could not upload CUDA bispectrum projection scales");
    }
    const std::size_t size = static_cast<std::size_t>(batch.atoms()) * static_cast<std::size_t>(features);
    double* output = context.output_buffer(size);
    DeviceBuffer<double> d_so3_basis;
    int so3_quadrature_count = 0;
    if (kind == 0) {
        const auto so3_basis = so3_basis_host(
            config.nmax, config.lmax, config.cutoff, config.alpha, &so3_quadrature_count);
        d_so3_basis.upload(
            so3_basis.data(), so3_basis.size(), context.stream(),
            "could not upload CUDA SO3 radial basis");
    }
    if (size > 0) {
        check_cuda(cudaMemsetAsync(output, 0, size * sizeof(double), context.stream()),
            "could not clear CUDA rotational output");
        constexpr unsigned block_size = 64;
        if (kind == 0) {
            so3_kernel<<<static_cast<unsigned>((batch.atoms() + block_size - 1) / block_size),
                block_size, 0, context.stream()>>>(
                batch.numbers(), graph.offsets(), graph.atoms(), graph.displacements(),
                graph.distance2(), config.nmax, config.lmax, config.cutoff, config.alpha,
                config.weight_on, so3_quadrature_count, d_so3_basis.get(), features,
                batch.atoms(), output);
            check_cuda(cudaGetLastError(), "CUDA SO3 kernel launch failed");
        } else {
            rotational_kernel<<<static_cast<unsigned>((batch.atoms() + block_size - 1) / block_size),
                block_size, 0, context.stream()>>>(
                batch.numbers(), graph.offsets(), graph.atoms(), graph.displacements(), graph.distance2(),
                d_weights.get(), d_radii.get(),
                rotational_plan_cache == nullptr ? d_bispectrum_z_inner_offsets.get()
                    : rotational_plan.z_inner_offsets,
                rotational_plan_cache == nullptr ? d_bispectrum_inner_term_offsets.get()
                    : rotational_plan.inner_term_offsets,
                rotational_plan_cache == nullptr ? d_bispectrum_inner_outer_coefficients.get()
                    : rotational_plan.inner_outer_coefficients,
                rotational_plan_cache == nullptr ? d_bispectrum_term_first_indices.get()
                    : rotational_plan.term_first_indices,
                rotational_plan_cache == nullptr ? d_bispectrum_term_second_indices.get()
                    : rotational_plan.term_second_indices,
                rotational_plan_cache == nullptr ? d_bispectrum_term_coefficients.get()
                    : rotational_plan.term_coefficients,
                rotational_plan_cache == nullptr ? d_bispectrum_projection_offsets.get()
                    : rotational_plan.projection_offsets,
                rotational_plan_cache == nullptr ? d_bispectrum_projection_u_indices.get()
                    : rotational_plan.projection_u_indices,
                rotational_plan_cache == nullptr ? d_bispectrum_projection_z_indices.get()
                    : rotational_plan.projection_z_indices,
                rotational_plan_cache == nullptr ? d_bispectrum_projection_scales.get()
                    : rotational_plan.projection_scales,
                kind, config.nmax, config.lmax,
                config.twojmax, config.cutoff, config.rfac0, config.rmin0, config.rcutfac,
                config.normalize_u, static_cast<int>(features), batch.atoms(), output);
            check_cuda(cudaGetLastError(), "CUDA rotational kernel launch failed");
        }
    }
    const auto values = context.download_output(size);
    return atom_result(values, batch.atoms(), features, name, options, false,
        std::vector<I64>(host_batch.offsets, host_batch.offsets + host_batch.structures + 1));
}

__device__ double c00_spherical_bessel(int angular, double x) {
    const double absolute = fabs(x);
    if (absolute < 1e-4) {
        if (angular == 0) {
            const double x2 = x * x;
            return 1.0 - x2 / 6.0 + x2 * x2 / 120.0;
        }
        double denominator = 1.0;
        for (int value = 1; value <= angular; ++value) {
            denominator *= 2.0 * value + 1.0;
        }
        return pow(x, angular) / denominator
            * (1.0 - x * x / (2.0 * (2.0 * angular + 3.0)));
    }
    const double j0 = sin(x) / x;
    if (angular == 0) return j0;
    const double j1 = sin(x) / (x * x) - cos(x) / x;
    if (angular == 1) return j1;
    double previous = j0;
    double current = j1;
    for (int degree = 1; degree < angular; ++degree) {
        const double next = (2.0 * degree + 1.0) * current / x - previous;
        previous = current;
        current = next;
    }
    return current;
}

__device__ double c00_cutoff_value(int kind, double distance, double cutoff) {
    if (distance > cutoff) return 0.0;
    if (kind == 0) {
        return 0.5 * (cos(kPi * distance / cutoff) + 1.0);
    }
    if (kind == 1) {
        const double x = 4.0 * distance / cutoff - 3.0;
        if (x < -1.0) return 1.0;
        if (x < 1.0) return 0.25 * (x * x * x - 3.0 * x + 2.0);
        return 0.0;
    }
    constexpr double delta = 0.5;
    const double r1 = cutoff > delta ? cutoff - delta : 0.5 * cutoff;
    double value = 1.0;
    if (distance > r1) {
        const double cutoff2 = cutoff * cutoff;
        const double distance2 = distance * distance;
        value = (cutoff2 - distance2) * (cutoff2 - distance2)
            * (cutoff2 + 2.0 * distance2 - 3.0 * r1 * r1)
            / pow(cutoff2 - r1 * r1, 3.0);
    }
    if (kind == 3) value /= 1.0 + pow(distance / 2.0, 7.0);
    return value;
}

__device__ double c00_radial_value(
    double distance,
    int angular,
    int radial,
    int cutoff_kind,
    double cutoff,
    double sigma,
    const double* zeros,
    const double* norms,
    const double* tables,
    const I64* zero_offsets,
    const I64* norm_offsets,
    const I64* table_offsets,
    const I32* radial_counts,
    int table_width) {
    const int count = radial_counts[angular];
    if (radial < 0 || radial >= count) return 0.0;
    if (sigma > 0.0) {
        const double coordinate = fmin(1.0, fmax(0.0, distance / cutoff))
            * static_cast<double>(table_width - 1);
        const int left = min(table_width - 2, static_cast<int>(coordinate));
        const double fraction = coordinate - static_cast<double>(left);
        const I64 base = table_offsets[angular] + static_cast<I64>(radial) * table_width;
        const auto value = [&](int point) {
            const int bounded = max(0, min(table_width - 1, point));
            return tables[base + bounded];
        };
        const double p0 = value(left - 1);
        const double p1 = value(left);
        const double p2 = value(left + 1);
        const double p3 = value(left + 2);
        return p1 + 0.5 * fraction * (
            p2 - p0 + fraction * (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3
                + fraction * (3.0 * (p1 - p2) + p3 - p0)));
    }
    const double basis = c00_spherical_bessel(
        angular,
        zeros[zero_offsets[angular] + radial] * distance / cutoff);
    return c00_cutoff_value(cutoff_kind, distance, cutoff)
        * basis / norms[norm_offsets[angular] + radial];
}

__global__ void c00ps_mlff_kernel(
    const I32* numbers,
    const I64* graph_offsets,
    const I32* graph_atoms,
    const double* graph_displacements,
    const double* graph_distance2,
    const I32* species,
    int species_count,
    const I32* radial_counts,
    const I64* zero_offsets,
    const I64* norm_offsets,
    const I64* table_offsets,
    const double* zeros,
    const double* norms,
    const double* tables,
    const I64* coefficient_offsets,
    int cutoff_kind,
    double cutoff,
    double sigma,
    bool include_radial,
    bool include_angular,
    bool normalize_radial,
    bool normalize_angular,
    bool super_vector,
    bool exclude_self,
    double radial_weight,
    double angular_weight,
    int max_angular,
    int table_width,
    I64 features,
    I64 atoms,
    I64 coefficient_stride,
    double* workspace,
    double* output) {
    const I64 center = static_cast<I64>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (center >= atoms) return;
    const int center_type = species_index(numbers[center], species, species_count);
    if (center_type < 0) return;
    constexpr int MaxAngular = 20;
    double* coefficients = workspace + center * coefficient_stride;
    for (I64 index = 0; index < coefficient_stride; ++index) coefficients[index] = 0.0;
    const I64 begin = graph_offsets[center];
    const I64 end = graph_offsets[center + 1];
    for (I64 edge = begin; edge < end; ++edge) {
        const double distance = sqrt(fmax(0.0, graph_distance2[edge]));
        if (distance <= 1e-12 || distance > cutoff) continue;
        const int type = species_index(numbers[graph_atoms[edge]], species, species_count);
        if (type < 0) continue;
        double harmonics[441]{};
        harmonic_values<MaxAngular>(graph_displacements + edge * 3, harmonics, max_angular);
        for (int angular = 0; angular <= max_angular; ++angular) {
            const int count = radial_counts[angular];
            const I64 coefficient_base = coefficient_offsets[angular]
                + static_cast<I64>(type) * count * (2 * angular + 1);
            for (int radial = 0; radial < count; ++radial) {
                const double value = c00_radial_value(
                    distance, angular, radial, cutoff_kind, cutoff, sigma,
                    zeros, norms, tables,
                    zero_offsets, norm_offsets, table_offsets, radial_counts, table_width);
                const I64 destination = coefficient_base
                    + static_cast<I64>(radial) * (2 * angular + 1);
                for (int m = 0; m <= 2 * angular; ++m) {
                    coefficients[destination + m] += value
                        * harmonics[angular * angular + m];
                }
            }
        }
    }
    double* target = output + center * features;
    I64 output_index = 0;
    const int radial_channels = species_count * radial_counts[0];
    if (include_radial) {
        for (int channel = 0; channel < radial_channels; ++channel) {
            const int type = channel / radial_counts[0];
            const int radial = channel % radial_counts[0];
            target[output_index++] = coefficients[
                coefficient_offsets[0] + type * radial_counts[0] + radial];
        }
        if (normalize_radial) {
            double norm2 = 0.0;
            for (int index = 0; index < radial_channels; ++index) norm2 += target[index] * target[index];
            if (norm2 > 1e-20) {
                const double scale = 1.0 / sqrt(norm2);
                for (int index = 0; index < radial_channels; ++index) target[index] *= scale;
            }
        }
    }
    if (include_angular) {
        const I64 angular_offset = include_radial ? radial_channels : 0;
        I64 angular_index = 0;
        for (int angular = 0; angular <= max_angular; ++angular) {
            const int count = radial_counts[angular];
            const int channels = species_count * count;
            const double prefactor = sqrt(8.0 * kPi * kPi / (2.0 * angular + 1.0));
            for (int first = 0; first < channels; ++first) {
                for (int second = first; second < channels; ++second) {
                    double value = 0.0;
                    const int first_type = first / count;
                    const int first_radial = first % count;
                    const int second_type = second / count;
                    const int second_radial = second % count;
                    const I64 first_base = coefficient_offsets[angular]
                        + static_cast<I64>(first_type * count + first_radial) * (2 * angular + 1);
                    const I64 second_base = coefficient_offsets[angular]
                        + static_cast<I64>(second_type * count + second_radial) * (2 * angular + 1);
                    for (int m = 0; m <= 2 * angular; ++m) {
                        value += coefficients[first_base + m] * coefficients[second_base + m];
                    }
                    if (exclude_self && first_type == center_type && second_type == center_type) {
                        const double addition = (2.0 * angular + 1.0) / (4.0 * kPi);
                        for (I64 edge = begin; edge < end; ++edge) {
                            const double distance = sqrt(fmax(0.0, graph_distance2[edge]));
                            if (distance <= 1e-12 || distance > cutoff) continue;
                            const int type = species_index(numbers[graph_atoms[edge]], species, species_count);
                            if (type != center_type) continue;
                            const double left = c00_radial_value(
                                distance, angular, first_radial, cutoff_kind, cutoff, sigma,
                                zeros, norms, tables,
                                zero_offsets, norm_offsets, table_offsets, radial_counts, table_width);
                            const double right = c00_radial_value(
                                distance, angular, second_radial, cutoff_kind, cutoff, sigma,
                                zeros, norms, tables,
                                zero_offsets, norm_offsets, table_offsets, radial_counts, table_width);
                            value -= addition * left * right;
                        }
                    }
                    const double pair_weight = first_radial == second_radial ? 1.0 : sqrt(2.0);
                    target[angular_offset + angular_index++] = pair_weight * prefactor * value;
                }
            }
        }
    }
    if (normalize_angular) {
        const I64 angular_offset = include_radial ? radial_channels : 0;
        const I64 angular_size = features - angular_offset;
        double norm2 = 0.0;
        for (I64 index = 0; index < angular_size; ++index) {
            norm2 += target[angular_offset + index] * target[angular_offset + index];
        }
        if (norm2 > 1e-20) {
            const double scale = 1.0 / sqrt(norm2);
            for (I64 index = 0; index < angular_size; ++index) target[angular_offset + index] *= scale;
        }
    }
    if (super_vector) {
        const I64 radial_end = include_radial ? radial_channels : 0;
        if (include_radial) {
            const double scale = sqrt(radial_weight);
            for (I64 index = 0; index < radial_end; ++index) target[index] *= scale;
        }
        if (include_angular) {
            const double scale = sqrt(angular_weight);
            for (I64 index = radial_end; index < features; ++index) target[index] *= scale;
        }
        double norm2 = 0.0;
        for (I64 index = 0; index < features; ++index) norm2 += target[index] * target[index];
        if (norm2 > 1e-20) {
            const double scale = 1.0 / sqrt(norm2);
            for (I64 index = 0; index < features; ++index) target[index] *= scale;
        }
    }
}

py::dict compute_c00ps_mlff_descriptor(
    CudaExecutionContext& context,
    DeviceBatch& batch,
    DeviceNeighborGraph& graph,
    const detail::StructureBatchView& host_batch,
    const py::dict& options) {
    const py::str payload_key("_cuda_payload");
    if (!options.contains(payload_key) || options[payload_key].is_none()) {
        throw std::invalid_argument("C00PSMLFF CUDA backend requires its prepared basis payload");
    }
    const py::dict payload = py::cast<py::dict>(options[payload_key]);
    const auto species = species_option(options);
    const auto radial_counts = py::cast<std::vector<I32>>(payload["radial_counts"]);
    const auto zeros_nested = nested_payload_vectors(payload, "basis_zeros");
    const auto norms_nested = nested_payload_vectors(payload, "basis_norms");
    const auto tables_nested = nested_payload_vectors(payload, "basis_values");
    const int max_angular = option(options, "l_max", 4);
    const double cutoff = option(options, "r_cut", option(options, "cutoff", 6.0));
    const double sigma = option(options, "radial_sigma", 0.5);
    if (species.empty() || max_angular < 0 || max_angular >= static_cast<int>(radial_counts.size())
        || cutoff <= 0.0 || sigma < 0.0
        || zeros_nested.size() != radial_counts.size()
        || norms_nested.size() != radial_counts.size()
        || tables_nested.size() != radial_counts.size()) {
        throw std::invalid_argument("invalid C00PSMLFF CUDA basis payload");
    }
    const bool include_radial = option(options, "include_radial", true);
    const bool include_angular = option(options, "include_angular", true);
    const bool normalize_radial = option(options, "normalize_radial", false);
    const bool normalize_angular = option(options, "normalize_angular", false);
    const bool super_vector = option(options, "super_vector", false);
    const bool exclude_self = option(options, "exclude_self_interaction", true);
    const double radial_weight = option(options, "radial_weight", 1.0);
    const double angular_weight = option(options, "angular_weight", 1.0);
    const std::string cutoff_name = option(options, "cutoff_function", std::string("bp"));
    int cutoff_kind = cutoff_name == "bp" ? 0 : cutoff_name == "mo" ? 1
        : cutoff_name == "rj" ? 2 : cutoff_name == "wmc" ? 3 : -1;
    if (cutoff_kind < 0) throw std::invalid_argument("invalid C00PSMLFF cutoff function");

    std::vector<I64> zero_offsets(radial_counts.size(), 0);
    std::vector<I64> norm_offsets(radial_counts.size(), 0);
    std::vector<I64> table_offsets(radial_counts.size(), 0);
    std::vector<double> zeros;
    std::vector<double> norms;
    std::vector<double> tables;
    I64 coefficient_stride = 0;
    for (std::size_t angular = 0; angular < radial_counts.size(); ++angular) {
        zero_offsets[angular] = static_cast<I64>(zeros.size());
        norm_offsets[angular] = static_cast<I64>(norms.size());
        table_offsets[angular] = static_cast<I64>(tables.size());
        zeros.insert(zeros.end(), zeros_nested[angular].begin(), zeros_nested[angular].end());
        norms.insert(norms.end(), norms_nested[angular].begin(), norms_nested[angular].end());
        tables.insert(tables.end(), tables_nested[angular].begin(), tables_nested[angular].end());
        coefficient_stride += static_cast<I64>(species.size()) * radial_counts[angular]
            * (2 * static_cast<int>(angular) + 1);
    }
    std::vector<I64> coefficient_offsets(radial_counts.size(), 0);
    I64 coefficient_offset = 0;
    for (std::size_t angular = 0; angular < radial_counts.size(); ++angular) {
        coefficient_offsets[angular] = coefficient_offset;
        coefficient_offset += static_cast<I64>(species.size()) * radial_counts[angular]
            * (2 * static_cast<int>(angular) + 1);
    }
    const int table_width = 10001;
    if (sigma > 0.0) {
        for (std::size_t angular = 0; angular < tables_nested.size(); ++angular) {
            const std::size_t expected = static_cast<std::size_t>(radial_counts[angular]) * table_width;
            if (tables_nested[angular].size() != expected) {
                throw std::invalid_argument("C00PSMLFF CUDA radial table has an unexpected size");
            }
        }
    }
    const I64 features = feature_count_option(options, 0);
    const I64 radial_features = include_radial
        ? static_cast<I64>(species.size()) * radial_counts[0] : 0;
    I64 angular_features = 0;
    if (include_angular) {
        for (int angular = 0; angular <= max_angular; ++angular) {
            const I64 channels = static_cast<I64>(species.size()) * radial_counts[angular];
            angular_features += channels * (channels + 1) / 2;
        }
    }
    const I64 computed_features = radial_features + angular_features;
    if (features != computed_features || computed_features <= 0) {
        throw std::invalid_argument("C00PSMLFF CUDA feature count mismatch");
    }
    graph.build_dpa(context, batch, host_batch, cutoff, true, false, false);

    DeviceBuffer<I32> d_species;
    DeviceBuffer<I32> d_radial_counts;
    DeviceBuffer<I64> d_zero_offsets;
    DeviceBuffer<I64> d_norm_offsets;
    DeviceBuffer<I64> d_table_offsets;
    DeviceBuffer<double> d_zeros;
    DeviceBuffer<double> d_norms;
    DeviceBuffer<double> d_tables;
    DeviceBuffer<I64> d_coefficient_offsets;
    d_species.upload(species.data(), species.size(), context.stream(), "could not upload C00PS species");
    d_radial_counts.upload(radial_counts.data(), radial_counts.size(), context.stream(), "could not upload C00PS radial counts");
    d_zero_offsets.upload(zero_offsets.data(), zero_offsets.size(), context.stream(), "could not upload C00PS zero offsets");
    d_norm_offsets.upload(norm_offsets.data(), norm_offsets.size(), context.stream(), "could not upload C00PS norm offsets");
    d_table_offsets.upload(table_offsets.data(), table_offsets.size(), context.stream(), "could not upload C00PS table offsets");
    d_zeros.upload(zeros.data(), zeros.size(), context.stream(), "could not upload C00PS zeros");
    d_norms.upload(norms.data(), norms.size(), context.stream(), "could not upload C00PS norms");
    d_tables.upload(tables.data(), tables.size(), context.stream(), "could not upload C00PS radial tables");
    d_coefficient_offsets.upload(coefficient_offsets.data(), coefficient_offsets.size(), context.stream(), "could not upload C00PS coefficient offsets");
    const std::size_t size = static_cast<std::size_t>(batch.atoms()) * static_cast<std::size_t>(features);
    double* output = context.output_buffer(size);
    auto* workspace = static_cast<double*>(context.workspace_buffer(
        static_cast<std::size_t>(batch.atoms()) * static_cast<std::size_t>(coefficient_stride)
        * sizeof(double)));
    if (size > 0) {
        check_cuda(cudaMemsetAsync(output, 0, size * sizeof(double), context.stream()),
            "could not clear C00PS output");
        constexpr unsigned block_size = 64;
        c00ps_mlff_kernel<<<static_cast<unsigned>((batch.atoms() + block_size - 1) / block_size),
            block_size, 0, context.stream()>>>(
            batch.numbers(), graph.offsets(), graph.atoms(), graph.displacements(), graph.distance2(),
            d_species.get(), static_cast<int>(species.size()), d_radial_counts.get(),
            d_zero_offsets.get(), d_norm_offsets.get(), d_table_offsets.get(),
            d_zeros.get(), d_norms.get(), d_tables.get(), d_coefficient_offsets.get(),
            cutoff_kind, cutoff, sigma, include_radial, include_angular, normalize_radial,
            normalize_angular, super_vector, exclude_self, radial_weight, angular_weight,
            max_angular, table_width, features, batch.atoms(), coefficient_stride, workspace, output);
        check_cuda(cudaGetLastError(), "CUDA C00PSMLFF kernel launch failed");
    }
    const auto values = context.download_output(size);
    return atom_result(values, batch.atoms(), features, "C00PSMLFF", options, false,
        std::vector<I64>(host_batch.offsets, host_batch.offsets + host_batch.structures + 1));
}

__device__ double turbo_radial_normalization(int n) {
    return sqrt(1.0 / (2.0 * static_cast<double>(n) + 5.0));
}

__device__ double turbo_smoothing_prefactor(
    double rj, double sigma, double soft, double hard, double nf) {
    if (hard == soft || soft - rj >= 4.0 * sigma) return 0.0;
    const double dr = hard - soft;
    const double sigma2 = sigma * sigma;
    return exp(-0.5 * (soft - rj) * (soft - rj)
        / (sigma2 + dr * dr / (nf * nf)));
}

__device__ void turbo_radial_coefficients_device(
    int type,
    double distance,
    bool central,
    int basis_kind,
    int basis_size,
    double rcut_hard,
    double rcut_soft,
    double nf,
    int radial_enhancement,
    double basis_sigma,
    double atom_sigma_r,
    double atom_sigma_r_scaling,
    double amplitude_scaling,
    double central_weight,
    const int* alpha_max,
    const double* transforms,
    double* result,
    double* primitive,
    double* filtered,
    double* raw) {
    for (int index = 0; index < basis_size; ++index) result[index] = 0.0;
    if (distance >= rcut_hard || (basis_kind == 1 && central)) return;

    const double hard = 1.0;
    const double soft = rcut_soft / rcut_hard;
    const double rj = distance / rcut_hard;
    const double dr = hard - soft;
    const double atom_sigma = atom_sigma_r / rcut_hard;
    const double atom_sigma_scaled = atom_sigma + atom_sigma_r_scaling * rj;
    const double sigma2 = atom_sigma_scaled * atom_sigma_scaled;
    double amplitude = 1.0 / atom_sigma_scaled;
    if (amplitude_scaling != 0.0) {
        const double rj2 = rj * rj;
        const double polynomial = 1.0 + 2.0 * rj2 * rj - 3.0 * rj2;
        if (polynomial <= 1e-10) return;
        amplitude *= pow(polynomial, amplitude_scaling);
    }
    if (central) amplitude *= central_weight;
    if (radial_enhancement == 1) {
        amplitude *= rj + sqrt(2.0 / kPi) * atom_sigma_scaled;
    } else if (radial_enhancement == 2) {
        amplitude *= rj * rj + sigma2
            + sqrt(8.0 / kPi) * atom_sigma_scaled * rj;
    }
    if (amplitude == 0.0) return;

    const int expansion_count = basis_kind == 1 ? basis_size - 1 : basis_size;
    for (int index = 0; index < basis_size; ++index) {
        primitive[index] = 0.0;
        filtered[index] = 0.0;
    }
    double integral_n = 0.0;
    double norm_n = 1.0;
    double norm_np1 = turbo_radial_normalization(-2);
    double integral_np1 = sqrt(kPi / 2.0) * atom_sigma_scaled
        * (erf((soft - rj) / (sqrt(2.0) * atom_sigma_scaled))
            - erf(-rj / (sqrt(2.0) * atom_sigma_scaled))) / norm_np1;
    double correction_soft = hard == soft ? 0.0
        : sigma2 / dr * exp(-0.5 * (soft - rj) * (soft - rj) / sigma2);
    double correction_hard = sigma2 * exp(-0.5 * rj * rj / sigma2);
    for (int n = -1; n <= expansion_count; ++n) {
        correction_soft *= dr;
        correction_hard *= hard;
        const double norm_np2 = turbo_radial_normalization(n);
        const double integral_np2 = sigma2 * static_cast<double>(n + 1)
                * norm_n / norm_np2 * integral_n
            - norm_np1 * (rj - hard) / norm_np2 * integral_np1
            + correction_soft / norm_np2 - correction_hard / norm_np2;
        if (n > 0) primitive[n - 1] = integral_np2;
        norm_n = norm_np1;
        norm_np1 = norm_np2;
        integral_n = integral_np1;
        integral_np1 = integral_np2;
    }

    const double prefactor = turbo_smoothing_prefactor(
        rj, atom_sigma_scaled, soft, hard, nf);
    if (prefactor != 0.0) {
        const double nf_width2 = dr * dr / (nf * nf);
        const double filtered_sigma = atom_sigma_scaled * dr / nf
            / sqrt(sigma2 + nf_width2);
        const double filtered_sigma2 = filtered_sigma * filtered_sigma;
        const double filtered_center = (sigma2 * soft + nf_width2 * rj)
            / (sigma2 + nf_width2);
        integral_n = 0.0;
        norm_n = 1.0;
        norm_np1 = turbo_radial_normalization(-2);
        integral_np1 = sqrt(kPi / 2.0) * filtered_sigma
            * (erf((hard - filtered_center) / (sqrt(2.0) * filtered_sigma))
                - erf((soft - filtered_center) / (sqrt(2.0) * filtered_sigma))) / norm_np1;
        double filtered_correction = filtered_sigma2 / dr
            * exp(-0.5 * (soft - filtered_center) * (soft - filtered_center)
                / filtered_sigma2);
        for (int n = -1; n <= expansion_count; ++n) {
            filtered_correction *= dr;
            const double norm_np2 = turbo_radial_normalization(n);
            const double integral_np2 = filtered_sigma2 * static_cast<double>(n + 1)
                    * norm_n / norm_np2 * integral_n
                - norm_np1 * (filtered_center - hard) / norm_np2 * integral_np1
                - filtered_correction / norm_np2;
            if (n > 0) filtered[n - 1] = integral_np2;
            norm_n = norm_np1;
            norm_np1 = norm_np2;
            integral_n = integral_np1;
            integral_np1 = integral_np2;
        }
    }

    if (basis_kind == 1) {
        const double sigma_star = sqrt(basis_sigma * basis_sigma + sigma2);
        primitive[basis_size - 1] = exp(-0.5 * rj * rj / (sigma_star * sigma_star))
            * sqrt(kPi / 2.0) * atom_sigma_scaled * basis_sigma / sigma_star
            * (1.0 + erf(basis_sigma / atom_sigma_scaled * rj
                / (sqrt(2.0) * sigma_star)))
            * sqrt(2.0 / basis_sigma) / pow(kPi, 0.25);
    }

    for (int index = 0; index < basis_size; ++index) {
        raw[index] = amplitude * (primitive[index] + prefactor * filtered[index]);
    }
    int transform_offset = 0;
    for (int previous = 0; previous < type; ++previous) {
        transform_offset += alpha_max[previous] * alpha_max[previous];
    }
    const double* transform = transforms + transform_offset;
    for (int target = 0; target < basis_size; ++target) {
        double value = 0.0;
        for (int source = 0; source < basis_size; ++source) {
            value += transform[target * basis_size + source] * raw[source];
        }
        result[target] = value * sqrt(rcut_hard);
    }
}

__device__ double turbo_associated_legendre(int l, int m, double x) {
    double pmm = 1.0;
    const double root = sqrt(fmax(0.0, 1.0 - x * x));
    for (int order = 1; order <= m; ++order) {
        pmm *= -(2.0 * order - 1.0) * root;
    }
    if (l == m) return pmm;
    double pmp1m = x * (2.0 * m + 1.0) * pmm;
    if (l == m + 1) return pmp1m;
    for (int degree = m + 2; degree <= l; ++degree) {
        const double value = (x * (2.0 * degree - 1.0) * pmp1m
            - (degree + m - 1.0) * pmm) / (degree - m);
        pmm = pmp1m;
        pmp1m = value;
    }
    return pmp1m;
}

__device__ void turbo_modified_spherical_bessel(
    int l_max, double x, double* values, double* semifactorial) {
    for (int l = 0; l <= l_max; ++l) semifactorial[l] = 1.0;
    for (int l = 1; l <= l_max; ++l) {
        semifactorial[l] = semifactorial[l - 1] * (2.0 * l + 1.0);
    }
    const double x2 = x * x;
    const double x4 = x2 * x2;
    const double xcut = 1e-7;
    double flm2 = 1.0;
    double flm1 = 0.0;
    if (x > 0.0) {
        flm2 = fabs((1.0 - exp(-2.0 * x2)) / (2.0 * x2));
        flm1 = fabs((x2 - 1.0 + exp(-2.0 * x2) * (x2 + 1.0)) / (2.0 * x4));
    }
    for (int l = 0; l <= l_max; ++l) {
        if (l == 0) {
            values[0] = x < xcut ? 1.0 - x2 : flm2;
        } else if (l == 1) {
            values[1] = x2 / 1000.0 < xcut ? (x2 - x4) / semifactorial[1] : flm1;
        } else if (pow(x2, l) / semifactorial[l] * l < xcut) {
            values[l] = pow(x2, l) / semifactorial[l];
        } else {
            values[l] = fabs(flm2 - (2.0 * l - 1.0) / x2 * flm1);
        }
        if (l >= 2) {
            flm2 = flm1;
            flm1 = values[l];
        }
    }
}

__device__ void turbo_angular_coefficients(
    int l_max,
    double distance,
    const double* displacement,
    double sigma,
    double rcut_hard,
    DeviceComplex* result,
    double* ilexp,
    double* semifactorial) {
    const int packed_count = 1 + l_max * (l_max + 1) / 2 + l_max;
    for (int index = 0; index < packed_count; ++index) result[index] = {0.0, 0.0};
    if (distance >= rcut_hard) return;
    const double x = distance < 1e-14 ? 1.0
        : fmax(-1.0, fmin(1.0, displacement[2] / distance));
    const double phi = distance < 1e-14 ? 0.0 : atan2(displacement[1], displacement[0]);
    turbo_modified_spherical_bessel(l_max, distance / sigma, ilexp, semifactorial);
    const double amplitude = rcut_hard * rcut_hard / (sigma * sigma);
    const double phase_real = phi == 0.0 ? 1.0 : cos(-phi);
    const double phase_imag = phi == 0.0 ? 0.0 : sin(-phi);
    int packed = 0;
    for (int l = 0; l <= l_max; ++l) {
        double phase_real_power = 1.0;
        double phase_imag_power = 0.0;
        for (int m = 0; m <= l; ++m) {
            if (m > 0) {
                const double next_real = phase_real_power * phase_real
                    - phase_imag_power * phase_imag;
                const double next_imag = phase_real_power * phase_imag
                    + phase_imag_power * phase_real;
                phase_real_power = next_real;
                phase_imag_power = next_imag;
            }
            const double factorial_minus = tgamma(static_cast<double>(l - m) + 1.0);
            const double factorial_plus = tgamma(static_cast<double>(l + m) + 1.0);
            const double prefactor = sqrt((2.0 * l + 1.0) / (4.0 * kPi)
                * factorial_minus / factorial_plus);
            const double scale = amplitude * prefactor
                * turbo_associated_legendre(l, m, x) * ilexp[l];
            result[packed++] = {scale * phase_real_power, scale * phase_imag_power};
        }
    }
}

__device__ void turbo_accumulate_atom(
    int center_type,
    int atom_type,
    double distance,
    const double* displacement,
    bool central,
    int basis_kind,
    int l_max,
    double rcut_hard,
    double rcut_soft,
    double nf,
    int radial_enhancement,
    double basis_sigma,
    const double* atom_sigma_r,
    const double* atom_sigma_r_scaling,
    const double* atom_sigma_t,
    const double* atom_sigma_t_scaling,
    const double* amplitude_scaling,
    const double* central_weights,
    const int* alpha_max,
    const int* channel_offsets,
    const double* transforms,
    DeviceComplex* coefficients) {
    constexpr int MaxBasis = 11;
    constexpr int MaxPacked = 231;
    const int basis_size = alpha_max[atom_type];
    double radial[MaxBasis]{};
    double primitive[MaxBasis]{};
    double filtered[MaxBasis]{};
    double raw[MaxBasis]{};
    turbo_radial_coefficients_device(
        atom_type, distance, central, basis_kind, basis_size, rcut_hard, rcut_soft,
        nf, radial_enhancement, basis_sigma, atom_sigma_r[atom_type], atom_sigma_r_scaling[atom_type],
        amplitude_scaling[atom_type], central_weights[center_type], alpha_max,
        transforms, radial, primitive, filtered, raw);
    const int packed_count = 1 + l_max * (l_max + 1) / 2 + l_max;
    DeviceComplex angular[MaxPacked]{};
    double ilexp[21]{};
    double semifactorial[21]{};
    const double sigma = atom_sigma_t[atom_type]
        + atom_sigma_t_scaling[atom_type] * distance;
    turbo_angular_coefficients(
        l_max, distance, displacement, sigma, rcut_hard,
        angular, ilexp, semifactorial);
    const int offset = channel_offsets[atom_type];
    for (int n = 0; n < basis_size; ++n) {
        DeviceComplex* target = coefficients + (offset + n) * packed_count;
        for (int index = 0; index < packed_count; ++index) {
            target[index] = complex_add(target[index], complex_scale(
                angular[index], 4.0 * kPi * radial[n]));
        }
    }
}

__global__ void soap_turbo_cuda_kernel(
    const I32* numbers,
    const I64* graph_offsets,
    const I32* graph_atoms,
    const I32* graph_shifts,
    const double* graph_displacements,
    const double* graph_distance2,
    const I32* species,
    int species_count,
    const I32* alpha_max,
    const I32* channel_offsets,
    const I32* central_allowed,
    const double* atom_sigma_r,
    const double* atom_sigma_r_scaling,
    const double* atom_sigma_t,
    const double* atom_sigma_t_scaling,
    const double* amplitude_scaling,
    const double* central_weights,
    const double* transforms,
    const double* gaussian_columns,
    int basis_kind,
    int l_max,
    double rcut_hard,
    double rcut_soft,
    double nf,
    int radial_enhancement,
    I64 channels,
    I64 packed_count,
    I64 dense_features,
    I64 features,
    const I64* compression_offsets,
    const I64* compression_sources,
    const double* compression_factors,
    I64 coefficient_stride,
    I64 atoms,
    double* coefficient_workspace,
    double* dense_workspace,
    double* output) {
    const I64 center = static_cast<I64>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (center >= atoms) return;
    const int center_type = species_index(numbers[center], species, species_count);
    if (center_type < 0 || central_allowed == nullptr || central_allowed[center_type] == 0) return;
    DeviceComplex* coefficients = reinterpret_cast<DeviceComplex*>(
        coefficient_workspace + center * coefficient_stride);
    for (I64 index = 0; index < channels * packed_count; ++index) {
        coefficients[index] = {0.0, 0.0};
    }
    const double origin[3] = {0.0, 0.0, 0.0};
    turbo_accumulate_atom(
        center_type, center_type, 0.0, origin, true, basis_kind, l_max,
        rcut_hard, rcut_soft, nf, radial_enhancement,
        atom_sigma_r[center_type] / rcut_hard, atom_sigma_r,
        atom_sigma_r_scaling, atom_sigma_t, atom_sigma_t_scaling,
        amplitude_scaling, central_weights, alpha_max, channel_offsets,
        transforms, coefficients);
    if (basis_kind == 1 && central_weights[center_type] != 0.0) {
        int transform_offset = 0;
        int gaussian_offset = 0;
        for (int type = 0; type < center_type; ++type) {
            transform_offset += alpha_max[type] * alpha_max[type];
            gaussian_offset += alpha_max[type];
        }
        const int basis_size = alpha_max[center_type];
        const double sigma_r = atom_sigma_r[center_type];
        const double sigma_t = atom_sigma_t[center_type];
        const double enhancement = radial_enhancement == 1
            ? sqrt(2.0 / kPi) * sigma_r / rcut_hard
            : radial_enhancement == 2
                ? sigma_r * sigma_r / (rcut_hard * rcut_hard) : 1.0;
        const double prefactor = enhancement * central_weights[center_type]
            * sqrt(4.0 * kPi) * pow(kPi, 0.25) * sqrt(sigma_r / 2.0)
            * pow(rcut_hard, 3.0) / (sigma_t * sigma_t * sigma_r);
        const double* transform = transforms + transform_offset;
        const double* column = gaussian_columns + gaussian_offset;
        const int coefficient_offset = channel_offsets[center_type] * packed_count;
        for (int n = 0; n < basis_size; ++n) {
            double value = 0.0;
            for (int source = 0; source < basis_size; ++source) {
                value += transform[n * basis_size + source] * column[source];
            }
            coefficients[coefficient_offset + n * packed_count].real += prefactor * value;
        }
    }
    const I64 begin = graph_offsets[center];
    const I64 end = graph_offsets[center + 1];
    for (I64 edge = begin; edge < end; ++edge) {
        const I32 atom = graph_atoms[edge];
        if (atom == center && graph_shifts[edge * 3] == 0
            && graph_shifts[edge * 3 + 1] == 0 && graph_shifts[edge * 3 + 2] == 0) continue;
        const int atom_type = species_index(numbers[atom], species, species_count);
        if (atom_type < 0) continue;
        const double distance = sqrt(fmax(0.0, graph_distance2[edge]));
        if (distance >= rcut_hard) continue;
        turbo_accumulate_atom(
            center_type, atom_type, distance, graph_displacements + edge * 3, false,
            basis_kind, l_max, rcut_hard, rcut_soft, nf, radial_enhancement,
            atom_sigma_r[atom_type] / rcut_hard,
            atom_sigma_r, atom_sigma_r_scaling, atom_sigma_t, atom_sigma_t_scaling,
            amplitude_scaling, central_weights, alpha_max, channel_offsets,
            transforms, coefficients);
    }

    double* row = output + center * features;
    double* dense = compression_offsets != nullptr && dense_features > 0
        ? dense_workspace + center * dense_features : row;
    I64 dense_index = 0;
    for (I64 first = 0; first < channels; ++first) {
        for (I64 second = first; second < channels; ++second) {
            for (int l = 0; l <= l_max; ++l) {
                double value = 0.0;
                const int packed_offset = l * (l + 1) / 2;
                const double multiplicity_pair = first == second ? 1.0 : sqrt(2.0);
                for (int m = 0; m <= l; ++m) {
                    const DeviceComplex left = coefficients[first * packed_count + packed_offset + m];
                    const DeviceComplex right = coefficients[second * packed_count + packed_offset + m];
                    value += multiplicity_pair * (m > 0 ? 2.0 : 1.0)
                        * (left.real * right.real + left.imag * right.imag);
                }
                dense[dense_index++] = value;
            }
        }
    }
    if (compression_offsets != nullptr && dense_features > 0
        && features != dense_features) {
        for (I64 feature = 0; feature < features; ++feature) {
            double value = 0.0;
            for (I64 term = compression_offsets[feature]; term < compression_offsets[feature + 1]; ++term) {
                value += compression_factors[term] * dense[compression_sources[term]];
            }
            row[feature] = value;
        }
    } else if (dense != row) {
        for (I64 feature = 0; feature < features; ++feature) row[feature] = dense[feature];
    }
    double norm = 0.0;
    for (I64 feature = 0; feature < features; ++feature) norm += row[feature] * row[feature];
    norm = sqrt(norm);
    if (norm < 1e-5) norm = 1.0;
    for (I64 feature = 0; feature < features; ++feature) row[feature] /= norm;
}

py::dict compute_soap_turbo_descriptor(
    CudaExecutionContext& context,
    DeviceBatch& batch,
    DeviceNeighborGraph& graph,
    const detail::StructureBatchView& host_batch,
    const py::dict& options) {
    const py::str payload_key("_cuda_payload");
    if (!options.contains(payload_key) || options[payload_key].is_none()) {
        throw std::invalid_argument("SOAPTurbo CUDA backend requires its prepared basis payload");
    }
    const py::dict payload = py::cast<py::dict>(options[payload_key]);
    const auto species = species_option(options);
    const auto alpha_max = py::cast<std::vector<I32>>(payload["alpha_max"]);
    const auto channel_offsets = py::cast<std::vector<I32>>(payload["channel_offsets"]);
    const auto central_allowed = py::cast<std::vector<I32>>(payload["central_allowed"]);
    const auto transforms = vector_child(payload, "basis_transforms");
    const auto gaussian_columns = vector_child(payload, "gaussian_columns");
    const auto compression_offsets = py::cast<std::vector<I64>>(payload["compression_offsets"]);
    const auto compression_sources = py::cast<std::vector<I64>>(payload["compression_sources"]);
    const auto compression_factors = vector_child(payload, "compression_factors");
    const std::string basis_name = option(options, "basis", std::string("poly3"));
    const int basis_kind = basis_name == "poly3gauss" ? 1 : 0;
    const int l_max = option(options, "l_max", 6);
    const double rcut_hard = option(options, "rcut_hard", 5.0);
    const double rcut_soft = option(options, "rcut_soft", rcut_hard);
    const double nf = option(options, "nf", 1.0);
    const int radial_enhancement = option(options, "radial_enhancement", 0);
    const auto atom_sigma_r = numeric_vector_option(options, "atom_sigma_r", 0.5, species.size());
    const auto atom_sigma_r_scaling = numeric_vector_option(options, "atom_sigma_r_scaling", 0.0, species.size());
    const auto atom_sigma_t = numeric_vector_option(options, "atom_sigma_t", 0.5, species.size());
    const auto atom_sigma_t_scaling = numeric_vector_option(options, "atom_sigma_t_scaling", 0.0, species.size());
    const auto amplitude_scaling = numeric_vector_option(options, "amplitude_scaling", 0.0, species.size());
    const auto central_weights = numeric_vector_option(options, "central_weight", 1.0, species.size());
    const I64 features = feature_count_option(options, 0);
    const I64 channels = channel_offsets.empty() ? 0 : channel_offsets.back();
    const I64 packed_count = payload.contains("packed_count")
        ? py::cast<I64>(payload["packed_count"])
        : 1 + l_max * (l_max + 1) / 2 + l_max;
    const I64 dense_features = payload.contains("dense_feature_count")
        ? py::cast<I64>(payload["dense_feature_count"])
        : channels * (channels + 1) / 2 * (l_max + 1);
    if (species.empty() || alpha_max.size() != species.size()
        || channel_offsets.size() != species.size() + 1
        || central_allowed.size() != species.size() || features <= 0
        || channels <= 0 || channels > 256 || l_max < 0 || l_max > 20
        || rcut_hard <= 0.0 || rcut_soft <= 0.0 || rcut_soft > rcut_hard
        || nf <= 0.0 || (basis_kind != 0 && basis_kind != 1)
        || transforms.empty() || (basis_kind == 1 && gaussian_columns.empty())
        || compression_offsets.empty()
        || compression_sources.size() != compression_factors.size()) {
        throw std::invalid_argument("invalid SOAPTurbo CUDA basis payload");
    }
    for (const I32 count : alpha_max) {
        if (count <= 0 || count > (basis_kind == 0 ? 10 : 11)) {
            throw std::invalid_argument("SOAPTurbo alpha_max exceeds the supported CUDA range");
        }
    }
    const bool identity = compression_sources.empty() && features == dense_features;
    if (!identity && (compression_offsets.size() != static_cast<std::size_t>(features + 1)
        || compression_offsets.back() != static_cast<I64>(compression_sources.size()))) {
        throw std::invalid_argument("invalid SOAPTurbo CUDA compression map");
    }
    graph.build_dpa(context, batch, host_batch, rcut_hard, true, false, false);

    DeviceBuffer<I32> d_species;
    DeviceBuffer<I32> d_alpha_max;
    DeviceBuffer<I32> d_channel_offsets;
    DeviceBuffer<I32> d_central_allowed;
    DeviceBuffer<double> d_atom_sigma_r;
    DeviceBuffer<double> d_atom_sigma_r_scaling;
    DeviceBuffer<double> d_atom_sigma_t;
    DeviceBuffer<double> d_atom_sigma_t_scaling;
    DeviceBuffer<double> d_amplitude_scaling;
    DeviceBuffer<double> d_central_weights;
    DeviceBuffer<double> d_transforms;
    DeviceBuffer<double> d_gaussian_columns;
    DeviceBuffer<I64> d_compression_offsets;
    DeviceBuffer<I64> d_compression_sources;
    DeviceBuffer<double> d_compression_factors;
    d_species.upload(species.data(), species.size(), context.stream(), "could not upload SOAPTurbo species");
    d_alpha_max.upload(alpha_max.data(), alpha_max.size(), context.stream(), "could not upload SOAPTurbo alpha_max");
    d_channel_offsets.upload(channel_offsets.data(), channel_offsets.size(), context.stream(), "could not upload SOAPTurbo channel offsets");
    d_central_allowed.upload(central_allowed.data(), central_allowed.size(), context.stream(), "could not upload SOAPTurbo central mask");
    d_atom_sigma_r.upload(atom_sigma_r.data(), atom_sigma_r.size(), context.stream(), "could not upload SOAPTurbo radial widths");
    d_atom_sigma_r_scaling.upload(atom_sigma_r_scaling.data(), atom_sigma_r_scaling.size(), context.stream(), "could not upload SOAPTurbo radial scaling");
    d_atom_sigma_t.upload(atom_sigma_t.data(), atom_sigma_t.size(), context.stream(), "could not upload SOAPTurbo angular widths");
    d_atom_sigma_t_scaling.upload(atom_sigma_t_scaling.data(), atom_sigma_t_scaling.size(), context.stream(), "could not upload SOAPTurbo angular scaling");
    d_amplitude_scaling.upload(amplitude_scaling.data(), amplitude_scaling.size(), context.stream(), "could not upload SOAPTurbo amplitudes");
    d_central_weights.upload(central_weights.data(), central_weights.size(), context.stream(), "could not upload SOAPTurbo center weights");
    d_transforms.upload(transforms.data(), transforms.size(), context.stream(), "could not upload SOAPTurbo radial transforms");
    d_gaussian_columns.upload(gaussian_columns.data(), gaussian_columns.size(), context.stream(), "could not upload SOAPTurbo Gaussian column");
    d_compression_offsets.upload(compression_offsets.data(), compression_offsets.size(), context.stream(), "could not upload SOAPTurbo compression offsets");
    d_compression_sources.upload(compression_sources.data(), compression_sources.size(), context.stream(), "could not upload SOAPTurbo compression sources");
    d_compression_factors.upload(compression_factors.data(), compression_factors.size(), context.stream(), "could not upload SOAPTurbo compression factors");
    const std::size_t size = static_cast<std::size_t>(batch.atoms()) * static_cast<std::size_t>(features);
    double* output = context.output_buffer(size);
    const I64 coefficient_stride = channels * packed_count * 2;
    const std::size_t coefficient_values = static_cast<std::size_t>(batch.atoms())
        * static_cast<std::size_t>(coefficient_stride);
    const std::size_t dense_values = identity ? 0U
        : static_cast<std::size_t>(batch.atoms()) * static_cast<std::size_t>(dense_features);
    if (coefficient_values > std::numeric_limits<std::size_t>::max() - dense_values
        || coefficient_values + dense_values > std::numeric_limits<std::size_t>::max() / sizeof(double)) {
        throw CudaOutOfMemory("SOAPTurbo CUDA workspace is too large");
    }
    auto* workspace = static_cast<unsigned char*>(context.workspace_buffer(
        (coefficient_values + dense_values) * sizeof(double)));
    auto* coefficient_workspace = reinterpret_cast<double*>(workspace);
    auto* dense_workspace = identity ? nullptr
        : reinterpret_cast<double*>(workspace + coefficient_values * sizeof(double));
    if (size > 0) {
        check_cuda(cudaMemsetAsync(output, 0, size * sizeof(double), context.stream()),
            "could not clear SOAPTurbo output");
        constexpr unsigned block_size = 64;
        soap_turbo_cuda_kernel<<<static_cast<unsigned>((batch.atoms() + block_size - 1) / block_size),
            block_size, 0, context.stream()>>>(
            batch.numbers(), graph.offsets(), graph.atoms(), graph.shifts(), graph.displacements(),
            graph.distance2(), d_species.get(), static_cast<int>(species.size()), d_alpha_max.get(),
            d_channel_offsets.get(), d_central_allowed.get(), d_atom_sigma_r.get(),
            d_atom_sigma_r_scaling.get(), d_atom_sigma_t.get(), d_atom_sigma_t_scaling.get(),
            d_amplitude_scaling.get(), d_central_weights.get(), d_transforms.get(),
            d_gaussian_columns.get(), basis_kind, l_max, rcut_hard, rcut_soft, nf,
            radial_enhancement, channels, packed_count, dense_features, features,
            identity ? nullptr : d_compression_offsets.get(),
            d_compression_sources.get(), d_compression_factors.get(), coefficient_stride,
            batch.atoms(), coefficient_workspace, dense_workspace, output);
        check_cuda(cudaGetLastError(), "SOAPTurbo CUDA kernel launch failed");
    }
    const auto values = context.download_output(size);
    return atom_result(values, batch.atoms(), features, "SOAPTurbo", options, false,
        std::vector<I64>(host_batch.offsets, host_batch.offsets + host_batch.structures + 1));
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

__global__ void mtp4_cuda_kernel(
    const I32* numbers,
    const I64* graph_offsets,
    const I32* graph_atoms,
    const I32* graph_shifts,
    const double* graph_displacements,
    const double* graph_distance2,
    const I32* species,
    int species_count,
    int radial_kind,
    int radial_basis_size,
    int radial_funcs_count,
    double min_dist,
    double max_dist,
    double max_dist_sq,
    double max_dist_sq_minus_eps,
    double exp_ratio,
    double radial_zeroth,
    const double* radial_recursive,
    const double* radial_vdw_params,
    const double* model_parameters,
    double radial_scaling,
    const I32* moments,
    I64 moment_count,
    const I32* eval_kinds,
    const I32* eval_linear_ids,
    const double* eval_linear_coefficients,
    I64 eval_count,
    const I64* eval_product_offsets,
    const I32* eval_product_left,
    const I32* eval_product_right,
    const double* eval_product_coefficients,
    const I32* scalar_output_ids,
    I64 features,
    I64 atoms,
    double* eval_workspace,
    double* output) {
    const I64 center = static_cast<I64>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (center >= atoms || eval_count <= 0 || eval_count > 65536) return;
    const int center_type = species_index(numbers[center], species, species_count);
    if (center_type < 0) return;
    double* raw = eval_workspace + center * eval_count;
    for (I64 index = 0; index < eval_count; ++index) raw[index] = 0.0;
    const I64 radial_count = static_cast<I64>(species_count) * species_count
        * radial_funcs_count * radial_basis_size;
    double radial_basis[64]{};
    double radial_values[32]{};
    const I64 begin = graph_offsets[center];
    const I64 end = graph_offsets[center + 1];
    for (I64 edge = begin; edge < end; ++edge) {
        const I32 atom = graph_atoms[edge];
        if (atom == center && graph_shifts[edge * 3] == 0
            && graph_shifts[edge * 3 + 1] == 0 && graph_shifts[edge * 3 + 2] == 0) continue;
        const double r_sq = fmax(0.0, graph_distance2[edge]);
        const double distance = sqrt(r_sq);
        if (distance <= 0.0 || distance > max_dist) continue;
        const int neighbor_type = species_index(numbers[atom], species, species_count);
        if (neighbor_type < 0) continue;
        mtp4_radial_basis_device(
            r_sq, radial_kind, radial_basis_size, min_dist, max_dist, max_dist_sq,
            max_dist_sq_minus_eps, exp_ratio, radial_zeroth, radial_recursive,
            radial_vdw_params, center_type, neighbor_type, radial_basis);
        const I64 pair_offset = (static_cast<I64>(center_type) * species_count + neighbor_type)
            * radial_funcs_count * radial_basis_size;
        for (int function = 0; function < radial_funcs_count; ++function) {
            double value = 0.0;
            for (int radial = 0; radial < radial_basis_size; ++radial) {
                value += model_parameters[pair_offset + function * radial_basis_size + radial]
                    * radial_basis[radial];
            }
            radial_values[function] = value * radial_scaling;
        }
        const double distance_power_limit = distance;
        const double* displacement = graph_displacements + edge * 3;
        for (I64 id = 0; id < moment_count; ++id) {
            const I32* moment = moments + id * 4;
            if (moment[3] > 1 || moment[0] < 0 || moment[0] >= radial_funcs_count) continue;
            double coordinate_product = 1.0;
            double coordinate_x = 1.0;
            double coordinate_y = 1.0;
            double coordinate_z = 1.0;
            double distance_power = 1.0;
            const int degree = moment[1] + moment[2] + moment[3];
            for (int power = 0; power < moment[1]; ++power) coordinate_x *= displacement[0];
            for (int power = 0; power < moment[2]; ++power) coordinate_y *= displacement[1];
            for (int power = 0; power < moment[3]; ++power) coordinate_z *= displacement[2];
            for (int power = 0; power < degree; ++power) distance_power *= distance_power_limit;
            coordinate_product = coordinate_x * coordinate_y * coordinate_z;
            raw[id] += radial_values[moment[0]] * coordinate_product / distance_power;
        }
    }
    for (I64 id = 0; id < eval_count; ++id) {
        const int kind = eval_kinds[id];
        if (kind == 0) continue;
        if (kind == 1) {
            double value = 0.0;
            for (int term = 0; term < 3; ++term) {
                const int dependency = eval_linear_ids[id * 3 + term];
                if (dependency >= 0) value += eval_linear_coefficients[id * 3 + term] * raw[dependency];
            }
            raw[id] = value;
        } else {
            double value = 0.0;
            for (I64 term = eval_product_offsets[id]; term < eval_product_offsets[id + 1]; ++term) {
                value += eval_product_coefficients[term]
                    * raw[eval_product_left[term]] * raw[eval_product_right[term]];
            }
            raw[id] = value;
        }
    }
    double* target = output + center * features;
    for (I64 feature = 0; feature < features; ++feature) {
        target[feature] = raw[scalar_output_ids[feature]];
    }
    (void)radial_count;
}

py::dict compute_mtp4_descriptor(
    CudaExecutionContext& context,
    DeviceBatch& batch,
    DeviceNeighborGraph& graph,
    const detail::StructureBatchView& host_batch,
    const py::dict& options) {
    const py::str payload_key("_cuda_payload");
    if (!options.contains(payload_key) || options[payload_key].is_none()) {
        throw std::invalid_argument("MLIP-4 MTP CUDA backend requires its evaluator payload");
    }
    const py::dict payload = py::cast<py::dict>(options[payload_key]);
    const auto species = species_option(options);
    const auto model_species = py::cast<std::vector<I32>>(payload["model_species"]);
    const auto model_parameters = vector_child(payload, "model_parameters");
    const auto radial_recursive = vector_child(payload, "radial_recursive");
    const auto radial_vdw_params = vector_child(payload, "radial_vdw_params");
    const auto moments = py::cast<std::vector<I32>>(payload["moments"]);
    const auto eval_kinds = py::cast<std::vector<I32>>(payload["eval_kinds"]);
    const auto eval_linear_ids = py::cast<std::vector<I32>>(payload["eval_linear_ids"]);
    const auto eval_linear_coefficients = vector_child(payload, "eval_linear_coefficients");
    const auto eval_product_offsets = py::cast<std::vector<I64>>(payload["eval_product_offsets"]);
    const auto eval_product_left = py::cast<std::vector<I32>>(payload["eval_product_left"]);
    const auto eval_product_right = py::cast<std::vector<I32>>(payload["eval_product_right"]);
    const auto eval_product_coefficients = vector_child(payload, "eval_product_coefficients");
    const auto scalar_output_ids = py::cast<std::vector<I32>>(payload["scalar_output_ids"]);
    const I64 features = feature_count_option(options, 0);
    const int radial_kind = py::cast<int>(payload["radial_kind"]);
    const int radial_basis_size = py::cast<int>(payload["radial_basis_size"]);
    const int radial_funcs_count = py::cast<int>(payload["radial_funcs_count"]);
    const double min_dist = py::cast<double>(payload["radial_min_dist"]);
    const double max_dist = py::cast<double>(payload["radial_max_dist"]);
    const double radial_scaling = py::cast<double>(payload["radial_scaling"]);
    const double radial_zeroth = py::cast<double>(payload["radial_zeroth"]);
    const double radial_exp_ratio = py::cast<double>(payload["radial_exp_ratio"]);
    const double radial_maxdist_sq = py::cast<double>(payload["radial_maxdist_sq"]);
    const double radial_maxdist_sq_minus_eps = py::cast<double>(payload["radial_maxdist_sq_minus_eps"]);
    const I64 moment_count = static_cast<I64>(moments.size() / 4U);
    const I64 eval_count = static_cast<I64>(eval_kinds.size());
    const I64 radial_parameter_count = static_cast<I64>(species.size()) * species.size()
        * radial_funcs_count * radial_basis_size;
    if (model_species.size() != species.size() || species.empty()
        || radial_kind < 0 || radial_kind > 3 || radial_basis_size <= 0
        || radial_basis_size > 64 || radial_funcs_count <= 0 || radial_funcs_count > 32
        || !std::isfinite(min_dist) || !std::isfinite(max_dist)
        || min_dist < 0.0 || max_dist <= min_dist
        || model_parameters.size() < static_cast<std::size_t>(radial_parameter_count)
        || moments.size() % 4U != 0 || moment_count <= 0 || eval_count <= 0
        || eval_linear_ids.size() != static_cast<std::size_t>(eval_count * 3)
        || eval_linear_coefficients.size() != static_cast<std::size_t>(eval_count * 3)
        || eval_product_offsets.size() != static_cast<std::size_t>(eval_count + 1)
        || eval_product_left.size() != eval_product_right.size()
        || eval_product_left.size() != eval_product_coefficients.size()
        || eval_product_offsets.back() != static_cast<I64>(eval_product_left.size())
        || scalar_output_ids.size() != static_cast<std::size_t>(features)) {
        throw std::invalid_argument("invalid MLIP-4 MTP CUDA evaluator payload");
    }
    if (radial_kind == 0 && radial_recursive.size() < static_cast<std::size_t>(3 * (radial_basis_size - 1))) {
        throw std::invalid_argument("MLIP-4 Cinf radial evaluator payload is incomplete");
    }
    if (radial_kind == 3 && radial_vdw_params.size() < species.size() + 2) {
        throw std::invalid_argument("MLIP-4 damped radial evaluator payload is incomplete");
    }
    for (const I32 id : scalar_output_ids) {
        if (id < 0 || id >= eval_count) throw std::invalid_argument("MLIP-4 scalar output id is out of range");
    }
    graph.build_dpa(context, batch, host_batch, max_dist, true, false, false);

    DeviceBuffer<I32> d_species;
    DeviceBuffer<double> d_parameters;
    DeviceBuffer<double> d_recursive;
    DeviceBuffer<double> d_vdw_params;
    DeviceBuffer<I32> d_moments;
    DeviceBuffer<I32> d_eval_kinds;
    DeviceBuffer<I32> d_eval_linear_ids;
    DeviceBuffer<double> d_eval_linear_coefficients;
    DeviceBuffer<I64> d_eval_product_offsets;
    DeviceBuffer<I32> d_eval_product_left;
    DeviceBuffer<I32> d_eval_product_right;
    DeviceBuffer<double> d_eval_product_coefficients;
    DeviceBuffer<I32> d_scalar_output_ids;
    d_species.upload(species.data(), species.size(), context.stream(), "could not upload MLIP-4 species");
    d_parameters.upload(model_parameters.data(), model_parameters.size(), context.stream(), "could not upload MLIP-4 radial parameters");
    d_recursive.upload(radial_recursive.data(), radial_recursive.size(), context.stream(), "could not upload MLIP-4 radial recurrence");
    d_vdw_params.upload(radial_vdw_params.data(), radial_vdw_params.size(), context.stream(), "could not upload MLIP-4 damped radial parameters");
    d_moments.upload(moments.data(), moments.size(), context.stream(), "could not upload MLIP-4 moments");
    d_eval_kinds.upload(eval_kinds.data(), eval_kinds.size(), context.stream(), "could not upload MLIP-4 evaluator kinds");
    d_eval_linear_ids.upload(eval_linear_ids.data(), eval_linear_ids.size(), context.stream(), "could not upload MLIP-4 linear evaluator ids");
    d_eval_linear_coefficients.upload(eval_linear_coefficients.data(), eval_linear_coefficients.size(), context.stream(), "could not upload MLIP-4 linear evaluator coefficients");
    d_eval_product_offsets.upload(eval_product_offsets.data(), eval_product_offsets.size(), context.stream(), "could not upload MLIP-4 product evaluator offsets");
    d_eval_product_left.upload(eval_product_left.data(), eval_product_left.size(), context.stream(), "could not upload MLIP-4 product evaluator left ids");
    d_eval_product_right.upload(eval_product_right.data(), eval_product_right.size(), context.stream(), "could not upload MLIP-4 product evaluator right ids");
    d_eval_product_coefficients.upload(eval_product_coefficients.data(), eval_product_coefficients.size(), context.stream(), "could not upload MLIP-4 product evaluator coefficients");
    d_scalar_output_ids.upload(scalar_output_ids.data(), scalar_output_ids.size(), context.stream(), "could not upload MLIP-4 scalar output ids");
    const std::size_t size = static_cast<std::size_t>(batch.atoms()) * static_cast<std::size_t>(features);
    double* output = context.output_buffer(size);
    auto* workspace = static_cast<double*>(context.workspace_buffer(
        static_cast<std::size_t>(batch.atoms()) * static_cast<std::size_t>(eval_count) * sizeof(double)));
    if (size > 0) {
        check_cuda(cudaMemsetAsync(output, 0, size * sizeof(double), context.stream()),
            "could not clear MLIP-4 MTP output");
        constexpr unsigned block_size = 64;
        mtp4_cuda_kernel<<<static_cast<unsigned>((batch.atoms() + block_size - 1) / block_size),
            block_size, 0, context.stream()>>>(
            batch.numbers(), graph.offsets(), graph.atoms(), graph.shifts(), graph.displacements(), graph.distance2(),
            d_species.get(), static_cast<int>(species.size()), radial_kind, radial_basis_size,
            radial_funcs_count, min_dist, max_dist, radial_maxdist_sq,
            radial_maxdist_sq_minus_eps, radial_exp_ratio, radial_zeroth, d_recursive.get(),
            d_vdw_params.get(), d_parameters.get(), radial_scaling, d_moments.get(), moment_count,
            d_eval_kinds.get(), d_eval_linear_ids.get(), d_eval_linear_coefficients.get(), eval_count,
            d_eval_product_offsets.get(), d_eval_product_left.get(), d_eval_product_right.get(),
            d_eval_product_coefficients.get(), d_scalar_output_ids.get(), features, batch.atoms(),
            workspace, output);
        check_cuda(cudaGetLastError(), "MLIP-4 MTP CUDA kernel launch failed");
    }
    const auto values = context.download_output(size);
    return atom_result(values, batch.atoms(), features, "MTP", options, false,
        std::vector<I64>(host_batch.offsets, host_batch.offsets + host_batch.structures + 1));
}

__device__ void mtp2_radial_basis_device(
    double distance,
    double min_dist,
    double max_dist,
    int basis_size,
    double scaling,
    bool repulsive,
    double* output) {
    const double radius = repulsive ? fmax(distance, min_dist) : distance;
    const double ksi = (2.0 * radius - (min_dist + max_dist)) / (max_dist - min_dist);
    const double edge = radius - max_dist;
    output[0] = edge * edge;
    if (basis_size > 1) output[1] = ksi * edge * edge;
    for (int index = 2; index < basis_size; ++index) {
        output[index] = 2.0 * ksi * output[index - 1] - output[index - 2];
    }
    for (int index = 0; index < basis_size; ++index) output[index] *= scaling;
}

__global__ void mtp2_cuda_kernel(
    const I32* numbers,
    const I64* graph_offsets,
    const I32* graph_atoms,
    const I32* graph_shifts,
    const double* graph_displacements,
    const double* graph_distance2,
    const I32* species,
    int species_count,
    int radial_basis_size,
    int radial_funcs_count,
    int alpha_moments_count,
    double min_dist,
    double max_dist,
    double scaling,
    bool repulsive,
    const double* radial_coefficients,
    const I32* alpha_basic,
    I64 alpha_basic_count,
    const I32* alpha_times,
    I64 alpha_times_count,
    const I32* moment_mapping,
    I64 features,
    I64 atoms,
    double* workspace,
    double* output) {
    const I64 center = static_cast<I64>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (center >= atoms || alpha_moments_count <= 0 || alpha_moments_count > 65536) return;
    const int center_type = species_index(numbers[center], species, species_count);
    if (center_type < 0) return;
    double* moments = workspace + center * alpha_moments_count;
    for (int index = 0; index < alpha_moments_count; ++index) moments[index] = 0.0;
    double rb[128]{};
    double radial_values[64]{};
    const I64 begin = graph_offsets[center];
    const I64 end = graph_offsets[center + 1];
    const I64 pair_stride = static_cast<I64>(radial_funcs_count) * radial_basis_size;
    for (I64 edge = begin; edge < end; ++edge) {
        const I32 atom = graph_atoms[edge];
        if (atom == center && graph_shifts[edge * 3] == 0
            && graph_shifts[edge * 3 + 1] == 0 && graph_shifts[edge * 3 + 2] == 0) continue;
        const double distance = sqrt(fmax(0.0, graph_distance2[edge]));
        if (distance <= 0.0 || distance > max_dist) continue;
        const int neighbor_type = species_index(numbers[atom], species, species_count);
        if (neighbor_type < 0) continue;
        mtp2_radial_basis_device(
            distance, min_dist, max_dist, radial_basis_size, scaling, repulsive, rb);
        const I64 pair_offset = (static_cast<I64>(center_type) * species_count + neighbor_type)
            * pair_stride;
        for (int function = 0; function < radial_funcs_count; ++function) {
            double value = 0.0;
            for (int radial = 0; radial < radial_basis_size; ++radial) {
                value += radial_coefficients[pair_offset + function * radial_basis_size + radial]
                    * rb[radial];
            }
            radial_values[function] = value;
        }
        const double* displacement = graph_displacements + edge * 3;
        for (I64 index = 0; index < alpha_basic_count; ++index) {
            const I32* alpha = alpha_basic + index * 4;
            if (alpha[0] < 0 || alpha[0] >= radial_funcs_count) continue;
            const int degree = alpha[1] + alpha[2] + alpha[3];
            double x_power = 1.0;
            double y_power = 1.0;
            double z_power = 1.0;
            double distance_power = 1.0;
            for (int power = 0; power < alpha[1]; ++power) x_power *= displacement[0];
            for (int power = 0; power < alpha[2]; ++power) y_power *= displacement[1];
            for (int power = 0; power < alpha[3]; ++power) z_power *= displacement[2];
            for (int power = 0; power < degree; ++power) distance_power *= distance;
            moments[index] += radial_values[alpha[0]] * x_power * y_power * z_power / distance_power;
        }
    }
    for (I64 index = 0; index < alpha_times_count; ++index) {
        const I32* alpha = alpha_times + index * 4;
        if (alpha[0] >= 0 && alpha[0] < alpha_moments_count
            && alpha[1] >= 0 && alpha[1] < alpha_moments_count
            && alpha[3] >= 0 && alpha[3] < alpha_moments_count) {
            moments[alpha[3]] += static_cast<double>(alpha[2])
                * moments[alpha[0]] * moments[alpha[1]];
        }
    }
    double* target = output + center * features;
    target[0] = 1.0;
    for (I64 index = 0; index < features - 1; ++index) {
        const int mapped = moment_mapping[index];
        if (mapped >= 0 && mapped < alpha_moments_count) target[index + 1] = moments[mapped];
    }
}

py::dict compute_mtp2_descriptor(
    CudaExecutionContext& context,
    DeviceBatch& batch,
    DeviceNeighborGraph& graph,
    const detail::StructureBatchView& host_batch,
    const py::dict& options) {
    const py::str payload_key("_cuda_payload");
    const py::dict payload = py::cast<py::dict>(options[payload_key]);
    const auto species = species_option(options);
    const auto radial_coefficients = vector_child(payload, "radial_coefficients");
    const auto alpha_basic = py::cast<std::vector<I32>>(payload["alpha_index_basic"]);
    const auto alpha_times = py::cast<std::vector<I32>>(payload["alpha_index_times"]);
    const auto moment_mapping = py::cast<std::vector<I32>>(payload["alpha_moment_mapping"]);
    const int species_count = py::cast<int>(payload["species_count"]);
    const int radial_basis_size = py::cast<int>(payload["radial_basis_size"]);
    const int radial_funcs_count = py::cast<int>(payload["radial_funcs_count"]);
    const int alpha_moments_count = py::cast<int>(payload["alpha_moments_count"]);
    const double min_dist = py::cast<double>(payload["radial_min_dist"]);
    const double max_dist = py::cast<double>(payload["radial_max_dist"]);
    const double scaling = py::cast<double>(payload["scaling"]);
    const std::string basis_type = option(payload, "radial_basis_type", std::string("RBChebyshev"));
    const bool repulsive = basis_type == "RBChebyshev_repuls";
    const I64 features = feature_count_option(options, 0);
    if (species.empty() || species_count != static_cast<int>(species.size())
        || radial_basis_size <= 0 || radial_basis_size > 128 || radial_funcs_count <= 0
        || radial_funcs_count > 64 || alpha_moments_count <= 0
        || alpha_basic.size() % 4U != 0 || alpha_times.size() % 4U != 0
        || moment_mapping.size() + 1 != static_cast<std::size_t>(features)
        || radial_coefficients.size() < static_cast<std::size_t>(species_count * species_count
            * radial_funcs_count * radial_basis_size)
        || min_dist < 0.0 || max_dist <= min_dist || features <= 0) {
        throw std::invalid_argument("invalid MLIP-2 MTP CUDA evaluator payload");
    }
    graph.build_dpa(context, batch, host_batch, max_dist, true, false, false);
    DeviceBuffer<I32> d_species;
    DeviceBuffer<double> d_radial_coefficients;
    DeviceBuffer<I32> d_alpha_basic;
    DeviceBuffer<I32> d_alpha_times;
    DeviceBuffer<I32> d_moment_mapping;
    d_species.upload(species.data(), species.size(), context.stream(), "could not upload MLIP-2 species");
    d_radial_coefficients.upload(radial_coefficients.data(), radial_coefficients.size(), context.stream(), "could not upload MLIP-2 radial coefficients");
    d_alpha_basic.upload(alpha_basic.data(), alpha_basic.size(), context.stream(), "could not upload MLIP-2 basic indices");
    d_alpha_times.upload(alpha_times.data(), alpha_times.size(), context.stream(), "could not upload MLIP-2 product indices");
    d_moment_mapping.upload(moment_mapping.data(), moment_mapping.size(), context.stream(), "could not upload MLIP-2 moment mapping");
    const std::size_t size = static_cast<std::size_t>(batch.atoms()) * static_cast<std::size_t>(features);
    double* output = context.output_buffer(size);
    auto* workspace = static_cast<double*>(context.workspace_buffer(
        static_cast<std::size_t>(batch.atoms()) * static_cast<std::size_t>(alpha_moments_count)
        * sizeof(double)));
    if (size > 0) {
        check_cuda(cudaMemsetAsync(output, 0, size * sizeof(double), context.stream()),
            "could not clear MLIP-2 MTP output");
        constexpr unsigned block_size = 64;
        mtp2_cuda_kernel<<<static_cast<unsigned>((batch.atoms() + block_size - 1) / block_size),
            block_size, 0, context.stream()>>>(
            batch.numbers(), graph.offsets(), graph.atoms(), graph.shifts(), graph.displacements(),
            graph.distance2(), d_species.get(), species_count, radial_basis_size,
            radial_funcs_count, alpha_moments_count, min_dist, max_dist, scaling, repulsive,
            d_radial_coefficients.get(), d_alpha_basic.get(), static_cast<I64>(alpha_basic.size() / 4U),
            d_alpha_times.get(), static_cast<I64>(alpha_times.size() / 4U), d_moment_mapping.get(),
            features, batch.atoms(), workspace, output);
        check_cuda(cudaGetLastError(), "MLIP-2 MTP CUDA kernel launch failed");
    }
    const auto values = context.download_output(size);
    return atom_result(values, batch.atoms(), features, "MTP", options, false,
        std::vector<I64>(host_batch.offsets, host_batch.offsets + host_batch.structures + 1));
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

__global__ void ace_cuda_kernel(
    const I32* numbers,
    const I64* graph_offsets,
    const I32* graph_atoms,
    const I32* graph_shifts,
    const double* graph_displacements,
    const double* graph_distance2,
    const I32* species,
    int species_count,
    const I32* base_species,
    const I32* base_radial,
    const I32* base_angular,
    const I32* base_magnetic,
    I64 base_channels,
    int max_radial,
    int max_angular,
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
    const I64* center_feature_offsets,
    const I64* feature_term_offsets,
    const I64* term_channel_offsets,
    const I32* term_channels,
    const double* term_coefficients,
    I64 features,
    I64 atoms,
    double* coefficient_workspace,
    double* output) {
    const I64 center = static_cast<I64>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (center >= atoms || base_channels > 2048) return;
    const int center_type = species_index(numbers[center], species, species_count);
    if (center_type < 0) return;
    DeviceComplex* values = reinterpret_cast<DeviceComplex*>(
        coefficient_workspace + center * base_channels * 2);
    for (I64 channel = 0; channel < base_channels; ++channel) values[channel] = {0.0, 0.0};
    const I64 begin = graph_offsets[center];
    const I64 end = graph_offsets[center + 1];
    double radial[256]{};
    DeviceComplex harmonics[441]{};
    for (I64 edge = begin; edge < end; ++edge) {
        const I32 atom = graph_atoms[edge];
        if (atom == center && graph_shifts[edge * 3] == 0
            && graph_shifts[edge * 3 + 1] == 0 && graph_shifts[edge * 3 + 2] == 0) continue;
        const double distance = sqrt(fmax(0.0, graph_distance2[edge]));
        if (distance <= 0.0) continue;
        const int atom_type = species_index(numbers[atom], species, species_count);
        if (atom_type < 0) continue;
        ace_radial_values(
            distance, transform_a, transform_p, transform_r0, t_left, t_right,
            p_left, p_right, radial_a, radial_b, radial_c, max_radial, radial);
        ace_spherical_harmonics(
            graph_displacements + edge * 3, max_angular, harmonics);
        for (I64 channel = 0; channel < base_channels; ++channel) {
            if (base_species[channel] != atom_type) continue;
            const int radial_index = base_radial[channel] - 1;
            const int angular = base_angular[channel];
            if (radial_index < 0 || radial_index >= max_radial) continue;
            const DeviceComplex angular_value = harmonics[
                angular * angular + angular + base_magnetic[channel]];
            values[channel] = complex_add(values[channel], complex_scale(
                angular_value, radial[radial_index]));
        }
    }
    const I64 feature_begin = center_feature_offsets[center_type];
    const I64 feature_end = center_feature_offsets[center_type + 1];
    double* target = output + center * features;
    for (I64 feature = feature_begin; feature < feature_end; ++feature) {
        double value = 0.0;
        const I64 term_begin = feature_term_offsets[feature];
        const I64 term_end = feature_term_offsets[feature + 1];
        for (I64 term = term_begin; term < term_end; ++term) {
            DeviceComplex product = {term_coefficients[term], 0.0};
            for (I64 index = term_channel_offsets[term];
                 index < term_channel_offsets[term + 1]; ++index) {
                product = complex_multiply(product, values[term_channels[index]]);
            }
            value += product.real;
        }
        target[feature - feature_begin] = value;
    }
}

py::dict compute_ace_descriptor(
    CudaExecutionContext& context,
    DeviceBatch& batch,
    DeviceNeighborGraph& graph,
    const detail::StructureBatchView& host_batch,
    const py::dict& options) {
    const py::str payload_key("_cuda_payload");
    if (!options.contains(payload_key) || options[payload_key].is_none()) {
        throw std::invalid_argument("ACE CUDA backend requires its generated basis payload");
    }
    const py::dict payload = py::cast<py::dict>(options[payload_key]);
    const auto species = species_option(options);
    const auto base_species = py::cast<std::vector<I32>>(payload["base_species"]);
    const auto base_radial = py::cast<std::vector<I32>>(payload["base_radial"]);
    const auto base_angular = py::cast<std::vector<I32>>(payload["base_angular"]);
    const auto base_magnetic = py::cast<std::vector<I32>>(payload["base_magnetic"]);
    const auto radial_a = vector_child(payload, "radial_a");
    const auto radial_b = vector_child(payload, "radial_b");
    const auto radial_c = vector_child(payload, "radial_c");
    const auto center_feature_offsets = py::cast<std::vector<I64>>(payload["center_feature_offsets"]);
    const auto feature_term_offsets = py::cast<std::vector<I64>>(payload["feature_term_offsets"]);
    const auto term_channel_offsets = py::cast<std::vector<I64>>(payload["term_channel_offsets"]);
    const auto term_channels = py::cast<std::vector<I32>>(payload["term_channels"]);
    const auto term_coefficients = vector_child(payload, "term_coefficients");
    const int max_radial = payload.contains("max_radial")
        ? py::cast<int>(payload["max_radial"]) : static_cast<int>(radial_a.size());
    const int max_angular = payload.contains("max_angular")
        ? py::cast<int>(payload["max_angular"]) : option(options, "N", 3);
    const py::dict transform = child_dict(options, "trans");
    const double transform_a = option(transform, "a", 1.0);
    const double transform_p = option(transform, "p", 2.0);
    const double transform_r0 = option(options, "r0", 2.5);
    const double cutoff = option(options, "rcut", 5.0);
    const I64 features = feature_count_option(options, 0);
    if (species.empty() || base_species.size() != base_radial.size()
        || base_species.size() != base_angular.size()
        || base_species.size() != base_magnetic.size() || base_species.empty()
        || base_species.size() > 2048 || radial_a.empty()
        || radial_a.size() != radial_b.size() || radial_a.size() != radial_c.size()
        || center_feature_offsets.size() != species.size() + 1
        || feature_term_offsets.size() != static_cast<std::size_t>(center_feature_offsets.back() + 1)
        || term_channel_offsets.empty()
        || term_channel_offsets.back() != static_cast<I64>(term_channels.size())
        || term_channel_offsets.size() != term_coefficients.size() + 1
        || features <= 0 || max_radial <= 0 || max_radial > 256
        || max_angular < 0 || max_angular > 20 || cutoff <= 0.0
        || transform_a + transform_r0 <= 0.0 || transform_p == 0.0) {
        throw std::invalid_argument("invalid ACE CUDA basis payload");
    }
    graph.build_dpa(context, batch, host_batch, cutoff, true, false, false);

    DeviceBuffer<I32> d_species;
    DeviceBuffer<I32> d_base_species;
    DeviceBuffer<I32> d_base_radial;
    DeviceBuffer<I32> d_base_angular;
    DeviceBuffer<I32> d_base_magnetic;
    DeviceBuffer<double> d_radial_a;
    DeviceBuffer<double> d_radial_b;
    DeviceBuffer<double> d_radial_c;
    DeviceBuffer<I64> d_center_feature_offsets;
    DeviceBuffer<I64> d_feature_term_offsets;
    DeviceBuffer<I64> d_term_channel_offsets;
    DeviceBuffer<I32> d_term_channels;
    DeviceBuffer<double> d_term_coefficients;
    d_species.upload(species.data(), species.size(), context.stream(), "could not upload ACE species");
    d_base_species.upload(base_species.data(), base_species.size(), context.stream(), "could not upload ACE base species");
    d_base_radial.upload(base_radial.data(), base_radial.size(), context.stream(), "could not upload ACE base radial indices");
    d_base_angular.upload(base_angular.data(), base_angular.size(), context.stream(), "could not upload ACE base angular indices");
    d_base_magnetic.upload(base_magnetic.data(), base_magnetic.size(), context.stream(), "could not upload ACE magnetic indices");
    d_radial_a.upload(radial_a.data(), radial_a.size(), context.stream(), "could not upload ACE radial recurrence");
    d_radial_b.upload(radial_b.data(), radial_b.size(), context.stream(), "could not upload ACE radial recurrence offset");
    d_radial_c.upload(radial_c.data(), radial_c.size(), context.stream(), "could not upload ACE radial recurrence second offset");
    d_center_feature_offsets.upload(center_feature_offsets.data(), center_feature_offsets.size(), context.stream(), "could not upload ACE center feature offsets");
    d_feature_term_offsets.upload(feature_term_offsets.data(), feature_term_offsets.size(), context.stream(), "could not upload ACE feature term offsets");
    d_term_channel_offsets.upload(term_channel_offsets.data(), term_channel_offsets.size(), context.stream(), "could not upload ACE term channel offsets");
    d_term_channels.upload(term_channels.data(), term_channels.size(), context.stream(), "could not upload ACE term channels");
    d_term_coefficients.upload(term_coefficients.data(), term_coefficients.size(), context.stream(), "could not upload ACE term coefficients");
    const std::size_t size = static_cast<std::size_t>(batch.atoms()) * static_cast<std::size_t>(features);
    double* output = context.output_buffer(size);
    auto* coefficient_workspace = static_cast<double*>(context.workspace_buffer(
        static_cast<std::size_t>(batch.atoms()) * base_species.size() * 2 * sizeof(double)));
    if (size > 0) {
        check_cuda(cudaMemsetAsync(output, 0, size * sizeof(double), context.stream()),
            "could not clear ACE output");
        constexpr unsigned block_size = 64;
        ace_cuda_kernel<<<static_cast<unsigned>((batch.atoms() + block_size - 1) / block_size),
            block_size, 0, context.stream()>>>(
            batch.numbers(), graph.offsets(), graph.atoms(), graph.shifts(), graph.displacements(), graph.distance2(),
            d_species.get(), static_cast<int>(species.size()), d_base_species.get(), d_base_radial.get(),
            d_base_angular.get(), d_base_magnetic.get(), static_cast<I64>(base_species.size()),
            max_radial, max_angular, transform_a, transform_p, transform_r0,
            py::cast<double>(payload["radial_t_left"]), py::cast<double>(payload["radial_t_right"]),
            py::cast<int>(payload["radial_p_left"]), py::cast<int>(payload["radial_p_right"]),
            d_radial_a.get(), d_radial_b.get(), d_radial_c.get(), d_center_feature_offsets.get(),
            d_feature_term_offsets.get(), d_term_channel_offsets.get(), d_term_channels.get(),
            d_term_coefficients.get(), features, batch.atoms(), coefficient_workspace, output);
        check_cuda(cudaGetLastError(), "ACE CUDA kernel launch failed");
    }
    const auto values = context.download_output(size);
    return atom_result(values, batch.atoms(), features, "ACE", options, false,
        std::vector<I64>(host_batch.offsets, host_batch.offsets + host_batch.structures + 1));
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

__global__ void generic_moment_kernel(
    const I32* numbers,
    const I64* graph_offsets,
    const I32* graph_atoms,
    const double* graph_displacements,
    const double* graph_distance2,
    const I32* species,
    int species_count,
    const I32* channel_species,
    const I32* channel_radial,
    int channels,
    int max_rank,
    int mode,
    int max_order,
    double min_dist,
    double max_dist,
    double soft_cutoff,
    double hard_cutoff,
    double radial_sigma,
    double radial_weight,
    double angular_weight,
    const double* center_weights,
    I64 features,
    I64 atoms,
    I64 moment_stride,
    int mtp_radial_basis_size,
    double* moment_workspace,
    double* output) {
    const I64 center = static_cast<I64>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (center >= atoms) return;
    if (channels > 256 || max_rank > 20) return;
    double local_moments[256 * 21]{};
    double* moments = local_moments;
    if (mode == 0) {
        if (moment_workspace == nullptr || max_rank > 5 || moment_stride <= 0) return;
        moments = moment_workspace + center * static_cast<I64>(channels) * moment_stride;
        for (I64 index = 0; index < static_cast<I64>(channels) * moment_stride; ++index) {
            moments[index] = 0.0;
        }
    }
    const I64 begin = graph_offsets[center];
    const I64 end = graph_offsets[center + 1];
    const int center_type = species_index(numbers[center], species, species_count);
    if (mode == 1 && center_type >= 0 && center_weights != nullptr) {
        for (int channel = 0; channel < channels; ++channel) {
            if (channel_species[channel] == center_type && channel_radial[channel] == 0) {
                moments[channel * (max_rank + 1)] += center_weights[center_type];
            }
        }
    }
    for (I64 edge = begin; edge < end; ++edge) {
        const double distance = sqrt(fmax(0.0, graph_distance2[edge]));
        if (distance < min_dist || distance > max_dist) continue;
        const int type = species_index(numbers[graph_atoms[edge]], species, species_count);
        if (type < 0) continue;
        const double reduced = (distance - min_dist) / fmax(max_dist - min_dist, 1e-12);
        const double cutoff = 0.5 * (1.0 + cos(kPi * reduced));
        double soft = 1.0;
        if (mode == 1) {
            if (distance >= hard_cutoff) continue;
            soft = distance <= soft_cutoff ? 1.0
                : 0.5 * (1.0 + cos(kPi * (distance - soft_cutoff)
                    / fmax(hard_cutoff - soft_cutoff, 1e-12)));
        }
        for (int channel = 0; channel < channels; ++channel) {
            if (channel_species[channel] != type) continue;
            const int radial = channel_radial[channel];
            double radial_value = 0.0;
            if (mode == 0) {
                const int basis = radial % max(1, mtp_radial_basis_size);
                const int radial_function = radial / max(1, mtp_radial_basis_size);
                const double radial_scale = radial_function == 0
                    ? 1.0 : pow(fmax(0.0, reduced), radial_function);
                radial_value = cutoff * cutoff * radial_scale
                    * chebyshev_device(basis, 2.0 * reduced - 1.0);
            } else {
                radial_value = cutoff * soft
                    * pow(fmax(0.0, reduced), radial % 4)
                    * cos((radial + 1.0) * kPi * reduced);
            }
            if (mode == 3 && radial_sigma > 0.0) {
                radial_value *= exp(-0.5 * distance * distance / (radial_sigma * radial_sigma));
            }
            if (mode == 0) {
                const double inverse_distance = 1.0 / distance;
                const double unit[3] = {
                    graph_displacements[edge * 3 + 0] * inverse_distance,
                    graph_displacements[edge * 3 + 1] * inverse_distance,
                    graph_displacements[edge * 3 + 2] * inverse_distance,
                };
                for (int rank = 0; rank <= max_rank; ++rank) {
                    I64 tensor_size = 1;
                    I64 tensor_offset = 0;
                    for (int previous = 0; previous < rank; ++previous) {
                        tensor_offset += tensor_size;
                        tensor_size *= 3;
                    }
                    for (I64 flat = 0; flat < tensor_size; ++flat) {
                        I64 value = flat;
                        double product = radial_value;
                        for (int component = 0; component < rank; ++component) {
                            const int axis = static_cast<int>(value % 3);
                            value /= 3;
                            product *= unit[axis];
                        }
                        moments[static_cast<I64>(channel) * moment_stride + tensor_offset + flat] += product;
                    }
                }
            } else {
                for (int rank = 0; rank <= max_rank; ++rank) {
                    moments[channel * (max_rank + 1) + rank] += radial_value
                        * pow(distance / fmax(max_dist, 1e-12), rank);
                }
            }
        }
    }
    double* target = output + center * features;
    for (I64 feature = 0; feature < features; ++feature) {
        double value = 0.0;
        if (mode == 0) {
            // Standalone MTP: traces first, followed by rank-wise pair
            // contractions, exactly matching its public feature count/order.
            I64 trace_count = static_cast<I64>(channels) * (max_rank / 2 + 1);
            if (feature < trace_count) {
                const int channel = static_cast<int>(feature / (max_rank / 2 + 1));
                const int rank = 2 * static_cast<int>(feature % (max_rank / 2 + 1));
                I64 tensor_offset = 0;
                I64 tensor_size = 1;
                for (int previous = 0; previous < rank; ++previous) {
                    tensor_offset += tensor_size;
                    tensor_size *= 3;
                }
                if (rank == 0) {
                    value = moments[static_cast<I64>(channel) * moment_stride];
                } else {
                    const int trace_pairs = rank / 2;
                    I64 combinations = 1;
                    for (int pair = 0; pair < trace_pairs; ++pair) combinations *= 3;
                    for (I64 combination = 0; combination < combinations; ++combination) {
                        I64 remainder = combination;
                        I64 flat = 0;
                        for (int pair = 0; pair < trace_pairs; ++pair) {
                            const int component = static_cast<int>(remainder % 3);
                            remainder /= 3;
                            flat = flat * 9 + component * 3 + component;
                        }
                        value += moments[static_cast<I64>(channel) * moment_stride
                            + tensor_offset + flat];
                    }
                }
            } else {
                I64 remainder = feature - trace_count;
                const I64 pair_count = static_cast<I64>(channels) * (channels + 1) / 2;
                const int rank = max_rank < 0 ? 0 : static_cast<int>(
                    remainder / pair_count) % (max_rank + 1);
                remainder %= pair_count;
                int first = 0;
                for (; first < channels; ++first) {
                    const int count = channels - first;
                    if (remainder < count) break;
                    remainder -= count;
                }
                const int second = first + static_cast<int>(remainder);
                I64 tensor_offset = 0;
                I64 tensor_size = 1;
                for (int previous = 0; previous < rank; ++previous) {
                    tensor_offset += tensor_size;
                    tensor_size *= 3;
                }
                for (I64 component = 0; component < tensor_size; ++component) {
                    value += moments[static_cast<I64>(first) * moment_stride
                        + tensor_offset + component]
                        * moments[static_cast<I64>(second) * moment_stride
                        + tensor_offset + component];
                }
            }
        } else {
            const int body = max(0, max_order);
            const I64 pair_count = static_cast<I64>(channels) * (channels + 1) / 2;
            const I64 pair = pair_count > 0 ? feature / (body + 1) % pair_count : 0;
            int first = 0;
            I64 remainder = pair;
            for (; first < channels; ++first) {
                const int count = channels - first;
                if (remainder < count) break;
                remainder -= count;
            }
            const int second = min(channels - 1, first + static_cast<int>(remainder));
            const int rank = max_rank == 0 ? 0 : static_cast<int>(feature % (max_rank + 1));
            value = moments[first * (max_rank + 1) + rank]
                * moments[second * (max_rank + 1) + rank];
            if (mode == 2) {
                const int order = body == 0 ? 1 : static_cast<int>(feature % (body + 1)) + 1;
                value *= pow(fabs(moments[first * (max_rank + 1)]) + 1e-12, order - 2);
            }
            if (mode == 3) {
                value *= feature < channels ? radial_weight : angular_weight;
            }
        }
        target[feature] = value;
    }
}

py::dict compute_generic_moment_descriptor(
    CudaExecutionContext& context,
    DeviceBatch& batch,
    DeviceNeighborGraph& graph,
    const detail::StructureBatchView& host_batch,
    const std::string& name,
    const py::dict& options) {
    const auto species = species_option(options);
    if (species.empty()) throw std::invalid_argument(name + " species must not be empty");

    int mode = 0;
    int max_rank = 2;
    int max_order = 0;
    double min_dist = 0.0;
    double max_dist = 5.0;
    double soft_cutoff = max_dist;
    double hard_cutoff = max_dist;
    double radial_sigma = 0.0;
    double radial_weight = 1.0;
    double angular_weight = 1.0;
    int mtp_radial_basis_size = 1;
    std::vector<I32> channel_species;
    std::vector<I32> channel_radial;
    std::vector<double> center_weights;
    I64 computed_features = 0;

    if (name == "SOAPTurbo") {
        mode = 1;
        const auto alpha = integer_vector_option(options, "alpha_max", 8, species.size());
        const int lmax = option(options, "l_max", 6);
        hard_cutoff = option(options, "rcut_hard", 5.0);
        soft_cutoff = option(options, "rcut_soft", hard_cutoff);
        max_dist = hard_cutoff;
        max_rank = lmax;
        radial_sigma = numeric_vector_option(options, "atom_sigma_r", 0.5, species.size())[0];
        center_weights = numeric_vector_option(options, "central_weight", 1.0, species.size());
        for (std::size_t type = 0; type < alpha.size(); ++type) {
            if (alpha[type] <= 0) throw std::invalid_argument("SOAPTurbo alpha_max must be positive");
            for (int radial = 0; radial < alpha[type]; ++radial) {
                channel_species.push_back(static_cast<I32>(type));
                channel_radial.push_back(static_cast<I32>(radial));
            }
        }
        const I64 channels = static_cast<I64>(channel_species.size());
        const std::string compression = option(options, "compression", std::string("off"));
        if (compression.empty() || compression == "off" || compression == "none") {
            computed_features = channels * (channels + 1) / 2 * (max_rank + 1);
        } else if (compression == "trivial") {
            std::vector<int> pivots;
            int pivot = 0;
            for (I32 count : alpha) {
                pivots.push_back(pivot);
                pivot += count;
            }
            I64 retained = 0;
            for (I64 first = 0; first < channels; ++first) {
                for (I64 second = first; second < channels; ++second) {
                    bool keep = false;
                    for (int value : pivots) keep = keep || first == value || second == value;
                    if (keep) ++retained;
                }
            }
            computed_features = retained * (max_rank + 1);
        } else if (compression.size() == 3 && compression[1] == '_') {
            const int nu_r = compression[0] - '0';
            const int nu_s = compression[2] - '0';
            if (nu_r < 0 || nu_r > 2 || nu_s < 0 || nu_s > 2
                || std::adjacent_find(alpha.begin(), alpha.end(), std::not_equal_to<>()) != alpha.end()) {
                throw std::invalid_argument("invalid SOAPTurbo compression");
            }
            const int n1 = nu_r > 0 ? alpha[0] : 1;
            const int n2 = nu_r == 2 ? alpha[0] : 1;
            const int s1 = nu_s > 0 ? static_cast<int>(species.size()) : 1;
            const int s2 = nu_s == 2 ? static_cast<int>(species.size()) : 1;
            if (nu_r % 2 == 0 && nu_s % 2 == 0) {
                const I64 compressed_channels = static_cast<I64>(n1) * s1;
                computed_features = compressed_channels * (compressed_channels + 1) / 2
                    * (max_rank + 1);
            } else {
                computed_features = static_cast<I64>(n1) * s1 * n2 * s2 * (max_rank + 1);
            }
        } else {
            throw std::invalid_argument("invalid SOAPTurbo compression");
        }
    } else if (name == "ACE") {
        mode = 2;
        const auto maxdeg = numeric_values_option(options, "maxdeg", 8.0);
        if (maxdeg.empty()) throw std::invalid_argument("ACE maxdeg must not be empty");
        const double largest_degree = *std::max_element(maxdeg.begin(), maxdeg.end());
        const int radial_count = std::max(1, static_cast<int>(std::ceil(largest_degree)) + 1);
        const int correlation = option(options, "N", 3);
        min_dist = option(options, "rin", 0.0);
        max_dist = option(options, "rcut", 5.0);
        max_rank = std::min(20, std::max(0, correlation));
        max_order = std::min(20, std::max(0, correlation));
        for (std::size_t type = 0; type < species.size(); ++type) {
            for (int radial = 0; radial < radial_count; ++radial) {
                channel_species.push_back(static_cast<I32>(type));
                channel_radial.push_back(static_cast<I32>(radial));
            }
        }
        const I64 channels = static_cast<I64>(channel_species.size());
        computed_features = channels * (max_rank / 2 + 1)
            + static_cast<I64>(max_rank + 1) * channels * (channels + 1) / 2;
    } else if (name == "MTP") {
        mode = 0;
        const int radial_basis_size = option(options, "radial_basis_size", 4);
        const int radial_funcs_count = option(options, "radial_funcs_count", 1);
        mtp_radial_basis_size = radial_basis_size;
        max_rank = option(options, "max_rank", 2);
        min_dist = option(options, "min_dist", 0.0);
        max_dist = option(options, "max_dist", option(options, "r_cut", 5.0));
        if (radial_basis_size <= 0 || radial_funcs_count <= 0) {
            throw std::invalid_argument("MTP radial basis sizes must be positive");
        }
        for (std::size_t type = 0; type < species.size(); ++type) {
            for (int function = 0; function < radial_funcs_count; ++function) {
                for (int radial = 0; radial < radial_basis_size; ++radial) {
                    channel_species.push_back(static_cast<I32>(type));
                    channel_radial.push_back(static_cast<I32>(function * radial_basis_size + radial));
                }
            }
        }
        const I64 channels = static_cast<I64>(channel_species.size());
        computed_features = channels * (max_rank / 2 + 1)
            + static_cast<I64>(max_rank + 1) * channels * (channels + 1) / 2;
    } else if (name == "C00PSMLFF") {
        mode = 3;
        const int radial_count = option(
            options, "n_radial", option(options, "n_max", 8));
        max_rank = option(options, "l_max", 4);
        max_dist = option(options, "r_cut", option(options, "cutoff", 6.0));
        radial_sigma = option(options, "radial_sigma", 0.5);
        radial_weight = option(options, "radial_weight", 1.0);
        angular_weight = option(options, "angular_weight", 1.0);
        const bool include_radial = option(options, "include_radial", true);
        const bool include_angular = option(options, "include_angular", true);
        if (radial_count <= 0) throw std::invalid_argument("C00PSMLFF n_radial must be positive");
        for (std::size_t type = 0; type < species.size(); ++type) {
            for (int radial = 0; radial < radial_count; ++radial) {
                channel_species.push_back(static_cast<I32>(type));
                channel_radial.push_back(static_cast<I32>(radial));
            }
        }
        const I64 channels = static_cast<I64>(channel_species.size());
        computed_features = include_radial ? channels : 0;
        if (include_angular) {
            computed_features += static_cast<I64>(max_rank + 1) * channels * (channels + 1) / 2;
        }
    } else {
        throw std::invalid_argument(name + " has no generic CUDA moment layout");
    }

    const I64 features = feature_count_option(options, computed_features);
    if (features <= 0 || channel_species.empty() || channel_species.size() > 256
        || max_rank < 0 || max_rank > 20 || min_dist < 0.0
        || max_dist <= min_dist || soft_cutoff < 0.0 || soft_cutoff > hard_cutoff
        || hard_cutoff <= 0.0) {
        throw std::invalid_argument("invalid CUDA generic descriptor parameters");
    }
    const I64 channels = static_cast<I64>(channel_species.size());
    I64 moment_stride = 0;
    if (mode == 0) {
        I64 tensor_size = 1;
        for (int rank = 0; rank <= max_rank; ++rank) {
            moment_stride += tensor_size;
            tensor_size *= 3;
        }
    }
    graph.build_dpa(context, batch, host_batch, max_dist, true, false, false);

    DeviceBuffer<I32> d_species;
    DeviceBuffer<I32> d_channel_species;
    DeviceBuffer<I32> d_channel_radial;
    DeviceBuffer<double> d_center_weights;
    d_species.upload(species.data(), species.size(), context.stream(),
        "could not upload generic descriptor species");
    d_channel_species.upload(channel_species.data(), channel_species.size(), context.stream(),
        "could not upload generic descriptor channel species");
    d_channel_radial.upload(channel_radial.data(), channel_radial.size(), context.stream(),
        "could not upload generic descriptor channel indices");
    d_center_weights.upload(center_weights.data(), center_weights.size(), context.stream(),
        "could not upload SOAPTurbo center weights");

    const std::size_t size = static_cast<std::size_t>(batch.atoms())
        * static_cast<std::size_t>(features);
    double* output = context.output_buffer(size);
    auto* moment_workspace = mode == 0
        ? static_cast<double*>(context.workspace_buffer(
            static_cast<std::size_t>(batch.atoms()) * static_cast<std::size_t>(channels)
            * static_cast<std::size_t>(moment_stride) * sizeof(double)))
        : nullptr;
    if (size > 0) {
        check_cuda(cudaMemsetAsync(output, 0, size * sizeof(double), context.stream()),
            "could not clear generic CUDA descriptor output");
        constexpr unsigned block_size = 64;
        generic_moment_kernel<<<static_cast<unsigned>((batch.atoms() + block_size - 1) / block_size),
            block_size, 0, context.stream()>>>(
            batch.numbers(), graph.offsets(), graph.atoms(), graph.displacements(), graph.distance2(), d_species.get(),
            static_cast<int>(species.size()), d_channel_species.get(), d_channel_radial.get(),
            static_cast<int>(channels), max_rank, mode, max_order, min_dist, max_dist,
            soft_cutoff, hard_cutoff, radial_sigma, radial_weight, angular_weight,
            d_center_weights.get(), features, batch.atoms(), moment_stride,
            mtp_radial_basis_size, moment_workspace, output);
        check_cuda(cudaGetLastError(), "generic CUDA descriptor kernel launch failed");
    }
    const auto values = context.download_output(size);
    return atom_result(values, batch.atoms(), features, name, options, false,
        std::vector<I64>(host_batch.offsets, host_batch.offsets + host_batch.structures + 1));
}

} // namespace

py::dict compute_extended_descriptor(
    CudaExecutionContext& context,
    DeviceBatch& batch,
    DeviceNeighborGraph& graph,
    const detail::StructureBatchView& host_batch,
    const std::string& name,
    const py::dict& options,
    const py::object& control,
    RotationalPlanCache* rotational_plan_cache) {
    (void)control;
    if (name == "AtomicComposition") {
        return compute_atomic_composition(
            context, batch, host_batch, species_option(options), option(options, "per_system", true),
            name, options);
    }
    if (name == "SortedDistances") {
        return compute_sorted_distances(
            context, batch, graph, host_batch, species_option(options), option(options, "cutoff", 6.0),
            option(options, "max_neighbors", 8), option(options, "separate_neighbor_types", true),
            name, options);
    }
    if (name == "SphericalExpansionByPair") {
        return compute_spherical_pair(
            context, batch, graph, host_batch, option(options, "cutoff", 6.0),
            option(options, "density_width", 0.3), option(options, "max_radial", 6),
            option(options, "max_angular", 4), name, options);
    }
    if (name == "SOAP") {
        return compute_soap_descriptor(context, batch, graph, host_batch, options);
    }
    if (name == "ACSF") {
        return compute_acsf_descriptor(context, batch, graph, host_batch, options);
    }
    if (name == "CoulombMatrix") {
        return compute_matrix_descriptor(context, batch, host_batch, kMatrixKindCoulomb, name, options);
    }
    if (name == "SineMatrix") {
        return compute_matrix_descriptor(context, batch, host_batch, kMatrixKindSine, name, options);
    }
    if (name == "EwaldSumMatrix") {
        return compute_matrix_descriptor(context, batch, host_batch, kMatrixKindEwald, name, options);
    }
    if (name == "MBTR" || name == "LMBTR" || name == "ValleOganov") {
        return compute_mbtr_descriptor(context, batch, graph, host_batch, name, options);
    }
    if (name == "EAD") {
        return compute_ead_descriptor(context, batch, graph, host_batch, options);
    }
    if (name == "LodeSphericalExpansion") {
        return compute_lode_descriptor(context, batch, graph, host_batch, options);
    }
    if (name == "SO3" || name == "SO4" || name == "SNAP" || name == "LBispectrum") {
        return compute_rotational_descriptor(
            context, batch, graph, host_batch, name, options, rotational_plan_cache);
    }
    if (name == "C00PSMLFF") {
        return compute_c00ps_mlff_descriptor(context, batch, graph, host_batch, options);
    }
    if (name == "SOAPTurbo") {
        return compute_soap_turbo_descriptor(context, batch, graph, host_batch, options);
    }
    if (name == "ACE") {
        return compute_ace_descriptor(context, batch, graph, host_batch, options);
    }
    if (name == "MTP") {
        const py::str payload_key("_cuda_payload");
        if (options.contains(payload_key) && !options[payload_key].is_none()) {
            const py::dict payload = py::cast<py::dict>(options[payload_key]);
            if (payload.contains("model_parameters")) {
                return compute_mtp4_descriptor(context, batch, graph, host_batch, options);
            }
            if (payload.contains("alpha_index_basic")) {
                return compute_mtp2_descriptor(context, batch, graph, host_batch, options);
            }
        }
        return compute_generic_moment_descriptor(context, batch, graph, host_batch, name, options);
    }
    throw std::invalid_argument("CUDA backend does not support this extended descriptor");
}

struct RotationalPlanCache::Impl {
    bool prepared = false;
    int expansion_order = -1;
    int diagonal = -1;
    bool l_bispectrum = false;
    std::int64_t features = 0;
    DeviceBuffer<I64> z_inner_offsets;
    DeviceBuffer<I64> inner_term_offsets;
    DeviceBuffer<double> inner_outer_coefficients;
    DeviceBuffer<I64> term_first_indices;
    DeviceBuffer<I64> term_second_indices;
    DeviceBuffer<double> term_coefficients;
    DeviceBuffer<I64> projection_offsets;
    DeviceBuffer<I64> projection_u_indices;
    DeviceBuffer<I64> projection_z_indices;
    DeviceBuffer<double> projection_scales;
};

RotationalPlanCache::RotationalPlanCache() : impl_(std::make_unique<Impl>()) {}

RotationalPlanCache::~RotationalPlanCache() noexcept {
    clear();
}

RotationalPlanDeviceView RotationalPlanCache::prepare(
    CudaExecutionContext& context,
    int expansion_order,
    int diagonal,
    bool l_bispectrum) {
    if (impl_->prepared && impl_->expansion_order == expansion_order
        && impl_->diagonal == diagonal && impl_->l_bispectrum == l_bispectrum) {
        return {
            impl_->z_inner_offsets.get(),
            impl_->inner_term_offsets.get(),
            impl_->inner_outer_coefficients.get(),
            impl_->term_first_indices.get(),
            impl_->term_second_indices.get(),
            impl_->term_coefficients.get(),
            impl_->projection_offsets.get(),
            impl_->projection_u_indices.get(),
            impl_->projection_z_indices.get(),
            impl_->projection_scales.get(),
            impl_->features,
        };
    }

    const auto bispectrum_plan = detail::rotational::make_bispectrum_plan(
        expansion_order, diagonal, l_bispectrum);
    const auto flattened_plan = detail::rotational::flatten(bispectrum_plan);
    impl_->prepared = false;
    impl_->z_inner_offsets.clear();
    impl_->inner_term_offsets.clear();
    impl_->inner_outer_coefficients.clear();
    impl_->term_first_indices.clear();
    impl_->term_second_indices.clear();
    impl_->term_coefficients.clear();
    impl_->projection_offsets.clear();
    impl_->projection_u_indices.clear();
    impl_->projection_z_indices.clear();
    impl_->projection_scales.clear();
    impl_->z_inner_offsets.upload(
        flattened_plan.z_inner_offsets.data(), flattened_plan.z_inner_offsets.size(),
        context.stream(), "could not upload CUDA bispectrum Z offsets");
    impl_->inner_term_offsets.upload(
        flattened_plan.inner_term_offsets.data(), flattened_plan.inner_term_offsets.size(),
        context.stream(), "could not upload CUDA bispectrum inner offsets");
    impl_->inner_outer_coefficients.upload(
        flattened_plan.inner_outer_coefficients.data(),
        flattened_plan.inner_outer_coefficients.size(), context.stream(),
        "could not upload CUDA bispectrum outer coefficients");
    impl_->term_first_indices.upload(
        flattened_plan.term_first_indices.data(), flattened_plan.term_first_indices.size(),
        context.stream(), "could not upload CUDA bispectrum first indices");
    impl_->term_second_indices.upload(
        flattened_plan.term_second_indices.data(), flattened_plan.term_second_indices.size(),
        context.stream(), "could not upload CUDA bispectrum second indices");
    impl_->term_coefficients.upload(
        flattened_plan.term_coefficients.data(), flattened_plan.term_coefficients.size(),
        context.stream(), "could not upload CUDA bispectrum CG coefficients");
    impl_->projection_offsets.upload(
        flattened_plan.projection_offsets.data(), flattened_plan.projection_offsets.size(),
        context.stream(), "could not upload CUDA bispectrum projection offsets");
    impl_->projection_u_indices.upload(
        flattened_plan.projection_u_indices.data(), flattened_plan.projection_u_indices.size(),
        context.stream(), "could not upload CUDA bispectrum projection U indices");
    impl_->projection_z_indices.upload(
        flattened_plan.projection_z_indices.data(), flattened_plan.projection_z_indices.size(),
        context.stream(), "could not upload CUDA bispectrum projection Z indices");
    impl_->projection_scales.upload(
        flattened_plan.projection_scales.data(), flattened_plan.projection_scales.size(),
        context.stream(), "could not upload CUDA bispectrum projection scales");
    impl_->expansion_order = expansion_order;
    impl_->diagonal = diagonal;
    impl_->l_bispectrum = l_bispectrum;
    impl_->features = static_cast<std::int64_t>(bispectrum_plan.components.size());
    impl_->prepared = true;
    return {
        impl_->z_inner_offsets.get(),
        impl_->inner_term_offsets.get(),
        impl_->inner_outer_coefficients.get(),
        impl_->term_first_indices.get(),
        impl_->term_second_indices.get(),
        impl_->term_coefficients.get(),
        impl_->projection_offsets.get(),
        impl_->projection_u_indices.get(),
        impl_->projection_z_indices.get(),
        impl_->projection_scales.get(),
        impl_->features,
    };
}

void RotationalPlanCache::clear() noexcept {
    if (impl_ == nullptr) return;
    impl_->prepared = false;
    impl_->z_inner_offsets.clear();
    impl_->inner_term_offsets.clear();
    impl_->inner_outer_coefficients.clear();
    impl_->term_first_indices.clear();
    impl_->term_second_indices.clear();
    impl_->term_coefficients.clear();
    impl_->projection_offsets.clear();
    impl_->projection_u_indices.clear();
    impl_->projection_z_indices.clear();
    impl_->projection_scales.clear();
}

} // namespace mdescriptor::cuda
