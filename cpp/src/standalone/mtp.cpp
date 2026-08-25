#include "mdescriptor/extra.hpp"
#include "mdescriptor/mtp4.hpp"
#include "mdescriptor/neighbor.hpp"
#include "descriptor_common.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <limits>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace mdescriptor {
using namespace detail;

struct OfficialMtpModel {
    bool native_mlip4 = false;
    std::shared_ptr<NativeMtp4Model> native_model;
    int species_count = 0;
    double scaling = 1.0;
    std::string radial_basis_type;
    double min_dist = 0.0;
    double max_dist = 0.0;
    int radial_basis_size = 0;
    int radial_funcs_count = 0;
    int alpha_moments_count = 0;
    std::vector<std::array<int, 4>> alpha_index_basic;
    std::vector<std::array<int, 4>> alpha_index_times;
    std::vector<int> alpha_moment_mapping;
    // [central species][outer species][radial function][Chebyshev term].
    std::vector<double> radial_coeffs;

    void load(const std::string& path);
    std::int64_t feature_count() const noexcept {
        if (native_mlip4) return native_model ? native_model->feature_count() : 0;
        return static_cast<std::int64_t>(alpha_moment_mapping.size()) + 1;
    }
};

namespace {
constexpr double kPi = 3.141592653589793238462643383279502884;

[[noreturn]] void invalid_model(const std::string& path, const std::string& message) {
    throw std::invalid_argument("invalid MLIP-2 MTP potential '" + path + "': " + message);
}

void expect_token(std::istream& input, const std::string& expected, const std::string& path) {
    std::string token;
    if (!(input >> token) || token != expected) {
        invalid_model(path, "expected '" + expected + "'");
    }
}

template <typename T>
T read_value(std::istream& input, const std::string& path, const std::string& field) {
    T value{};
    if (!(input >> value)) {
        invalid_model(path, "could not read " + field);
    }
    return value;
}

void read_assignment(std::istream& input, const std::string& path, const std::string& field) {
    expect_token(input, field, path);
    expect_token(input, "=", path);
}

void read_tuple_array(
    std::istream& input,
    const std::string& path,
    const std::string& field,
    std::size_t count,
    std::vector<std::array<int, 4>>& values
) {
    read_assignment(input, path, field);
    char character = read_value<char>(input, path, field + " opening brace");
    if (character != '{') {
        invalid_model(path, "expected opening brace for " + field);
    }
    values.resize(count);
    for (std::size_t row = 0; row < count; ++row) {
        do {
            character = read_value<char>(input, path, field + " row");
        } while (character != '{');
        for (int column = 0; column < 4; ++column) {
            values[row][static_cast<std::size_t>(column)] =
                read_value<int>(input, path, field + " value");
            if (column != 3) {
                character = read_value<char>(input, path, field + " separator");
            }
        }
        character = read_value<char>(input, path, field + " row closing brace");
        if (character != '}') {
            invalid_model(path, "expected row closing brace for " + field);
        }
        if (row + 1 != count) {
            character = read_value<char>(input, path, field + " row separator");
        }
    }
    character = read_value<char>(input, path, field + " closing brace");
    if (character != '}') {
        invalid_model(path, "expected closing brace for " + field);
    }
}

void read_int_array(
    std::istream& input,
    const std::string& path,
    const std::string& field,
    std::size_t count,
    std::vector<int>& values
) {
    read_assignment(input, path, field);
    char character = read_value<char>(input, path, field + " opening brace");
    if (character != '{') {
        invalid_model(path, "expected opening brace for " + field);
    }
    values.resize(count);
    for (std::size_t index = 0; index < count; ++index) {
        values[index] = read_value<int>(input, path, field + " value");
        if (index + 1 != count) {
            character = read_value<char>(input, path, field + " separator");
        }
    }
    character = read_value<char>(input, path, field + " closing brace");
    if (character != '}') {
        invalid_model(path, "expected closing brace for " + field);
    }
}

std::vector<double> read_double_row(
    std::istream& input,
    const std::string& path,
    int count,
    const std::string& field
) {
    char character = read_value<char>(input, path, field + " opening brace");
    if (character != '{') {
        invalid_model(path, "expected opening brace for " + field);
    }
    std::vector<double> row(static_cast<std::size_t>(count));
    for (int index = 0; index < count; ++index) {
        row[static_cast<std::size_t>(index)] = read_value<double>(input, path, field + " value");
        if (index + 1 != count) {
            character = read_value<char>(input, path, field + " separator");
        }
    }
    character = read_value<char>(input, path, field + " closing brace");
    if (character != '}') {
        invalid_model(path, "expected closing brace for " + field);
    }
    return row;
}

void compute_rbc_chebyshev(
    double distance,
    double min_dist,
    double max_dist,
    int basis_size,
    double scaling,
    double* values
) {
    const double ksi = (2.0 * distance - (min_dist + max_dist)) / (max_dist - min_dist);
    const double edge = distance - max_dist;
    values[0] = edge * edge;
    if (basis_size > 1) {
        values[1] = ksi * edge * edge;
    }
    for (int index = 2; index < basis_size; ++index) {
        values[index] = 2.0 * ksi * values[index - 1] - values[index - 2];
    }
    for (int index = 0; index < basis_size; ++index) {
        values[index] *= scaling;
    }
}

std::int64_t channel_count(const MtpOptions& options) {
    return static_cast<std::int64_t>(options.species.size())
        * options.radial_funcs_count * options.radial_basis_size;
}

std::int64_t tensor_size(int rank) {
    std::int64_t result = 1;
    for (int index = 0; index < rank; ++index) {
        result *= 3;
    }
    return result;
}

std::int64_t basic_count(const MtpOptions& options) {
    const auto channels = channel_count(options);
    return channels * (options.max_rank / 2 + 1);
}

std::int64_t pair_count(const MtpOptions& options) {
    const auto channels = channel_count(options);
    return static_cast<std::int64_t>(options.max_rank + 1)
        * channels * (channels + 1) / 2;
}

double chebyshev(int order, double x) {
    if (order == 0) {
        return 1.0;
    }
    double previous = 1.0;
    double current = x;
    for (int index = 2; index <= order; ++index) {
        const double next = 2.0 * x * current - previous;
        previous = current;
        current = next;
    }
    return current;
}

double trace_tensor(const double* tensor, int rank) {
    if (rank == 0) {
        return tensor[0];
    }
    double result = 0.0;
    const int trace_pairs = rank / 2;
    const std::int64_t combinations = tensor_size(trace_pairs);
    for (std::int64_t combination = 0; combination < combinations; ++combination) {
        std::int64_t value = combination;
        std::int64_t flat = 0;
        for (int pair = 0; pair < trace_pairs; ++pair) {
            const int component = static_cast<int>(value % 3);
            value /= 3;
            flat = flat * 9 + component * 3 + component;
        }
        result += tensor[static_cast<std::size_t>(flat)];
    }
    return result;
}

void add_outer_product(
    double* tensor,
    int rank,
    const Vec3& unit_vector,
    double weight
) {
    if (rank == 0) {
        tensor[0] += weight;
        return;
    }
    const std::int64_t components = tensor_size(rank);
    for (std::int64_t flat = 0; flat < components; ++flat) {
        std::int64_t value = flat;
        double product = weight;
        for (int index = 0; index < rank; ++index) {
            const int component = static_cast<int>(value % 3);
            value /= 3;
            product *= component == 0 ? unit_vector.x
                : component == 1 ? unit_vector.y : unit_vector.z;
        }
        tensor[static_cast<std::size_t>(flat)] += product;
    }
}

void compute_mtp_impl(
    const StructureBatchView& batch,
    const MtpOptions& options,
    double* output,
    const std::shared_ptr<ComputeControl>& control
) {
    const auto graph = build_neighbor_graph(batch, options.max_dist, control, options.num_threads);
    const auto mapping = species_map(options.species);
    const auto channels = static_cast<std::size_t>(channel_count(options));
    const auto features = static_cast<std::size_t>(mtp_feature_count(options));
    const auto tensor_offsets = [channels, &options](int rank) {
        std::size_t offset = 0;
        for (int previous = 0; previous < rank; ++previous) {
            offset += channels * static_cast<std::size_t>(tensor_size(previous));
        }
        return offset;
    };

    run_parallel_structures(batch.structures, options.num_threads, control, [&](std::int64_t structure) {
        const std::int64_t begin = batch.offsets[structure];
        const std::int64_t end = batch.offsets[structure + 1];
        for (std::int64_t center = begin; center < end; ++center) {
            if (cancelled(control)) {
                return;
            }
            double* values = output + center * static_cast<std::int64_t>(features);
            std::fill(values, values + features, 0.0);

            std::vector<double> moments;
            std::size_t moment_size = 0;
            for (int rank = 0; rank <= options.max_rank; ++rank) {
                moment_size += channels * static_cast<std::size_t>(tensor_size(rank));
            }
            moments.assign(moment_size, 0.0);
            const auto neighbors = graph.for_center(center);
            for (std::size_t neighbor_index = 0; neighbor_index < neighbors.size; ++neighbor_index) {
                if (neighbors.exact_self(neighbor_index, center)) {
                    continue;
                }
                const auto atom = neighbors.atoms[neighbor_index];
                const auto species_it = mapping.find(batch.numbers[atom]);
                if (species_it == mapping.end()) {
                    continue;
                }
                const double distance2 = neighbors.distance2[neighbor_index];
                const double distance = std::sqrt(std::max(distance2, 0.0));
                if (distance <= 0.0 || distance < options.min_dist || distance > options.max_dist) {
                    continue;
                }
                const double span = options.max_dist - options.min_dist;
                const double reduced = (distance - options.min_dist) / span;
                const double cutoff = 0.5 * (1.0 + std::cos(kPi * reduced));
                const double inv_distance = 1.0 / distance;
                const Vec3 unit_vector{
                    neighbors.displacements[neighbor_index * 3] * inv_distance,
                    neighbors.displacements[neighbor_index * 3 + 1] * inv_distance,
                    neighbors.displacements[neighbor_index * 3 + 2] * inv_distance,
                };
                const double x = 2.0 * reduced - 1.0;
                const std::int64_t species_offset = static_cast<std::int64_t>(species_it->second)
                    * options.radial_funcs_count * options.radial_basis_size;
                for (int radial_function = 0; radial_function < options.radial_funcs_count; ++radial_function) {
                    const double radial_scale = radial_function == 0 ? 1.0 : std::pow(reduced, radial_function);
                    for (int radial = 0; radial < options.radial_basis_size; ++radial) {
                        const double radial_value = cutoff * cutoff * radial_scale * chebyshev(radial, x);
                        const auto channel = static_cast<std::size_t>(species_offset
                            + radial_function * options.radial_basis_size + radial);
                        for (int rank = 0; rank <= options.max_rank; ++rank) {
                            const std::size_t offset = tensor_offsets(rank)
                                + channel * static_cast<std::size_t>(tensor_size(rank));
                            add_outer_product(moments.data() + offset, rank, unit_vector, radial_value);
                        }
                    }
                }
            }

            std::size_t output_offset = 0;
            for (std::size_t channel = 0; channel < channels; ++channel) {
                for (int rank = 0; rank <= options.max_rank; rank += 2) {
                    const std::size_t offset = tensor_offsets(rank)
                        + channel * static_cast<std::size_t>(tensor_size(rank));
                    values[output_offset++] = trace_tensor(moments.data() + offset, rank);
                }
            }
            for (int rank = 0; rank <= options.max_rank; ++rank) {
                const std::size_t rank_offset = tensor_offsets(rank);
                const std::size_t components = static_cast<std::size_t>(tensor_size(rank));
                for (std::size_t first = 0; first < channels; ++first) {
                    const std::size_t first_offset = rank_offset + first * components;
                    for (std::size_t second = first; second < channels; ++second) {
                        const std::size_t second_offset = rank_offset + second * components;
                        double dot = 0.0;
                        for (std::size_t component = 0; component < components; ++component) {
                            dot += moments[first_offset + component] * moments[second_offset + component];
                        }
                        values[output_offset++] = dot;
                    }
                }
            }
        }
    });
}

void compute_official_mtp_impl(
    const StructureBatchView& batch,
    const MtpOptions& options,
    const OfficialMtpModel& model,
    double* output,
    const std::shared_ptr<ComputeControl>& control
) {
    const auto graph = build_neighbor_graph(batch, model.max_dist, control, options.num_threads);
    const auto mapping = species_map(options.species);
    const auto features = static_cast<std::size_t>(model.feature_count());
    int max_coordinate_power = 0;
    for (const auto& index : model.alpha_index_basic) {
        max_coordinate_power = std::max(
            max_coordinate_power,
            index[1] + index[2] + index[3]);
    }

    run_parallel_structures(batch.structures, options.num_threads, control, [&](std::int64_t structure) {
        const std::int64_t begin = batch.offsets[structure];
        const std::int64_t end = batch.offsets[structure + 1];
        for (std::int64_t center = begin; center < end; ++center) {
            if (cancelled(control)) {
                return;
            }
            double* values = output + center * static_cast<std::int64_t>(features);
            std::fill(values, values + features, 0.0);
            std::vector<double> moments(static_cast<std::size_t>(model.alpha_moments_count), 0.0);
            std::vector<double> rb(static_cast<std::size_t>(model.radial_basis_size));
            std::vector<double> radial_values(static_cast<std::size_t>(model.radial_funcs_count));
            std::vector<double> distance_powers(static_cast<std::size_t>(max_coordinate_power + 1), 1.0);
            std::vector<std::array<double, 3>> coordinate_powers(
                static_cast<std::size_t>(max_coordinate_power + 1), {1.0, 1.0, 1.0});

            const auto central_it = mapping.find(batch.numbers[center]);
            if (central_it == mapping.end()) {
                throw std::invalid_argument("batch contains an atomic number outside calculator species");
            }
            const int central_type = central_it->second;
            const auto neighbors = graph.for_center(center);
            for (std::size_t neighbor_index = 0; neighbor_index < neighbors.size; ++neighbor_index) {
                if (neighbors.exact_self(neighbor_index, center)) {
                    continue;
                }
                const auto atom = neighbors.atoms[neighbor_index];
                const auto outer_it = mapping.find(batch.numbers[atom]);
                if (outer_it == mapping.end()) {
                    continue;
                }
                const double distance2 = neighbors.distance2[neighbor_index];
                const double distance = std::sqrt(std::max(distance2, 0.0));
                if (distance <= 0.0 || distance > model.max_dist) {
                    continue;
                }
                const double basis_distance = model.radial_basis_type == "RBChebyshev_repuls"
                    ? std::max(distance, model.min_dist)
                    : distance;
                compute_rbc_chebyshev(
                    basis_distance,
                    model.min_dist,
                    model.max_dist,
                    model.radial_basis_size,
                    model.scaling,
                    rb.data());
                const std::size_t pair_offset = (
                    static_cast<std::size_t>(central_type) * static_cast<std::size_t>(model.species_count)
                    + static_cast<std::size_t>(outer_it->second))
                    * static_cast<std::size_t>(model.radial_funcs_count)
                    * static_cast<std::size_t>(model.radial_basis_size);
                for (int radial_function = 0; radial_function < model.radial_funcs_count; ++radial_function) {
                    double value = 0.0;
                    const std::size_t radial_offset = pair_offset
                        + static_cast<std::size_t>(radial_function) * static_cast<std::size_t>(model.radial_basis_size);
                    for (int radial = 0; radial < model.radial_basis_size; ++radial) {
                        value += model.radial_coeffs[radial_offset + static_cast<std::size_t>(radial)]
                            * rb[static_cast<std::size_t>(radial)];
                    }
                    radial_values[static_cast<std::size_t>(radial_function)] = value;
                }

                distance_powers[0] = 1.0;
                coordinate_powers[0] = {1.0, 1.0, 1.0};
                const double* displacement = neighbors.displacements + neighbor_index * 3;
                for (int power = 1; power <= max_coordinate_power; ++power) {
                    distance_powers[static_cast<std::size_t>(power)]
                        = distance_powers[static_cast<std::size_t>(power - 1)] * distance;
                    coordinate_powers[static_cast<std::size_t>(power)] = {
                        coordinate_powers[static_cast<std::size_t>(power - 1)][0] * displacement[0],
                        coordinate_powers[static_cast<std::size_t>(power - 1)][1] * displacement[1],
                        coordinate_powers[static_cast<std::size_t>(power - 1)][2] * displacement[2],
                    };
                }

                for (std::size_t index = 0; index < model.alpha_index_basic.size(); ++index) {
                    const auto& alpha = model.alpha_index_basic[index];
                    const int coordinate_degree = alpha[1] + alpha[2] + alpha[3];
                    const double coordinate_product =
                        coordinate_powers[static_cast<std::size_t>(alpha[1])][0]
                        * coordinate_powers[static_cast<std::size_t>(alpha[2])][1]
                        * coordinate_powers[static_cast<std::size_t>(alpha[3])][2];
                    moments[index] += radial_values[static_cast<std::size_t>(alpha[0])]
                        * coordinate_product / distance_powers[static_cast<std::size_t>(coordinate_degree)];
                }
            }

            for (const auto& alpha : model.alpha_index_times) {
                moments[static_cast<std::size_t>(alpha[3])] += static_cast<double>(alpha[2])
                    * moments[static_cast<std::size_t>(alpha[0])]
                    * moments[static_cast<std::size_t>(alpha[1])];
            }
            values[0] = 1.0;
            for (std::size_t index = 0; index < model.alpha_moment_mapping.size(); ++index) {
                values[index + 1] = moments[static_cast<std::size_t>(model.alpha_moment_mapping[index])];
            }
        }
    });
}

} // namespace

void OfficialMtpModel::load(const std::string& path) {
    std::ifstream input(path);
    if (!input.is_open()) {
        invalid_model(path, "cannot open file");
    }
    std::string line;
    input >> std::ws;
    if (input.peek() == '{' || input.peek() == '[') {
        native_mlip4 = true;
        native_model = std::make_shared<NativeMtp4Model>();
        native_model->load(path);
        species_count = native_model->species_count();
        min_dist = native_model->min_dist();
        max_dist = native_model->max_dist();
        radial_basis_size = native_model->radial_basis_size();
        radial_funcs_count = native_model->radial_funcs_count();
        radial_basis_type = native_model->radial_basis_type();
        return;
    }
    input.clear();
    input.seekg(0);
    if (!std::getline(input, line) || (line != "MTP" && line != "MTP\r")) {
        invalid_model(path, "file is not in MTP format");
    }
    if (!std::getline(input, line) || (line != "version = 1.1.0" && line != "version = 1.1.0\r")) {
        invalid_model(path, "only MTP version 1.1.0 is supported");
    }

    std::string token;
    if (!(input >> token)) {
        invalid_model(path, "missing potential header");
    }
    if (token == "potential_name") {
        expect_token(input, "=", path);
        (void)read_value<std::string>(input, path, "potential_name");
        if (!(input >> token)) {
            invalid_model(path, "missing species_count");
        }
    }
    if (token == "scaling") {
        expect_token(input, "=", path);
        scaling = read_value<double>(input, path, "scaling");
        if (!(input >> token)) {
            invalid_model(path, "missing species_count");
        }
    }
    if (token != "species_count") {
        invalid_model(path, "expected species_count");
    }
    expect_token(input, "=", path);
    species_count = read_value<int>(input, path, "species_count");
    if (species_count <= 0) {
        invalid_model(path, "species_count must be positive");
    }

    if (!(input >> token)) {
        invalid_model(path, "missing radial_basis_type");
    }
    if (token == "potential_tag") {
        std::string ignored;
        std::getline(input, ignored);
        if (!(input >> token)) {
            invalid_model(path, "missing radial_basis_type");
        }
    }
    if (token != "radial_basis_type") {
        invalid_model(path, "expected radial_basis_type");
    }
    expect_token(input, "=", path);
    radial_basis_type = read_value<std::string>(input, path, "radial_basis_type");
    if (radial_basis_type != "RBChebyshev" && radial_basis_type != "RBChebyshev_repuls") {
        invalid_model(path, "only RBChebyshev and RBChebyshev_repuls potentials are currently supported");
    }

    if (!(input >> token)) {
        invalid_model(path, "missing radial basis parameters");
    }
    if (token == "scaling") {
        expect_token(input, "=", path);
        scaling *= read_value<double>(input, path, "radial basis scaling");
        if (!(input >> token)) {
            invalid_model(path, "missing min_dist");
        }
    }
    if (token != "min_dist") {
        invalid_model(path, "expected min_dist");
    }
    expect_token(input, "=", path);
    min_dist = read_value<double>(input, path, "min_dist");
    read_assignment(input, path, "max_dist");
    max_dist = read_value<double>(input, path, "max_dist");
    read_assignment(input, path, "radial_basis_size");
    radial_basis_size = read_value<int>(input, path, "radial_basis_size");
    if (!std::isfinite(min_dist) || !std::isfinite(max_dist) || min_dist < 0.0
        || max_dist <= min_dist || radial_basis_size <= 0) {
        invalid_model(path, "invalid radial basis parameters");
    }
    read_assignment(input, path, "radial_funcs_count");
    radial_funcs_count = read_value<int>(input, path, "radial_funcs_count");
    if (radial_funcs_count <= 0) {
        invalid_model(path, "radial_funcs_count must be positive");
    }

    expect_token(input, "radial_coeffs", path);
    const std::size_t coeff_count = static_cast<std::size_t>(species_count)
        * static_cast<std::size_t>(species_count)
        * static_cast<std::size_t>(radial_funcs_count)
        * static_cast<std::size_t>(radial_basis_size);
    radial_coeffs.assign(coeff_count, 0.0);
    for (int central = 0; central < species_count; ++central) {
        for (int outer = 0; outer < species_count; ++outer) {
            const std::string pair = read_value<std::string>(input, path, "radial species pair");
            const std::string expected = std::to_string(central) + "-" + std::to_string(outer);
            if (pair != expected) {
                invalid_model(path, "unexpected radial coefficient block " + pair);
            }
            const std::size_t pair_offset = (
                static_cast<std::size_t>(central) * static_cast<std::size_t>(species_count)
                + static_cast<std::size_t>(outer))
                * static_cast<std::size_t>(radial_funcs_count)
                * static_cast<std::size_t>(radial_basis_size);
            for (int radial_function = 0; radial_function < radial_funcs_count; ++radial_function) {
                const auto row = read_double_row(input, path, radial_basis_size, "radial coefficients");
                std::copy(
                    row.begin(), row.end(),
                    radial_coeffs.begin() + pair_offset
                        + static_cast<std::size_t>(radial_function) * static_cast<std::size_t>(radial_basis_size));
            }
        }
    }

    read_assignment(input, path, "alpha_moments_count");
    alpha_moments_count = read_value<int>(input, path, "alpha_moments_count");
    read_assignment(input, path, "alpha_index_basic_count");
    const int alpha_basic_count = read_value<int>(input, path, "alpha_index_basic_count");
    if (alpha_moments_count <= 0 || alpha_basic_count <= 0) {
        invalid_model(path, "alpha counts must be positive");
    }
    read_tuple_array(input, path, "alpha_index_basic", static_cast<std::size_t>(alpha_basic_count), alpha_index_basic);
    read_assignment(input, path, "alpha_index_times_count");
    const int alpha_times_count = read_value<int>(input, path, "alpha_index_times_count");
    if (alpha_times_count < 0) {
        invalid_model(path, "alpha_index_times_count must not be negative");
    }
    read_tuple_array(input, path, "alpha_index_times", static_cast<std::size_t>(alpha_times_count), alpha_index_times);
    read_assignment(input, path, "alpha_scalar_moments");
    const int scalar_count = read_value<int>(input, path, "alpha_scalar_moments");
    if (scalar_count <= 0) {
        invalid_model(path, "alpha_scalar_moments must be positive");
    }
    read_int_array(input, path, "alpha_moment_mapping", static_cast<std::size_t>(scalar_count), alpha_moment_mapping);

    for (const auto& alpha : alpha_index_basic) {
        if (alpha[0] < 0 || alpha[0] >= radial_funcs_count
            || alpha[1] < 0 || alpha[2] < 0 || alpha[3] < 0) {
            invalid_model(path, "alpha_index_basic contains an out-of-range value");
        }
    }
    for (const auto& alpha : alpha_index_times) {
        if (alpha[0] < 0 || alpha[1] < 0 || alpha[0] >= alpha_moments_count
            || alpha[1] >= alpha_moments_count || alpha[3] < 0 || alpha[3] >= alpha_moments_count) {
            invalid_model(path, "alpha_index_times contains an out-of-range value");
        }
    }
    for (const int index : alpha_moment_mapping) {
        if (index < 0 || index >= alpha_moments_count) {
            invalid_model(path, "alpha_moment_mapping contains an out-of-range value");
        }
    }
    if (!std::isfinite(scaling)) {
        invalid_model(path, "scaling must be finite");
    }
}

std::int64_t mtp_feature_count(const MtpOptions& options) {
    // ponytail: rank traces and pair contractions are the current compact basis; MLIP alpha-index recursion is the upgrade path.
    return basic_count(options) + pair_count(options);
}

void compute_mtp(
    const StructureBatchView& batch,
    const MtpOptions& options,
    double* output,
    const std::shared_ptr<ComputeControl>& control
) {
    validate_batch(batch);
    validate_species(batch, options.species);
    if (!std::isfinite(options.min_dist) || !std::isfinite(options.max_dist)
        || options.min_dist < 0.0 || options.max_dist <= options.min_dist
        || options.radial_basis_size <= 0 || options.radial_funcs_count <= 0
        || options.max_rank < 0 || options.max_rank > 5 || options.num_threads < 0) {
        throw std::invalid_argument("invalid MTP parameters");
    }
    if (batch.atoms == 0) {
        return;
    }
    compute_mtp_impl(batch, options, output, control);
}

MtpCalculator::MtpCalculator(MtpOptions options) : options_(std::move(options)) {
    if (!options_.potential_path.empty()) {
        static std::mutex cache_mutex;
        static std::unordered_map<std::string, std::weak_ptr<OfficialMtpModel>> cache;
        const std::string cache_key = options_.model_digest.empty()
            ? options_.potential_path
            : options_.model_digest;
        std::lock_guard<std::mutex> cache_lock(cache_mutex);
        if (const auto found = cache.find(cache_key); found != cache.end()) {
            official_model_ = found->second.lock();
        }
        if (!official_model_) {
            official_model_ = std::make_shared<OfficialMtpModel>();
            official_model_->load(options_.potential_path);
            cache[cache_key] = official_model_;
        }
    } else {
        (void)mtp_feature_count(options_);
    }
}

std::int64_t MtpCalculator::feature_count() const noexcept {
    return official_model_ ? official_model_->feature_count() : mtp_feature_count(options_);
}
const std::vector<std::int32_t>& MtpCalculator::species() const noexcept { return options_.species; }
bool MtpCalculator::official_model() const noexcept { return static_cast<bool>(official_model_); }
bool MtpCalculator::official_mlip4() const noexcept {
    return official_model_ && official_model_->native_mlip4;
}
const std::string& MtpCalculator::official_format() const noexcept {
    static const std::string empty;
    static const std::string mlip2 = "MLIP-2";
    static const std::string mlip4 = "MLIP-4";
    if (!official_model_) return empty;
    return official_model_->native_mlip4 ? mlip4 : mlip2;
}
const std::vector<int>& MtpCalculator::official_alpha_moment_mapping() const noexcept {
    static const std::vector<int> empty;
    return official_model_ ? official_model_->alpha_moment_mapping : empty;
}
double MtpCalculator::official_min_dist() const noexcept {
    return official_model_ ? official_model_->min_dist : 0.0;
}
double MtpCalculator::official_max_dist() const noexcept {
    return official_model_ ? official_model_->max_dist : 0.0;
}
int MtpCalculator::official_radial_basis_size() const noexcept {
    return official_model_ ? official_model_->radial_basis_size : 0;
}
int MtpCalculator::official_radial_funcs_count() const noexcept {
    return official_model_ ? official_model_->radial_funcs_count : 0;
}
const std::string& MtpCalculator::official_radial_basis_type() const noexcept {
    static const std::string empty;
    return official_model_ ? official_model_->radial_basis_type : empty;
}
void MtpCalculator::close() noexcept { closed_.store(true, std::memory_order_release); }
bool MtpCalculator::closed() const noexcept { return closed_.load(std::memory_order_acquire); }

void MtpCalculator::compute(
    const StructureBatchView& batch,
    double* output,
    const std::shared_ptr<ComputeControl>& control
) const {
    if (closed()) {
        throw std::runtime_error("MTP calculator is closed");
    }
    std::lock_guard<std::mutex> lock(compute_mutex_);
    if (official_model_) {
        if (official_model_->native_mlip4) {
            official_model_->native_model->compute(batch, options_.species, options_.num_threads, output, control);
            return;
        }
        validate_batch(batch);
        validate_species(batch, options_.species);
        if (static_cast<int>(options_.species.size()) != official_model_->species_count) {
            throw std::invalid_argument("MLIP-2 potential species_count does not match calculator species");
        }
        if (batch.atoms == 0) {
            return;
        }
        compute_official_mtp_impl(batch, options_, *official_model_, output, control);
    } else {
        compute_mtp(batch, options_, output, control);
    }
}

} // namespace mdescriptor
