#include "mdescriptor/extra.hpp"
#include "mdescriptor/neighbor.hpp"
#include "extra_common.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace mdescriptor {
using namespace detail;

namespace {
double gaussian_bin(double value, double weight, const MBTROptions& options, int bin) {
    const double dx = (options.grid_max - options.grid_min) / (options.grid_n - 1);
    const double lower = options.grid_min - 0.5 * dx + bin * dx;
    const double upper = lower + dx;
    const double sigma_root = options.grid_sigma * std::sqrt(2.0);
    double result = 0.5 * (std::erf((upper - value) / sigma_root) - std::erf((lower - value) / sigma_root)) / dx;
    if (!options.normalize_gaussians) {
        result *= options.grid_sigma * std::sqrt(2.0 * kPi);
    }
    return weight * result;
}

void add_histogram(double* target, double value, double weight, const MBTROptions& options) {
    if (weight == 0.0 || value < options.grid_min - options.grid_sigma * 8.0 || value > options.grid_max + options.grid_sigma * 8.0) {
        return;
    }
    for (int bin = 0; bin < options.grid_n; ++bin) {
        target[bin] += gaussian_bin(value, weight, options, bin);
    }
}

double mbtr_weight(const MBTROptions& options, double first, double second, double third = 0.0) {
    switch (options.weighting) {
    case MBTRWeighting::Exponential: {
        const double value = std::exp(-options.scale * (first + second + third));
        return value >= options.threshold ? value : 0.0;
    }
    case MBTRWeighting::InverseSquare:
        return first <= options.r_cut ? 1.0 / std::max(first * first, 1e-30) : 0.0;
    case MBTRWeighting::SmoothCutoff: {
        auto smooth = [&](double distance) {
            const double x = std::clamp(distance / options.r_cut, 0.0, 1.0);
            return 1.0 + options.sharpness * std::pow(x, options.sharpness + 1.0)
                - (options.sharpness + 1.0) * std::pow(x, options.sharpness);
        };
        return first <= options.r_cut && second <= options.r_cut ? smooth(first) * smooth(second) : 0.0;
    }
    default:
        return 1.0;
    }
}

int pair_channel(int first, int second, int species_count) {
    const int lower = std::min(first, second);
    const int upper = std::max(first, second);
    return lower * species_count - lower * (lower + 1) / 2 + upper;
}

double cell_volume(const double* cell) {
    return std::abs(
        cell[0] * (cell[4] * cell[8] - cell[5] * cell[7])
        - cell[1] * (cell[3] * cell[8] - cell[5] * cell[6])
        + cell[2] * (cell[3] * cell[7] - cell[4] * cell[6]));
}

void normalize_histogram(
    double* values,
    std::int64_t size,
    const MBTROptions& options,
    int atom_count,
    const std::vector<int>& species_counts,
    double volume) {
    if (options.normalization == MBTRNormalization::L2) {
        double sum = 0.0;
        for (std::int64_t index = 0; index < size; ++index) {
            sum += values[index] * values[index];
        }
        const double scale = std::sqrt(sum);
        if (scale > 0.0) {
            for (std::int64_t index = 0; index < size; ++index) {
                values[index] /= scale;
            }
        }
    } else if (options.normalization == MBTRNormalization::NAtoms && atom_count > 0) {
        for (std::int64_t index = 0; index < size; ++index) {
            values[index] /= atom_count;
        }
    } else if (options.normalization == MBTRNormalization::ValleOganov && !options.local
               && options.geometry != MBTRGeometry::AtomicNumber) {
        const int species_count = static_cast<int>(species_counts.size());
        const int pair_count = species_count * (species_count + 1) / 2;
        if (options.geometry == MBTRGeometry::Distance || options.geometry == MBTRGeometry::InverseDistance) {
            for (int first = 0; first < species_count; ++first) {
                for (int second = first; second < species_count; ++second) {
                    const double count_product = first == second
                        ? 0.5 * species_counts[first] * species_counts[second]
                        : static_cast<double>(species_counts[first]) * species_counts[second];
                    if (count_product <= 0.0) {
                        continue;
                    }
                    const double factor = volume / (count_product * 4.0 * kPi);
                    const std::int64_t begin = static_cast<std::int64_t>(pair_channel(first, second, species_count)) * options.grid_n;
                    for (int bin = 0; bin < options.grid_n; ++bin) {
                        values[begin + bin] *= factor;
                    }
                }
            }
        } else {
            for (int first = 0; first < species_count; ++first) {
                for (int center = 0; center < species_count; ++center) {
                    for (int third = first; third < species_count; ++third) {
                        const double count_product = static_cast<double>(species_counts[first])
                            * species_counts[center] * species_counts[third];
                        if (count_product <= 0.0) {
                            continue;
                        }
                        const int channel = center * pair_count + pair_channel(first, third, species_count);
                        const double factor = volume / count_product;
                        const std::int64_t begin = static_cast<std::int64_t>(channel) * options.grid_n;
                        for (int bin = 0; bin < options.grid_n; ++bin) {
                            values[begin + bin] *= factor;
                        }
                    }
                }
            }
        }
    }
}
} // namespace

std::int64_t mbtr_feature_count(const MBTROptions& options) {
    const auto count = static_cast<std::int64_t>(options.species.size());
    const auto pair = count * (count + 1) / 2;
    if (options.geometry == MBTRGeometry::AtomicNumber) {
        return count * options.grid_n;
    }
    if (options.local) {
        const auto channels = options.geometry == MBTRGeometry::Distance || options.geometry == MBTRGeometry::InverseDistance
            ? count + 1
            : (count + 1) * (3 * (count + 1) - 1) / 2;
        return channels * options.grid_n;
    }
    return (options.geometry == MBTRGeometry::Distance || options.geometry == MBTRGeometry::InverseDistance
        ? pair
        : count * pair) * options.grid_n;
}

void compute_mbtr(
    const StructureBatchView& batch,
    const MBTROptions& options,
    double* output,
    const std::shared_ptr<ComputeControl>& control) {
    validate_batch(batch);
    validate_species(batch, options.species);
    if (options.grid_n < 2 || options.grid_max <= options.grid_min || options.grid_sigma <= 0.0) {
        throw std::invalid_argument("invalid MBTR grid");
    }
    const auto mapping = type_map(options.species);
    const std::int64_t features = mbtr_feature_count(options);
    const std::int64_t rows = options.local ? batch.atoms : batch.structures;
    std::fill(output, output + rows * features, 0.0);
    check_cancelled(control);
    if (control) {
        control->reset(batch.structures);
    }
    double cutoff = options.r_cut > 0.0 ? options.r_cut
        : (options.weighting == MBTRWeighting::Unity ? options.grid_max : 0.0);
    if (options.weighting == MBTRWeighting::Exponential && options.scale > 0.0) {
        const double multiplier = options.geometry == MBTRGeometry::Angle || options.geometry == MBTRGeometry::Cosine ? 0.5 : 1.0;
        cutoff = std::max(cutoff, multiplier * -std::log(options.threshold) / options.scale);
    }
    if (cutoff <= 0.0) {
        cutoff = options.grid_max;
    }
    const int species_count = static_cast<int>(options.species.size());
    const int pair_count = species_count * (species_count + 1) / 2;
    NeighborGraph graph;
    if (options.geometry != MBTRGeometry::AtomicNumber) {
        // reference implementation's periodic extension uses a strict distance cutoff. The shared
        // neighbor graph keeps its inclusive boundary for the other descriptors.
        graph = build_neighbor_graph(batch, cutoff, control, 0, true, true);
    }
    const bool needs_species_counts = options.normalization == MBTRNormalization::ValleOganov
        && !options.local && options.geometry != MBTRGeometry::AtomicNumber;
    for (std::int64_t structure = 0; structure < batch.structures; ++structure) {
        check_cancelled(control);
        const std::int64_t begin = batch.offsets[structure];
        const std::int64_t end = batch.offsets[structure + 1];
        const int atom_count = static_cast<int>(end - begin);
        std::vector<int> species_counts;
        double volume = 0.0;
        if (needs_species_counts) {
            species_counts.assign(species_count, 0);
            for (std::int64_t atom = begin; atom < end; ++atom) {
                ++species_counts[static_cast<std::size_t>(mapping.at(batch.numbers[atom]))];
            }
            volume = cell_volume(batch.cells + structure * 9);
        }
        if (options.geometry == MBTRGeometry::AtomicNumber) {
            double* target = output + structure * features;
            for (std::int64_t atom = begin; atom < end; ++atom) {
                const int channel = static_cast<int>(mapping.at(batch.numbers[atom]));
                add_histogram(target + channel * options.grid_n, static_cast<double>(batch.numbers[atom]), 1.0, options);
            }
            normalize_histogram(target, features, options, atom_count, species_counts, volume);
        } else if (!options.local) {
            double* target = output + structure * features;
            for (int center = 0; center < atom_count; ++center) {
                const NeighborView neighbors = graph.for_center(begin + center);
                for (std::size_t first_index = 0; first_index < neighbors.size; ++first_index) {
                    const int first_atom = neighbors.atoms[first_index];
                    const bool first_periodic = neighbors.shifts[first_index * 3] != 0 || neighbors.shifts[first_index * 3 + 1] != 0 || neighbors.shifts[first_index * 3 + 2] != 0;
                    const double first_distance = std::sqrt(std::max(0.0, neighbors.distance2[first_index]));
                    if (first_distance <= 1e-12) {
                        continue;
                    }
                    if (options.geometry != MBTRGeometry::Angle && options.geometry != MBTRGeometry::Cosine
                        && !first_periodic && first_atom < begin + center) {
                        continue;
                    }
                    const int first_type = static_cast<int>(mapping.at(batch.numbers[first_atom]));
                    if (options.geometry == MBTRGeometry::Distance || options.geometry == MBTRGeometry::InverseDistance) {
                        const double weight = mbtr_weight(options, first_distance, 0.0) * (first_periodic ? 0.5 : 1.0);
                        const double value = options.geometry == MBTRGeometry::Distance ? first_distance : 1.0 / first_distance;
                        add_histogram(target + pair_channel(static_cast<int>(mapping.at(batch.numbers[begin + center])), first_type, species_count) * options.grid_n, value, weight, options);
                    }
                    if (options.geometry != MBTRGeometry::Angle && options.geometry != MBTRGeometry::Cosine) {
                        continue;
                    }
                    for (std::size_t second_index = first_index + 1; second_index < neighbors.size; ++second_index) {
                        const int second_atom = neighbors.atoms[second_index];
                        const double second_distance = std::sqrt(std::max(0.0, neighbors.distance2[second_index]));
                        if (second_distance <= 1e-12) {
                            continue;
                        }
                        const Vec3 first_vector{neighbors.displacements[first_index * 3], neighbors.displacements[first_index * 3 + 1], neighbors.displacements[first_index * 3 + 2]};
                        const Vec3 second_vector{neighbors.displacements[second_index * 3], neighbors.displacements[second_index * 3 + 1], neighbors.displacements[second_index * 3 + 2]};
                        const double third_distance = norm(first_vector - second_vector);
                        // Reuse the stored radial squares and keep the opposite side
                        // opaque so fast-math cannot reduce the formula to dot/acos.
                        const volatile double cosine_third_distance = third_distance;
                        const double cosine = std::clamp(
                            (neighbors.distance2[first_index] + neighbors.distance2[second_index]
                                - cosine_third_distance * cosine_third_distance)
                                / (2.0 * first_distance * second_distance),
                            -1.0, 1.0);
                        const double value = options.geometry == MBTRGeometry::Cosine ? cosine : std::acos(cosine) * 180.0 / kPi;
                        const double weight = mbtr_weight(options, first_distance, second_distance, third_distance);
                        const int first_type = static_cast<int>(mapping.at(batch.numbers[first_atom]));
                        const int second_type = static_cast<int>(mapping.at(batch.numbers[second_atom]));
                        const int channel = static_cast<int>(mapping.at(batch.numbers[begin + center])) * pair_count + pair_channel(first_type, second_type, species_count);
                        add_histogram(target + channel * options.grid_n, value, weight, options);
                    }
                }
            }
            normalize_histogram(target, features, options, atom_count, species_counts, volume);
        } else {
            for (int center = 0; center < atom_count; ++center) {
                double* target = output + (begin + center) * features;
                const NeighborView neighbors = graph.for_center(begin + center);
                if (options.geometry == MBTRGeometry::Distance || options.geometry == MBTRGeometry::InverseDistance) {
                    for (std::size_t index = 0; index < neighbors.size; ++index) {
                        const double distance = std::sqrt(std::max(0.0, neighbors.distance2[index]));
                        if (distance <= 1e-12 || (neighbors.atoms[index] == begin + center && neighbors.exact_self(index, begin + center))) {
                            continue;
                        }
                        const int channel = static_cast<int>(mapping.at(batch.numbers[neighbors.atoms[index]])) + 1;
                        const double value = options.geometry == MBTRGeometry::Distance ? distance : 1.0 / distance;
                        add_histogram(target + channel * options.grid_n, value, mbtr_weight(options, distance, 0.0), options);
                    }
                } else {
                    const int element_count = species_count + 1; // Include reference implementation's ghost center X.
                    const int reserved = element_count * (element_count + 1) / 2;
                    auto add_angle = [&](int channel, double first_distance, double second_distance, double opposite_distance, double weight) {
                        const double denominator = 2.0 * first_distance * second_distance;
                        const double cosine = denominator > 0.0
                            ? std::clamp(
                                (first_distance * first_distance + second_distance * second_distance
                                    - opposite_distance * opposite_distance) / denominator,
                                -1.0, 1.0)
                            : 1.0;
                        const double value = options.geometry == MBTRGeometry::Cosine
                            ? cosine : std::acos(cosine) * 180.0 / kPi;
                        add_histogram(target + channel * options.grid_n, value, weight, options);
                    };
                    for (std::size_t first_index = 0; first_index < neighbors.size; ++first_index) {
                        const double first_distance = std::sqrt(std::max(0.0, neighbors.distance2[first_index]));
                        if (first_distance <= 1e-12 || neighbors.exact_self(first_index, begin + center)) {
                            continue;
                        }
                        for (std::size_t second_index = first_index + 1; second_index < neighbors.size; ++second_index) {
                            const double second_distance = std::sqrt(std::max(0.0, neighbors.distance2[second_index]));
                            if (second_distance <= 1e-12 || neighbors.exact_self(second_index, begin + center)) {
                                continue;
                            }
                            const Vec3 first_vector{neighbors.displacements[first_index * 3], neighbors.displacements[first_index * 3 + 1], neighbors.displacements[first_index * 3 + 2]};
                            const Vec3 second_vector{neighbors.displacements[second_index * 3], neighbors.displacements[second_index * 3 + 1], neighbors.displacements[second_index * 3 + 2]};
                            const double third_distance = norm(first_vector - second_vector);
                            const double weight = mbtr_weight(options, first_distance, second_distance, third_distance);
                            const int first_type = static_cast<int>(mapping.at(batch.numbers[neighbors.atoms[first_index]])) + 1;
                            const int second_type = static_cast<int>(mapping.at(batch.numbers[neighbors.atoms[second_index]])) + 1;

                            // The center atom is the angle vertex: (first, X, second).
                            add_angle(
                                pair_channel(first_type, second_type, element_count),
                                first_distance, second_distance, third_distance, weight);
                            // The center atom is an endpoint: (X, first, second) and (X, second, first).
                            add_angle(
                                reserved + (first_type - 1) * element_count + second_type,
                                first_distance, third_distance, second_distance, weight);
                            add_angle(
                                reserved + (second_type - 1) * element_count + first_type,
                                second_distance, third_distance, first_distance, weight);
                        }
                    }
                }
                normalize_histogram(target, features, options, atom_count, species_counts, volume);
            }
        }
        mark_completed(control);
    }
}
} // namespace mdescriptor
