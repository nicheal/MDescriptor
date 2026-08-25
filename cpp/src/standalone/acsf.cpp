#include "mdescriptor/descriptor.hpp"
#include "mdescriptor/neighbor.hpp"
#include "descriptor_common.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <utility>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace mdescriptor {
using namespace detail;

namespace {
constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr double kSqrt2 = 1.414213562373095048801688724209698079;

std::int64_t acsf_features(const AcsfOptions& options) {
    const std::int64_t types = static_cast<std::int64_t>(options.species.size());
    return (1 + options.n_g2 + options.n_g3) * types
        + (options.n_g4 + options.n_g5) * types * (types + 1) / 2;
}

struct AcsfG4Plan {
    std::vector<double> eta_values;
    std::vector<std::pair<double, double>> angular_values;
    std::vector<std::size_t> eta_slots;
    std::vector<std::size_t> angular_slots;
};

AcsfG4Plan prepare_acsf_g4(const AcsfOptions& options) {
    AcsfG4Plan plan;
    plan.eta_slots.resize(static_cast<std::size_t>(options.n_g4));
    plan.angular_slots.resize(static_cast<std::size_t>(options.n_g4));
    for (std::int64_t p = 0; p < options.n_g4; ++p) {
        const double eta = options.g4_params[p * 3];
        const std::pair<double, double> angular{
            options.g4_params[p * 3 + 1],
            options.g4_params[p * 3 + 2],
        };
        auto eta_it = std::find(plan.eta_values.begin(), plan.eta_values.end(), eta);
        if (eta_it == plan.eta_values.end()) {
            plan.eta_slots[static_cast<std::size_t>(p)] = plan.eta_values.size();
            plan.eta_values.push_back(eta);
        } else {
            plan.eta_slots[static_cast<std::size_t>(p)] = static_cast<std::size_t>(eta_it - plan.eta_values.begin());
        }
        auto angular_it = std::find(plan.angular_values.begin(), plan.angular_values.end(), angular);
        if (angular_it == plan.angular_values.end()) {
            plan.angular_slots[static_cast<std::size_t>(p)] = plan.angular_values.size();
            plan.angular_values.push_back(angular);
        } else {
            plan.angular_slots[static_cast<std::size_t>(p)] = static_cast<std::size_t>(angular_it - plan.angular_values.begin());
        }
    }
    return plan;
}

void compute_acsf_structure(
    const StructureBatchView& batch,
    const AcsfOptions& options,
    const NeighborGraph& neighbor_graph,
    std::int64_t structure,
    double* output,
    const std::shared_ptr<ComputeControl>& control,
    bool parallel_centers,
    const AcsfG4Plan& g4_plan
) {
    const std::int64_t begin = batch.offsets[structure];
    const std::int64_t end = batch.offsets[structure + 1];
    const std::int64_t types = static_cast<std::int64_t>(options.species.size());
    const std::int64_t per_type = 1 + options.n_g2 + options.n_g3;
    const std::size_t feature_count = static_cast<std::size_t>(acsf_features(options));
    const auto mapping = species_map(options.species);

    auto cutoff = [&](double distance) {
        return 0.5 * (std::cos(kPi * distance / options.r_cut) + 1.0);
    };
    auto pair_index = [](std::int64_t a, std::int64_t b) {
        return a >= b ? a * (a + 1) / 2 + b : b * (b + 1) / 2 + a;
    };
    auto angular_power = [](double base, double zeta) {
        if (zeta == 1.0) {
            return base;
        }
        if (zeta == 2.0 && base >= 0.0) {
            return base * base;
        }
        return std::pow(base, zeta);
    };

    struct AcsfNeighbor {
        Vec3 displacement;
        double distance2;
        double distance;
        double cutoff;
        std::int64_t type;
    };
    thread_local std::vector<AcsfNeighbor> cached;
    thread_local std::vector<double> radial_values;
    thread_local std::vector<double> angular_values;
    const bool has_angular = options.n_g4 > 0 || options.n_g5 > 0;
    const bool cache_radial = g4_plan.eta_values.size() < static_cast<std::size_t>(options.n_g4);
    const bool cache_angular = g4_plan.angular_values.size() < static_cast<std::size_t>(options.n_g4);

#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(options.num_threads > 0 ? options.num_threads : omp_get_max_threads()) if(parallel_centers)
#endif
    for (std::int64_t center = begin; center < end; ++center) {
        if (cancelled(control)) {
            continue;
        }
        const std::int64_t row = center - begin;
        const NeighborView neighbors = neighbor_graph.for_center(center);
        double* values = output + row * feature_count;
        std::fill(values, values + feature_count, 0.0);
        if (has_angular) {
            cached.clear();
            cached.reserve(neighbors.size);
        }
        for (std::size_t j_index = 0; j_index < neighbors.size; ++j_index) {
            if (neighbors.exact_self(j_index, center)) {
                continue;
            }
            const auto j_atom = neighbors.atoms[j_index];
            const auto j_type_it = mapping.find(batch.numbers[j_atom]);
            if (j_type_it == mapping.end()) {
                throw std::invalid_argument("batch contains an atomic number outside calculator species");
            }
            const std::int64_t j_type = j_type_it->second;
            const double rij2 = neighbors.distance2[j_index];
            const double rij = std::sqrt(rij2);
            const double fc_ij = cutoff(rij);
            std::size_t offset = static_cast<std::size_t>(j_type * per_type);
            values[offset++] += fc_ij; // G1
            for (std::int64_t p = 0; p < options.n_g2; ++p) {
                const double eta = options.g2_params[p * 2];
                const double rs = options.g2_params[p * 2 + 1];
                values[offset++] += std::exp(-eta * (rij - rs) * (rij - rs)) * fc_ij;
            }
            for (std::int64_t p = 0; p < options.n_g3; ++p) {
                values[offset++] += std::cos(options.g3_params[p] * rij) * fc_ij;
            }
            if (has_angular) {
                cached.push_back({
                    Vec3{
                        neighbors.displacements[j_index * 3 + 0],
                        neighbors.displacements[j_index * 3 + 1],
                        neighbors.displacements[j_index * 3 + 2],
                    },
                    rij2,
                    rij,
                    fc_ij,
                    j_type,
                });
            }
        }

        const std::size_t angular_offset = static_cast<std::size_t>(types * per_type);
        for (std::size_t j_index = 0; j_index < cached.size(); ++j_index) {
            const AcsfNeighbor& j = cached[j_index];
            for (std::size_t k_index = 0; k_index < j_index; ++k_index) {
                const AcsfNeighbor& k = cached[k_index];
                const double rjk2 = norm2(j.displacement - k.displacement);
                if (j.distance == 0.0 || k.distance == 0.0) {
                    continue;
                }
                const double rjk = std::sqrt(rjk2);
                const double cosine = dot(j.displacement, k.displacement) / (j.distance * k.distance);
                const double fc4 = j.cutoff * k.cutoff * (rjk <= options.r_cut ? cutoff(rjk) : 0.0);
                const double fc5 = j.cutoff * k.cutoff;
                const double distance_sum = j.distance2 + k.distance2 + rjk2;
                if (cache_radial && rjk <= options.r_cut) {
                    radial_values.resize(g4_plan.eta_values.size());
                    for (std::size_t slot = 0; slot < g4_plan.eta_values.size(); ++slot) {
                        radial_values[slot] = std::exp(-g4_plan.eta_values[slot] * distance_sum);
                    }
                }
                if (cache_angular) {
                    angular_values.resize(g4_plan.angular_values.size());
                    for (std::size_t slot = 0; slot < g4_plan.angular_values.size(); ++slot) {
                        const auto [zeta, lambda] = g4_plan.angular_values[slot];
                        angular_values[slot] = angular_power(0.5 * (1.0 + lambda * cosine), zeta);
                    }
                }
                const std::size_t offset = angular_offset
                    + static_cast<std::size_t>(pair_index(j.type, k.type) * (options.n_g4 + options.n_g5));
                for (std::int64_t p = 0; p < options.n_g4; ++p) {
                    const double eta = options.g4_params[p * 3];
                    const double zeta = options.g4_params[p * 3 + 1];
                    const double lambda = options.g4_params[p * 3 + 2];
                    const double angular = rjk <= options.r_cut && cache_angular
                        ? angular_values[g4_plan.angular_slots[static_cast<std::size_t>(p)]]
                        : rjk <= options.r_cut ? angular_power(0.5 * (1.0 + lambda * cosine), zeta) : 0.0;
                    const double radial = rjk <= options.r_cut && cache_radial
                        ? radial_values[g4_plan.eta_slots[static_cast<std::size_t>(p)]]
                        : rjk <= options.r_cut ? std::exp(-eta * distance_sum) : 0.0;
                    values[offset + static_cast<std::size_t>(p)] +=
                        2.0 * angular * radial * fc4;
                }
                for (std::int64_t p = 0; p < options.n_g5; ++p) {
                    const double eta = options.g5_params[p * 3];
                    const double zeta = options.g5_params[p * 3 + 1];
                    const double lambda = options.g5_params[p * 3 + 2];
                    values[offset + static_cast<std::size_t>(options.n_g4 + p)] +=
                        2.0 * angular_power(0.5 * (1.0 + lambda * cosine), zeta)
                        * std::exp(-eta * (j.distance2 + k.distance2)) * fc5;
                }
            }
        }
    }
    if (parallel_centers && cancelled(control)) {
        throw CancelledError();
    }
}
} // namespace

void compute_acsf(const StructureBatchView& batch, const AcsfOptions& options, double* output, const std::shared_ptr<ComputeControl>& control) {
    validate_batch(batch);
    validate_species(batch, options.species);
    if (!std::isfinite(options.r_cut) || options.r_cut <= 0.0 || options.n_g2 < 0 || options.n_g3 < 0
        || options.n_g4 < 0 || options.n_g5 < 0
        || options.g2_params.size() != static_cast<std::size_t>(options.n_g2 * 2)
        || options.g3_params.size() != static_cast<std::size_t>(options.n_g3)
        || options.g4_params.size() != static_cast<std::size_t>(options.n_g4 * 3)
        || options.g5_params.size() != static_cast<std::size_t>(options.n_g5 * 3)
        || options.num_threads < 0) {
        throw std::invalid_argument("invalid ACSF parameters");
    }
    for (double value : options.g2_params) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument("ACSF parameters must be finite");
        }
    }
    for (std::int64_t p = 0; p < options.n_g2; ++p) {
        if (options.g2_params[static_cast<std::size_t>(p * 2)] <= 0.0) {
            throw std::invalid_argument("ACSF G2 eta parameters must be positive");
        }
    }
    for (double value : options.g4_params) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument("ACSF parameters must be finite");
        }
    }
    for (std::int64_t p = 0; p < options.n_g4; ++p) {
        const std::size_t offset = static_cast<std::size_t>(p * 3);
        if (options.g4_params[offset] <= 0.0 || options.g4_params[offset + 1] <= 0.0) {
            throw std::invalid_argument("ACSF G4 eta and zeta parameters must be positive");
        }
    }
    for (double value : options.g3_params) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument("ACSF parameters must be finite");
        }
    }
    for (double value : options.g5_params) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument("ACSF parameters must be finite");
        }
    }
    for (std::int64_t p = 0; p < options.n_g5; ++p) {
        const std::size_t offset = static_cast<std::size_t>(p * 3);
        if (options.g5_params[offset] <= 0.0 || options.g5_params[offset + 1] <= 0.0) {
            throw std::invalid_argument("ACSF G5 eta and zeta parameters must be positive");
        }
    }
    if (control) {
        control->reset(batch.structures);
    }
    const AcsfG4Plan g4_plan = prepare_acsf_g4(options);
    const std::int64_t features = acsf_features(options);
    if (batch.structures == 1) {
        const NeighborGraph neighbor_graph = build_neighbor_graph(batch, options.r_cut, control, options.num_threads);
        compute_acsf_structure(batch, options, neighbor_graph, 0, output, control, true, g4_plan);
        if (control && control->cancelled()) {
            throw CancelledError();
        }
        if (control) {
            control->mark_completed();
        }
        return;
    }
    run_parallel_structures(batch.structures, options.num_threads, control, [&](std::int64_t s) {
        const std::int64_t begin = batch.offsets[s];
        const std::int64_t end = batch.offsets[s + 1];
        const std::int64_t offsets[2] = {0, end - begin};
        const StructureBatchView structure_batch{
            batch.numbers + begin,
            batch.positions + begin * 3,
            batch.cells + s * 9,
            batch.pbc + s * 3,
            offsets,
            1,
            end - begin,
        };
        const NeighborGraph neighbor_graph = build_neighbor_graph(
            structure_batch, options.r_cut, control, 1);
        compute_acsf_structure(
            structure_batch, options, neighbor_graph, 0,
            output + begin * features, control, false, g4_plan);
    });
}

AcsfCalculator::AcsfCalculator(AcsfOptions options) : options_(std::move(options)) {}
std::int64_t AcsfCalculator::feature_count() const noexcept { return acsf_features(options_); }
const std::vector<std::int32_t>& AcsfCalculator::species() const noexcept { return options_.species; }
void AcsfCalculator::close() noexcept { closed_.store(true, std::memory_order_release); }
bool AcsfCalculator::closed() const noexcept { return closed_.load(std::memory_order_acquire); }
void AcsfCalculator::compute(const StructureBatchView& batch, double* output, const std::shared_ptr<ComputeControl>& control) const {
    std::lock_guard<std::mutex> lock(compute_mutex_);
    if (closed()) {
        throw std::runtime_error("ACSF calculator is closed");
    }
    compute_acsf(batch, options_, output, control);
}
} // namespace mdescriptor
