#include "mdescriptor/extra.hpp"
#include "mdescriptor/neighbor.hpp"
#include "descriptor_common.hpp"
#include "extra_common.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

namespace mdescriptor {
using namespace detail;

std::int64_t ead_feature_count(const EadOptions& options) {
    return static_cast<std::int64_t>(options.max_degree + 1) * options.eta.size() * options.rs.size();
}

void compute_ead(
    const StructureBatchView& batch,
    const EadOptions& options,
    double* output,
    const std::shared_ptr<ComputeControl>& control) {
    validate_batch(batch);
    if (options.max_degree < 0 || options.cutoff <= 0.0 || options.eta.empty() || options.rs.empty()) {
        throw std::invalid_argument("invalid EAD parameters");
    }
    const auto features = ead_feature_count(options);
    std::fill(output, output + batch.atoms * features, 0.0);
    check_cancelled(control);
    if (control) {
        control->reset(batch.structures);
    }
    if (options.num_threads < 0) {
        throw std::invalid_argument("num_threads must be non-negative");
    }
    const NeighborGraph graph = build_neighbor_graph(batch, options.cutoff, control, options.num_threads);
    std::vector<std::array<int, 4>> powers;
    for (int degree = 0; degree <= options.max_degree; ++degree) {
        for (int lx = 0; lx <= degree; ++lx) {
            for (int ly = 0; ly <= degree - lx; ++ly) {
                powers.push_back({lx, ly, degree - lx - ly, degree});
            }
        }
    }
    auto compute_structure = [&](std::int64_t structure) {
        const std::int64_t begin = batch.offsets[structure];
        const std::int64_t end = batch.offsets[structure + 1];
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(options.num_threads > 0 ? options.num_threads : omp_get_max_threads()) if(!omp_in_parallel())
#endif
        for (std::int64_t center = begin; center < end; ++center) {
            if (cancelled(control)) {
                continue;
            }
            std::vector<double> terms(powers.size() * options.eta.size() * options.rs.size(), 0.0);
            const NeighborView neighbors = graph.for_center(center);
            for (std::size_t index = 0; index < neighbors.size; ++index) {
                if (neighbors.exact_self(index, center)) {
                    continue;
                }
                const Vec3 vector{neighbors.displacements[index * 3], neighbors.displacements[index * 3 + 1], neighbors.displacements[index * 3 + 2]};
                const double distance = std::sqrt(std::max(0.0, neighbors.distance2[index]));
                if (distance <= 0.0 || distance >= options.cutoff) {
                    continue;
                }
                const double smooth = 0.5 * (1.0 + std::cos(kPi * distance / options.cutoff));
                const double z = static_cast<double>(batch.numbers[neighbors.atoms[index]]);
                for (std::size_t power = 0; power < powers.size(); ++power) {
                    const auto [lx, ly, lz, degree] = powers[power];
                    const double monomial = std::pow(vector.x, lx) * std::pow(vector.y, ly) * std::pow(vector.z, lz)
                        / std::sqrt(std::tgamma(lx + 1.0) * std::tgamma(ly + 1.0) * std::tgamma(lz + 1.0));
                    for (std::size_t eta = 0; eta < options.eta.size(); ++eta) {
                        for (std::size_t rs = 0; rs < options.rs.size(); ++rs) {
                            const std::size_t offset = (power * options.eta.size() + eta) * options.rs.size() + rs;
                            terms[offset] += monomial * std::exp(-options.eta[eta] * (distance - options.rs[rs]) * (distance - options.rs[rs])) * z * smooth;
                        }
                    }
                }
            }
            double* target = output + center * features;
            std::int64_t offset = 0;
            for (int degree = 0; degree <= options.max_degree; ++degree) {
                for (std::size_t eta = 0; eta < options.eta.size(); ++eta) {
                    for (std::size_t rs = 0; rs < options.rs.size(); ++rs) {
                        double sum = 0.0;
                        for (std::size_t power = 0; power < powers.size(); ++power) {
                            if (powers[power][3] <= degree) {
                                const auto value = terms[(power * options.eta.size() + eta) * options.rs.size() + rs];
                                sum += value * value;
                            }
                        }
                        target[offset++] = std::tgamma(degree + 1.0) * sum / std::pow(options.cutoff, 2.0 * degree);
                    }
                }
            }
        }
    };
    if (batch.structures == 1) {
        compute_structure(0);
        check_cancelled(control);
        mark_completed(control);
        return;
    }
    run_parallel_structures(batch.structures, options.num_threads, control, compute_structure);
}
} // namespace mdescriptor
