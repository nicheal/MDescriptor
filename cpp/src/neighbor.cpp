#include "mdescriptor/neighbor.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace mdescriptor {
namespace {

struct Vec3 {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};

Vec3 operator+(Vec3 a, Vec3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
Vec3 operator-(Vec3 a, Vec3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
Vec3 operator*(double scale, Vec3 value) { return {scale * value.x, scale * value.y, scale * value.z}; }
double norm2(Vec3 value) { return value.x * value.x + value.y * value.y + value.z * value.z; }

struct Mat3 {
    double a[3][3]{};
};

Vec3 row(const Mat3& matrix, int index) {
    return {matrix.a[index][0], matrix.a[index][1], matrix.a[index][2]};
}

Mat3 load_cell(const StructureBatchView& batch, std::int64_t structure) {
    Mat3 result;
    const double* source = batch.cells + structure * 9;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            result.a[i][j] = source[i * 3 + j];
        }
    }
    return result;
}

double determinant(const Mat3& matrix) {
    return matrix.a[0][0] * (matrix.a[1][1] * matrix.a[2][2] - matrix.a[1][2] * matrix.a[2][1])
         - matrix.a[0][1] * (matrix.a[1][0] * matrix.a[2][2] - matrix.a[1][2] * matrix.a[2][0])
         + matrix.a[0][2] * (matrix.a[1][0] * matrix.a[2][1] - matrix.a[1][1] * matrix.a[2][0]);
}

Mat3 inverse(const Mat3& matrix) {
    const double det = determinant(matrix);
    if (!std::isfinite(det) || std::abs(det) < 1e-14) {
        throw std::invalid_argument("cell matrix is singular");
    }
    Mat3 result;
    result.a[0][0] = (matrix.a[1][1] * matrix.a[2][2] - matrix.a[1][2] * matrix.a[2][1]) / det;
    result.a[0][1] = (matrix.a[0][2] * matrix.a[2][1] - matrix.a[0][1] * matrix.a[2][2]) / det;
    result.a[0][2] = (matrix.a[0][1] * matrix.a[1][2] - matrix.a[0][2] * matrix.a[1][1]) / det;
    result.a[1][0] = (matrix.a[1][2] * matrix.a[2][0] - matrix.a[1][0] * matrix.a[2][2]) / det;
    result.a[1][1] = (matrix.a[0][0] * matrix.a[2][2] - matrix.a[0][2] * matrix.a[2][0]) / det;
    result.a[1][2] = (matrix.a[0][2] * matrix.a[1][0] - matrix.a[0][0] * matrix.a[1][2]) / det;
    result.a[2][0] = (matrix.a[1][0] * matrix.a[2][1] - matrix.a[1][1] * matrix.a[2][0]) / det;
    result.a[2][1] = (matrix.a[0][1] * matrix.a[2][0] - matrix.a[0][0] * matrix.a[2][1]) / det;
    result.a[2][2] = (matrix.a[0][0] * matrix.a[1][1] - matrix.a[0][1] * matrix.a[1][0]) / det;
    return result;
}

Vec3 position(const StructureBatchView& batch, std::int64_t atom) {
    const double* source = batch.positions + atom * 3;
    return {source[0], source[1], source[2]};
}

Vec3 fractional_position(Vec3 value, const Mat3& inverse_cell) {
    return {
        value.x * inverse_cell.a[0][0] + value.y * inverse_cell.a[1][0] + value.z * inverse_cell.a[2][0],
        value.x * inverse_cell.a[0][1] + value.y * inverse_cell.a[1][1] + value.z * inverse_cell.a[2][1],
        value.x * inverse_cell.a[0][2] + value.y * inverse_cell.a[1][2] + value.z * inverse_cell.a[2][2],
    };
}

Vec3 cartesian_position(Vec3 value, const Mat3& cell) {
    return value.x * row(cell, 0) + value.y * row(cell, 1) + value.z * row(cell, 2);
}

struct ExtendedAtom {
    std::int32_t atom = 0;
    std::int32_t shift[3] = {};
    Vec3 position;
};

struct CellGrid {
    Vec3 minimum;
    Vec3 maximum;
    Vec3 spacing;
    std::array<int, 3> dimensions{1, 1, 1};
    std::vector<std::int32_t> offsets;
    std::vector<std::int32_t> atoms;

    int index(int x, int y, int z) const noexcept {
        return x + dimensions[0] * (y + dimensions[1] * z);
    }

    int coordinate(double value, int axis) const noexcept {
        const double origin = axis == 0 ? minimum.x : axis == 1 ? minimum.y : minimum.z;
        const double width = axis == 0 ? spacing.x : axis == 1 ? spacing.y : spacing.z;
        const int dimension = dimensions[axis];
        const int cell = static_cast<int>((value - origin) / width);
        return std::max(0, std::min(dimension - 1, cell));
    }
};

CellGrid build_grid(const std::vector<ExtendedAtom>& extended, double cutoff) {
    CellGrid grid;
    if (extended.empty()) {
        return grid;
    }
    grid.minimum = grid.maximum = extended.front().position;
    for (const ExtendedAtom& atom : extended) {
        grid.minimum.x = std::min(grid.minimum.x, atom.position.x);
        grid.minimum.y = std::min(grid.minimum.y, atom.position.y);
        grid.minimum.z = std::min(grid.minimum.z, atom.position.z);
        grid.maximum.x = std::max(grid.maximum.x, atom.position.x);
        grid.maximum.y = std::max(grid.maximum.y, atom.position.y);
        grid.maximum.z = std::max(grid.maximum.z, atom.position.z);
    }
    constexpr double padding = 1e-10;
    grid.minimum.x -= padding;
    grid.minimum.y -= padding;
    grid.minimum.z -= padding;
    grid.maximum.x += padding;
    grid.maximum.y += padding;
    grid.maximum.z += padding;
    const double ranges[3] = {
        grid.maximum.x - grid.minimum.x,
        grid.maximum.y - grid.minimum.y,
        grid.maximum.z - grid.minimum.z,
    };
    double* spacing[3] = {&grid.spacing.x, &grid.spacing.y, &grid.spacing.z};
    for (int axis = 0; axis < 3; ++axis) {
        grid.dimensions[axis] = std::max(1, static_cast<int>(ranges[axis] / cutoff));
        *spacing[axis] = std::max(cutoff, ranges[axis] / grid.dimensions[axis]);
    }

    const std::size_t cell_count = static_cast<std::size_t>(grid.dimensions[0])
        * static_cast<std::size_t>(grid.dimensions[1])
        * static_cast<std::size_t>(grid.dimensions[2]);
    std::vector<std::int32_t> counts(cell_count, 0);
    for (const ExtendedAtom& atom : extended) {
        const int cell = grid.index(
            grid.coordinate(atom.position.x, 0),
            grid.coordinate(atom.position.y, 1),
            grid.coordinate(atom.position.z, 2));
        ++counts[static_cast<std::size_t>(cell)];
    }
    grid.offsets.resize(cell_count + 1, 0);
    for (std::size_t cell = 0; cell < cell_count; ++cell) {
        grid.offsets[cell + 1] = grid.offsets[cell] + counts[cell];
    }
    grid.atoms.resize(extended.size());
    std::vector<std::int32_t> fill(cell_count, 0);
    for (std::size_t index = 0; index < extended.size(); ++index) {
        const ExtendedAtom& atom = extended[index];
        const int cell = grid.index(
            grid.coordinate(atom.position.x, 0),
            grid.coordinate(atom.position.y, 1),
            grid.coordinate(atom.position.z, 2));
        grid.atoms[static_cast<std::size_t>(grid.offsets[cell] + fill[cell]++)] = static_cast<std::int32_t>(index);
    }
    return grid;
}

struct LocalGraph {
    std::vector<std::int64_t> counts;
    std::vector<std::int32_t> atoms;
    std::vector<std::int32_t> shifts;
    std::vector<double> displacements;
    std::vector<double> distance2;
};

LocalGraph build_structure_graph(
    const StructureBatchView& batch,
    std::int64_t structure,
    double cutoff,
    const std::shared_ptr<ComputeControl>& control,
    int num_threads,
    bool include_boundary,
    bool use_scaled_periodic_images,
    bool store_shifts) {
    const std::int64_t begin = batch.offsets[structure];
    const std::int64_t end = batch.offsets[structure + 1];
    const std::int64_t atom_count = end - begin;
    const Mat3 cell = load_cell(batch, structure);
    const Mat3 inv = inverse(cell);
    const double bounds[3] = {
        std::floor(cutoff * std::sqrt(inv.a[0][0] * inv.a[0][0] + inv.a[1][0] * inv.a[1][0] + inv.a[2][0] * inv.a[2][0]) + 1.0),
        std::floor(cutoff * std::sqrt(inv.a[0][1] * inv.a[0][1] + inv.a[1][1] * inv.a[1][1] + inv.a[2][1] * inv.a[2][1]) + 1.0),
        std::floor(cutoff * std::sqrt(inv.a[0][2] * inv.a[0][2] + inv.a[1][2] * inv.a[1][2] + inv.a[2][2] * inv.a[2][2]) + 1.0),
    };

    std::vector<ExtendedAtom> extended;
    auto append_cell = [&](int n0, int n1, int n2) {
        const Vec3 shift = n0 * row(cell, 0) + n1 * row(cell, 1) + n2 * row(cell, 2);
        for (std::int64_t atom = begin; atom < end; ++atom) {
            const Vec3 original = position(batch, atom);
            const bool is_original_cell = n0 == 0 && n1 == 0 && n2 == 0;
            Vec3 image_position = original + shift;
            std::int32_t image_shift[3] = {n0, n1, n2};
            if (use_scaled_periodic_images && !is_original_cell) {
                const Vec3 fractional = fractional_position(original, inv)
                    - Vec3{static_cast<double>(n0), static_cast<double>(n1), static_cast<double>(n2)};
                image_position = cartesian_position(fractional, cell);
                image_shift[0] = -n0;
                image_shift[1] = -n1;
                image_shift[2] = -n2;
            }
            extended.push_back({
                static_cast<std::int32_t>(atom),
                {image_shift[0], image_shift[1], image_shift[2]},
                image_position,
            });
        }
    };
    if (use_scaled_periodic_images) {
        append_cell(0, 0, 0);
    }
    for (int n0 = -static_cast<int>(bounds[0]); n0 <= static_cast<int>(bounds[0]); ++n0) {
        for (int n1 = -static_cast<int>(bounds[1]); n1 <= static_cast<int>(bounds[1]); ++n1) {
            for (int n2 = -static_cast<int>(bounds[2]); n2 <= static_cast<int>(bounds[2]); ++n2) {
                if (use_scaled_periodic_images && n0 == 0 && n1 == 0 && n2 == 0) {
                    continue;
                }
                append_cell(n0, n1, n2);
            }
        }
    }
    if (use_scaled_periodic_images) {
        std::vector<ExtendedAtom> filtered;
        filtered.reserve(extended.size());
        for (const ExtendedAtom& candidate : extended) {
            const bool is_original_cell = candidate.shift[0] == 0
                && candidate.shift[1] == 0 && candidate.shift[2] == 0;
            bool within_interaction_limit = is_original_cell;
            if (!within_interaction_limit) {
                for (std::int64_t center = begin; center < end; ++center) {
                    if (std::sqrt(norm2(candidate.position - position(batch, center))) < cutoff) {
                        within_interaction_limit = true;
                        break;
                    }
                }
            }
            if (within_interaction_limit) {
                filtered.push_back(candidate);
            }
        }
        extended.swap(filtered);
    }
    const CellGrid grid = build_grid(extended, cutoff);
    const double cutoff2 = cutoff * cutoff;
    const auto within_cutoff = [cutoff2, include_boundary](double distance2) {
        return include_boundary ? distance2 <= cutoff2 : distance2 < cutoff2;
    };
    LocalGraph result;
    // reference implementation consumes one cell-list query per center. For the serial SOAP
    // path, do the same and append neighbors directly; the old count-then-fill
    // pass scanned the same candidate cells twice.
    if (num_threads == 1) {
        result.counts.resize(static_cast<std::size_t>(atom_count) + 1, 0);
        for (std::int64_t local = 0; local < atom_count; ++local) {
            if (control && control->cancelled()) {
                continue;
            }
            const std::int64_t center = begin + local;
            const Vec3 center_position = position(batch, center);
            const int ix = grid.coordinate(center_position.x, 0);
            const int iy = grid.coordinate(center_position.y, 1);
            const int iz = grid.coordinate(center_position.z, 2);
            result.counts[static_cast<std::size_t>(local)] = static_cast<std::int64_t>(result.atoms.size());
            for (int z = std::max(0, iz - 1); z <= std::min(grid.dimensions[2] - 1, iz + 1); ++z) {
                for (int y = std::max(0, iy - 1); y <= std::min(grid.dimensions[1] - 1, iy + 1); ++y) {
                    for (int x = std::max(0, ix - 1); x <= std::min(grid.dimensions[0] - 1, ix + 1); ++x) {
                        const int cell_index = grid.index(x, y, z);
                        for (std::int32_t offset = grid.offsets[cell_index]; offset < grid.offsets[cell_index + 1]; ++offset) {
                            const ExtendedAtom& neighbor = extended[static_cast<std::size_t>(grid.atoms[offset])];
                            const Vec3 displacement = neighbor.position - center_position;
                            const double distance2 = norm2(displacement);
                            if (!within_cutoff(distance2)) {
                                continue;
                            }
                            result.atoms.push_back(neighbor.atom);
                            if (store_shifts) {
                                result.shifts.push_back(neighbor.shift[0]);
                                result.shifts.push_back(neighbor.shift[1]);
                                result.shifts.push_back(neighbor.shift[2]);
                            }
                            result.displacements.push_back(displacement.x);
                            result.displacements.push_back(displacement.y);
                            result.displacements.push_back(displacement.z);
                            result.distance2.push_back(distance2);
                        }
                    }
                }
            }
            result.counts[static_cast<std::size_t>(local + 1)] = static_cast<std::int64_t>(result.atoms.size());
        }
        if (control && control->cancelled()) {
            throw CancelledError();
        }
        return result;
    }
    result.counts.resize(static_cast<std::size_t>(atom_count), 0);

    auto count_center = [&](std::int64_t center) {
        const Vec3 center_position = position(batch, center);
        const int ix = grid.coordinate(center_position.x, 0);
        const int iy = grid.coordinate(center_position.y, 1);
        const int iz = grid.coordinate(center_position.z, 2);
        std::int64_t count = 0;
        for (int z = std::max(0, iz - 1); z <= std::min(grid.dimensions[2] - 1, iz + 1); ++z) {
            for (int y = std::max(0, iy - 1); y <= std::min(grid.dimensions[1] - 1, iy + 1); ++y) {
                for (int x = std::max(0, ix - 1); x <= std::min(grid.dimensions[0] - 1, ix + 1); ++x) {
                    const int cell_index = grid.index(x, y, z);
                    for (std::int32_t offset = grid.offsets[cell_index]; offset < grid.offsets[cell_index + 1]; ++offset) {
                        const ExtendedAtom& neighbor = extended[static_cast<std::size_t>(grid.atoms[offset])];
                        if (within_cutoff(norm2(neighbor.position - center_position))) {
                            ++count;
                        }
                    }
                }
            }
        }
        return count;
    };

#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(num_threads > 0 ? num_threads : omp_get_max_threads())
#endif
    for (std::int64_t local = 0; local < atom_count; ++local) {
        if (control && control->cancelled()) {
            continue;
        }
        result.counts[static_cast<std::size_t>(local)] = count_center(begin + local);
    }
    if (control && control->cancelled()) {
        throw CancelledError();
    }

    std::vector<std::int64_t> offsets(static_cast<std::size_t>(atom_count) + 1, 0);
    for (std::int64_t local = 0; local < atom_count; ++local) {
        offsets[static_cast<std::size_t>(local + 1)] = offsets[static_cast<std::size_t>(local)] + result.counts[static_cast<std::size_t>(local)];
    }
    const std::size_t total = static_cast<std::size_t>(offsets.back());
    result.atoms.resize(total);
    if (store_shifts) result.shifts.resize(total * 3);
    result.displacements.resize(total * 3);
    result.distance2.resize(total);

    auto fill_center = [&](std::int64_t center, std::int64_t local) {
        const Vec3 center_position = position(batch, center);
        const int ix = grid.coordinate(center_position.x, 0);
        const int iy = grid.coordinate(center_position.y, 1);
        const int iz = grid.coordinate(center_position.z, 2);
        std::int64_t output = offsets[static_cast<std::size_t>(local)];
        for (int z = std::max(0, iz - 1); z <= std::min(grid.dimensions[2] - 1, iz + 1); ++z) {
            for (int y = std::max(0, iy - 1); y <= std::min(grid.dimensions[1] - 1, iy + 1); ++y) {
                for (int x = std::max(0, ix - 1); x <= std::min(grid.dimensions[0] - 1, ix + 1); ++x) {
                    const int cell_index = grid.index(x, y, z);
                    for (std::int32_t offset = grid.offsets[cell_index]; offset < grid.offsets[cell_index + 1]; ++offset) {
                        const ExtendedAtom& neighbor = extended[static_cast<std::size_t>(grid.atoms[offset])];
                        const Vec3 displacement = neighbor.position - center_position;
                        const double distance2 = norm2(displacement);
                        if (!within_cutoff(distance2)) {
                            continue;
                        }
                        result.atoms[static_cast<std::size_t>(output)] = neighbor.atom;
                        if (store_shifts) {
                            result.shifts[static_cast<std::size_t>(output) * 3 + 0] = neighbor.shift[0];
                            result.shifts[static_cast<std::size_t>(output) * 3 + 1] = neighbor.shift[1];
                            result.shifts[static_cast<std::size_t>(output) * 3 + 2] = neighbor.shift[2];
                        }
                        result.displacements[static_cast<std::size_t>(output) * 3 + 0] = displacement.x;
                        result.displacements[static_cast<std::size_t>(output) * 3 + 1] = displacement.y;
                        result.displacements[static_cast<std::size_t>(output) * 3 + 2] = displacement.z;
                        result.distance2[static_cast<std::size_t>(output)] = distance2;
                        ++output;
                    }
                }
            }
        }
    };

#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(num_threads > 0 ? num_threads : omp_get_max_threads())
#endif
    for (std::int64_t local = 0; local < atom_count; ++local) {
        if (control && control->cancelled()) {
            continue;
        }
        fill_center(begin + local, local);
    }
    if (control && control->cancelled()) {
        throw CancelledError();
    }
    result.counts = std::move(offsets);
    return result;
}

} // namespace

bool NeighborView::exact_self(std::size_t index, std::int64_t center) const noexcept {
    if (shifts == nullptr) return false;
    return atoms[index] == center
        && shifts[index * 3 + 0] == 0
        && shifts[index * 3 + 1] == 0
        && shifts[index * 3 + 2] == 0;
}

NeighborView NeighborGraph::for_center(std::int64_t center) const noexcept {
    const std::size_t begin = static_cast<std::size_t>(offsets_[static_cast<std::size_t>(center)]);
    const std::size_t end = static_cast<std::size_t>(offsets_[static_cast<std::size_t>(center + 1)]);
    return {
        atoms_.data() + begin,
        shifts_.empty() ? nullptr : shifts_.data() + begin * 3,
        displacements_.data() + begin * 3,
        distance2_.data() + begin,
        end - begin,
    };
}

NeighborGraph build_neighbor_graph(
    const StructureBatchView& batch,
    double cutoff,
    const std::shared_ptr<ComputeControl>& control,
    int num_threads,
    bool include_boundary,
    bool use_scaled_periodic_images,
    bool store_shifts) {
    if (!std::isfinite(cutoff) || cutoff <= 0.0) {
        throw std::invalid_argument("neighbor cutoff must be finite and positive");
    }
    NeighborGraph graph;
    graph.cutoff_ = cutoff;
    graph.offsets_.resize(static_cast<std::size_t>(batch.atoms) + 1, 0);
    if (batch.structures == 0) {
        return graph;
    }
    if (batch.structures == 1) {
        LocalGraph current = build_structure_graph(
            batch, 0, cutoff, control, num_threads, include_boundary,
            use_scaled_periodic_images, store_shifts);
        graph.offsets_ = std::move(current.counts);
        graph.atoms_ = std::move(current.atoms);
        if (store_shifts) graph.shifts_ = std::move(current.shifts);
        graph.displacements_ = std::move(current.displacements);
        graph.distance2_ = std::move(current.distance2);
        return graph;
    }
    std::vector<LocalGraph> local(static_cast<std::size_t>(batch.structures));
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(num_threads > 0 ? num_threads : omp_get_max_threads())
#endif
    for (std::int64_t structure = 0; structure < batch.structures; ++structure) {
        if (control && control->cancelled()) {
            continue;
        }
        local[static_cast<std::size_t>(structure)] = build_structure_graph(
            batch, structure, cutoff, control, num_threads, include_boundary,
            use_scaled_periodic_images, store_shifts);
    }
    if (control && control->cancelled()) {
        throw CancelledError();
    }

    for (std::int64_t structure = 0; structure < batch.structures; ++structure) {
        const std::int64_t begin = batch.offsets[structure];
        const std::int64_t end = batch.offsets[structure + 1];
        const LocalGraph& current = local[static_cast<std::size_t>(structure)];
        for (std::int64_t atom = begin; atom < end; ++atom) {
            const std::int64_t local_atom = atom - begin;
            graph.offsets_[static_cast<std::size_t>(atom + 1)] = graph.offsets_[static_cast<std::size_t>(atom)]
                + current.counts[static_cast<std::size_t>(local_atom + 1)]
                - current.counts[static_cast<std::size_t>(local_atom)];
        }
    }
    const std::size_t total = static_cast<std::size_t>(graph.offsets_.back());
    graph.atoms_.reserve(total);
    if (store_shifts) graph.shifts_.reserve(total * 3);
    graph.displacements_.reserve(total * 3);
    graph.distance2_.reserve(total);
    for (std::int64_t structure = 0; structure < batch.structures; ++structure) {
        const LocalGraph& current = local[static_cast<std::size_t>(structure)];
        graph.atoms_.insert(graph.atoms_.end(), current.atoms.begin(), current.atoms.end());
        if (store_shifts) {
            graph.shifts_.insert(graph.shifts_.end(), current.shifts.begin(), current.shifts.end());
        }
        graph.displacements_.insert(graph.displacements_.end(), current.displacements.begin(), current.displacements.end());
        graph.distance2_.insert(graph.distance2_.end(), current.distance2.begin(), current.distance2.end());
    }
    return graph;
}

} // namespace mdescriptor
