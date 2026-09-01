#include "mdescriptor/neighbor.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <typename Error, typename Function>
void require_throws(Function&& function, const std::string& message) {
    try {
        function();
    } catch (const Error&) {
        return;
    }
    throw std::runtime_error(message);
}

mdescriptor::StructureBatchView isolated_pair() {
    static const std::array<std::int32_t, 2> numbers{1, 8};
    static const std::array<double, 6> positions{0.0, 0.0, 0.0, 1.0, 0.0, 0.0};
    static const std::array<double, 9> cell{};
    static const std::array<std::int32_t, 3> pbc{};
    static const std::array<std::int64_t, 2> offsets{0, 2};
    return {
        numbers.data(), positions.data(), cell.data(), pbc.data(), offsets.data(), 1, 2,
    };
}

void test_compute_control_contract() {
    auto control = std::make_shared<mdescriptor::ComputeControl>();
    control->reset(2);
    require(control->total() == 2, "reset must publish the requested total");
    require(control->completed() == 0, "reset must clear completed work");
    require(!control->cancelled(), "reset must clear cancellation");
    control->mark_completed();
    require(control->completed() == 1, "mark_completed must advance progress");
    control->cancel();
    require(control->cancelled(), "cancel must be observable");
    require_throws<std::invalid_argument>(
        [&] { control->reset(-1); },
        "negative work totals must be rejected");
}

void test_neighbor_graph_boundary_contract() {
    const auto batch = isolated_pair();
    const auto inclusive = mdescriptor::build_neighbor_graph(batch, 1.0, nullptr, 1, true);
    require(inclusive.atoms() == 2, "graph must retain the batch atom count");
    require(inclusive.offsets() == std::vector<std::int64_t>({0, 2, 4}),
            "inclusive cutoff must retain self and boundary neighbor entries");
    for (std::int64_t center = 0; center < 2; ++center) {
        const auto neighbors = inclusive.for_center(center);
        require(neighbors.size == 2, "each center must expose both atoms");
        bool found_self = false;
        bool found_boundary = false;
        for (std::size_t index = 0; index < neighbors.size; ++index) {
            found_self = found_self || neighbors.exact_self(index, center);
            found_boundary = found_boundary || neighbors.distance2[index] == 1.0;
        }
        require(found_self, "neighbor views must identify exact self entries");
        require(found_boundary, "inclusive cutoff must retain distance-equal neighbors");
    }

    const auto exclusive = mdescriptor::build_neighbor_graph(batch, 1.0, nullptr, 1, false);
    require(exclusive.offsets() == std::vector<std::int64_t>({0, 1, 2}),
            "exclusive cutoff must discard distance-equal neighbors");
}

void test_neighbor_graph_rejects_invalid_or_cancelled_work() {
    auto mixed = isolated_pair();
    const std::array<std::int32_t, 3> mixed_pbc{1, 0, 1};
    mixed.pbc = mixed_pbc.data();
    require_throws<std::invalid_argument>(
        [&] { (void)mdescriptor::build_neighbor_graph(mixed, 1.0); },
        "mixed periodicity must be rejected by the C++ boundary");

    auto control = std::make_shared<mdescriptor::ComputeControl>();
    control->reset(1);
    control->cancel();
    require_throws<mdescriptor::CancelledError>(
        [&] { (void)mdescriptor::build_neighbor_graph(isolated_pair(), 1.0, control, 1); },
        "pre-cancelled work must raise CancelledError");
}

struct NeighborRecord {
    std::int32_t atom = 0;
    std::array<double, 3> displacement{};
    double distance2 = 0.0;
};

std::vector<NeighborRecord> records(const mdescriptor::NeighborView& view) {
    std::vector<NeighborRecord> result;
    result.reserve(view.size);
    for (std::size_t index = 0; index < view.size; ++index) {
        result.push_back({
            view.atoms[index],
            {view.displacements[index * 3 + 0], view.displacements[index * 3 + 1],
             view.displacements[index * 3 + 2]},
            view.distance2[index],
        });
    }
    return result;
}

void test_compact_periodic_graph_matches_image_graph() {
    const std::array<std::int32_t, 5> numbers{1, 6, 8, 14, 1};
    const std::array<double, 15> positions{
        -0.2, 0.1, 0.2,
        19.8, 0.2, 0.1,
        0.4, 21.7, 0.3,
        0.3, 0.4, 23.8,
        10.0, 11.0, 12.0,
    };
    const std::array<double, 9> cell{
        20.0, 0.0, 0.0,
        0.0, 22.0, 0.0,
        0.0, 0.0, 24.0,
    };
    const std::array<std::int32_t, 3> pbc{1, 1, 1};
    const std::array<std::int64_t, 2> offsets{0, 5};
    const mdescriptor::StructureBatchView batch{
        numbers.data(), positions.data(), cell.data(), pbc.data(), offsets.data(), 1, 5,
    };
    const auto compact = mdescriptor::build_neighbor_graph(
        batch, 6.0, nullptr, 1, true, false, false);
    const auto image_graph = mdescriptor::build_neighbor_graph(
        batch, 6.0, nullptr, 1, true, false, true);
    require(compact.shifts().empty(), "compact NEP graph must not allocate shifts");
    require(compact.offsets().back() == image_graph.offsets().back(),
            "compact and image graph edge counts must match");

    for (std::int64_t center = 0; center < batch.atoms; ++center) {
        const auto compact_records = records(compact.for_center(center));
        const auto image_records = records(image_graph.for_center(center));
        require(compact_records.size() == image_records.size(),
                "compact and image graph row sizes must match");
        std::vector<bool> matched(image_records.size(), false);
        for (const auto& compact_record : compact_records) {
            bool found = false;
            for (std::size_t index = 0; index < image_records.size(); ++index) {
                const auto& image_record = image_records[index];
                if (matched[index] || compact_record.atom != image_record.atom) continue;
                if (std::abs(compact_record.distance2 - image_record.distance2) > 1e-12) {
                    continue;
                }
                bool same_displacement = true;
                for (int component = 0; component < 3; ++component) {
                    if (std::abs(compact_record.displacement[component]
                                 - image_record.displacement[component]) > 1e-12) {
                        same_displacement = false;
                        break;
                    }
                }
                if (same_displacement) {
                    matched[index] = true;
                    found = true;
                    break;
                }
            }
            require(found, "compact graph lost or changed a periodic neighbor");
        }
    }
}

} // namespace

int main() {
    try {
        test_compute_control_contract();
        test_neighbor_graph_boundary_contract();
        test_neighbor_graph_rejects_invalid_or_cancelled_work();
        test_compact_periodic_graph_matches_image_graph();
    } catch (const std::exception& error) {
        std::cerr << "C++ public API test failed: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
