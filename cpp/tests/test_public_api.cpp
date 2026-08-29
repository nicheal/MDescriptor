#include "mdescriptor/neighbor.hpp"

#include <array>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

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

} // namespace

int main() {
    try {
        test_compute_control_contract();
        test_neighbor_graph_boundary_contract();
        test_neighbor_graph_rejects_invalid_or_cancelled_work();
    } catch (const std::exception& error) {
        std::cerr << "C++ public API test failed: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
