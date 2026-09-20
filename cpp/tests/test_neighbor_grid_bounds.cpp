#include "mdescriptor/neighbor.hpp"

#include <array>
#include <cstdint>
#include <iostream>
#include <limits>
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

void test_nonperiodic_grid_dimension_conversion_is_checked() {
    const std::array<std::int32_t, 2> numbers{1, 1};
    const std::array<double, 6> positions{0.0, 0.0, 0.0, 1.0, 0.0, 0.0};
    const std::array<double, 9> cell{};
    const std::array<std::int32_t, 3> pbc{};
    const std::array<std::int64_t, 2> offsets{0, 2};
    const mdescriptor::StructureBatchView batch{
        numbers.data(), positions.data(), cell.data(), pbc.data(), offsets.data(), 1, 2,
    };

    require_throws<std::invalid_argument>(
        [&] { (void)mdescriptor::build_neighbor_graph(batch, 1e-12, nullptr, 1); },
        "an unrepresentable non-periodic grid dimension must be rejected");
}

void test_compact_grid_cell_count_overflow_is_checked() {
    const std::array<std::int32_t, 1> numbers{1};
    const std::array<double, 3> positions{0.0, 0.0, 0.0};
    const double length = static_cast<double>(std::numeric_limits<int>::max());
    const std::array<double, 9> cell{
        length, 0.0, 0.0,
        0.0, length, 0.0,
        0.0, 0.0, length,
    };
    const std::array<std::int32_t, 3> pbc{1, 1, 1};
    const std::array<std::int64_t, 2> offsets{0, 1};
    const mdescriptor::StructureBatchView batch{
        numbers.data(), positions.data(), cell.data(), pbc.data(), offsets.data(), 1, 1,
    };

    require_throws<std::invalid_argument>(
        [&] {
            (void)mdescriptor::build_neighbor_graph(
                batch, 1.0, nullptr, 1, true, false, false);
        },
        "a compact grid whose cell count overflows size_t must be rejected");
}

void test_periodic_image_bound_conversion_is_checked() {
    const std::array<std::int32_t, 1> numbers{1};
    const std::array<double, 3> positions{0.0, 0.0, 0.0};
    const std::array<double, 9> cell{
        1e-14, 0.0, 0.0,
        0.0, 1e7, 0.0,
        0.0, 0.0, 1e7,
    };
    const std::array<std::int32_t, 3> pbc{1, 1, 1};
    const std::array<std::int64_t, 2> offsets{0, 1};
    const mdescriptor::StructureBatchView batch{
        numbers.data(), positions.data(), cell.data(), pbc.data(), offsets.data(), 1, 1,
    };

    require_throws<std::invalid_argument>(
        [&] { (void)mdescriptor::build_neighbor_graph(batch, 1.0, nullptr, 1); },
        "an unrepresentable periodic image range must be rejected");
}

void test_atom_index_width_is_checked_before_allocation() {
    const std::array<std::int32_t, 1> numbers{1};
    const std::array<double, 3> positions{0.0, 0.0, 0.0};
    const std::array<double, 9> cell{};
    const std::array<std::int32_t, 3> pbc{};
    const std::int64_t atom_count =
        static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::max()) + 1;
    const std::array<std::int64_t, 2> offsets{0, atom_count};
    const mdescriptor::StructureBatchView batch{
        numbers.data(), positions.data(), cell.data(), pbc.data(), offsets.data(), 1, atom_count,
    };

    require_throws<std::invalid_argument>(
        [&] { (void)mdescriptor::build_neighbor_graph(batch, 1.0, nullptr, 1); },
        "atom counts beyond the graph index width must be rejected before allocation");
}

} // namespace

int main() {
    try {
        test_nonperiodic_grid_dimension_conversion_is_checked();
        test_compact_grid_cell_count_overflow_is_checked();
        test_periodic_image_bound_conversion_is_checked();
        test_atom_index_width_is_checked_before_allocation();
    } catch (const std::exception& error) {
        std::cerr << "C++ neighbor grid bounds test failed: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
