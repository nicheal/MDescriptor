#include "extended_descriptors_common.cuh"

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
    const auto values = download_output_with_gil_release(context, size);
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
    const auto values = download_output_with_gil_release(context, size);
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
    graph.build_dpa(
        context, batch, host_batch, cutoff, true, false, true, false, false,
        NeighborGraphOrdering::Canonical);
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
    auto values = download_output_with_gil_release(context, output_size);
    auto records_host = download(
        records.get(), records_size, context, "could not download CUDA pair records");
    // DeviceNeighborGraph already emitted the public cell-list order.  Keep
    // the host boundary limited to the final arrays; pair features and their
    // canonical ordering are both produced on the CUDA stream.
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


} // namespace

py::dict compute_extended_basic(
    CudaExecutionContext& context,
    DeviceBatch& batch,
    DeviceNeighborGraph& graph,
    const detail::StructureBatchView& host_batch,
    const std::string& name,
    const py::dict& options,
    const py::object& control,
    RotationalPlanCache* rotational_plan) {
    (void)control;
    (void)rotational_plan;
    if (name == "AtomicComposition") {
        return compute_atomic_composition(
            context, batch, host_batch, species_option(options),
            option(options, "per_system", true), name, options);
    }
    if (name == "SortedDistances") {
        return compute_sorted_distances(
            context, batch, graph, host_batch, species_option(options),
            option(options, "cutoff", 6.0), option(options, "max_neighbors", 8),
            option(options, "separate_neighbor_types", true), name, options);
    }
    if (name == "SphericalExpansionByPair") {
        return compute_spherical_pair(
            context, batch, graph, host_batch, option(options, "cutoff", 6.0),
            option(options, "density_width", 0.3), option(options, "max_radial", 6),
            option(options, "max_angular", 4), name, options);
    }
    throw std::invalid_argument("unknown basic CUDA descriptor: " + name);
}

} // namespace mdescriptor::cuda
