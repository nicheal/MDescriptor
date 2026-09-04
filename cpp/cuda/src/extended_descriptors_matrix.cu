#include "extended_descriptors_common.cuh"

py::dict compute_matrix_descriptor(
    CudaExecutionContext& context,
    DeviceBatch& batch,
    const detail::StructureBatchView& host_batch,
    int kind,
    const std::string& name,
    const py::dict& options) {
    int n_atoms_max = option(options, "n_atoms_max", 0);
    const std::string permutation_name = option(options, "permutation", std::string("sorted_l2"));
    const int permutation = permutation_name == "none" ? kMatrixPermutationNone
        : permutation_name == "sorted_l2" ? kMatrixPermutationSortedL2 : kMatrixPermutationEigenspectrum;
    if (permutation_name != "none" && permutation_name != "sorted_l2"
        && permutation_name != "eigenspectrum") {
        throw std::invalid_argument("invalid CUDA matrix permutation");
    }
    if (n_atoms_max <= 0) {
        for (I64 structure = 0; structure < host_batch.structures; ++structure) {
            n_atoms_max = std::max(
                n_atoms_max,
                static_cast<int>(host_batch.offsets[structure + 1] - host_batch.offsets[structure]));
        }
    }
    if (n_atoms_max <= 0 && host_batch.atoms == 0) {
        py::dict result;
        result["values"] = values_array({}, host_batch.structures, 0);
        result["level"] = "structure";
        result["labels"] = labels_option(options, name, 0);
        result["metadata"] = metadata(options, name);
        return result;
    }
    if (n_atoms_max <= 0 || n_atoms_max > 256) {
        throw std::invalid_argument(
            "CUDA matrix descriptors require 1 <= n_atoms_max <= 256");
    }
    for (I64 structure = 0; structure < host_batch.structures; ++structure) {
        const I64 count = host_batch.offsets[structure + 1] - host_batch.offsets[structure];
        if (count > n_atoms_max) {
            throw std::invalid_argument("structure exceeds n_atoms_max");
        }
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
    const auto values = download_output_with_gil_release(context, output_size);
    py::dict result;
    result["values"] = values_array(values, host_batch.structures, columns);
    result["level"] = "structure";
    result["labels"] = labels_option(options, name, columns);
    result["metadata"] = metadata(options, name);
    return result;
}

} // namespace

py::dict compute_extended_matrix(
    CudaExecutionContext& context,
    DeviceBatch& batch,
    DeviceNeighborGraph& graph,
    const detail::StructureBatchView& host_batch,
    const std::string& name,
    const py::dict& options,
    const py::object& control,
    RotationalPlanCache* rotational_plan) {
    (void)graph;
    (void)control;
    (void)rotational_plan;
    const int kind = name == "CoulombMatrix" ? kMatrixKindCoulomb
        : name == "SineMatrix" ? kMatrixKindSine : kMatrixKindEwald;
    return compute_matrix_descriptor(context, batch, host_batch, kind, name, options);
}

} // namespace mdescriptor::cuda
