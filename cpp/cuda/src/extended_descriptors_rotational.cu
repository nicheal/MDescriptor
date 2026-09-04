#include "extended_descriptors_common.cuh"

py::dict compute_rotational_descriptor(
    CudaExecutionContext& context,
    DeviceBatch& batch,
    DeviceNeighborGraph& graph,
    const detail::StructureBatchView& host_batch,
    const std::string& name,
    const py::dict& options,
    RotationalPlanCache* rotational_plan_cache) {
    const auto species = batch_species(host_batch);
    if (species.empty()) throw std::invalid_argument(name + " requires at least one atom");
    const RotationalCudaOptions config = rotational_options(name, options);
    const int kind = config.kind;
    if (!std::isfinite(config.cutoff) || !std::isfinite(config.alpha)
        || !std::isfinite(config.rfac0) || !std::isfinite(config.rmin0)
        || !std::isfinite(config.rcutfac)
        || config.cutoff <= 0.0 || config.alpha <= 0.0 || config.rfac0 <= 0.0
        || config.rmin0 < 0.0 || config.rcutfac <= 0.0
        || config.lmax < 0 || config.nmax < 1
        || config.diagonal < 0 || config.diagonal > 3
        || (kind != 0 && config.twojmax < 0)) {
        throw std::invalid_argument("invalid CUDA rotational descriptor parameters");
    }
    if (kind == 0 && (config.nmax > 8 || config.lmax > 8)) {
        throw std::invalid_argument(
            "CUDA SO3 supports nmax and lmax up to 8 in the current kernel");
    }
    I64 computed = 0;
    detail::rotational::BispectrumPlan bispectrum_plan;
    detail::rotational::FlattenedBispectrumPlan flattened_plan;
    RotationalPlanDeviceView rotational_plan;
    if (kind == 0) {
        computed = static_cast<I64>(config.lmax + 1) * config.nmax * (config.nmax + 1) / 2;
    } else {
        const int order = mdescriptor::detail::rotational::expansion_order(
            kind, config.lmax, config.twojmax);
        if (order > 10) {
            throw std::invalid_argument(
                name + " CUDA expansion order is limited to 10");
        }
        if (rotational_plan_cache != nullptr) {
            rotational_plan = rotational_plan_cache->prepare(
                context, order, config.diagonal, name == "LBispectrum");
            computed = rotational_plan.features;
        } else {
            bispectrum_plan = detail::rotational::make_bispectrum_plan(
                order, config.diagonal, name == "LBispectrum");
            flattened_plan = detail::rotational::flatten(bispectrum_plan);
            computed = static_cast<I64>(bispectrum_plan.components.size());
        }
    }
    const I64 features = payload_or_option_feature_count(options, computed, name);
    if (features != computed) throw std::invalid_argument(name + " CUDA feature count mismatch");
    const py::str weights_key("weights");
    const bool custom_weights = options.contains(weights_key)
        && !options[weights_key].is_none();
    std::vector<double> weights;
    std::vector<double> radii;
    // SO4 uses atomic numbers directly in the rotational kernel.  For SNAP
    // and L-Bispectrum a null pointer is the common all-ones fast path.
    if (kind != 1 && custom_weights) {
        weights.assign(static_cast<std::size_t>(host_batch.atoms), 1.0);
        const auto configured = species_dictionary_values(options[weights_key], species, 1.0);
        for (I64 atom = 0; atom < host_batch.atoms; ++atom) {
            const auto found = std::find(species.begin(), species.end(), host_batch.numbers[atom]);
            if (found != species.end()) weights[static_cast<std::size_t>(atom)]
                = configured[static_cast<std::size_t>(found - species.begin())];
        }
    }
    const py::str radii_key("element_radii");
    if (name == "LBispectrum" && options.contains(radii_key) && !options[radii_key].is_none()) {
        const auto configured = species_dictionary_values(options[radii_key], species, config.cutoff * 0.5);
        radii.resize(static_cast<std::size_t>(host_batch.atoms), config.cutoff * 0.5);
        for (I64 atom = 0; atom < host_batch.atoms; ++atom) {
            const auto found = std::find(species.begin(), species.end(), host_batch.numbers[atom]);
            if (found != species.end()) radii[static_cast<std::size_t>(atom)]
                = configured[static_cast<std::size_t>(found - species.begin())];
        }
    }
    const double graph_cutoff = radii.empty() ? config.cutoff
        : 2.0 * *std::max_element(radii.begin(), radii.end()) * config.rcutfac;
    graph.build_dpa(
        context, batch, host_batch, graph_cutoff, true, false,
        mdescriptor::detail::rotational::kBispectrumIncludeExactSelf);
    DeviceBuffer<double> d_weights;
    DeviceBuffer<double> d_radii;
    d_weights.upload(weights.data(), weights.size(), context.stream(), "could not upload rotational weights");
    d_radii.upload(radii.data(), radii.size(), context.stream(), "could not upload rotational radii");
    DeviceBuffer<I64> d_bispectrum_z_inner_offsets;
    DeviceBuffer<I64> d_bispectrum_inner_term_offsets;
    DeviceBuffer<double> d_bispectrum_inner_outer_coefficients;
    DeviceBuffer<I64> d_bispectrum_term_first_indices;
    DeviceBuffer<I64> d_bispectrum_term_second_indices;
    DeviceBuffer<double> d_bispectrum_term_coefficients;
    DeviceBuffer<I64> d_bispectrum_projection_offsets;
    DeviceBuffer<I64> d_bispectrum_projection_u_indices;
    DeviceBuffer<I64> d_bispectrum_projection_z_indices;
    DeviceBuffer<double> d_bispectrum_projection_scales;
    if (kind != 0 && rotational_plan_cache == nullptr) {
        d_bispectrum_z_inner_offsets.upload(
            flattened_plan.z_inner_offsets.data(), flattened_plan.z_inner_offsets.size(),
            context.stream(), "could not upload CUDA bispectrum Z offsets");
        d_bispectrum_inner_term_offsets.upload(
            flattened_plan.inner_term_offsets.data(), flattened_plan.inner_term_offsets.size(),
            context.stream(), "could not upload CUDA bispectrum inner offsets");
        d_bispectrum_inner_outer_coefficients.upload(
            flattened_plan.inner_outer_coefficients.data(),
            flattened_plan.inner_outer_coefficients.size(), context.stream(),
            "could not upload CUDA bispectrum outer coefficients");
        d_bispectrum_term_first_indices.upload(
            flattened_plan.term_first_indices.data(), flattened_plan.term_first_indices.size(),
            context.stream(), "could not upload CUDA bispectrum first indices");
        d_bispectrum_term_second_indices.upload(
            flattened_plan.term_second_indices.data(), flattened_plan.term_second_indices.size(),
            context.stream(), "could not upload CUDA bispectrum second indices");
        d_bispectrum_term_coefficients.upload(
            flattened_plan.term_coefficients.data(), flattened_plan.term_coefficients.size(),
            context.stream(), "could not upload CUDA bispectrum CG coefficients");
        d_bispectrum_projection_offsets.upload(
            flattened_plan.projection_offsets.data(), flattened_plan.projection_offsets.size(),
            context.stream(), "could not upload CUDA bispectrum projection offsets");
        d_bispectrum_projection_u_indices.upload(
            flattened_plan.projection_u_indices.data(), flattened_plan.projection_u_indices.size(),
            context.stream(), "could not upload CUDA bispectrum projection U indices");
        d_bispectrum_projection_z_indices.upload(
            flattened_plan.projection_z_indices.data(), flattened_plan.projection_z_indices.size(),
            context.stream(), "could not upload CUDA bispectrum projection Z indices");
        d_bispectrum_projection_scales.upload(
            flattened_plan.projection_scales.data(), flattened_plan.projection_scales.size(),
            context.stream(), "could not upload CUDA bispectrum projection scales");
    }
    const std::size_t size = static_cast<std::size_t>(batch.atoms()) * static_cast<std::size_t>(features);
    double* output = context.output_buffer(size);
    DeviceBuffer<double> d_so3_basis;
    int so3_quadrature_count = 0;
    if (kind == 0) {
        const auto so3_basis = so3_basis_host(
            config.nmax, config.lmax, config.cutoff, config.alpha, &so3_quadrature_count);
        d_so3_basis.upload(
            so3_basis.data(), so3_basis.size(), context.stream(),
            "could not upload CUDA SO3 radial basis");
    }
    if (size > 0) {
        check_cuda(cudaMemsetAsync(output, 0, size * sizeof(double), context.stream()),
            "could not clear CUDA rotational output");
        constexpr unsigned block_size = 64;
        if (kind == 0) {
            so3_kernel<<<static_cast<unsigned>((batch.atoms() + block_size - 1) / block_size),
                block_size, 0, context.stream()>>>(
                batch.numbers(), graph.offsets(), graph.atoms(), graph.displacements(),
                graph.distance2(), config.nmax, config.lmax, config.cutoff, config.alpha,
                config.weight_on, so3_quadrature_count, d_so3_basis.get(), features,
                batch.atoms(), output);
            check_cuda(cudaGetLastError(), "CUDA SO3 kernel launch failed");
        } else {
            rotational_kernel<<<static_cast<unsigned>((batch.atoms() + block_size - 1) / block_size),
                block_size, 0, context.stream()>>>(
                batch.numbers(), graph.offsets(), graph.atoms(), graph.displacements(), graph.distance2(),
                d_weights.get(), d_radii.get(),
                rotational_plan_cache == nullptr ? d_bispectrum_z_inner_offsets.get()
                    : rotational_plan.z_inner_offsets,
                rotational_plan_cache == nullptr ? d_bispectrum_inner_term_offsets.get()
                    : rotational_plan.inner_term_offsets,
                rotational_plan_cache == nullptr ? d_bispectrum_inner_outer_coefficients.get()
                    : rotational_plan.inner_outer_coefficients,
                rotational_plan_cache == nullptr ? d_bispectrum_term_first_indices.get()
                    : rotational_plan.term_first_indices,
                rotational_plan_cache == nullptr ? d_bispectrum_term_second_indices.get()
                    : rotational_plan.term_second_indices,
                rotational_plan_cache == nullptr ? d_bispectrum_term_coefficients.get()
                    : rotational_plan.term_coefficients,
                rotational_plan_cache == nullptr ? d_bispectrum_projection_offsets.get()
                    : rotational_plan.projection_offsets,
                rotational_plan_cache == nullptr ? d_bispectrum_projection_u_indices.get()
                    : rotational_plan.projection_u_indices,
                rotational_plan_cache == nullptr ? d_bispectrum_projection_z_indices.get()
                    : rotational_plan.projection_z_indices,
                rotational_plan_cache == nullptr ? d_bispectrum_projection_scales.get()
                    : rotational_plan.projection_scales,
                kind, config.nmax, config.lmax,
                config.twojmax, config.cutoff, config.rfac0, config.rmin0, config.rcutfac,
                config.normalize_u, static_cast<int>(features), batch.atoms(), output);
            check_cuda(cudaGetLastError(), "CUDA rotational kernel launch failed");
        }
    }
    const auto values = download_output_with_gil_release(context, size);
    return atom_result(values, batch.atoms(), features, name, options, false,
        std::vector<I64>(host_batch.offsets, host_batch.offsets + host_batch.structures + 1));
}

} // namespace

py::dict compute_extended_rotational(
    CudaExecutionContext& context,
    DeviceBatch& batch,
    DeviceNeighborGraph& graph,
    const detail::StructureBatchView& host_batch,
    const std::string& name,
    const py::dict& options,
    const py::object& control,
    RotationalPlanCache* rotational_plan) {
    (void)control;
    return compute_rotational_descriptor(
        context, batch, graph, host_batch, name, options, rotational_plan);
}

} // namespace mdescriptor::cuda
