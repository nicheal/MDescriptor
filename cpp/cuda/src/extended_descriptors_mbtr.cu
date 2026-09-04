#include "extended_descriptors_common.cuh"

__device__ void add_histogram_device(
    double* target,
    double value,
    double weight,
    double grid_min,
    double grid_max,
    double grid_sigma,
    int grid_n,
    bool normalize) {
    mdescriptor::detail::mbtr::add_histogram(
        target, value, weight, grid_min, grid_max, grid_sigma, grid_n, normalize);
}

__device__ double mbtr_weight_device(
    int weighting,
    double scale,
    double threshold,
    double r_cut,
    double sharpness,
    double first,
    double second,
    double third) {
    return mdescriptor::detail::mbtr::weight(
        weighting, scale, threshold, r_cut, sharpness, first, second, third);
}

__device__ void normalize_mbtr_device(
    double* values,
    I64 features,
    int normalization,
    int atom_count,
    const int* species_counts,
    int species_count,
    double volume,
    int geometry,
    int grid_n,
    bool local) {
    if (normalization == mdescriptor::detail::mbtr::kNormalizationL2) {
        mdescriptor::detail::mbtr::normalize_l2(values, features);
    } else if (normalization == mdescriptor::detail::mbtr::kNormalizationNAtoms) {
        mdescriptor::detail::mbtr::normalize_n_atoms(values, features, atom_count);
    } else if (normalization == mdescriptor::detail::mbtr::kNormalizationValleOganov) {
        mdescriptor::detail::mbtr::normalize_valle_oganov(
            values, volume, species_counts, species_count, geometry, grid_n, local);
    }
}

__global__ void mbtr_kernel(
    const I32* numbers,
    const I32* atom_types,
    const double* cells,
    const I64* offsets,
    const I64* graph_offsets,
    const I32* graph_atoms,
    const I32* graph_shifts,
    const double* graph_displacements,
    const double* graph_distance2,
    int species_count,
    int geometry,
    int weighting,
    int normalization,
    double grid_min,
    double grid_max,
    double grid_sigma,
    int grid_n,
    bool normalize_gaussians,
    double scale,
    double threshold,
    double r_cut,
    double sharpness,
    bool local,
    I64 structures,
    I64 atoms,
    I64 features,
    double* output) {
    const I64 row = static_cast<I64>(blockIdx.x) * blockDim.x + threadIdx.x;
    const I64 rows = local ? atoms : structures;
    if (row >= rows) return;
    double* target = output + row * features;
    I64 begin = 0;
    I64 end = 0;
    I64 structure = row;
    if (local) {
        begin = row;
        end = row + 1;
        structure = 0;
        while (structure + 1 < structures && offsets[structure + 1] <= row) ++structure;
    } else {
        begin = offsets[row];
        end = offsets[row + 1];
    }
    const int atom_count = static_cast<int>(offsets[structure + 1] - offsets[structure]);
    int species_counts[64]{};
    const bool needs_species_counts = normalization
        == mdescriptor::detail::mbtr::kNormalizationValleOganov
        && !local && geometry != mbtr::kGeometryAtomicNumber;
    if (needs_species_counts && species_count <= 64) {
        for (I64 atom = offsets[structure]; atom < offsets[structure + 1]; ++atom) {
            const int type = atom_types[atom];
            if (type >= 0) ++species_counts[type];
        }
    }
    if (geometry == mbtr::kGeometryAtomicNumber) {
        if (local) {
            const int type = atom_types[row];
            if (type >= 0) add_histogram_device(
                target + type * grid_n, static_cast<double>(numbers[row]), 1.0,
                grid_min, grid_max, grid_sigma, grid_n, normalize_gaussians);
        } else {
            for (I64 atom = offsets[row]; atom < offsets[row + 1]; ++atom) {
                const int type = atom_types[atom];
                if (type >= 0) add_histogram_device(
                    target + type * grid_n, static_cast<double>(numbers[atom]), 1.0,
                    grid_min, grid_max, grid_sigma, grid_n, normalize_gaussians);
            }
        }
        normalize_mbtr_device(
            target, features, normalization, atom_count, species_counts,
            species_count, 0.0, geometry, grid_n, local);
        return;
    }
    if (local) {
        const I64 center = row;
        const I64 graph_begin = graph_offsets[center];
        const I64 graph_end = graph_offsets[center + 1];
        if (geometry == mbtr::kGeometryDistance
            || geometry == mbtr::kGeometryInverseDistance) {
            for (I64 edge = graph_begin; edge < graph_end; ++edge) {
                const I32 atom = graph_atoms[edge];
                const double distance = sqrt(fmax(0.0, graph_distance2[edge]));
                if (distance <= 1e-12
                    || (atom == center && graph_shifts[edge * 3] == 0
                        && graph_shifts[edge * 3 + 1] == 0
                        && graph_shifts[edge * 3 + 2] == 0)) continue;
                const int type = atom_types[atom];
                if (type < 0) continue;
                const double value = geometry == mbtr::kGeometryDistance
                    ? distance : 1.0 / distance;
                add_histogram_device(
                    target + (type + 1) * grid_n, value,
                    mbtr_weight_device(weighting, scale, threshold, r_cut, sharpness,
                        distance, 0.0, 0.0), grid_min, grid_max, grid_sigma,
                    grid_n, normalize_gaussians);
            }
        } else {
            const int element_count = species_count + 1;
            const int reserved = element_count * (element_count + 1) / 2;
            for (I64 first = graph_begin; first < graph_end; ++first) {
                const double first_distance = sqrt(fmax(0.0, graph_distance2[first]));
                if (first_distance <= 1e-12) continue;
                for (I64 second = graph_begin; second < first; ++second) {
                    const double second_distance = sqrt(fmax(0.0, graph_distance2[second]));
                    if (second_distance <= 1e-12) continue;
                    const double dx = graph_displacements[first * 3] - graph_displacements[second * 3];
                    const double dy = graph_displacements[first * 3 + 1] - graph_displacements[second * 3 + 1];
                    const double dz = graph_displacements[first * 3 + 2] - graph_displacements[second * 3 + 2];
                    const double third_distance = sqrt(dx * dx + dy * dy + dz * dz);
                    const double cosine = fmin(1.0, fmax(-1.0,
                        (first_distance * first_distance + second_distance * second_distance
                            - third_distance * third_distance)
                            / (2.0 * first_distance * second_distance)));
                    const double value = geometry == mbtr::kGeometryCosine ? cosine
                        : acos(cosine) * 180.0 / kPi;
                    const int first_type = atom_types[graph_atoms[first]] + 1;
                    const int second_type = atom_types[graph_atoms[second]] + 1;
                    if (first_type <= 0 || second_type <= 0) continue;
                    const double weight = mbtr_weight_device(
                        weighting, scale, threshold, r_cut, sharpness,
                        first_distance, second_distance, third_distance);
                    add_histogram_device(
                        target + pair_channel_device(first_type, second_type, element_count) * grid_n,
                        value, weight, grid_min, grid_max, grid_sigma,
                        grid_n, normalize_gaussians);
                    add_histogram_device(
                        target + (reserved + (first_type - 1) * element_count + second_type) * grid_n,
                        (geometry == mbtr::kGeometryCosine ? cosine
                            : acos(fmin(1.0, fmax(-1.0,
                                (first_distance * first_distance + third_distance * third_distance
                                    - second_distance * second_distance)
                                    / (2.0 * first_distance * third_distance))))
                                * 180.0 / kPi),
                        weight, grid_min, grid_max, grid_sigma, grid_n, normalize_gaussians);
                    add_histogram_device(
                        target + (reserved + (second_type - 1) * element_count + first_type) * grid_n,
                        (geometry == mbtr::kGeometryCosine ? cosine
                            : acos(fmin(1.0, fmax(-1.0,
                                (second_distance * second_distance + third_distance * third_distance
                                    - first_distance * first_distance)
                                    / (2.0 * second_distance * third_distance))))
                                * 180.0 / kPi),
                        weight, grid_min, grid_max, grid_sigma, grid_n, normalize_gaussians);
                }
            }
        }
        normalize_mbtr_device(
            target, features, normalization, atom_count, species_counts,
            species_count, 0.0, geometry, grid_n, local);
        return;
    }
    const int pair_count = species_count * (species_count + 1) / 2;
    // A non-local MBTR row is one structure, so all of its centers contribute
    // to the same target histogram.  The local branch above intentionally
    // handles one center per CUDA thread.
    for (I64 center = begin; center < end; ++center) {
        const int center_type = atom_types[center];
        if (center_type < 0) continue;
        const I64 graph_begin = graph_offsets[center];
        const I64 graph_end = graph_offsets[center + 1];
        if (geometry == mbtr::kGeometryDistance
            || geometry == mbtr::kGeometryInverseDistance) {
            for (I64 first = graph_begin; first < graph_end; ++first) {
                const I32 first_atom = graph_atoms[first];
                const double distance = sqrt(fmax(0.0, graph_distance2[first]));
                if (distance <= 1e-12) continue;
                const bool periodic = graph_shifts[first * 3] != 0
                    || graph_shifts[first * 3 + 1] != 0
                    || graph_shifts[first * 3 + 2] != 0;
                if (!periodic && first_atom < center) continue;
                const int first_type = atom_types[first_atom];
                if (first_type < 0) continue;
                const double value = geometry == mbtr::kGeometryDistance
                    ? distance : 1.0 / distance;
                const double pair_weight = mbtr_weight_device(
                    weighting, scale, threshold, r_cut, sharpness, distance, 0.0, 0.0)
                    * (periodic ? 0.5 : 1.0);
                add_histogram_device(
                    target + pair_channel_device(center_type, first_type, species_count) * grid_n,
                    value, pair_weight, grid_min, grid_max, grid_sigma,
                    grid_n, normalize_gaussians);
            }
        } else {
            for (I64 first = graph_begin; first < graph_end; ++first) {
                const double first_distance = sqrt(fmax(0.0, graph_distance2[first]));
                if (first_distance <= 1e-12) continue;
                for (I64 second = graph_begin; second < first; ++second) {
                    const double second_distance = sqrt(fmax(0.0, graph_distance2[second]));
                    if (second_distance <= 1e-12) continue;
                    const double dx = graph_displacements[first * 3] - graph_displacements[second * 3];
                    const double dy = graph_displacements[first * 3 + 1] - graph_displacements[second * 3 + 1];
                    const double dz = graph_displacements[first * 3 + 2] - graph_displacements[second * 3 + 2];
                    const double third_distance = sqrt(dx * dx + dy * dy + dz * dz);
                    const double cosine = fmin(1.0, fmax(-1.0,
                        (first_distance * first_distance + second_distance * second_distance
                            - third_distance * third_distance)
                            / (2.0 * first_distance * second_distance)));
                    const double value = geometry == mbtr::kGeometryCosine ? cosine
                        : acos(cosine) * 180.0 / kPi;
                    const int first_type = atom_types[graph_atoms[first]];
                    const int second_type = atom_types[graph_atoms[second]];
                    if (first_type < 0 || second_type < 0) continue;
                    const double weight = mbtr_weight_device(
                        weighting, scale, threshold, r_cut, sharpness,
                        first_distance, second_distance, third_distance);
                    const int channel = center_type * pair_count
                        + pair_channel_device(first_type, second_type, species_count);
                    add_histogram_device(
                        target + channel * grid_n, value, weight, grid_min, grid_max,
                        grid_sigma, grid_n, normalize_gaussians);
                }
            }
        }
    }
    const double* cell = cells + structure * 9;
    const double volume = cell_volume_device(cell);
    normalize_mbtr_device(
        target, features, normalization, atom_count, species_counts,
        species_count, volume, geometry, grid_n, false);
}

double nested_number(
    const py::dict& object, const char* key, double fallback) {
    const py::str name(key);
    if (!object.contains(name) || object[name].is_none()) return fallback;
    return py::cast<double>(object[name]);
}

std::string nested_string(
    const py::dict& object, const char* key, const std::string& fallback) {
    const py::str name(key);
    if (!object.contains(name) || object[name].is_none()) return fallback;
    return py::cast<std::string>(object[name]);
}

py::dict nested_dict_option(
    const py::dict& options, const char* key) {
    const py::str name(key);
    if (!options.contains(name) || options[name].is_none()) return py::dict();
    try {
        return py::cast<py::dict>(options[name]);
    } catch (const py::cast_error&) {
        throw std::invalid_argument(std::string(key) + " must be an object");
    }
}

py::dict mbtr_config_option(const py::dict& options) {
    const py::str payload_key("_cuda_payload");
    const py::str config_key("mbtr_config");
    if (!options.contains(payload_key) || options[payload_key].is_none()) {
        return py::dict();
    }
    py::dict payload;
    try {
        payload = py::cast<py::dict>(options[payload_key]);
    } catch (const py::cast_error&) {
        throw std::invalid_argument("_cuda_payload must be an object");
    }
    if (!payload.contains(config_key) || payload[config_key].is_none()) {
        return py::dict();
    }
    try {
        return py::cast<py::dict>(payload[config_key]);
    } catch (const py::cast_error&) {
        throw std::invalid_argument("_cuda_payload.mbtr_config must be an object");
    }
}

py::dict compute_mbtr_descriptor(
    CudaExecutionContext& context,
    DeviceBatch& batch,
    DeviceNeighborGraph& graph,
    const detail::StructureBatchView& host_batch,
    const std::string& name,
    const py::dict& options) {
    const py::dict canonical = mbtr_config_option(options);
    const bool has_canonical = py::len(canonical) != 0;
    const auto species = has_canonical ? species_option(canonical) : species_option(options);
    if (species.empty()) throw std::invalid_argument(name + " species must not be empty");
    int geometry = mbtr::kGeometryDistance;
    int weighting = mbtr::kWeightingUnity;
    int normalization = mbtr::kNormalizationNone;
    double grid_min = 0.0;
    double grid_max = 6.0;
    double grid_sigma = 0.1;
    int grid_n = 50;
    double scale = 0.5;
    double threshold = 1e-3;
    double r_cut = 0.0;
    double sharpness = 2.0;
    bool normalize_gaussians = option(options, "normalize_gaussians", true);
    bool local = name == "LMBTR";
    if (has_canonical) {
        geometry = option(canonical, "geometry", mbtr::kGeometryDistance);
        weighting = option(canonical, "weighting", mbtr::kWeightingUnity);
        normalization = option(canonical, "normalization", mbtr::kNormalizationNone);
        grid_min = option(canonical, "grid_min", 0.0);
        grid_max = option(canonical, "grid_max", 6.0);
        grid_sigma = option(canonical, "grid_sigma", 0.1);
        grid_n = option(canonical, "grid_n", 50);
        normalize_gaussians = option(canonical, "normalize_gaussians", true);
        scale = option(canonical, "scale", 0.5);
        threshold = option(canonical, "threshold", 1e-3);
        r_cut = option(canonical, "r_cut", 0.0);
        sharpness = option(canonical, "sharpness", 2.0);
        local = option(canonical, "local", local);
    } else if (name == "ValleOganov") {
        const std::string function = option(options, "function", std::string("distance"));
        geometry = function == "angle" ? mbtr::kGeometryAngle : mbtr::kGeometryDistance;
        grid_n = option(options, "n", 50);
        grid_sigma = option(options, "sigma", 0.1);
        r_cut = option(options, "r_cut", 6.0);
        grid_min = 0.0;
        grid_max = geometry == mbtr::kGeometryAngle ? 180.0 : r_cut;
        weighting = geometry == mbtr::kGeometryAngle
            ? mbtr::kWeightingSmoothCutoff : mbtr::kWeightingInverseSquare;
        sharpness = 2.0;
        normalization = mbtr::kNormalizationValleOganov;
        const std::string normalization_name = option(
            options, "normalization", std::string("valle_oganov"));
        if (normalization_name == "none") normalization = mbtr::kNormalizationNone;
        else if (normalization_name == "l2") normalization = mbtr::kNormalizationL2;
        else if (normalization_name == "n_atoms") normalization = mbtr::kNormalizationNAtoms;
    } else {
        const py::dict geometry_object = nested_dict_option(options, "geometry");
        const py::dict grid_object = nested_dict_option(options, "grid");
        const py::dict weighting_object = nested_dict_option(options, "weighting");
        const std::string geometry_name = nested_string(
            geometry_object, "function", "distance");
        if (geometry_name == "atomic_number") geometry = mbtr::kGeometryAtomicNumber;
        else if (geometry_name == "distance") geometry = mbtr::kGeometryDistance;
        else if (geometry_name == "inverse_distance") geometry = mbtr::kGeometryInverseDistance;
        else if (geometry_name == "angle") geometry = mbtr::kGeometryAngle;
        else if (geometry_name == "cosine") geometry = mbtr::kGeometryCosine;
        else throw std::invalid_argument("unsupported CUDA MBTR geometry");
        const std::string weighting_name = nested_string(
            weighting_object, "function", "unity");
        if (weighting_name == "unity" || weighting_name == "none") weighting = mbtr::kWeightingUnity;
        else if (weighting_name == "exp") weighting = mbtr::kWeightingExponential;
        else if (weighting_name == "inverse_square") weighting = mbtr::kWeightingInverseSquare;
        else if (weighting_name == "smooth_cutoff") weighting = mbtr::kWeightingSmoothCutoff;
        else throw std::invalid_argument("unsupported CUDA MBTR weighting");
        grid_min = nested_number(grid_object, "min", 0.0);
        grid_max = nested_number(grid_object, "max", 6.0);
        grid_sigma = nested_number(grid_object, "sigma", 0.1);
        grid_n = static_cast<int>(nested_number(grid_object, "n", 50));
        scale = nested_number(weighting_object, "scale", 0.5);
        threshold = nested_number(weighting_object, "threshold", 1e-3);
        const double default_cutoff = weighting == mbtr::kWeightingUnity ? 0.0 : grid_max;
        r_cut = nested_number(weighting_object, "r_cut", default_cutoff);
        sharpness = nested_number(weighting_object, "sharpness", 2.0);
        const std::string normalization_name = option(
            options, "normalization", std::string("none"));
        if (normalization_name == "l2") normalization = mbtr::kNormalizationL2;
        else if (normalization_name == "n_atoms") normalization = mbtr::kNormalizationNAtoms;
        else if (normalization_name == "valle_oganov") normalization = mbtr::kNormalizationValleOganov;
    }
    if (geometry != mbtr::kGeometryAtomicNumber && r_cut <= 0.0) {
        r_cut = grid_max;
    }
    if (weighting == mbtr::kWeightingExponential && scale > 0.0) {
        const double multiplier = geometry == mbtr::kGeometryAngle
            || geometry == mbtr::kGeometryCosine ? 0.5 : 1.0;
        r_cut = std::max(r_cut, multiplier * -log(threshold) / scale);
    }
    const bool distance_geometry = geometry == mbtr::kGeometryDistance
        || geometry == mbtr::kGeometryInverseDistance;
    const int species_count = static_cast<int>(species.size());
    const int pair_count = species_count * (species_count + 1) / 2;
    const I64 channels = geometry == mbtr::kGeometryAtomicNumber ? species_count
        : local ? (distance_geometry ? species_count + 1
            : (species_count + 1) * (3 * (species_count + 1) - 1) / 2)
        : distance_geometry ? pair_count : species_count * pair_count;
    const I64 features = channels * grid_n;
    const I64 rows = local ? batch.atoms() : batch.structures();
    const std::size_t size = static_cast<std::size_t>(rows)
        * static_cast<std::size_t>(features);
    double* output = context.output_buffer(size);
    check_cuda(cudaMemsetAsync(output, 0, size * sizeof(double), context.stream()),
        "could not clear CUDA MBTR output");
    // Resolve element channels once on the host.  The previous kernel did a
    // linear species-table search for every edge of every angle, which made
    // the lookup part of the hottest inner loop.
    std::vector<I32> atom_types(static_cast<std::size_t>(batch.atoms()), -1);
    for (I64 atom = 0; atom < batch.atoms(); ++atom) {
        const auto found = std::find(species.begin(), species.end(), host_batch.numbers[atom]);
        if (found != species.end()) {
            atom_types[static_cast<std::size_t>(atom)] = static_cast<I32>(
                std::distance(species.begin(), found));
        }
    }
    DeviceBuffer<I32> d_atom_types;
    d_atom_types.upload(
        atom_types.data(), atom_types.size(), context.stream(),
        "could not upload MBTR atom types");
    if (geometry != mbtr::kGeometryAtomicNumber) {
        graph.build_dpa(context, batch, host_batch, r_cut, true, false, false);
    }
    if (rows > 0) {
        constexpr unsigned block_size = 64;
        mbtr_kernel<<<static_cast<unsigned>((rows + block_size - 1) / block_size),
            block_size, 0, context.stream()>>>(
            batch.numbers(), d_atom_types.get(), batch.cells(), batch.offsets(), graph.offsets(), graph.atoms(),
            graph.shifts(), graph.displacements(), graph.distance2(),
            species_count, geometry, weighting, normalization, grid_min, grid_max,
            grid_sigma, grid_n, normalize_gaussians, scale, threshold, r_cut, sharpness,
            local, batch.structures(), batch.atoms(), features, output);
        check_cuda(cudaGetLastError(), "CUDA MBTR kernel launch failed");
    }
    const auto values = download_output_with_gil_release(context, size);
    if (local) {
        return atom_result(values, rows, features, name, options, false,
            std::vector<I64>(host_batch.offsets, host_batch.offsets + host_batch.structures + 1));
    }
    py::dict result;
    result["values"] = values_array(values, rows, features);
    result["level"] = "structure";
    result["labels"] = labels_option(options, name, features);
    result["metadata"] = metadata(options, name);
    return result;
}


} // namespace

py::dict compute_extended_mbtr(
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
    return compute_mbtr_descriptor(context, batch, graph, host_batch, name, options);
}

} // namespace mdescriptor::cuda
