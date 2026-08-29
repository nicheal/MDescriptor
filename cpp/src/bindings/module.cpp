#include "mdescriptor/descriptor.hpp"
#include "mdescriptor/ace.hpp"
#include "mdescriptor/dpa4.hpp"
#include "mdescriptor/dpa4c.hpp"
#include "mdescriptor/extra.hpp"
#include "mdescriptor/local_descriptors.hpp"
#include "mdescriptor/nep.hpp"
#include "mdescriptor/neighbor.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <limits>
#include <string>
#include <vector>

namespace py = pybind11;
using mdescriptor::AcsfCalculator;
using mdescriptor::AcsfOptions;
using mdescriptor::AceCalculator;
using mdescriptor::AceOptions;
using mdescriptor::ComputeControl;
using mdescriptor::MtpCalculator;
using mdescriptor::MtpOptions;
using mdescriptor::NepCalculator;
using mdescriptor::NepOptions;
using mdescriptor::SoapCalculator;
using mdescriptor::SoapOptions;
using mdescriptor::SoapTurboCalculator;
using mdescriptor::SoapTurboOptions;
using mdescriptor::C00PSMlffCalculator;
using mdescriptor::C00PSMlffOptions;
using mdescriptor::Dpa4cCalculator;
using mdescriptor::Dpa4cOptions;
using mdescriptor::Dpa4Calculator;
using mdescriptor::Dpa4Options;
using mdescriptor::Dpa4BlockOptions;
using mdescriptor::StructureBatchView;

namespace {

using I32Array = py::array_t<std::int32_t, py::array::c_style | py::array::forcecast>;
using I64Array = py::array_t<std::int64_t, py::array::c_style | py::array::forcecast>;
using F64Array = py::array_t<double, py::array::c_style | py::array::forcecast>;

template <typename Value>
std::vector<Value> vector_from_array(py::handle value, const char* name) {
    using Array = py::array_t<Value, py::array::c_style | py::array::forcecast>;
    auto array = Array::ensure(value);
    if (!array) {
        throw std::invalid_argument(std::string(name) + " must be a numeric array");
    }
    const auto info = array.request();
    const auto* data = static_cast<const Value*>(info.ptr);
    return std::vector<Value>(data, data + info.size);
}

py::handle required_payload_value(const py::dict& payload, const char* name) {
    if (!payload.contains(name)) {
        throw std::invalid_argument(std::string("DPA4C payload is missing ") + name);
    }
    return payload[name];
}

Dpa4cOptions dpa4c_options_from_payload(const py::dict& payload) {
    Dpa4cOptions options;
    options.rcut = py::cast<double>(required_payload_value(payload, "rcut"));
    options.ntypes = py::cast<int>(required_payload_value(payload, "ntypes"));
    options.channels = py::cast<int>(required_payload_value(payload, "channels"));
    options.lmax = py::cast<int>(required_payload_value(payload, "lmax"));
    options.n_radial = py::cast<int>(required_payload_value(payload, "n_radial"));
    options.radial_modes = py::cast<int>(required_payload_value(payload, "radial_modes"));
    options.radial_hidden = py::cast<int>(required_payload_value(payload, "radial_hidden"));
    options.pair_hidden = py::cast<int>(required_payload_value(payload, "pair_hidden"));
    if (payload.contains("num_threads")) {
        options.num_threads = py::cast<int>(payload["num_threads"]);
    }
    if (payload.contains("calibrate")) {
        options.calibrate = py::cast<bool>(payload["calibrate"]);
    }

    options.type_embedding = vector_from_array<float>(
        required_payload_value(payload, "type_embedding"), "type_embedding");
    options.radial_freqs = vector_from_array<float>(
        required_payload_value(payload, "radial_freqs"), "radial_freqs");
    options.radial_w0 = vector_from_array<float>(
        required_payload_value(payload, "radial_w0"), "radial_w0");
    options.radial_w1 = vector_from_array<float>(
        required_payload_value(payload, "radial_w1"), "radial_w1");
    options.radial_mode_w = vector_from_array<float>(
        required_payload_value(payload, "radial_mode_w"), "radial_mode_w");
    options.pair_w0 = vector_from_array<float>(
        required_payload_value(payload, "pair_w0"), "pair_w0");
    options.pair_w1 = vector_from_array<float>(
        required_payload_value(payload, "pair_w1"), "pair_w1");

    options.degree_channels = py::cast<std::vector<int>>(
        required_payload_value(payload, "degree_channels"));
    options.bispectrum_ranks = py::cast<std::vector<int>>(
        required_payload_value(payload, "bispectrum_ranks"));
    options.readout_alignment = vector_from_array<float>(
        required_payload_value(payload, "readout_alignment"), "readout_alignment");
    options.readout_projections = vector_from_array<float>(
        required_payload_value(payload, "readout_projections"), "readout_projections");
    options.readout_alignment_offsets = vector_from_array<std::int64_t>(
        required_payload_value(payload, "readout_alignment_offsets"),
        "readout_alignment_offsets");
    options.readout_projection_offsets = vector_from_array<std::int64_t>(
        required_payload_value(payload, "readout_projection_offsets"),
        "readout_projection_offsets");

    options.bispectrum_coupling = vector_from_array<float>(
        required_payload_value(payload, "bispectrum_coupling"), "bispectrum_coupling");
    options.coupling_offsets = vector_from_array<std::int64_t>(
        required_payload_value(payload, "coupling_offsets"), "coupling_offsets");
    options.degree_triples = py::cast<std::vector<int>>(
        required_payload_value(payload, "degree_triples"));
    options.probe_offsets = vector_from_array<std::int64_t>(
        required_payload_value(payload, "probe_offsets"), "probe_offsets");
    options.probe_index = vector_from_array<std::int64_t>(
        required_payload_value(payload, "probe_index"), "probe_index");
    options.probe_scale = vector_from_array<float>(
        required_payload_value(payload, "probe_scale"), "probe_scale");
    options.output_mean = vector_from_array<float>(
        required_payload_value(payload, "output_mean"), "output_mean");
    options.output_stddev = vector_from_array<float>(
        required_payload_value(payload, "output_stddev"), "output_stddev");
    return options;
}

py::handle dpa4_required_payload_value(const py::dict& payload, const char* name) {
    if (!payload.contains(name)) {
        throw std::invalid_argument(std::string("DPA4 payload is missing ") + name);
    }
    return payload[name];
}

std::vector<std::vector<float>> dpa4_float_sequence(
    py::handle value,
    std::size_t expected_count,
    const char* name) {
    py::sequence sequence = py::cast<py::sequence>(value);
    if (sequence.size() != static_cast<py::ssize_t>(expected_count)) {
        throw std::invalid_argument(std::string("DPA4 ") + name + " has an unexpected count");
    }
    std::vector<std::vector<float>> result;
    result.reserve(expected_count);
    for (py::ssize_t index = 0; index < sequence.size(); ++index) {
        result.push_back(vector_from_array<float>(sequence[index], name));
    }
    return result;
}

Dpa4Options dpa4_options_from_payload(const py::dict& payload) {
    Dpa4Options options;
    options.rcut = py::cast<double>(dpa4_required_payload_value(payload, "rcut"));
    options.ntypes = py::cast<int>(dpa4_required_payload_value(payload, "ntypes"));
    options.channels = py::cast<int>(dpa4_required_payload_value(payload, "channels"));
    options.n_radial = py::cast<int>(dpa4_required_payload_value(payload, "n_radial"));
    if (payload.contains("num_threads")) {
        options.num_threads = py::cast<int>(payload["num_threads"]);
    }

    auto required_float = [&](const char* name) {
        return vector_from_array<float>(dpa4_required_payload_value(payload, name), name);
    };
    options.type_embedding = required_float("type_embedding");
    options.env_rbf_layer1 = required_float("env_rbf_layer1");
    options.env_rbf_layer2 = required_float("env_rbf_layer2");
    options.env_type_embedding = required_float("env_type_embedding");
    options.env_g_layer1 = required_float("env_g_layer1");
    options.env_g_layer2 = required_float("env_g_layer2");
    options.env_output_projection = required_float("env_output_projection");
    options.film_scale_norm = required_float("film_scale_norm");
    options.film_shift_norm = required_float("film_shift_norm");
    options.film_scale_strength_log = py::cast<float>(
        dpa4_required_payload_value(payload, "film_scale_strength_log"));
    options.film_shift_strength_log = py::cast<float>(
        dpa4_required_payload_value(payload, "film_shift_strength_log"));
    options.radial_freqs = required_float("radial_freqs");
    options.radial_layer1 = required_float("radial_layer1");
    options.radial_norm_scale = required_float("radial_norm_scale");
    options.radial_layer2 = required_float("radial_layer2");
    options.wigner_l2_tensor = required_float("wigner_l2_tensor");
    options.wigner_l3_coefficients = required_float("wigner_l3_coefficients");
    options.wigner_l3_exponents = vector_from_array<std::int64_t>(
        dpa4_required_payload_value(payload, "wigner_l3_exponents"),
        "wigner_l3_exponents");
    options.gie_row_index = vector_from_array<std::int64_t>(
        dpa4_required_payload_value(payload, "gie_row_index"), "gie_row_index");
    options.gie_m0_index = vector_from_array<std::int64_t>(
        dpa4_required_payload_value(payload, "gie_m0_index"), "gie_m0_index");
    options.gie_radial_index = vector_from_array<std::int64_t>(
        dpa4_required_payload_value(payload, "gie_radial_index"), "gie_radial_index");
    options.grid_to = required_float("grid_to");
    options.grid_from = required_float("grid_from");
    options.output_linear1 = required_float("output_linear1");
    options.output_linear2 = required_float("output_linear2");
    options.output_scalar_gate = required_float("output_scalar_gate");
    options.output_grid_left = required_float("output_grid_left");
    options.output_grid_right = required_float("output_grid_right");
    options.output_grid_out = required_float("output_grid_out");

    py::sequence blocks = py::cast<py::sequence>(
        dpa4_required_payload_value(payload, "blocks"));
    if (blocks.size() != 3) {
        throw std::invalid_argument("DPA4 payload must contain three blocks");
    }
    for (py::ssize_t block_index = 0; block_index < blocks.size(); ++block_index) {
        const py::dict block_payload = py::cast<py::dict>(blocks[block_index]);
        Dpa4BlockOptions& block = options.blocks[static_cast<std::size_t>(block_index)];
        auto block_required = [&](const char* name) {
            return dpa4_required_payload_value(block_payload, name);
        };
        if (block_payload.contains("pre_norm_enabled")) {
            block.pre_norm_enabled = py::cast<bool>(block_payload["pre_norm_enabled"]);
        }
        if (block_payload.contains("post_norm_enabled")) {
            block.post_norm_enabled = py::cast<bool>(block_payload["post_norm_enabled"]);
        }
        if (block_payload.contains("ffn_norm_enabled")) {
            block.ffn_norm_enabled = py::cast<bool>(block_payload["ffn_norm_enabled"]);
        }
        auto block_float = [&](const char* name) {
            return vector_from_array<float>(block_required(name), name);
        };
        block.pre_norm_scale = block_float("pre_norm_scale");
        block.pre_norm_bias = block_float("pre_norm_bias");
        block.pre_norm_balance = block_float("pre_norm_balance");
        block.post_norm_scale = block_float("post_norm_scale");
        block.post_norm_bias = block_float("post_norm_bias");
        block.post_norm_balance = block_float("post_norm_balance");
        block.ffn_norm_scale = block_float("ffn_norm_scale");
        block.ffn_norm_bias = block_float("ffn_norm_bias");
        block.ffn_norm_balance = block_float("ffn_norm_balance");
        block.pre_focus_weight = block_float("pre_focus_weight");
        block.post_focus_weight = block_float("post_focus_weight");
        block.radial_mixer_weight = block_float("radial_mixer_weight");
        block.radial_channel_basis = block_float("radial_channel_basis");

        const auto m0 = dpa4_float_sequence(block_required("so2_weight_m0"), 4, "so2_weight_m0");
        const auto m1 = dpa4_float_sequence(block_required("so2_weight_m1"), 4, "so2_weight_m1");
        const auto gates = dpa4_float_sequence(block_required("so2_gate_weight"), 3, "so2_gate_weight");
        for (std::size_t index = 0; index < 4; ++index) {
            block.so2_weight_m0[index] = m0[index];
            block.so2_weight_m1[index] = m1[index];
        }
        for (std::size_t index = 0; index < 3; ++index) {
            block.so2_gate_weight[index] = gates[index];
        }
        block.attn_qk_scale = block_float("attn_qk_scale");
        block.attn_q_weight = block_float("attn_q_weight");
        block.attn_k_weight = block_float("attn_k_weight");
        block.attn_output_gate_scale = block_float("attn_output_gate_scale");
        block.attn_logit_weight = block_float("attn_logit_weight");
        block.attn_z_bias_raw = block_float("attn_z_bias_raw");
        block.attn_gate_weight = block_float("attn_gate_weight");
        block.message_scalar_gate = block_float("message_scalar_gate");
        block.message_frame_expand = block_float("message_frame_expand");
        block.message_frame_contract = block_float("message_frame_contract");
        block.message_residual_scale = block_float("message_residual_scale");
        block.ffn_linear1 = block_float("ffn_linear1");
        block.ffn_linear2 = block_float("ffn_linear2");
        block.ffn_scalar_gate = block_float("ffn_scalar_gate");
        block.ffn_grid_left = block_float("ffn_grid_left");
        block.ffn_grid_right = block_float("ffn_grid_right");
        block.ffn_grid_router = block_float("ffn_grid_router");
        block.ffn_grid_out = block_float("ffn_grid_out");
    }
    return options;
}

StructureBatchView view_batch(
    const I32Array& numbers,
    const F64Array& positions,
    const F64Array& cells,
    const I32Array& pbc,
    const I64Array& offsets
) {
    if (numbers.ndim() != 1 || positions.ndim() != 2 || positions.shape(1) != 3
        || cells.ndim() != 3 || cells.shape(1) != 3 || cells.shape(2) != 3
        || pbc.ndim() != 2 || pbc.shape(1) != 3 || offsets.ndim() != 1
        || offsets.shape(0) < 1) {
        throw std::invalid_argument("invalid StructureBatch array shapes");
    }
    const auto structures = static_cast<std::int64_t>(offsets.shape(0) - 1);
    if (positions.shape(0) != numbers.shape(0) || cells.shape(0) != structures || pbc.shape(0) != structures) {
        throw std::invalid_argument("StructureBatch arrays have inconsistent lengths");
    }
    auto numbers_info = numbers.request();
    auto positions_info = positions.request();
    auto cells_info = cells.request();
    auto pbc_info = pbc.request();
    auto offsets_info = offsets.request();
    return {
        static_cast<const std::int32_t*>(numbers_info.ptr),
        static_cast<const double*>(positions_info.ptr),
        static_cast<const double*>(cells_info.ptr),
        static_cast<const std::int32_t*>(pbc_info.ptr),
        static_cast<const std::int64_t*>(offsets_info.ptr),
        structures,
        static_cast<std::int64_t>(numbers.shape(0)),
    };
}

std::shared_ptr<ComputeControl> control_or_default(const std::shared_ptr<ComputeControl>& control) {
    return control ? control : std::make_shared<ComputeControl>();
}

py::array compute_soap_array(
    const SoapCalculator& calculator,
    const I32Array& numbers,
    const F64Array& positions,
    const F64Array& cells,
    const I32Array& pbc,
    const I64Array& offsets,
    const std::shared_ptr<ComputeControl>& control,
    std::int32_t num_threads,
    bool inner_average,
    bool outer_average
) {
    const auto batch = view_batch(numbers, positions, cells, pbc, offsets);
    const auto rows = (inner_average || outer_average) ? batch.structures : batch.atoms;
    py::array_t<double> output({rows, calculator.feature_count()});
    auto output_info = output.request();
    auto ctrl = control_or_default(control);
    {
        py::gil_scoped_release release;
        (void)num_threads;
        calculator.compute(batch, static_cast<double*>(output_info.ptr), ctrl);
    }
    return output;
}

py::array compute_soap_turbo_array(
    const SoapTurboCalculator& calculator,
    const I32Array& numbers,
    const F64Array& positions,
    const F64Array& cells,
    const I32Array& pbc,
    const I64Array& offsets,
    const std::shared_ptr<ComputeControl>& control
) {
    const auto batch = view_batch(numbers, positions, cells, pbc, offsets);
    py::array_t<double> output({batch.atoms, calculator.feature_count()});
    auto output_info = output.request();
    auto ctrl = control_or_default(control);
    {
        py::gil_scoped_release release;
        calculator.compute(batch, static_cast<double*>(output_info.ptr), ctrl);
    }
    return output;
}

py::array compute_acsf_array(
    const AcsfCalculator& calculator,
    const I32Array& numbers,
    const F64Array& positions,
    const F64Array& cells,
    const I32Array& pbc,
    const I64Array& offsets,
    const std::shared_ptr<ComputeControl>& control
) {
    const auto batch = view_batch(numbers, positions, cells, pbc, offsets);
    py::array_t<double> output({batch.atoms, calculator.feature_count()});
    auto output_info = output.request();
    auto ctrl = control_or_default(control);
    {
        py::gil_scoped_release release;
        calculator.compute(batch, static_cast<double*>(output_info.ptr), ctrl);
    }
    return output;
}

py::array compute_c00ps_mlff_array(
    const C00PSMlffCalculator& calculator,
    const I32Array& numbers,
    const F64Array& positions,
    const F64Array& cells,
    const I32Array& pbc,
    const I64Array& offsets,
    const std::shared_ptr<ComputeControl>& control
) {
    const auto batch = view_batch(numbers, positions, cells, pbc, offsets);
    py::array_t<double> output({batch.atoms, calculator.feature_count()});
    auto ctrl = control_or_default(control);
    {
        py::gil_scoped_release release;
        calculator.compute(batch, output.mutable_data(), ctrl);
    }
    return output;
}

py::array compute_mtp_array(
    const MtpCalculator& calculator,
    const I32Array& numbers,
    const F64Array& positions,
    const F64Array& cells,
    const I32Array& pbc,
    const I64Array& offsets,
    const std::shared_ptr<ComputeControl>& control
) {
    const auto batch = view_batch(numbers, positions, cells, pbc, offsets);
    py::array_t<double> output({batch.atoms, calculator.feature_count()});
    auto ctrl = control_or_default(control);
    {
        py::gil_scoped_release release;
        calculator.compute(batch, output.mutable_data(), ctrl);
    }
    return output;
}

py::array compute_ace_array(
    const AceCalculator& calculator,
    const I32Array& numbers,
    const F64Array& positions,
    const F64Array& cells,
    const I32Array& pbc,
    const I64Array& offsets,
    const std::shared_ptr<ComputeControl>& control
) {
    const auto batch = view_batch(numbers, positions, cells, pbc, offsets);
    py::array_t<double> output({batch.atoms, calculator.feature_count()});
    auto ctrl = control_or_default(control);
    {
        py::gil_scoped_release release;
        calculator.compute(batch, output.mutable_data(), ctrl);
    }
    return output;
}

py::array compute_nep_array(
    const NepCalculator& calculator,
    const I32Array& numbers,
    const F64Array& positions,
    const F64Array& cells,
    const I32Array& pbc,
    const I64Array& offsets,
    const std::shared_ptr<ComputeControl>& control
) {
    const auto batch = view_batch(numbers, positions, cells, pbc, offsets);
    py::array_t<double> output({batch.atoms, calculator.feature_count()});
    auto output_info = output.request();
    auto ctrl = control_or_default(control);
    {
        py::gil_scoped_release release;
        calculator.compute(batch, static_cast<double*>(output_info.ptr), ctrl);
    }
    return output;
}

py::array compute_dpa4c_array(
    const Dpa4cCalculator& calculator,
    const I32Array& numbers,
    const F64Array& positions,
    const F64Array& cells,
    const I32Array& pbc,
    const I64Array& offsets,
    const I32Array& type_indices,
    const std::shared_ptr<ComputeControl>& control
) {
    const auto batch = view_batch(numbers, positions, cells, pbc, offsets);
    if (type_indices.ndim() != 1 || type_indices.shape(0) != batch.atoms) {
        throw std::invalid_argument("DPA4C type_indices must have one entry per atom");
    }
    py::array_t<double> output({batch.atoms, calculator.feature_count()});
    auto ctrl = control_or_default(control);
    ctrl->reset(batch.structures);
    {
        py::gil_scoped_release release;
        calculator.compute(batch, type_indices.data(), output.mutable_data(), ctrl);
    }
    return output;
}

py::array compute_dpa4_array(
    const Dpa4Calculator& calculator,
    const I32Array& numbers,
    const F64Array& positions,
    const F64Array& cells,
    const I32Array& pbc,
    const I64Array& offsets,
    const I32Array& type_indices,
    const std::shared_ptr<ComputeControl>& control
) {
    const auto batch = view_batch(numbers, positions, cells, pbc, offsets);
    if (type_indices.ndim() != 1 || type_indices.shape(0) != batch.atoms) {
        throw std::invalid_argument("DPA4 type_indices must have one entry per atom");
    }
    py::array_t<double> output({batch.atoms, calculator.feature_count()});
    auto ctrl = control_or_default(control);
    ctrl->reset(batch.structures);
    {
        py::gil_scoped_release release;
        calculator.compute(batch, type_indices.data(), output.mutable_data(), ctrl);
    }
    return output;
}

py::array compute_coulomb_matrix_array(
    const I32Array& numbers,
    const F64Array& positions,
    const F64Array& cells,
    const I32Array& pbc,
    const I64Array& offsets,
    std::int64_t n_atoms_max,
    const std::string& permutation,
    double exponent,
    std::int32_t num_threads,
    const std::shared_ptr<ComputeControl>& control
) {
    const auto batch = view_batch(numbers, positions, cells, pbc, offsets);
    if (n_atoms_max <= 0 || n_atoms_max > std::numeric_limits<std::int64_t>::max() / n_atoms_max) {
        throw std::invalid_argument("n_atoms_max must be a positive value");
    }
    const auto features = permutation == "eigenspectrum" ? n_atoms_max : n_atoms_max * n_atoms_max;
    py::array_t<double> output({batch.structures, features});
    auto output_info = output.request();
    std::fill(
        static_cast<double*>(output_info.ptr),
        static_cast<double*>(output_info.ptr) + batch.structures * features,
        0.0);
    auto ctrl = control_or_default(control);
    {
        py::gil_scoped_release release;
        mdescriptor::compute_coulomb_matrix(
            batch, n_atoms_max, permutation, exponent, num_threads,
            static_cast<double*>(output_info.ptr), ctrl);
    }
    return output;
}

py::array compute_matrix_array(
    const I32Array& numbers,
    const F64Array& positions,
    const F64Array& cells,
    const I32Array& pbc,
    const I64Array& offsets,
    std::int64_t n_atoms_max,
    const std::string& permutation,
    double exponent,
    std::int32_t kind,
    double accuracy,
    double w,
    double r_cut,
    double g_cut,
    double a,
    std::int32_t num_threads,
    const std::shared_ptr<ComputeControl>& control
) {
    const auto batch = view_batch(numbers, positions, cells, pbc, offsets);
    const auto columns = permutation == "eigenspectrum" ? n_atoms_max : n_atoms_max * n_atoms_max;
    py::array_t<double> output({batch.structures, columns});
    auto ctrl = control_or_default(control);
    {
        py::gil_scoped_release release;
        mdescriptor::compute_matrix(
            batch, n_atoms_max, permutation, exponent,
            static_cast<mdescriptor::MatrixKind>(kind), accuracy, w, r_cut, g_cut, a,
            num_threads,
            output.mutable_data(), ctrl);
    }
    return output;
}

py::array compute_mbtr_array(
    const I32Array& numbers,
    const F64Array& positions,
    const F64Array& cells,
    const I32Array& pbc,
    const I64Array& offsets,
    const std::vector<std::int32_t>& species,
    std::int32_t geometry,
    std::int32_t weighting,
    std::int32_t normalization,
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
    std::int32_t num_threads,
    const std::shared_ptr<ComputeControl>& control
) {
    const auto batch = view_batch(numbers, positions, cells, pbc, offsets);
    mdescriptor::MBTROptions options;
    options.species = species;
    options.geometry = static_cast<mdescriptor::MBTRGeometry>(geometry);
    options.weighting = static_cast<mdescriptor::MBTRWeighting>(weighting);
    options.normalization = static_cast<mdescriptor::MBTRNormalization>(normalization);
    options.grid_min = grid_min;
    options.grid_max = grid_max;
    options.grid_sigma = grid_sigma;
    options.grid_n = grid_n;
    options.normalize_gaussians = normalize_gaussians;
    options.scale = scale;
    options.threshold = threshold;
    options.r_cut = r_cut;
    options.sharpness = sharpness;
    options.local = local;
    options.num_threads = num_threads;
    const auto rows = local ? batch.atoms : batch.structures;
    const auto columns = mdescriptor::mbtr_feature_count(options);
    py::array_t<double> output({rows, columns});
    auto ctrl = control_or_default(control);
    {
        py::gil_scoped_release release;
        mdescriptor::compute_mbtr(batch, options, output.mutable_data(), ctrl);
    }
    return output;
}

py::array compute_ead_array(
    const I32Array& numbers,
    const F64Array& positions,
    const F64Array& cells,
    const I32Array& pbc,
    const I64Array& offsets,
    int max_degree,
    double cutoff,
    const std::vector<double>& eta,
    const std::vector<double>& rs,
    std::int32_t num_threads,
    const std::shared_ptr<ComputeControl>& control
) {
    const auto batch = view_batch(numbers, positions, cells, pbc, offsets);
    mdescriptor::EadOptions options;
    options.max_degree = max_degree;
    options.cutoff = cutoff;
    options.eta = eta;
    options.rs = rs;
    options.num_threads = num_threads;
    const auto columns = mdescriptor::ead_feature_count(options);
    py::array_t<double> output({batch.atoms, columns});
    auto ctrl = control_or_default(control);
    {
        py::gil_scoped_release release;
        mdescriptor::compute_ead(batch, options, output.mutable_data(), ctrl);
    }
    return output;
}

py::array compute_rotational_descriptors_array(
    const I32Array& numbers,
    const F64Array& positions,
    const F64Array& cells,
    const I32Array& pbc,
    const I64Array& offsets,
    std::int32_t kind,
    int n_max,
    int l_max,
    double cutoff,
    double alpha,
    bool weight_on,
    bool normalize_u,
    double weight_scale,
    int twojmax,
    int diagonal,
    std::int32_t num_threads,
    const std::shared_ptr<ComputeControl>& control,
    double rfac0,
    const std::vector<double>& neighbor_weights,
    double rmin0,
    double rcutfac,
    const std::vector<double>& neighbor_radii
) {
    const auto batch = view_batch(numbers, positions, cells, pbc, offsets);
    mdescriptor::RotationalDescriptorOptions options;
    options.kind = static_cast<mdescriptor::RotationalDescriptorKind>(kind);
    options.n_max = n_max;
    options.l_max = l_max;
    options.cutoff = cutoff;
    options.alpha = alpha;
    options.weight_on = weight_on;
    options.normalize_u = normalize_u;
    options.weight_scale = weight_scale;
    options.rfac0 = rfac0;
    options.rmin0 = rmin0;
    options.rcutfac = rcutfac;
    options.neighbor_weights = neighbor_weights;
    options.neighbor_radii = neighbor_radii;
    options.twojmax = twojmax;
    options.diagonal = diagonal;
    options.num_threads = num_threads;
    const auto columns = mdescriptor::rotational_feature_count(options);
    py::array_t<double> output({batch.atoms, columns});
    auto ctrl = control_or_default(control);
    {
        py::gil_scoped_release release;
        mdescriptor::compute_rotational_descriptors(batch, options, output.mutable_data(), ctrl);
    }
    return output;
}

py::array compute_atomic_composition_array(
    const I32Array& numbers,
    const F64Array& positions,
    const F64Array& cells,
    const I32Array& pbc,
    const I64Array& offsets,
    const std::vector<std::int32_t>& species,
    bool per_system,
    std::int32_t num_threads,
    const std::shared_ptr<ComputeControl>& control
) {
    const auto batch = view_batch(numbers, positions, cells, pbc, offsets);
    const auto rows = per_system ? batch.structures : batch.atoms;
    py::array_t<double> output({rows, static_cast<std::int64_t>(species.size())});
    auto output_info = output.request();
    auto ctrl = control_or_default(control);
    {
        py::gil_scoped_release release;
        mdescriptor::compute_atomic_composition(
            batch, species, per_system, num_threads,
            static_cast<double*>(output_info.ptr), ctrl);
    }
    return output;
}

py::array compute_sorted_distances_array(
    const I32Array& numbers,
    const F64Array& positions,
    const F64Array& cells,
    const I32Array& pbc,
    const I64Array& offsets,
    const std::vector<std::int32_t>& species,
    double cutoff,
    int max_neighbors,
    bool separate_neighbor_types,
    std::int32_t num_threads,
    const std::shared_ptr<ComputeControl>& control
) {
    const auto batch = view_batch(numbers, positions, cells, pbc, offsets);
    const auto columns = separate_neighbor_types
        ? static_cast<std::int64_t>(species.size()) * max_neighbors
        : max_neighbors;
    py::array_t<double> output({batch.atoms, columns});
    auto output_info = output.request();
    mdescriptor::LocalDescriptorOptions options;
    options.species = species;
    options.cutoff = cutoff;
    options.num_threads = num_threads;
    auto ctrl = control_or_default(control);
    {
        py::gil_scoped_release release;
        mdescriptor::compute_sorted_distances(
            batch, options, max_neighbors, separate_neighbor_types,
            static_cast<double*>(output_info.ptr), ctrl);
    }
    return output;
}

py::tuple compute_neighbor_list_array(
    const I32Array& numbers,
    const F64Array& positions,
    const F64Array& cells,
    const I32Array& pbc,
    const I64Array& offsets,
    double cutoff,
    bool full_neighbor_list,
    bool self_pairs,
    std::int32_t num_threads,
    const std::shared_ptr<ComputeControl>& control
) {
    const auto batch = view_batch(numbers, positions, cells, pbc, offsets);
    auto ctrl = control_or_default(control);
    mdescriptor::DescriptorPairTable pairs;
    {
        py::gil_scoped_release release;
        pairs = mdescriptor::compute_neighbor_list(
            batch, cutoff, full_neighbor_list, self_pairs, num_threads, ctrl);
    }
    py::array_t<double> values(std::vector<py::ssize_t>{
        static_cast<py::ssize_t>(pairs.values.size() / 9), 9});
    py::array_t<std::int64_t> pair_offsets(pairs.offsets.size());
    std::copy(pairs.values.begin(), pairs.values.end(), values.mutable_data());
    std::copy(pairs.offsets.begin(), pairs.offsets.end(), pair_offsets.mutable_data());
    return py::make_tuple(values, pair_offsets);
}

py::array compute_spherical_expansion_array(
    const I32Array& numbers,
    const F64Array& positions,
    const F64Array& cells,
    const I32Array& pbc,
    const I64Array& offsets,
    const std::vector<std::int32_t>& species,
    double cutoff,
    double density_width,
    int max_radial,
    int max_angular,
    std::int32_t kind,
    double k_cutoff,
    int exponent,
    double radial_radius,
    std::int32_t num_threads,
    const std::shared_ptr<ComputeControl>& control
) {
    const auto batch = view_batch(numbers, positions, cells, pbc, offsets);
    mdescriptor::LocalDescriptorOptions options;
    options.species = species;
    options.cutoff = cutoff;
    options.density_width = density_width;
    options.max_radial = max_radial;
    options.max_angular = max_angular;
    options.k_cutoff = k_cutoff;
    options.exponent = exponent;
    options.radial_radius = radial_radius > 0.0 ? radial_radius : cutoff;
    options.num_threads = num_threads;
    const auto descriptor_kind = static_cast<mdescriptor::LocalDescriptorKind>(kind);
    const auto features = mdescriptor::local_descriptor_feature_count(options, descriptor_kind);
    py::array_t<double> output({batch.atoms, features});
    auto output_info = output.request();
    auto ctrl = control_or_default(control);
    {
        py::gil_scoped_release release;
        mdescriptor::compute_spherical_expansion(
            batch, options, descriptor_kind, static_cast<double*>(output_info.ptr), ctrl);
    }
    return output;
}

py::tuple compute_spherical_expansion_by_pair_array(
    const I32Array& numbers,
    const F64Array& positions,
    const F64Array& cells,
    const I32Array& pbc,
    const I64Array& offsets,
    const std::vector<std::int32_t>& species,
    double cutoff,
    double density_width,
    int max_radial,
    int max_angular,
    std::int32_t num_threads,
    const std::shared_ptr<ComputeControl>& control
) {
    const auto batch = view_batch(numbers, positions, cells, pbc, offsets);
    mdescriptor::LocalDescriptorOptions options;
    options.species = species;
    options.cutoff = cutoff;
    options.density_width = density_width;
    options.max_radial = max_radial;
    options.max_angular = max_angular;
    options.num_threads = num_threads;
    const auto features = static_cast<std::int64_t>(
        (max_angular + 1) * (max_angular + 1) * (max_radial + 1));
    auto ctrl = control_or_default(control);
    mdescriptor::DescriptorPairTable pairs;
    {
        py::gil_scoped_release release;
        pairs = mdescriptor::compute_spherical_expansion_by_pair(batch, options, ctrl);
    }
    const auto rows = static_cast<py::ssize_t>(pairs.values.size() / static_cast<std::size_t>(9 + features));
    py::array_t<double> output(std::vector<py::ssize_t>{rows, static_cast<py::ssize_t>(features)});
    py::array_t<std::int64_t> pair_offsets(pairs.offsets.size());
    py::array_t<double> identifiers(std::vector<py::ssize_t>{rows, 9});
    auto output_ptr = output.mutable_data();
    auto identifiers_ptr = identifiers.mutable_data();
    for (py::ssize_t row = 0; row < rows; ++row) {
        const double* source = pairs.values.data() + static_cast<std::size_t>(row) * static_cast<std::size_t>(9 + features);
        std::copy(source, source + 9, identifiers_ptr + row * 9);
        std::copy(source + 9, source + 9 + features, output_ptr + row * features);
    }
    std::copy(pairs.offsets.begin(), pairs.offsets.end(), pair_offsets.mutable_data());
    return py::make_tuple(output, pair_offsets, identifiers);
}

py::tuple build_neighbor_graph_arrays(
    const I32Array& numbers,
    const F64Array& positions,
    const F64Array& cell,
    const I32Array& pbc,
    double cutoff
) {
    if (cell.ndim() != 2 || cell.shape(0) != 3 || cell.shape(1) != 3
        || pbc.ndim() != 1 || pbc.shape(0) != 3) {
        throw std::invalid_argument("invalid single-structure neighbor graph shapes");
    }
    if (positions.ndim() != 2 || positions.shape(1) != 3
        || positions.shape(0) != numbers.shape(0)) {
        throw std::invalid_argument("invalid single-structure neighbor positions");
    }
    const std::int64_t offsets_data[2] = {0, static_cast<std::int64_t>(numbers.shape(0))};
    StructureBatchView batch{
        static_cast<const std::int32_t*>(numbers.request().ptr),
        static_cast<const double*>(positions.request().ptr),
        static_cast<const double*>(cell.request().ptr),
        static_cast<const std::int32_t*>(pbc.request().ptr),
        offsets_data,
        1,
        static_cast<std::int64_t>(numbers.shape(0)),
    };
    mdescriptor::NeighborGraph graph;
    {
        py::gil_scoped_release release;
        graph = mdescriptor::build_neighbor_graph(batch, cutoff);
    }

    const auto& graph_offsets = graph.offsets();
    const auto& graph_atoms = graph.atoms_data();
    const auto& graph_shifts = graph.shifts();
    const auto& graph_displacements = graph.displacements();
    const auto& graph_distance2 = graph.distance2();
    py::array_t<std::int64_t> output_offsets(graph_offsets.size());
    py::array_t<std::int32_t> output_atoms(graph_atoms.size());
    py::array_t<std::int32_t> output_shifts({graph_atoms.size(), std::size_t(3)});
    py::array_t<double> output_displacements({graph_atoms.size(), std::size_t(3)});
    py::array_t<double> output_distance2(graph_distance2.size());
    std::copy(graph_offsets.begin(), graph_offsets.end(), output_offsets.mutable_data());
    std::copy(graph_atoms.begin(), graph_atoms.end(), output_atoms.mutable_data());
    std::copy(graph_shifts.begin(), graph_shifts.end(), output_shifts.mutable_data());
    std::copy(graph_displacements.begin(), graph_displacements.end(), output_displacements.mutable_data());
    std::copy(graph_distance2.begin(), graph_distance2.end(), output_distance2.mutable_data());
    return py::make_tuple(output_offsets, output_atoms, output_shifts, output_displacements, output_distance2);
}

} // namespace

PYBIND11_MODULE(_native, module) {
    module.doc() = "MDescriptor periodic descriptor kernels";
    py::register_exception<mdescriptor::CancelledError>(module, "CancelledError");

    py::class_<ComputeControl, std::shared_ptr<ComputeControl>>(module, "ComputeControl")
        .def(py::init<>())
        .def("cancel", &ComputeControl::cancel)
        .def("cancelled", &ComputeControl::cancelled)
        .def("completed", &ComputeControl::completed)
        .def("total", &ComputeControl::total)
        .def("mark_completed", &ComputeControl::mark_completed)
        .def("reset", &ComputeControl::reset);

    module.def("build_neighbor_graph", &build_neighbor_graph_arrays,
               py::arg("numbers"), py::arg("positions"), py::arg("cell"),
               py::arg("pbc"), py::arg("cutoff"));

    module.def("compute_coulomb_matrix", &compute_coulomb_matrix_array,
               py::arg("numbers"), py::arg("positions"), py::arg("cells"), py::arg("pbc"),
               py::arg("offsets"), py::arg("n_atoms_max"), py::arg("permutation"),
               py::arg("exponent") = 2.4, py::arg("num_threads") = 0,
               py::arg("control") = nullptr);

    module.def("compute_matrix", &compute_matrix_array,
               py::arg("numbers"), py::arg("positions"), py::arg("cells"), py::arg("pbc"),
               py::arg("offsets"), py::arg("n_atoms_max"), py::arg("permutation"),
               py::arg("exponent") = 2.4, py::arg("kind") = 0,
               py::arg("accuracy") = 1e-5, py::arg("w") = 1.0,
               py::arg("r_cut") = 0.0, py::arg("g_cut") = 0.0, py::arg("a") = 0.0,
               py::arg("num_threads") = 0,
               py::arg("control") = nullptr);

    module.def("compute_mbtr", &compute_mbtr_array,
               py::arg("numbers"), py::arg("positions"), py::arg("cells"), py::arg("pbc"),
               py::arg("offsets"), py::arg("species"), py::arg("geometry") = 1,
               py::arg("weighting") = 1, py::arg("normalization") = 0,
               py::arg("grid_min") = 0.0, py::arg("grid_max") = 6.0,
               py::arg("grid_sigma") = 0.1, py::arg("grid_n") = 50,
               py::arg("normalize_gaussians") = true, py::arg("scale") = 0.5,
               py::arg("threshold") = 1e-3, py::arg("r_cut") = 6.0,
               py::arg("sharpness") = 2.0, py::arg("local") = false,
               py::arg("num_threads") = 0,
               py::arg("control") = nullptr);

    module.def("compute_ead", &compute_ead_array,
               py::arg("numbers"), py::arg("positions"), py::arg("cells"), py::arg("pbc"),
               py::arg("offsets"), py::arg("max_degree") = 3, py::arg("cutoff") = 6.0,
               py::arg("eta"), py::arg("rs"), py::arg("num_threads") = 0,
               py::arg("control") = nullptr);

    module.def("compute_rotational_descriptors", &compute_rotational_descriptors_array,
               py::arg("numbers"), py::arg("positions"), py::arg("cells"), py::arg("pbc"),
               py::arg("offsets"), py::arg("kind") = 0, py::arg("n_max") = 3,
               py::arg("l_max") = 3, py::arg("cutoff") = 3.5, py::arg("alpha") = 2.0,
               py::arg("weight_on") = false, py::arg("normalize_u") = false,
               py::arg("weight_scale") = 1.0, py::arg("twojmax") = 3,
               py::arg("diagonal") = 3, py::arg("num_threads") = 0,
               py::arg("control") = nullptr,
               py::arg("rfac0") = 1.0, py::arg("neighbor_weights") = std::vector<double>{},
               py::arg("rmin0") = 0.0, py::arg("rcutfac") = 1.0,
               py::arg("neighbor_radii") = std::vector<double>{});

    module.def("compute_atomic_composition", &compute_atomic_composition_array,
               py::arg("numbers"), py::arg("positions"), py::arg("cells"), py::arg("pbc"),
               py::arg("offsets"), py::arg("species"), py::arg("per_system") = true,
               py::arg("num_threads") = 0,
               py::arg("control") = nullptr);

    module.def("compute_sorted_distances", &compute_sorted_distances_array,
               py::arg("numbers"), py::arg("positions"), py::arg("cells"), py::arg("pbc"),
               py::arg("offsets"), py::arg("species"), py::arg("cutoff"),
               py::arg("max_neighbors"), py::arg("separate_neighbor_types") = true,
               py::arg("num_threads") = 0, py::arg("control") = nullptr);

    module.def("compute_neighbor_list", &compute_neighbor_list_array,
               py::arg("numbers"), py::arg("positions"), py::arg("cells"), py::arg("pbc"),
               py::arg("offsets"), py::arg("cutoff"), py::arg("full_neighbor_list") = true,
               py::arg("self_pairs") = false, py::arg("num_threads") = 0,
               py::arg("control") = nullptr);

    module.def("compute_spherical_expansion", &compute_spherical_expansion_array,
               py::arg("numbers"), py::arg("positions"), py::arg("cells"), py::arg("pbc"),
               py::arg("offsets"), py::arg("species"), py::arg("cutoff") = 6.0,
               py::arg("density_width") = 0.3, py::arg("max_radial") = 6,
               py::arg("max_angular") = 4, py::arg("kind") = 0,
               py::arg("k_cutoff") = 2.5, py::arg("exponent") = 1,
               py::arg("radial_radius") = 0.0, py::arg("num_threads") = 0,
               py::arg("control") = nullptr);

    module.def("compute_spherical_expansion_by_pair", &compute_spherical_expansion_by_pair_array,
               py::arg("numbers"), py::arg("positions"), py::arg("cells"), py::arg("pbc"),
               py::arg("offsets"), py::arg("species"), py::arg("cutoff") = 6.0,
               py::arg("density_width") = 0.3, py::arg("max_radial") = 6,
               py::arg("max_angular") = 4, py::arg("num_threads") = 0,
               py::arg("control") = nullptr);

    py::class_<SoapCalculator>(module, "SoapCalculator")
        .def(py::init<SoapOptions>())
        .def_property_readonly("feature_count", &SoapCalculator::feature_count)
        .def_property_readonly("species", &SoapCalculator::species)
        .def("close", &SoapCalculator::close)
        .def("closed", &SoapCalculator::closed)
        .def("compute", &compute_soap_array,
             py::arg("numbers"), py::arg("positions"), py::arg("cells"), py::arg("pbc"),
             py::arg("offsets"), py::arg("control") = nullptr, py::arg("num_threads") = 0,
             py::arg("inner_average") = true, py::arg("outer_average") = false);

    py::class_<SoapTurboCalculator>(module, "SoapTurboCalculator")
        .def(py::init<SoapTurboOptions>())
        .def_property_readonly("feature_count", &SoapTurboCalculator::feature_count)
        .def_property_readonly("species", &SoapTurboCalculator::species)
        .def("close", &SoapTurboCalculator::close)
        .def("closed", &SoapTurboCalculator::closed)
        .def("compute", &compute_soap_turbo_array,
             py::arg("numbers"), py::arg("positions"), py::arg("cells"), py::arg("pbc"),
             py::arg("offsets"), py::arg("control") = nullptr);

    py::class_<AcsfCalculator>(module, "AcsfCalculator")
        .def(py::init<AcsfOptions>())
        .def_property_readonly("feature_count", &AcsfCalculator::feature_count)
        .def_property_readonly("species", &AcsfCalculator::species)
        .def("close", &AcsfCalculator::close)
        .def("closed", &AcsfCalculator::closed)
        .def("compute", &compute_acsf_array,
             py::arg("numbers"), py::arg("positions"), py::arg("cells"), py::arg("pbc"),
             py::arg("offsets"), py::arg("control") = nullptr);

    py::class_<C00PSMlffCalculator>(module, "C00PSMlffCalculator")
        .def(py::init<C00PSMlffOptions>())
        .def_property_readonly("feature_count", &C00PSMlffCalculator::feature_count)
        .def_property_readonly("species", &C00PSMlffCalculator::species)
        .def_property_readonly("radial_counts", &C00PSMlffCalculator::radial_counts)
        .def("close", &C00PSMlffCalculator::close)
        .def("closed", &C00PSMlffCalculator::closed)
        .def("compute", &compute_c00ps_mlff_array,
             py::arg("numbers"), py::arg("positions"), py::arg("cells"), py::arg("pbc"),
             py::arg("offsets"), py::arg("control") = nullptr);

    py::class_<MtpCalculator>(module, "MtpCalculator")
        .def(py::init<MtpOptions>())
        .def_property_readonly("feature_count", &MtpCalculator::feature_count)
        .def_property_readonly("species", &MtpCalculator::species)
        .def_property_readonly("official_model", &MtpCalculator::official_model)
        .def_property_readonly("official_mlip4", &MtpCalculator::official_mlip4)
        .def_property_readonly("official_format", &MtpCalculator::official_format)
        .def_property_readonly("official_alpha_moment_mapping", &MtpCalculator::official_alpha_moment_mapping)
        .def_property_readonly("official_min_dist", &MtpCalculator::official_min_dist)
        .def_property_readonly("official_max_dist", &MtpCalculator::official_max_dist)
        .def_property_readonly("official_radial_basis_size", &MtpCalculator::official_radial_basis_size)
        .def_property_readonly("official_radial_funcs_count", &MtpCalculator::official_radial_funcs_count)
        .def_property_readonly("official_radial_basis_type", &MtpCalculator::official_radial_basis_type)
        .def("close", &MtpCalculator::close)
        .def("closed", &MtpCalculator::closed)
        .def("compute", &compute_mtp_array,
             py::arg("numbers"), py::arg("positions"), py::arg("cells"), py::arg("pbc"),
             py::arg("offsets"), py::arg("control") = nullptr);

    py::class_<AceCalculator>(module, "AceCalculator")
        .def(py::init<AceOptions>())
        .def_property_readonly("feature_count", &AceCalculator::feature_count)
        .def_property_readonly("feature_counts", &AceCalculator::feature_counts)
        .def_property_readonly("species", &AceCalculator::species)
        .def_property_readonly("max_angular", &AceCalculator::max_angular)
        .def_property_readonly("max_radial", &AceCalculator::max_radial)
        .def("close", &AceCalculator::close)
        .def("closed", &AceCalculator::closed)
        .def("compute", &compute_ace_array,
             py::arg("numbers"), py::arg("positions"), py::arg("cells"), py::arg("pbc"),
             py::arg("offsets"), py::arg("control") = nullptr);

    py::class_<NepCalculator>(module, "NepCalculator")
        .def(py::init<NepOptions>())
        .def_property_readonly("feature_count", &NepCalculator::feature_count)
        .def_property_readonly("species", &NepCalculator::species)
        .def_property_readonly("model_path", &NepCalculator::model_path)
        .def_property_readonly("radial_cutoff", &NepCalculator::radial_cutoff)
        .def_property_readonly("angular_cutoff", &NepCalculator::angular_cutoff)
        .def_property_readonly("n_max_radial", &NepCalculator::n_max_radial)
        .def_property_readonly("n_max_angular", &NepCalculator::n_max_angular)
        .def_property_readonly("l_max", &NepCalculator::l_max)
        .def("close", &NepCalculator::close)
        .def("closed", &NepCalculator::closed)
        .def("compute", &compute_nep_array,
             py::arg("numbers"), py::arg("positions"), py::arg("cells"), py::arg("pbc"),
             py::arg("offsets"), py::arg("control") = nullptr);

    py::class_<Dpa4cCalculator, std::shared_ptr<Dpa4cCalculator>>(
        module, "Dpa4cCalculator")
        .def(py::init([](const py::dict& payload) {
            return std::make_shared<Dpa4cCalculator>(
                dpa4c_options_from_payload(payload));
        }))
        .def_property_readonly("feature_count", &Dpa4cCalculator::feature_count)
        .def("close", &Dpa4cCalculator::close)
        .def("closed", &Dpa4cCalculator::closed)
        .def("compute", &compute_dpa4c_array,
             py::arg("numbers"), py::arg("positions"), py::arg("cells"), py::arg("pbc"),
             py::arg("offsets"), py::arg("type_indices"), py::arg("control") = nullptr);

    py::class_<Dpa4Calculator, std::shared_ptr<Dpa4Calculator>>(
        module, "Dpa4Calculator")
        .def(py::init([](const py::dict& payload) {
            return std::make_shared<Dpa4Calculator>(
                dpa4_options_from_payload(payload));
        }))
        .def_property_readonly("feature_count", &Dpa4Calculator::feature_count)
        .def("close", &Dpa4Calculator::close)
        .def("closed", &Dpa4Calculator::closed)
        .def("compute", &compute_dpa4_array,
             py::arg("numbers"), py::arg("positions"), py::arg("cells"), py::arg("pbc"),
             py::arg("offsets"), py::arg("type_indices"), py::arg("control") = nullptr);

    py::class_<SoapOptions>(module, "SoapOptions")
        .def(py::init<>())
        .def_readwrite("species", &SoapOptions::species)
        .def_readwrite("r_cut", &SoapOptions::r_cut)
        .def_readwrite("n_max", &SoapOptions::n_max)
        .def_readwrite("l_max", &SoapOptions::l_max)
        .def_readwrite("sigma", &SoapOptions::sigma)
        .def_readwrite("radial_basis", &SoapOptions::radial_basis)
        .def_readwrite("alphas", &SoapOptions::alphas)
        .def_readwrite("betas", &SoapOptions::betas)
        .def_readwrite("radial_grid", &SoapOptions::radial_grid)
        .def_readwrite("radial_weights", &SoapOptions::radial_weights)
        .def_readwrite("radial_values", &SoapOptions::radial_values)
        .def_readwrite("weighting_function", &SoapOptions::weighting_function)
        .def_readwrite("weighting_has_w0", &SoapOptions::weighting_has_w0)
        .def_readwrite("weighting_has_function", &SoapOptions::weighting_has_function)
        .def_readwrite("weighting_r0", &SoapOptions::weighting_r0)
        .def_readwrite("weighting_c", &SoapOptions::weighting_c)
        .def_readwrite("weighting_d", &SoapOptions::weighting_d)
        .def_readwrite("weighting_m", &SoapOptions::weighting_m)
        .def_readwrite("weighting_threshold", &SoapOptions::weighting_threshold)
        .def_readwrite("weighting_w0", &SoapOptions::weighting_w0)
        .def_readwrite("species_weights", &SoapOptions::species_weights)
        .def_readwrite("compression", &SoapOptions::compression)
        .def_readwrite("inner_average", &SoapOptions::inner_average)
        .def_readwrite("outer_average", &SoapOptions::outer_average)
        .def_readwrite("num_threads", &SoapOptions::num_threads);

    py::class_<C00PSMlffOptions>(module, "C00PSMlffOptions")
        .def(py::init<>())
        .def_readwrite("species", &C00PSMlffOptions::species)
        .def_readwrite("r_cut", &C00PSMlffOptions::r_cut)
        .def_readwrite("n_radial", &C00PSMlffOptions::n_radial)
        .def_readwrite("l_max", &C00PSMlffOptions::l_max)
        .def_readwrite("cutoff_function", &C00PSMlffOptions::cutoff_function)
        .def_readwrite("radial_sigma", &C00PSMlffOptions::radial_sigma)
        .def_readwrite("include_radial", &C00PSMlffOptions::include_radial)
        .def_readwrite("include_angular", &C00PSMlffOptions::include_angular)
        .def_readwrite("normalize_radial", &C00PSMlffOptions::normalize_radial)
        .def_readwrite("normalize_angular", &C00PSMlffOptions::normalize_angular)
        .def_readwrite("super_vector", &C00PSMlffOptions::super_vector)
        .def_readwrite("radial_weight", &C00PSMlffOptions::radial_weight)
        .def_readwrite("angular_weight", &C00PSMlffOptions::angular_weight)
        .def_readwrite("exclude_self_interaction", &C00PSMlffOptions::exclude_self_interaction)
        .def_readwrite("num_threads", &C00PSMlffOptions::num_threads);

    py::class_<SoapTurboOptions>(module, "SoapTurboOptions")
        .def(py::init<>())
        .def_readwrite("species", &SoapTurboOptions::species)
        .def_readwrite("alpha_max", &SoapTurboOptions::alpha_max)
        .def_readwrite("central_species", &SoapTurboOptions::central_species)
        .def_readwrite("atom_sigma_r", &SoapTurboOptions::atom_sigma_r)
        .def_readwrite("atom_sigma_r_scaling", &SoapTurboOptions::atom_sigma_r_scaling)
        .def_readwrite("atom_sigma_t", &SoapTurboOptions::atom_sigma_t)
        .def_readwrite("atom_sigma_t_scaling", &SoapTurboOptions::atom_sigma_t_scaling)
        .def_readwrite("amplitude_scaling", &SoapTurboOptions::amplitude_scaling)
        .def_readwrite("central_weight", &SoapTurboOptions::central_weight)
        .def_readwrite("l_max", &SoapTurboOptions::l_max)
        .def_readwrite("rcut_hard", &SoapTurboOptions::rcut_hard)
        .def_readwrite("rcut_soft", &SoapTurboOptions::rcut_soft)
        .def_readwrite("nf", &SoapTurboOptions::nf)
        .def_readwrite("radial_enhancement", &SoapTurboOptions::radial_enhancement)
        .def_readwrite("basis", &SoapTurboOptions::basis)
        .def_readwrite("compression", &SoapTurboOptions::compression)
        .def_readwrite("num_threads", &SoapTurboOptions::num_threads);

    py::class_<AcsfOptions>(module, "AcsfOptions")
        .def(py::init<>())
        .def_readwrite("species", &AcsfOptions::species)
        .def_readwrite("r_cut", &AcsfOptions::r_cut)
        .def_readwrite("g2_params", &AcsfOptions::g2_params)
        .def_readwrite("g3_params", &AcsfOptions::g3_params)
        .def_readwrite("g4_params", &AcsfOptions::g4_params)
        .def_readwrite("g5_params", &AcsfOptions::g5_params)
        .def_readwrite("n_g2", &AcsfOptions::n_g2)
        .def_readwrite("n_g3", &AcsfOptions::n_g3)
        .def_readwrite("n_g4", &AcsfOptions::n_g4)
        .def_readwrite("n_g5", &AcsfOptions::n_g5)
        .def_readwrite("num_threads", &AcsfOptions::num_threads);

    py::class_<MtpOptions>(module, "MtpOptions")
        .def(py::init<>())
        .def_readwrite("species", &MtpOptions::species)
        .def_readwrite("potential_path", &MtpOptions::potential_path)
        .def_readwrite("model_digest", &MtpOptions::model_digest)
        .def_readwrite("min_dist", &MtpOptions::min_dist)
        .def_readwrite("max_dist", &MtpOptions::max_dist)
        .def_readwrite("radial_basis_size", &MtpOptions::radial_basis_size)
        .def_readwrite("radial_funcs_count", &MtpOptions::radial_funcs_count)
        .def_readwrite("max_rank", &MtpOptions::max_rank)
        .def_readwrite("num_threads", &MtpOptions::num_threads);

    py::class_<AceOptions>(module, "AceOptions")
        .def(py::init<>())
        .def_readwrite("species", &AceOptions::species)
        .def_readwrite("max_order", &AceOptions::max_order)
        .def_readwrite("r0", &AceOptions::r0)
        .def_readwrite("transform_p", &AceOptions::transform_p)
        .def_readwrite("transform_a", &AceOptions::transform_a)
        .def_readwrite("w_l", &AceOptions::w_l)
        .def_readwrite("max_degree", &AceOptions::max_degree)
        .def_readwrite("degree_csp", &AceOptions::degree_csp)
        .def_readwrite("degree_chc", &AceOptions::degree_chc)
        .def_readwrite("degree_ahc", &AceOptions::degree_ahc)
        .def_readwrite("degree_bhc", &AceOptions::degree_bhc)
        .def_readwrite("degree_by_order", &AceOptions::degree_by_order)
        .def_readwrite("angular_weight_by_order", &AceOptions::angular_weight_by_order)
        .def_readwrite("r_cut", &AceOptions::r_cut)
        .def_readwrite("r_in", &AceOptions::r_in)
        .def_readwrite("p_cut", &AceOptions::p_cut)
        .def_readwrite("p_in", &AceOptions::p_in)
        .def_readwrite("constants", &AceOptions::constants)
        .def_readwrite("num_threads", &AceOptions::num_threads);

    py::class_<NepOptions>(module, "NepOptions")
        .def(py::init<>())
        .def_readwrite("model_path", &NepOptions::model_path)
        .def_readwrite("model_digest", &NepOptions::model_digest)
        .def_readwrite("num_threads", &NepOptions::num_threads);
}
