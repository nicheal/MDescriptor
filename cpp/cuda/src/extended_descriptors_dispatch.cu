#include "extended_dispatch.hpp"
#include "mdescriptor/cuda/descriptor_dispatch.hpp"

# pytypes.h only declares object::cast<T>; the definitions live here and a
# missing include surfaces only as an undefined-symbol link error.
#include <pybind11/cast.h>
#include <array>
#include <stdexcept>
#include <string_view>

namespace mdescriptor::cuda {
namespace {

bool cancelled(const py::object& control) {
    return !control.is_none() && control.attr("cancelled")().cast<bool>();
}

// Names and family handlers live in one table.  Feature discovery and compute
// dispatch both query this registry, so adding a descriptor cannot leave an
// advertised name without a callable implementation (or vice versa).
struct ExtendedDescriptorEntry {
    std::string_view name;
    ExtendedDescriptorHandler handler;
};

inline constexpr std::array<ExtendedDescriptorEntry, kExtendedDescriptorCount>
    kExtendedDescriptorRegistry = {{
        {"AtomicComposition", compute_extended_basic},
        {"SortedDistances", compute_extended_basic},
        {"SphericalExpansionByPair", compute_extended_basic},
        {"SOAP", compute_extended_soap},
        {"SOAPTurbo", compute_extended_soap_turbo},
        {"ACSF", compute_extended_acsf},
        {"ACE", compute_extended_ace},
        {"LodeSphericalExpansion", compute_extended_ead_lode},
        {"CoulombMatrix", compute_extended_matrix},
        {"SineMatrix", compute_extended_matrix},
        {"EwaldSumMatrix", compute_extended_matrix},
        {"MBTR", compute_extended_mbtr},
        {"LMBTR", compute_extended_mbtr},
        {"ValleOganov", compute_extended_mbtr},
        {"EAD", compute_extended_ead_lode},
        {"SO3", compute_extended_rotational},
        {"SO4", compute_extended_rotational},
        {"SNAP", compute_extended_rotational},
        {"LBispectrum", compute_extended_rotational},
        {"MTP", compute_extended_mtp},
        {"C00PSMLFF", compute_extended_c00ps},
    }};

} // namespace

std::size_t extended_descriptor_index(std::string_view name) noexcept {
    for (std::size_t index = 0; index < kExtendedDescriptorRegistry.size(); ++index) {
        if (kExtendedDescriptorRegistry[index].name == name) {
            return index;
        }
    }
    return kExtendedDescriptorRegistry.size();
}

bool is_extended_descriptor(std::string_view name) noexcept {
    return extended_descriptor_index(name) < kExtendedDescriptorRegistry.size();
}

py::dict compute_extended_descriptor(
    CudaExecutionContext& context,
    DeviceBatch& batch,
    DeviceNeighborGraph& graph,
    const detail::StructureBatchView& host_batch,
    const std::string& name,
    const py::dict& options,
    const py::object& control,
    RotationalPlanCache* rotational_plan_cache) {
    if (cancelled(control)) {
        throw std::runtime_error("descriptor computation cancelled");
    }
    const auto index = extended_descriptor_index(name);
    if (index >= kExtendedDescriptorRegistry.size()) {
        throw std::invalid_argument("CUDA backend does not support this extended descriptor");
    }
    return kExtendedDescriptorRegistry[index].handler(
        context, batch, graph, host_batch, name, options, control, rotational_plan_cache);
}

} // namespace mdescriptor::cuda
