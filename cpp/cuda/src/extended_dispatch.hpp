#pragma once

#include "mdescriptor/cuda/extended_descriptors.hpp"

#include <string>

namespace py = pybind11;

namespace mdescriptor::cuda {

using ExtendedDescriptorHandler = py::dict (*) (
    CudaExecutionContext&,
    DeviceBatch&,
    DeviceNeighborGraph&,
    const detail::StructureBatchView&,
    const std::string&,
    const py::dict&,
    const py::object&,
    RotationalPlanCache*);

py::dict compute_extended_soap(
    CudaExecutionContext&, DeviceBatch&, DeviceNeighborGraph&,
    const detail::StructureBatchView&, const std::string&, const py::dict&,
    const py::object&, RotationalPlanCache*);
py::dict compute_extended_matrix(
    CudaExecutionContext&, DeviceBatch&, DeviceNeighborGraph&,
    const detail::StructureBatchView&, const std::string&, const py::dict&,
    const py::object&, RotationalPlanCache*);
py::dict compute_extended_acsf(
    CudaExecutionContext&, DeviceBatch&, DeviceNeighborGraph&,
    const detail::StructureBatchView&, const std::string&, const py::dict&,
    const py::object&, RotationalPlanCache*);
py::dict compute_extended_mbtr(
    CudaExecutionContext&, DeviceBatch&, DeviceNeighborGraph&,
    const detail::StructureBatchView&, const std::string&, const py::dict&,
    const py::object&, RotationalPlanCache*);
py::dict compute_extended_basic(
    CudaExecutionContext&, DeviceBatch&, DeviceNeighborGraph&,
    const detail::StructureBatchView&, const std::string&, const py::dict&,
    const py::object&, RotationalPlanCache*);
py::dict compute_extended_ead_lode(
    CudaExecutionContext&, DeviceBatch&, DeviceNeighborGraph&,
    const detail::StructureBatchView&, const std::string&, const py::dict&,
    const py::object&, RotationalPlanCache*);
py::dict compute_extended_rotational(
    CudaExecutionContext&, DeviceBatch&, DeviceNeighborGraph&,
    const detail::StructureBatchView&, const std::string&, const py::dict&,
    const py::object&, RotationalPlanCache*);
py::dict compute_extended_c00ps(
    CudaExecutionContext&, DeviceBatch&, DeviceNeighborGraph&,
    const detail::StructureBatchView&, const std::string&, const py::dict&,
    const py::object&, RotationalPlanCache*);
py::dict compute_extended_soap_turbo(
    CudaExecutionContext&, DeviceBatch&, DeviceNeighborGraph&,
    const detail::StructureBatchView&, const std::string&, const py::dict&,
    const py::object&, RotationalPlanCache*);
py::dict compute_extended_mtp(
    CudaExecutionContext&, DeviceBatch&, DeviceNeighborGraph&,
    const detail::StructureBatchView&, const std::string&, const py::dict&,
    const py::object&, RotationalPlanCache*);
py::dict compute_extended_ace(
    CudaExecutionContext&, DeviceBatch&, DeviceNeighborGraph&,
    const detail::StructureBatchView&, const std::string&, const py::dict&,
    const py::object&, RotationalPlanCache*);
py::dict compute_extended_generic(
    CudaExecutionContext&, DeviceBatch&, DeviceNeighborGraph&,
    const detail::StructureBatchView&, const std::string&, const py::dict&,
    const py::object&, RotationalPlanCache*);

} // namespace mdescriptor::cuda
