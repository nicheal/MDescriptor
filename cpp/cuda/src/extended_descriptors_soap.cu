#include "extended_descriptors_common.cuh"



} // namespace

py::dict compute_extended_soap(
    CudaExecutionContext& context,
    DeviceBatch& batch,
    DeviceNeighborGraph& graph,
    const detail::StructureBatchView& host_batch,
    const std::string& name,
    const py::dict& options,
    const py::object& control,
    RotationalPlanCache* rotational_plan) {
    (void)name;
    (void)control;
    (void)rotational_plan;
    return compute_soap_descriptor(context, batch, graph, host_batch, options);
}

} // namespace mdescriptor::cuda
