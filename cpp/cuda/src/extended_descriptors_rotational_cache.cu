#include "extended_descriptors_common.cuh"

} // namespace

struct RotationalPlanCache::Impl {
    bool prepared = false;
    int expansion_order = -1;
    int diagonal = -1;
    bool l_bispectrum = false;
    std::int64_t features = 0;
    DeviceBuffer<I64> z_inner_offsets;
    DeviceBuffer<I64> inner_term_offsets;
    DeviceBuffer<double> inner_outer_coefficients;
    DeviceBuffer<I64> term_first_indices;
    DeviceBuffer<I64> term_second_indices;
    DeviceBuffer<double> term_coefficients;
    DeviceBuffer<I64> projection_offsets;
    DeviceBuffer<I64> projection_u_indices;
    DeviceBuffer<I64> projection_z_indices;
    DeviceBuffer<double> projection_scales;
};

RotationalPlanCache::RotationalPlanCache() : impl_(std::make_unique<Impl>()) {}

RotationalPlanCache::~RotationalPlanCache() noexcept {
    clear();
}

RotationalPlanDeviceView RotationalPlanCache::prepare(
    CudaExecutionContext& context,
    int expansion_order,
    int diagonal,
    bool l_bispectrum) {
    if (impl_->prepared && impl_->expansion_order == expansion_order
        && impl_->diagonal == diagonal && impl_->l_bispectrum == l_bispectrum) {
        return {
            impl_->z_inner_offsets.get(),
            impl_->inner_term_offsets.get(),
            impl_->inner_outer_coefficients.get(),
            impl_->term_first_indices.get(),
            impl_->term_second_indices.get(),
            impl_->term_coefficients.get(),
            impl_->projection_offsets.get(),
            impl_->projection_u_indices.get(),
            impl_->projection_z_indices.get(),
            impl_->projection_scales.get(),
            impl_->features,
        };
    }

    const auto bispectrum_plan = detail::rotational::make_bispectrum_plan(
        expansion_order, diagonal, l_bispectrum);
    const auto flattened_plan = detail::rotational::flatten(bispectrum_plan);
    impl_->prepared = false;
    impl_->z_inner_offsets.clear();
    impl_->inner_term_offsets.clear();
    impl_->inner_outer_coefficients.clear();
    impl_->term_first_indices.clear();
    impl_->term_second_indices.clear();
    impl_->term_coefficients.clear();
    impl_->projection_offsets.clear();
    impl_->projection_u_indices.clear();
    impl_->projection_z_indices.clear();
    impl_->projection_scales.clear();
    impl_->z_inner_offsets.upload(
        flattened_plan.z_inner_offsets.data(), flattened_plan.z_inner_offsets.size(),
        context.stream(), "could not upload CUDA bispectrum Z offsets");
    impl_->inner_term_offsets.upload(
        flattened_plan.inner_term_offsets.data(), flattened_plan.inner_term_offsets.size(),
        context.stream(), "could not upload CUDA bispectrum inner offsets");
    impl_->inner_outer_coefficients.upload(
        flattened_plan.inner_outer_coefficients.data(),
        flattened_plan.inner_outer_coefficients.size(), context.stream(),
        "could not upload CUDA bispectrum outer coefficients");
    impl_->term_first_indices.upload(
        flattened_plan.term_first_indices.data(), flattened_plan.term_first_indices.size(),
        context.stream(), "could not upload CUDA bispectrum first indices");
    impl_->term_second_indices.upload(
        flattened_plan.term_second_indices.data(), flattened_plan.term_second_indices.size(),
        context.stream(), "could not upload CUDA bispectrum second indices");
    impl_->term_coefficients.upload(
        flattened_plan.term_coefficients.data(), flattened_plan.term_coefficients.size(),
        context.stream(), "could not upload CUDA bispectrum CG coefficients");
    impl_->projection_offsets.upload(
        flattened_plan.projection_offsets.data(), flattened_plan.projection_offsets.size(),
        context.stream(), "could not upload CUDA bispectrum projection offsets");
    impl_->projection_u_indices.upload(
        flattened_plan.projection_u_indices.data(), flattened_plan.projection_u_indices.size(),
        context.stream(), "could not upload CUDA bispectrum projection U indices");
    impl_->projection_z_indices.upload(
        flattened_plan.projection_z_indices.data(), flattened_plan.projection_z_indices.size(),
        context.stream(), "could not upload CUDA bispectrum projection Z indices");
    impl_->projection_scales.upload(
        flattened_plan.projection_scales.data(), flattened_plan.projection_scales.size(),
        context.stream(), "could not upload CUDA bispectrum projection scales");
    impl_->expansion_order = expansion_order;
    impl_->diagonal = diagonal;
    impl_->l_bispectrum = l_bispectrum;
    impl_->features = static_cast<std::int64_t>(bispectrum_plan.components.size());
    impl_->prepared = true;
    return {
        impl_->z_inner_offsets.get(),
        impl_->inner_term_offsets.get(),
        impl_->inner_outer_coefficients.get(),
        impl_->term_first_indices.get(),
        impl_->term_second_indices.get(),
        impl_->term_coefficients.get(),
        impl_->projection_offsets.get(),
        impl_->projection_u_indices.get(),
        impl_->projection_z_indices.get(),
        impl_->projection_scales.get(),
        impl_->features,
    };
}

void RotationalPlanCache::clear() noexcept {
    if (impl_ == nullptr) return;
    impl_->prepared = false;
    impl_->z_inner_offsets.clear();
    impl_->inner_term_offsets.clear();
    impl_->inner_outer_coefficients.clear();
    impl_->term_first_indices.clear();
    impl_->term_second_indices.clear();
    impl_->term_coefficients.clear();
    impl_->projection_offsets.clear();
    impl_->projection_u_indices.clear();
    impl_->projection_z_indices.clear();
    impl_->projection_scales.clear();
}


} // namespace mdescriptor::cuda
