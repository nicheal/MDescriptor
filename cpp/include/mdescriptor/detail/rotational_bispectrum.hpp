#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace mdescriptor::detail::rotational {

#if defined(__CUDACC__)
#define MDESCRIPTOR_ROTATIONAL_HD __host__ __device__
#else
#define MDESCRIPTOR_ROTATIONAL_HD
#endif

constexpr double kPi = 3.141592653589793238462643383279502884;

inline int expansion_order(int kind, int lmax, int twojmax) noexcept {
    return kind == 3 ? std::max(0, twojmax) : 2 * lmax;
}

// This deliberately small complex type is the storage seam between the host
// and device implementations.  The recurrence below is compiled for both
// targets; std::complex is kept in the SO3-only path, where it is not needed
// by CUDA.
struct Complex {
    double real = 0.0;
    double imag = 0.0;
};

MDESCRIPTOR_ROTATIONAL_HD inline Complex complex_add(Complex first, Complex second) {
    return {first.real + second.real, first.imag + second.imag};
}

MDESCRIPTOR_ROTATIONAL_HD inline Complex complex_scale(Complex value, double scale) {
    return {value.real * scale, value.imag * scale};
}

MDESCRIPTOR_ROTATIONAL_HD inline Complex complex_conjugate(Complex value) {
    return {value.real, -value.imag};
}

MDESCRIPTOR_ROTATIONAL_HD inline Complex complex_multiply(Complex first, Complex second) {
    return {
        first.real * second.real - first.imag * second.imag,
        first.real * second.imag + first.imag * second.real,
    };
}

MDESCRIPTOR_ROTATIONAL_HD constexpr std::size_t u_block_offset(int angular) noexcept {
    return static_cast<std::size_t>(angular)
        * static_cast<std::size_t>(angular + 1)
        * static_cast<std::size_t>(2 * angular + 1) / 6U;
}

MDESCRIPTOR_ROTATIONAL_HD constexpr std::size_t u_total_size(int order) noexcept {
    return u_block_offset(order)
        + static_cast<std::size_t>(order + 1) * static_cast<std::size_t>(order + 1);
}

MDESCRIPTOR_ROTATIONAL_HD inline void hyperspherical_u(
    double x,
    double y,
    double z,
    int max_order,
    double cutoff,
    double rfac0,
    double rmin0,
    Complex* output) {
    const std::size_t size = u_total_size(max_order);
    for (std::size_t index = 0; index < size; ++index) {
        output[index] = {};
    }

    const double radius = sqrt(x * x + y * y + z * z);
    if (radius <= 1e-14) {
        output[0] = {1.0, 0.0};
        return;
    }

    const double radial_width = cutoff - rmin0;
    const double psi = rfac0 * kPi * (radius - rmin0)
        / (radial_width > 1e-12 ? radial_width : 1e-12);
    const double sine = sin(psi);
    const Complex a{cos(psi), -sine * z / radius};
    const Complex b{sine * y / radius, -sine * x / radius};
    const Complex conjugate_a = complex_conjugate(a);
    const Complex conjugate_b = complex_conjugate(b);
    output[0] = {1.0, 0.0};

    for (int angular = 1; angular <= max_order; ++angular) {
        const std::size_t base = u_block_offset(angular);
        const std::size_t previous = u_block_offset(angular - 1);
        int mb = 0;
        while (2 * mb <= angular) {
            const std::size_t row = base
                + static_cast<std::size_t>(mb * (angular + 1));
            const std::size_t previous_row = previous
                + static_cast<std::size_t>(mb * angular);
            output[row] = {};
            for (int ma = 0; ma < angular; ++ma) {
                const double first_root = sqrt(static_cast<double>(angular - ma)
                    / static_cast<double>(angular - mb));
                const double second_root = sqrt(static_cast<double>(ma + 1)
                    / static_cast<double>(angular - mb));
                output[row + static_cast<std::size_t>(ma)] = complex_add(
                    output[row + static_cast<std::size_t>(ma)],
                    complex_multiply(
                        complex_scale(conjugate_a, first_root),
                        output[previous_row + static_cast<std::size_t>(ma)]));
                output[row + static_cast<std::size_t>(ma + 1)] = complex_add(
                    output[row + static_cast<std::size_t>(ma + 1)],
                    complex_multiply(
                        complex_scale(conjugate_b, -second_root),
                        output[previous_row + static_cast<std::size_t>(ma)]));
            }
            ++mb;
        }

        std::size_t left = base;
        std::size_t right = base
            + static_cast<std::size_t>((angular + 1) * (angular + 1) - 1);
        int mbpar = 1;
        mb = 0;
        while (2 * mb <= angular) {
            int mapar = mbpar;
            for (int ma = 0; ma <= angular; ++ma) {
                output[right] = mapar == 1
                    ? complex_conjugate(output[left])
                    : complex_scale(complex_conjugate(output[left]), -1.0);
                mapar = -mapar;
                ++left;
                --right;
            }
            mbpar = -mbpar;
            ++mb;
        }
    }
}

MDESCRIPTOR_ROTATIONAL_HD inline double bispectrum_cutoff(
    double radius, double cutoff, double rmin0) {
    if (radius <= rmin0) return 1.0;
    if (radius > cutoff || cutoff <= rmin0) return 0.0;
    return 0.5 * (cos(kPi * (radius - rmin0) / (cutoff - rmin0)) + 1.0);
}

// The graph builders are different host/device adapters, but both consume the
// same descriptor-level policy.  In particular, the exact center is not a
// bispectrum neighbor and the cutoff boundary is retained.
constexpr bool kBispectrumIncludeExactSelf = false;
constexpr bool kBispectrumIncludeCutoffBoundary = true;
constexpr double kBispectrumMinimumRadius = 1e-8;

struct BispectrumComponent {
    int l1 = 0;
    int l2 = 0;
    int l = 0;
};

inline std::vector<std::size_t> u_offsets(int max_order) {
    std::vector<std::size_t> offsets(static_cast<std::size_t>(max_order + 1), 0);
    for (int angular = 0; angular <= max_order; ++angular) {
        offsets[static_cast<std::size_t>(angular)] = u_block_offset(angular);
    }
    return offsets;
}

inline double factorial_value(int value) {
    return value < 0 ? 0.0 : std::tgamma(static_cast<double>(value) + 1.0);
}

inline double clebsch_gordan(int l1, int l2, int l, int m1, int m2) {
    const int aa2 = 2 * m1 - l1;
    const int bb2 = 2 * m2 - l2;
    const int numerator = aa2 + bb2 + l;
    if (numerator % 2 != 0) return 0.0;
    const int m = numerator / 2;
    if (m < 0 || m > l) return 0.0;
    const int z_min = std::max(0, std::max(
        -(l - l2 + aa2) / 2,
        -(l - l1 - bb2) / 2));
    const int z_max = std::min(
        (l1 + l2 - l) / 2,
        std::min((l1 - aa2) / 2, (l2 + bb2) / 2));
    double sum = 0.0;
    for (int z = z_min; z <= z_max; ++z) {
        sum += (z % 2 == 0 ? 1.0 : -1.0) / (
            factorial_value(z)
            * factorial_value((l1 + l2 - l) / 2 - z)
            * factorial_value((l1 - aa2) / 2 - z)
            * factorial_value((l2 + bb2) / 2 - z)
            * factorial_value((l - l2 + aa2) / 2 + z)
            * factorial_value((l - l1 - bb2) / 2 + z));
    }
    const double delta = std::sqrt(
        factorial_value((l1 + l2 - l) / 2)
        * factorial_value((l1 - l2 + l) / 2)
        * factorial_value((-l1 + l2 + l) / 2)
        / factorial_value((l1 + l2 + l) / 2 + 1));
    const double scale = std::sqrt(
        factorial_value((l1 + aa2) / 2)
        * factorial_value((l1 - aa2) / 2)
        * factorial_value((l2 + bb2) / 2)
        * factorial_value((l2 - bb2) / 2)
        * factorial_value(m)
        * factorial_value(l - m)
        * (l + 1.0));
    return sum * delta * scale;
}

struct BispectrumContractionTerm {
    std::size_t first_index = 0;
    std::size_t second_index = 0;
    double coefficient = 0.0;
};

struct BispectrumInnerPlan {
    double outer_coefficient = 0.0;
    std::vector<BispectrumContractionTerm> terms;
};

struct BispectrumZPlan {
    std::vector<BispectrumInnerPlan> inner;
};

struct BispectrumProjectionTerm {
    std::size_t u_index = 0;
    std::size_t z_index = 0;
    double scale = 1.0;
};

struct BispectrumComponentPlan {
    BispectrumComponent component;
    std::vector<BispectrumZPlan> z;
    std::vector<BispectrumProjectionTerm> projection;
};

struct BispectrumPlan {
    int order = 0;
    std::vector<std::size_t> offsets{0};
    std::vector<BispectrumComponentPlan> components;

    std::size_t u_size() const noexcept {
        return u_total_size(order);
    }
};

inline void add_component_contraction(
    BispectrumPlan& plan, BispectrumComponent component) {
    BispectrumComponentPlan component_plan;
    component_plan.component = component;
    if ((component.l1 + component.l2 + component.l) % 2 != 0) {
        plan.components.push_back(std::move(component_plan));
        return;
    }

    const auto u_index = [&plan](int angular, int mb, int ma) {
        return plan.offsets[static_cast<std::size_t>(angular)]
            + static_cast<std::size_t>(mb * (angular + 1) + ma);
    };
    for (int mb = 0; 2 * mb <= component.l; ++mb) {
        for (int ma = 0; ma <= component.l; ++ma) {
            const int ma1_min = std::max(0,
                (2 * ma - component.l - component.l2 + component.l1) / 2);
            const int ma2_max = (2 * ma - component.l
                - (2 * ma1_min - component.l1) + component.l2) / 2;
            const int na = std::min(component.l1,
                (2 * ma - component.l + component.l2 + component.l1) / 2)
                - ma1_min + 1;
            const int mb1_min = std::max(0,
                (2 * mb - component.l - component.l2 + component.l1) / 2);
            const int mb2_max = (2 * mb - component.l
                - (2 * mb1_min - component.l1) + component.l2) / 2;
            const int nb = std::min(component.l1,
                (2 * mb - component.l + component.l2 + component.l1) / 2)
                - mb1_min + 1;

            BispectrumZPlan z_plan;
            for (int ib = 0; ib < nb; ++ib) {
                BispectrumInnerPlan inner_plan;
                const int mb1 = mb1_min + ib;
                const int mb2 = mb2_max - ib;
                inner_plan.outer_coefficient = clebsch_gordan(
                    component.l1, component.l2, component.l, mb1, mb2);
                inner_plan.terms.reserve(static_cast<std::size_t>(std::max(0, na)));
                for (int ia = 0; ia < na; ++ia) {
                    const int ma1 = ma1_min + ia;
                    const int ma2 = ma2_max - ia;
                    inner_plan.terms.push_back({
                        u_index(component.l1, mb1, ma1),
                        u_index(component.l2, mb2, ma2),
                        clebsch_gordan(component.l1, component.l2, component.l, ma1, ma2),
                    });
                }
                z_plan.inner.push_back(std::move(inner_plan));
            }
            component_plan.z.push_back(std::move(z_plan));
        }
    }

    std::size_t z_index = 0;
    for (int mb = 0; 2 * mb < component.l; ++mb) {
        for (int ma = 0; ma <= component.l; ++ma) {
            component_plan.projection.push_back({
                u_index(component.l, mb, ma), z_index++, 1.0,
            });
        }
    }
    if (component.l % 2 == 0) {
        const int mb = component.l / 2;
        for (int ma = 0; ma < mb; ++ma) {
            component_plan.projection.push_back({
                u_index(component.l, mb, ma), z_index++, 1.0,
            });
        }
        component_plan.projection.push_back({
            u_index(component.l, mb, mb), z_index++, 0.5,
        });
    }
    plan.components.push_back(std::move(component_plan));
}

inline BispectrumPlan make_bispectrum_plan(
    int max_order, int diagonal, bool l_bispectrum) {
    BispectrumPlan plan;
    plan.order = max_order;
    plan.offsets = u_offsets(max_order);
    for (int l1 = 0; l1 <= max_order; ++l1) {
        if (l_bispectrum && diagonal == 2) {
            add_component_contraction(plan, {l1, l1, l1});
            continue;
        }
        for (int l2 = 0; l2 <= l1; ++l2) {
            if (l_bispectrum && diagonal == 2 && l2 != l1) continue;
            for (int l = l1 - l2; l <= std::min(max_order, l1 + l2); l += 2) {
                if (l_bispectrum) {
                    if (diagonal == 1 && l2 != l1) continue;
                    if (diagonal == 2 && l != l1) continue;
                    if (diagonal == 3 && l < l1) continue;
                } else if (l < l1) {
                    continue;
                }
                add_component_contraction(plan, {l1, l2, l});
            }
        }
    }
    return plan;
}

// A flat representation is the device upload seam.  The host builds this
// once per descriptor, while the CUDA kernel only follows offsets and never
// recomputes CG coefficients or component enumeration per center.
struct FlattenedBispectrumPlan {
    std::vector<std::int64_t> z_inner_offsets{0};
    std::vector<std::int64_t> component_z_offsets{0};
    std::vector<std::int64_t> inner_term_offsets{0};
    std::vector<double> inner_outer_coefficients;
    std::vector<std::int64_t> term_first_indices;
    std::vector<std::int64_t> term_second_indices;
    std::vector<double> term_coefficients;
    std::vector<std::int64_t> projection_offsets{0};
    std::vector<std::int64_t> projection_u_indices;
    std::vector<std::int64_t> projection_z_indices;
    std::vector<double> projection_scales;
};

inline FlattenedBispectrumPlan flatten(const BispectrumPlan& plan) {
    FlattenedBispectrumPlan result;
    std::int64_t z_base = 0;
    for (const BispectrumComponentPlan& component : plan.components) {
        const std::int64_t component_z_base = z_base;
        for (const BispectrumZPlan& z : component.z) {
            for (const BispectrumInnerPlan& inner : z.inner) {
                result.inner_outer_coefficients.push_back(inner.outer_coefficient);
                for (const BispectrumContractionTerm& term : inner.terms) {
                    result.term_first_indices.push_back(static_cast<std::int64_t>(term.first_index));
                    result.term_second_indices.push_back(static_cast<std::int64_t>(term.second_index));
                    result.term_coefficients.push_back(term.coefficient);
                }
                result.inner_term_offsets.push_back(
                    static_cast<std::int64_t>(result.term_first_indices.size()));
            }
            result.z_inner_offsets.push_back(
                static_cast<std::int64_t>(result.inner_outer_coefficients.size()));
            ++z_base;
        }
        for (const BispectrumProjectionTerm& projection : component.projection) {
            result.projection_u_indices.push_back(static_cast<std::int64_t>(projection.u_index));
            result.projection_z_indices.push_back(
                component_z_base + static_cast<std::int64_t>(projection.z_index));
            result.projection_scales.push_back(projection.scale);
        }
        result.projection_offsets.push_back(
            static_cast<std::int64_t>(result.projection_u_indices.size()));
        result.component_z_offsets.push_back(z_base);
    }
    return result;
}

#undef MDESCRIPTOR_ROTATIONAL_HD

} // namespace mdescriptor::detail::rotational
