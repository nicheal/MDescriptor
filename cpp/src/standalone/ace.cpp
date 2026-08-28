#include "mdescriptor/ace.hpp"

#include "mdescriptor/neighbor.hpp"
#include "descriptor_common.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

namespace mdescriptor {
namespace {

// scipy-openblas32 exposes a private, prefixed LAPACKE ABI.  Keep the tiny
// declaration local so ACE does not depend on the platform's unprefixed
// LAPACK headers (which differ across MSVC, macOS and Linux toolchains).
extern "C" int scipy_LAPACKE_dgesdd(
    int matrix_layout,
    char jobz,
    int m,
    int n,
    double* a,
    int lda,
    double* s,
    double* u,
    int ldu,
    double* vt,
    int ldvt);

constexpr int kLapackRowMajor = 101;

using detail::StructureBatchView;
using detail::Vec3;

constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr int kQuadraturePoints = 1000;
constexpr double kRankTolerance = 1e-7;

struct OneParticle {
    int species = 0;
    int n = 0;
    int l = 0;
    int m = 0;
    double degree = 0.0;
    int channel = -1;
};

struct FeatureTerm {
    std::vector<int> channels;
    double coefficient = 0.0;
};

struct Feature {
    int order = 0;
    std::vector<FeatureTerm> terms;
};

struct SvdResult {
    std::vector<double> values;
    std::vector<double> left; // row-major left singular vectors
};

struct RotationalBasis {
    std::vector<std::vector<int>> m_values;
    std::vector<double> uri; // row-major, ri_rank x m_values.size()
    int count = 0;
    int rank = 0;
};

double factorial_log(int value) {
    if (value < 0) return -std::numeric_limits<double>::infinity();
    return std::lgamma(static_cast<double>(value) + 1.0);
}

double binomial(int n, int k) {
    if (k < 0 || k > n || n < 0) return 0.0;
    return std::exp(factorial_log(n) - factorial_log(k) - factorial_log(n - k));
}

double clebsch_gordan(int j1, int m1, int j2, int m2, int J, int M) {
    if (std::abs(j1 - j2) > J || J > j1 + j2 || M != m1 + m2
        || std::abs(m1) > j1 || std::abs(m2) > j2 || std::abs(M) > J) {
        return 0.0;
    }
    const long double log_norm = std::log(static_cast<long double>(2 * J + 1))
        + factorial_log(j1 + m1) + factorial_log(j1 - m1)
        + factorial_log(j2 + m2) + factorial_log(j2 - m2)
        + factorial_log(J + M) + factorial_log(J - M)
        - factorial_log(j1 + j2 - J)
        - factorial_log(j1 - j2 + J)
        - factorial_log(-j1 + j2 + J)
        - factorial_log(j1 + j2 + J + 1);
    const int lower = std::max({0, j2 - J - m1, j1 - J + m2});
    const int upper = std::min({j1 + j2 - J, j1 - m1, j2 + m2});
    long double sum = 0.0L;
    for (int k = lower; k <= upper; ++k) {
        const long double term = static_cast<long double>(binomial(j1 + j2 - J, k))
            * static_cast<long double>(binomial(j1 - j2 + J, j1 - m1 - k))
            * static_cast<long double>(binomial(-j1 + j2 + J, j2 + m2 - k));
        sum += (k & 1) ? -term : term;
    }
    return static_cast<double>(std::sqrt(std::exp(log_norm)) * sum);
}

SvdResult singular_vectors(const std::vector<double>& input, int size) {
    SvdResult result;
    result.values.resize(static_cast<std::size_t>(size), 0.0);
    result.left.assign(static_cast<std::size_t>(size) * static_cast<std::size_t>(size), 0.0);
    if (size == 0) return result;
    if (size > std::numeric_limits<int>::max()) {
        throw std::invalid_argument("ACE rotational block is too large for LAPACK");
    }
    std::vector<double> matrix = input;
    std::vector<double> right(
        static_cast<std::size_t>(size) * static_cast<std::size_t>(size), 0.0);
    const int dimension = size;
    const int status = scipy_LAPACKE_dgesdd(
        kLapackRowMajor,
        'A',
        dimension,
        dimension,
        matrix.data(),
        dimension,
        result.values.data(),
        result.left.data(),
        dimension,
        right.data(),
        dimension);
    if (status != 0) {
        throw std::invalid_argument("ACE LAPACK SVD failed");
    }
    return result;
}

std::vector<std::vector<int>> m_range(const std::vector<int>& angular) {
    const int order = static_cast<int>(angular.size());
    std::vector<std::vector<int>> result;
    if (order == 0) return result;
    if (order == 1) {
        result.push_back({0});
        return result;
    }
    std::vector<int> current(static_cast<std::size_t>(order), 0);
    std::function<void(int)> visit = [&](int position) {
        if (position < 0) {
            int sum = 0;
            for (int i = 0; i < order - 1; ++i) sum += current[static_cast<std::size_t>(i)];
            const int last = -sum;
            if (std::abs(last) <= angular.back()) {
                current.back() = last;
                result.push_back(current);
            }
            return;
        }
        for (int value = -angular[static_cast<std::size_t>(position)];
             value <= angular[static_cast<std::size_t>(position)]; ++value) {
            current[static_cast<std::size_t>(position)] = value;
            visit(position - 1);
        }
    };
    // CartesianIndices in Julia iterates its first coordinate fastest.  The
    // descending recursion below reproduces that order.
    visit(order - 2);
    return result;
}

struct CouplingBuilder {
    std::map<std::tuple<std::vector<int>, std::vector<int>, std::vector<int>>, double> cache;

    double coefficient(
        const std::vector<int>& angular,
        const std::vector<int>& m,
        const std::vector<int>& k) {
        if (angular.size() != m.size() || angular.size() != k.size() || angular.empty()) {
            return 0.0;
        }
        const auto key = std::make_tuple(angular, m, k);
        const auto found = cache.find(key);
        if (found != cache.end()) return found->second;
        int sum_m = 0;
        int sum_k = 0;
        for (std::size_t i = 0; i < angular.size(); ++i) {
            if (std::abs(m[i]) > angular[i] || std::abs(k[i]) > angular[i]) {
                cache.emplace(key, 0.0);
                return 0.0;
            }
            sum_m += m[i];
            sum_k += k[i];
        }
        if (sum_m != 0 || sum_k != 0) {
            cache.emplace(key, 0.0);
            return 0.0;
        }
        double value = 0.0;
        if (angular.size() == 1) {
            value = (angular[0] == 0 && m[0] == 0 && k[0] == 0) ? 1.0 : 0.0;
        } else if (angular.size() == 2) {
            if (angular[0] == angular[1]) {
                const int exponent = m[0] - k[0];
                value = (exponent % 2 == 0 ? 1.0 : -1.0)
                    * (8.0 * kPi * kPi / static_cast<double>(2 * angular[0] + 1));
            }
        } else {
            const std::size_t prefix = angular.size() - 2;
            std::vector<int> angular_prefix(angular.begin(), angular.begin() + static_cast<std::ptrdiff_t>(prefix));
            std::vector<int> m_prefix(m.begin(), m.begin() + static_cast<std::ptrdiff_t>(prefix));
            std::vector<int> k_prefix(k.begin(), k.begin() + static_cast<std::ptrdiff_t>(prefix));
            for (int j = std::abs(angular[angular.size() - 2] - angular.back());
                 j <= angular[angular.size() - 2] + angular.back(); ++j) {
                const int m_sum = m[angular.size() - 2] + m.back();
                const int k_sum = k[angular.size() - 2] + k.back();
                if (std::abs(m_sum) > j || std::abs(k_sum) > j) continue;
                const double cg_k = clebsch_gordan(
                    angular[angular.size() - 2], k[angular.size() - 2],
                    angular.back(), k.back(), j, k_sum);
                const double cg_m = clebsch_gordan(
                    angular[angular.size() - 2], m[angular.size() - 2],
                    angular.back(), m.back(), j, m_sum);
                if (cg_k == 0.0 || cg_m == 0.0) continue;
                angular_prefix.push_back(j);
                m_prefix.push_back(m_sum);
                k_prefix.push_back(k_sum);
                value += cg_k * cg_m * coefficient(angular_prefix, m_prefix, k_prefix);
                angular_prefix.pop_back();
                m_prefix.pop_back();
                k_prefix.pop_back();
            }
        }
        cache.emplace(key, value);
        return value;
    }
};

RotationalBasis build_rotational_basis(
    const std::vector<int>& angular,
    CouplingBuilder& coupling) {
    RotationalBasis result;
    result.m_values = m_range(angular);
    result.count = static_cast<int>(result.m_values.size());
    if (result.count == 0) return result;
    std::vector<double> cc(
        static_cast<std::size_t>(result.count) * static_cast<std::size_t>(result.count), 0.0);
    for (int column = 0; column < result.count; ++column) {
        for (int row = 0; row < result.count; ++row) {
            cc[static_cast<std::size_t>(row) * result.count + column] = coupling.coefficient(
                angular,
                result.m_values[static_cast<std::size_t>(column)],
                result.m_values[static_cast<std::size_t>(row)]);
        }
    }
    const auto svd_ri = singular_vectors(cc, result.count);
    const double max_singular = svd_ri.values.empty() ? 0.0 : svd_ri.values.front();
    if (!(max_singular > 0.0)) return result;
    const double threshold = static_cast<double>(result.count)
        * std::numeric_limits<double>::epsilon() * max_singular;
    std::vector<int> ri_columns;
    for (int i = 0; i < result.count; ++i) {
        if (svd_ri.values[static_cast<std::size_t>(i)] > threshold) ri_columns.push_back(i);
    }
    result.rank = static_cast<int>(ri_columns.size());
    result.uri.assign(static_cast<std::size_t>(result.rank) * result.count, 0.0);
    for (int row = 0; row < result.rank; ++row) {
        const int column = ri_columns[static_cast<std::size_t>(row)];
        for (int m = 0; m < result.count; ++m) {
            result.uri[static_cast<std::size_t>(row) * result.count + m] =
                svd_ri.left[static_cast<std::size_t>(m) * result.count + column];
        }
    }
    return result;
}

std::vector<std::vector<double>> invariant_rows(
    const std::vector<int>& angular,
    const std::vector<int>& species,
    const std::vector<int>& radial,
    const RotationalBasis& rotational) {
    const auto& m_values = rotational.m_values;
    const int count = rotational.count;
    const int ri_rank = rotational.rank;
    if (count == 0 || ri_rank == 0) return {};
    const auto& uri = rotational.uri;

    std::vector<int> permutation(static_cast<std::size_t>(angular.size()));
    std::iota(permutation.begin(), permutation.end(), 0);
    std::vector<double> sym_gram(static_cast<std::size_t>(ri_rank) * static_cast<std::size_t>(ri_rank), 0.0);
    do {
        for (int first = 0; first < count; ++first) {
            for (int second = 0; second < count; ++second) {
                bool matches = true;
                for (std::size_t p = 0; p < permutation.size(); ++p) {
                    if (species[permutation[p]] != species[p]
                        || radial[permutation[p]] != radial[p]
                        || angular[permutation[p]] != angular[p]
                        || m_values[static_cast<std::size_t>(first)][permutation[p]]
                        != m_values[static_cast<std::size_t>(second)][p]) {
                        matches = false;
                        break;
                    }
                }
                if (!matches) continue;
                for (int i = 0; i < ri_rank; ++i) {
                    for (int j = 0; j < ri_rank; ++j) {
                        sym_gram[static_cast<std::size_t>(i) * ri_rank + j] +=
                            uri[static_cast<std::size_t>(i) * count + first]
                            * uri[static_cast<std::size_t>(j) * count + second];
                    }
                }
            }
        }
    } while (std::next_permutation(permutation.begin(), permutation.end()));
    const auto svd_sym = singular_vectors(sym_gram, ri_rank);
    const double max_value = svd_sym.values.empty() ? 0.0 : svd_sym.values.front();
    if (!(max_value > 0.0)) return {};
    std::vector<int> sym_columns;
    for (int i = 0; i < ri_rank; ++i) {
        if (svd_sym.values[static_cast<std::size_t>(i)] > kRankTolerance * max_value) {
            sym_columns.push_back(i);
        }
    }
    std::vector<std::vector<double>> rows;
    rows.reserve(sym_columns.size());
    for (int selected : sym_columns) {
        const double scale = std::sqrt(svd_sym.values[static_cast<std::size_t>(selected)]);
        std::vector<double> row(static_cast<std::size_t>(count), 0.0);
        for (int m = 0; m < count; ++m) {
            double value = 0.0;
            for (int i = 0; i < ri_rank; ++i) {
                value += svd_sym.left[static_cast<std::size_t>(i) * ri_rank + selected]
                    * uri[static_cast<std::size_t>(i) * count + m];
            }
            row[static_cast<std::size_t>(m)] = scale * value;
        }
        rows.push_back(std::move(row));
    }
    return rows;
}

std::size_t p_index(int l, int m) {
    return static_cast<std::size_t>(l * (l + 1) / 2 + m);
}

std::size_t y_index(int l, int m) {
    return static_cast<std::size_t>(l * l + l + m);
}

void spherical_harmonics(int max_l, const Vec3& displacement, std::vector<std::complex<double>>& values) {
    const double distance = std::sqrt(std::max(detail::norm2(displacement), 0.0));
    values.assign(static_cast<std::size_t>((max_l + 1) * (max_l + 1)), {0.0, 0.0});
    if (distance <= 0.0) return;
    const double cos_theta = std::max(-1.0, std::min(1.0, displacement.z / distance));
    const double sin_theta = std::sqrt(std::max(0.0, 1.0 - cos_theta * cos_theta));
    const double phi = std::atan2(displacement.y, displacement.x);
    std::vector<double> associated(static_cast<std::size_t>((max_l + 1) * (max_l + 2) / 2), 0.0);
    double temp = std::sqrt(0.5 / kPi);
    associated[p_index(0, 0)] = temp;
    if (max_l > 0) {
        associated[p_index(1, 0)] = cos_theta * std::sqrt(3.0) * temp;
        temp = -std::sqrt(1.5) * sin_theta * temp;
        associated[p_index(1, 1)] = temp;
        for (int l = 2; l <= max_l; ++l) {
            const int ls = l * l;
            const int previous_ls = (l - 1) * (l - 1);
            for (int m = 0; m <= l - 2; ++m) {
                const double a = std::sqrt((4.0 * ls - 1.0) / (ls - m * m));
                const double b = -std::sqrt(static_cast<double>(previous_ls - m * m)
                    / static_cast<double>(4 * previous_ls - 1));
                associated[p_index(l, m)] = a * (cos_theta * associated[p_index(l - 1, m)]
                    + b * associated[p_index(l - 2, m)]);
            }
            associated[p_index(l, l - 1)] = cos_theta * std::sqrt(2.0 * (l - 1) + 3.0) * temp;
            temp = -std::sqrt(1.0 + 0.5 / static_cast<double>(l)) * sin_theta * temp;
            associated[p_index(l, l)] = temp;
        }
    }
    const std::complex<double> factor(std::cos(phi), std::sin(phi));
    std::complex<double> positive(1.0 / std::sqrt(2.0), 0.0);
    int sign = 1;
    for (int l = 0; l <= max_l; ++l) {
        values[y_index(l, 0)] = associated[p_index(l, 0)] / std::sqrt(2.0);
    }
    for (int m = 1; m <= max_l; ++m) {
        sign *= -1;
        positive *= factor;
        const std::complex<double> negative = static_cast<double>(sign) * std::conj(positive);
        for (int l = m; l <= max_l; ++l) {
            values[y_index(l, -m)] = negative * associated[p_index(l, m)];
            values[y_index(l, m)] = positive * associated[p_index(l, m)];
        }
    }
}

} // namespace

struct AceCalculator::Impl {
    std::vector<OneParticle> aspec;
    std::vector<OneParticle> base_spec;
    std::vector<std::vector<Feature>> features;
    std::vector<double> radial_a;
    std::vector<double> radial_b;
    std::vector<double> radial_c;
    std::vector<double> radial_t;
    int max_n = 0;
    int max_l = 0;
    double t_left = 0.0;
    double t_right = 0.0;
    int p_left = 0;
    int p_right = 0;
    std::map<std::tuple<int, int, int, int>, int> channel_map;
};

namespace {

double transformed_distance(const AceOptions& options, double radius) {
    return std::pow((options.transform_a + options.r0)
        / (options.transform_a + radius), options.transform_p);
}

double one_particle_degree(const AceOptions& options, int order, int n, int l) {
    if (!options.degree_by_order.empty()) {
        const std::size_t index = static_cast<std::size_t>(order - 1);
        return (static_cast<double>(n) + options.angular_weight_by_order[index] * static_cast<double>(l))
            / options.degree_by_order[index];
    }
    return static_cast<double>(n) + options.w_l * static_cast<double>(l);
}

double product_degree(const AceOptions& options, const std::vector<OneParticle>& values) {
    double degree = 0.0;
    const int order = static_cast<int>(values.size());
    if (order == 0) return 0.0;
    if (!options.degree_by_order.empty()) {
        const std::size_t index = static_cast<std::size_t>(order - 1);
        for (const auto& value : values) {
            degree += (static_cast<double>(value.n)
                + options.angular_weight_by_order[index] * static_cast<double>(value.l))
                / options.degree_by_order[index];
        }
        return degree;
    }
    double hyperbolic = 1.0;
    for (const auto& value : values) {
        degree += value.degree;
        hyperbolic *= std::max(options.degree_ahc,
            options.degree_bhc + value.degree);
    }
    return options.degree_csp * degree + options.degree_chc * hyperbolic;
}

void generate_combinations(
    const std::vector<OneParticle>& aspec,
    const AceOptions& options,
    int order,
    int start,
    std::vector<int>& current,
    std::vector<std::vector<int>>& result) {
    if (static_cast<int>(current.size()) == order) {
        std::vector<OneParticle> values;
        values.reserve(current.size());
        for (int index : current) values.push_back(aspec[static_cast<std::size_t>(index)]);
        const double degree = product_degree(options, values);
        const double limit = options.degree_by_order.empty() ? options.max_degree : 1.0;
        if (degree > limit + 1e-12) return;
        if (order == 0) {
            if (options.constants) result.push_back(current);
            return;
        }
        if (order == 1 && values[0].l != 0) return;
        if (order >= 2) {
            int sum_l = 0;
            int sum_m = 0;
            for (const auto& value : values) {
                sum_l += value.l;
                sum_m += value.m;
            }
            if ((sum_l & 1) != 0 || sum_m != 0) return;
        }
        result.push_back(current);
        return;
    }
    for (int index = start; index < static_cast<int>(aspec.size()); ++index) {
        current.push_back(index);
        generate_combinations(aspec, options, order, index, current, result);
        current.pop_back();
    }
}

void build_radial_basis(const AceOptions& options, AceCalculator::Impl& impl) {
    const int max_n = impl.max_n;
    impl.radial_a.assign(static_cast<std::size_t>(max_n), 0.0);
    impl.radial_b.assign(static_cast<std::size_t>(max_n), 0.0);
    impl.radial_c.assign(static_cast<std::size_t>(max_n), 0.0);
    const double t_cut = transformed_distance(options, options.r_cut);
    const double t_in = transformed_distance(options, options.r_in);
    if (t_cut < t_in) {
        impl.t_left = t_cut;
        impl.t_right = t_in;
        impl.p_left = options.p_cut;
        impl.p_right = options.p_in;
    } else {
        impl.t_left = t_in;
        impl.t_right = t_cut;
        impl.p_left = options.p_in;
        impl.p_right = options.p_cut;
    }
    if (!(impl.t_right > impl.t_left)) {
        throw std::invalid_argument("ACE transformed radial interval must be non-empty");
    }
    const double step = (impl.t_right - impl.t_left) / static_cast<double>(kQuadraturePoints);
    impl.radial_t.resize(kQuadraturePoints);
    for (int i = 0; i < kQuadraturePoints; ++i) {
        impl.radial_t[static_cast<std::size_t>(i)] = impl.t_left
            + step * (static_cast<double>(i) + 0.5);
    }
    auto envelope = [&](double t) {
        if ((impl.p_left > 0 && t < impl.t_left) || (impl.p_right > 0 && t > impl.t_right)) {
            return 0.0;
        }
        return std::pow(t - impl.t_left, impl.p_left)
            * std::pow(t - impl.t_right, impl.p_right);
    };
    auto dot_mean = [&](const std::vector<double>& left, const std::vector<double>& right) {
        double value = 0.0;
        for (int i = 0; i < kQuadraturePoints; ++i) {
            value += left[static_cast<std::size_t>(i)] * right[static_cast<std::size_t>(i)];
        }
        return value / static_cast<double>(kQuadraturePoints);
    };
    auto weighted_moment = [&](const std::vector<double>& left,
                               const std::vector<double>& right) {
        double value = 0.0;
        for (int i = 0; i < kQuadraturePoints; ++i) {
            const double previous = left[static_cast<std::size_t>(i)];
            value += impl.radial_t[static_cast<std::size_t>(i)] * previous
                * right[static_cast<std::size_t>(i)];
        }
        return value / static_cast<double>(kQuadraturePoints);
    };
    std::vector<double> first(kQuadraturePoints, 0.0);
    for (int i = 0; i < kQuadraturePoints; ++i) first[static_cast<std::size_t>(i)] = envelope(impl.radial_t[static_cast<std::size_t>(i)]);
    double norm_first = std::sqrt(dot_mean(first, first));
    if (!(norm_first > 0.0) || !std::isfinite(norm_first)) {
        throw std::invalid_argument("ACE radial cutoff envelope has zero norm");
    }
    impl.radial_a[0] = 1.0 / norm_first;
    if (max_n == 1) return;
    std::vector<double> previous(kQuadraturePoints, 0.0);
    for (int i = 0; i < kQuadraturePoints; ++i) previous[static_cast<std::size_t>(i)] = impl.radial_a[0] * first[static_cast<std::size_t>(i)];
    // OrthPolyBasis uses <t J_n, J_n>, not <t, J_n>.  Keeping the
    // polynomial factor on both sides is essential for ACE1's recurrence
    // coefficients and high-order radial values.
    double b = weighted_moment(previous, previous);
    std::vector<double> current_raw(kQuadraturePoints, 0.0);
    for (int i = 0; i < kQuadraturePoints; ++i) current_raw[static_cast<std::size_t>(i)] = (impl.radial_t[static_cast<std::size_t>(i)] - b) * previous[static_cast<std::size_t>(i)];
    double norm_current = std::sqrt(dot_mean(current_raw, current_raw));
    if (!(norm_current > 0.0) || !std::isfinite(norm_current)) {
        throw std::invalid_argument("ACE radial quadrature is rank deficient");
    }
    impl.radial_a[1] = 1.0 / norm_current;
    impl.radial_b[1] = -b / norm_current;
    std::vector<double> previous_previous = previous;
    previous.resize(kQuadraturePoints);
    for (int i = 0; i < kQuadraturePoints; ++i) previous[static_cast<std::size_t>(i)] =
        (impl.radial_a[1] * impl.radial_t[static_cast<std::size_t>(i)] + impl.radial_b[1])
        * previous_previous[static_cast<std::size_t>(i)];
    for (int n = 2; n < max_n; ++n) {
        b = weighted_moment(previous, previous);
        const double c = weighted_moment(previous, previous_previous);
        for (int i = 0; i < kQuadraturePoints; ++i) {
            current_raw[static_cast<std::size_t>(i)] =
                (impl.radial_t[static_cast<std::size_t>(i)] - b) * previous[static_cast<std::size_t>(i)]
                - c * previous_previous[static_cast<std::size_t>(i)];
        }
        norm_current = std::sqrt(dot_mean(current_raw, current_raw));
        if (!(norm_current > 0.0) || !std::isfinite(norm_current)) {
            throw std::invalid_argument("ACE radial quadrature is rank deficient");
        }
        impl.radial_a[static_cast<std::size_t>(n)] = 1.0 / norm_current;
        impl.radial_b[static_cast<std::size_t>(n)] = -b / norm_current;
        impl.radial_c[static_cast<std::size_t>(n)] = -c / norm_current;
        std::vector<double> next(kQuadraturePoints, 0.0);
        for (int i = 0; i < kQuadraturePoints; ++i) {
            next[static_cast<std::size_t>(i)] =
                (impl.radial_a[static_cast<std::size_t>(n)] * impl.radial_t[static_cast<std::size_t>(i)]
                    + impl.radial_b[static_cast<std::size_t>(n)]) * previous[static_cast<std::size_t>(i)]
                + impl.radial_c[static_cast<std::size_t>(n)] * previous_previous[static_cast<std::size_t>(i)];
        }
        previous_previous.swap(previous);
        previous = std::move(next);
    }
}

} // namespace

AceCalculator::AceCalculator(AceOptions options) : options_(std::move(options)) {
    if (options_.species.empty()) throw std::invalid_argument("ACE requires at least one species");
    if (options_.max_order < 1) throw std::invalid_argument("ACE N must be at least one");
    if (!std::isfinite(options_.r0) || !std::isfinite(options_.transform_p)
        || !std::isfinite(options_.transform_a) || options_.transform_p == 0.0
        || options_.transform_a + options_.r0 <= 0.0) {
        throw std::invalid_argument("invalid ACE PolyTransform parameters");
    }
    if (!std::isfinite(options_.r_cut) || !std::isfinite(options_.r_in)
        || options_.r_cut <= 0.0 || options_.r_in < 0.0 || options_.r_in >= options_.r_cut) {
        throw std::invalid_argument("ACE requires 0 <= rin < rcut");
    }
    if (options_.p_cut < 0 || options_.p_in < 0) {
        throw std::invalid_argument("ACE pcut and pin must be non-negative");
    }
    if (options_.num_threads < 0) throw std::invalid_argument("ACE num_threads must be non-negative");
    if (!options_.degree_by_order.empty()) {
        if (options_.degree_by_order.size() != static_cast<std::size_t>(options_.max_order)
            || options_.angular_weight_by_order.size() != options_.degree_by_order.size()) {
            throw std::invalid_argument("ACE degree vectors must have length N");
        }
        for (std::size_t i = 0; i < options_.degree_by_order.size(); ++i) {
            if (!std::isfinite(options_.degree_by_order[i]) || options_.degree_by_order[i] <= 0.0
                || !std::isfinite(options_.angular_weight_by_order[i])
                || options_.angular_weight_by_order[i] <= 0.0) {
                throw std::invalid_argument("ACE degree vectors must be finite and positive");
            }
        }
    } else if (!std::isfinite(options_.max_degree) || options_.max_degree <= 0.0
               || !std::isfinite(options_.w_l) || options_.w_l <= 0.0
               || !std::isfinite(options_.degree_csp) || options_.degree_csp < 0.0
               || !std::isfinite(options_.degree_chc) || options_.degree_chc < 0.0
               || !std::isfinite(options_.degree_ahc) || options_.degree_ahc < 0.0
               || !std::isfinite(options_.degree_bhc) || options_.degree_bhc < 0.0) {
        throw std::invalid_argument("ACE maxdeg and wL must be finite and positive");
    }

    auto impl = std::make_shared<Impl>();
    int max_n = 1;
    const double radial_degree_limit = options_.degree_by_order.empty() ? options_.max_degree : 1.0;
    while (one_particle_degree(options_, 1, max_n, 0) < radial_degree_limit) {
        if (max_n == std::numeric_limits<int>::max()) {
            throw std::invalid_argument("ACE radial basis is too large");
        }
        ++max_n;
    }
    impl->max_n = max_n;
    const double max_single_degree = one_particle_degree(options_, 1, max_n, 0);
    for (std::size_t species_index = 0; species_index < options_.species.size(); ++species_index) {
        for (int n = 1; n <= max_n; ++n) {
            for (int l = 0;; ++l) {
                const double degree = one_particle_degree(options_, 1, n, l);
                if (degree > max_single_degree + 1e-12) break;
                for (int m = -l; m <= l; ++m) {
                    impl->base_spec.push_back({static_cast<int>(species_index), n, l, m, degree, -1});
                }
                impl->max_l = std::max(impl->max_l, l);
            }
        }
    }
    // The species/n/l/m loop above is the l-major order used by ACE1's
    // BasicPSH1pBasis specification (with m running from -l through l).
    impl->max_n = max_n;
    impl->aspec = impl->base_spec;
    std::stable_sort(impl->aspec.begin(), impl->aspec.end(), [](const OneParticle& left, const OneParticle& right) {
        return left.degree < right.degree;
    });
    for (std::size_t index = 0; index < impl->base_spec.size(); ++index) {
        const auto& value = impl->base_spec[index];
        impl->channel_map.emplace(std::make_tuple(value.species, value.n, value.l, value.m), static_cast<int>(index));
    }
    for (auto& value : impl->aspec) {
        value.channel = impl->channel_map.at(std::make_tuple(value.species, value.n, value.l, value.m));
    }
    build_radial_basis(options_, *impl);

    CouplingBuilder coupling;
    std::map<std::vector<int>, RotationalBasis> rotational_cache;
    // RPIBasis only retains product specifications whose one-particle
    // magnetic indices are all zero.  Enumerating the remaining m channels
    // and discarding them later is combinatorially expensive at high degree,
    // so keep the same degree-sorted order but generate combinations from the
    // zero-m representatives directly.
    std::vector<OneParticle> zero_m_aspec;
    zero_m_aspec.reserve(impl->aspec.size());
    for (const auto& value : impl->aspec) {
        if (value.m == 0) zero_m_aspec.push_back(value);
    }
    std::vector<std::vector<int>> combinations;
    for (int order = 0; order <= options_.max_order; ++order) {
        std::vector<int> current;
        generate_combinations(zero_m_aspec, options_, order, 0, current, combinations);
    }
    impl->features.resize(options_.species.size());
    for (std::size_t center = 0; center < options_.species.size(); ++center) {
        // The supported ACE1 degree specifications are independent of the
        // centre species.  RPIBasis therefore has the same feature rows for
        // every centre species; build them once and share the immutable
        // representation instead of repeating all CG/SVD work per centre.
        if (center != 0) {
            impl->features[center] = impl->features.front();
            feature_counts_.push_back(feature_counts_.front());
            continue;
        }
        auto& destination = impl->features[center];
        std::vector<std::pair<std::pair<int, std::vector<int>>, Feature>> generated;
        for (const auto& combination : combinations) {
            if (combination.empty()) {
                generated.push_back({{0, {}}, {0, {{{}, 1.0}}}});
                continue;
            }
            // ``gensparse(..., ordered=true)`` first enumerates indices in the
            // degree-sorted one-particle list, but InnerPIBasis canonicalises
            // every tuple back to the original BasicPSH1pBasis order before
            // sorting it.  Work in that canonical order here as well; this is
            // what fixes both ACE1 feature ordering and coupling orientation.
            std::vector<int> channels;
            channels.reserve(combination.size());
            for (int index : combination) {
                channels.push_back(zero_m_aspec[static_cast<std::size_t>(index)].channel);
            }
            std::sort(channels.begin(), channels.end());
            std::vector<OneParticle> canonical_values;
            canonical_values.reserve(channels.size());
            for (int channel : channels) {
                canonical_values.push_back(impl->base_spec[static_cast<std::size_t>(channel)]);
            }
            bool all_zero_m = true;
            std::vector<int> angular;
            std::vector<int> species;
            std::vector<int> radial;
            angular.reserve(combination.size());
            species.reserve(combination.size());
            radial.reserve(combination.size());
            for (const auto& value : canonical_values) {
                all_zero_m = all_zero_m && value.m == 0;
                angular.push_back(value.l);
                species.push_back(value.species);
                radial.push_back(value.n);
            }
            if (!all_zero_m) continue;
            if (combination.size() == 1) {
                generated.push_back({{1, channels}, {1, {{{channels[0]}, 1.0}}}});
                continue;
            }
            const auto cache_it = rotational_cache.find(angular);
            if (cache_it == rotational_cache.end()) {
                const auto inserted = rotational_cache.emplace(
                    angular, build_rotational_basis(angular, coupling));
                if (!inserted.second) {
                    throw std::runtime_error("ACE rotational cache insertion failed");
                }
            }
            const auto& rotational = rotational_cache.at(angular);
            const auto rows = invariant_rows(angular, species, radial, rotational);
            const auto& m_values = rotational.m_values;
            for (const auto& row : rows) {
                Feature feature;
                feature.order = static_cast<int>(combination.size());
                for (std::size_t m_index = 0; m_index < m_values.size(); ++m_index) {
                    if (std::abs(row[m_index]) <= 1e-15) continue;
                    FeatureTerm term;
                    term.coefficient = row[m_index];
                    for (std::size_t position = 0; position < canonical_values.size(); ++position) {
                        const auto& value = canonical_values[position];
                        term.channels.push_back(impl->channel_map.at(std::make_tuple(
                            value.species, value.n, value.l, m_values[m_index][position])));
                    }
                    feature.terms.push_back(std::move(term));
                }
                if (!feature.terms.empty()) generated.push_back({{feature.order, channels}, std::move(feature)});
            }
        }
        std::stable_sort(generated.begin(), generated.end(), [](const auto& left, const auto& right) {
            if (left.first.first != right.first.first) return left.first.first < right.first.first;
            return left.first.second < right.first.second;
        });
        destination.reserve(generated.size());
        for (auto& item : generated) destination.push_back(std::move(item.second));
        feature_counts_.push_back(static_cast<std::int64_t>(destination.size()));
    }
    feature_count_ = feature_counts_.empty() ? 0 : *std::max_element(feature_counts_.begin(), feature_counts_.end());
    max_angular_ = impl->max_l;
    max_radial_ = impl->max_n;
    impl_ = std::move(impl);
}

std::int64_t AceCalculator::feature_count() const noexcept { return feature_count_; }
const std::vector<std::int32_t>& AceCalculator::species() const noexcept { return options_.species; }
const std::vector<std::int64_t>& AceCalculator::feature_counts() const noexcept { return feature_counts_; }
std::int32_t AceCalculator::max_angular() const noexcept { return max_angular_; }
std::int32_t AceCalculator::max_radial() const noexcept { return max_radial_; }
void AceCalculator::close() noexcept {
    closed_.store(true, std::memory_order_release);
}
bool AceCalculator::closed() const noexcept { return closed_.load(std::memory_order_acquire); }

void AceCalculator::compute(
    const StructureBatchView& batch,
    double* output,
    const std::shared_ptr<ComputeControl>& control) const {
    if (closed()) throw std::runtime_error("ACE calculator is closed");
    std::lock_guard<std::mutex> lock(compute_mutex_);
    const auto graph = build_neighbor_graph(batch, options_.r_cut, control, options_.num_threads);
    std::map<std::int32_t, int> species_index;
    for (std::size_t i = 0; i < options_.species.size(); ++i) {
        species_index.emplace(options_.species[i], static_cast<int>(i));
    }
    for (std::int64_t atom = 0; atom < batch.atoms; ++atom) {
        if (species_index.find(batch.numbers[atom]) == species_index.end()) {
            throw std::invalid_argument("ACE input contains an undeclared species");
        }
    }
    const std::size_t total = static_cast<std::size_t>(batch.atoms)
        * static_cast<std::size_t>(feature_count_);
    std::fill(output, output + total, 0.0);
    const std::size_t channels = impl_->base_spec.size();
    // The one-particle specification is generated once per declared species,
    // with the same (n,l,m) block for each species.  Restrict each neighbour
    // accumulation to its contiguous block instead of scanning every species
    // channel and branching on `spec.species`.
    const std::size_t channels_per_species = channels / options_.species.size();
    const auto evaluate_radial = [&](double radius, std::vector<double>& values) {
        values.assign(static_cast<std::size_t>(impl_->max_n), 0.0);
        const double t = transformed_distance(options_, radius);
        if ((impl_->p_left > 0 && t < impl_->t_left)
            || (impl_->p_right > 0 && t > impl_->t_right)) return;
        const double envelope = std::pow(t - impl_->t_left, impl_->p_left)
            * std::pow(t - impl_->t_right, impl_->p_right);
        values[0] = impl_->radial_a[0] * envelope;
        if (impl_->max_n == 1) return;
        values[1] = (impl_->radial_a[1] * t + impl_->radial_b[1]) * values[0];
        for (int n = 2; n < impl_->max_n; ++n) {
            values[static_cast<std::size_t>(n)] =
                (impl_->radial_a[static_cast<std::size_t>(n)] * t
                    + impl_->radial_b[static_cast<std::size_t>(n)]) * values[static_cast<std::size_t>(n - 1)]
                + impl_->radial_c[static_cast<std::size_t>(n)] * values[static_cast<std::size_t>(n - 2)];
        }
    };
    detail::run_parallel_structures(batch.structures, options_.num_threads, control, [&](std::int64_t structure) {
        const std::int64_t begin = batch.offsets[structure];
        const std::int64_t end = batch.offsets[structure + 1];
        for (std::int64_t center = begin; center < end; ++center) {
            if (detail::cancelled(control)) return;
            std::vector<std::complex<double>> values(channels, {0.0, 0.0});
            const auto neighbors = graph.for_center(center);
            std::vector<double> radial;
            std::vector<std::complex<double>> harmonics;
            for (std::size_t index = 0; index < neighbors.size; ++index) {
                if (neighbors.exact_self(index, center)) continue;
                const double distance2 = neighbors.distance2[index];
                const double distance = std::sqrt(std::max(distance2, 0.0));
                if (distance <= 0.0) continue;
                const auto neighbor_atom = neighbors.atoms[index];
                const auto species_it = species_index.find(batch.numbers[neighbor_atom]);
                if (species_it == species_index.end()) continue;
                const Vec3 displacement{
                    neighbors.displacements[index * 3 + 0],
                    neighbors.displacements[index * 3 + 1],
                    neighbors.displacements[index * 3 + 2],
                };
                evaluate_radial(distance, radial);
                spherical_harmonics(impl_->max_l, displacement, harmonics);
                const std::size_t species_begin = static_cast<std::size_t>(species_it->second)
                    * channels_per_species;
                const std::size_t species_end = species_begin + channels_per_species;
                for (std::size_t spec_index = species_begin; spec_index < species_end; ++spec_index) {
                    const auto& spec = impl_->base_spec[spec_index];
                    values[spec_index] += radial[static_cast<std::size_t>(spec.n - 1)]
                        * harmonics[y_index(spec.l, spec.m)];
                }
            }
            const auto center_it = species_index.find(batch.numbers[center]);
            if (center_it == species_index.end()) {
                throw std::invalid_argument("ACE input contains an undeclared center species");
            }
            const auto& basis = impl_->features[static_cast<std::size_t>(center_it->second)];
            double* destination = output + static_cast<std::size_t>(center) * static_cast<std::size_t>(feature_count_);
            for (std::size_t feature = 0; feature < basis.size(); ++feature) {
                double result = 0.0;
                for (const auto& term : basis[feature].terms) {
                    std::complex<double> product(term.coefficient, 0.0);
                    for (int channel : term.channels) product *= values[static_cast<std::size_t>(channel)];
                    result += std::real(product);
                }
                destination[feature] = result;
            }
        }
    });
}

} // namespace mdescriptor
