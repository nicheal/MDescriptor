#include "mdescriptor/nep.hpp"

#include "mdescriptor/neighbor.hpp"
#include "descriptor_common.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <limits>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace mdescriptor {
namespace {

using detail::kPi;
constexpr int kNumAngularTerms = 80;

// These constants are the fixed angular normalization coefficients used by
// NEP_CPU. They are part of the descriptor definition, not fitted model data.
constexpr double kC3B[kNumAngularTerms] = {
    0.238732414637843, 0.119366207318922, 0.119366207318922, 0.099471839432435,
    0.596831036594608, 0.596831036594608, 0.149207759148652, 0.149207759148652,
    0.139260575205408, 0.104445431404056, 0.104445431404056, 1.044454314040563,
    1.044454314040563, 0.174075719006761, 0.174075719006761, 0.011190581936149,
    0.223811638722978, 0.223811638722978, 0.111905819361489, 0.111905819361489,
    1.566681471060845, 1.566681471060845, 0.195835183882606, 0.195835183882606,
    0.013677377921960, 0.102580334414698, 0.102580334414698, 2.872249363611549,
    2.872249363611549, 0.119677056817148, 0.119677056817148, 2.154187022708661,
    2.154187022708661, 0.215418702270866, 0.215418702270866, 0.004041043476943,
    0.169723826031592, 0.169723826031592, 0.106077391269745, 0.106077391269745,
    0.424309565078979, 0.424309565078979, 0.127292869523694, 0.127292869523694,
    2.800443129521260, 2.800443129521260, 0.233370260793438, 0.233370260793438,
    0.004662742473395, 0.004079899664221, 0.004079899664221, 0.024479397985326,
    0.024479397985326, 0.012239698992663, 0.012239698992663, 0.538546755677165,
    0.538546755677165, 0.134636688919291, 0.134636688919291, 3.500553911901575,
    3.500553911901575, 0.250039565135827, 0.250039565135827, 0.000082569397966,
    0.005944996653579, 0.005944996653579, 0.104037441437634, 0.104037441437634,
    0.762941237209318, 0.762941237209318, 0.114441185581398, 0.114441185581398,
    5.950941650232678, 5.950941650232678, 0.141689086910302, 0.141689086910302,
    4.250672607309055, 4.250672607309055, 0.265667037956816, 0.265667037956816,
};
constexpr double kC4B[5] = {
    -0.007499480826664, -0.134990654879954, 0.067495327439977,
    0.404971964639861, -0.809943929279723,
};
constexpr double kC5B[3] = {
    0.026596810706114, 0.053193621412227, 0.026596810706114,
};
constexpr double kC4B2[5] = {
    0.027493550848847, 0.164961305093080, -0.013746775424423,
    0.041240326273270, 0.082480652546540,
};
constexpr double kC4B123[7] = {
    -0.008418146349617, -0.016836292699234, -0.033672585398469,
    -0.042090731748086, -0.067345170796937, -0.084181463496172,
    -0.168362926992344,
};
constexpr double kC4B233[10] = {
    0.008572620635186, 0.009644198214584, 0.019288396429168,
    0.025717861905558, 0.026789439484956, 0.032147327381947,
    0.038576792858337, 0.128589309527790, 0.192883964291685,
    0.321473273819474,
};
constexpr double kC4B134[10] = {
    0.003645164295772, 0.004860219061029, 0.006075273826286,
    0.018225821478859, 0.024301095305146, 0.036451642957719,
    0.042526916784005, 0.072903285915437, 0.085053833568010,
    0.255161500704030,
};

constexpr double kZ1[2][2] = {{0.0, 1.0}, {1.0, 0.0}};
constexpr double kZ2[3][3] = {{-1.0, 0.0, 3.0}, {0.0, 1.0, 0.0}, {1.0, 0.0, 0.0}};
constexpr double kZ3[4][4] = {
    {0.0, -3.0, 0.0, 5.0}, {-1.0, 0.0, 5.0, 0.0},
    {0.0, 1.0, 0.0, 0.0}, {1.0, 0.0, 0.0, 0.0},
};
constexpr double kZ4[5][5] = {
    {3.0, 0.0, -30.0, 0.0, 35.0}, {0.0, -3.0, 0.0, 7.0, 0.0},
    {-1.0, 0.0, 7.0, 0.0, 0.0}, {0.0, 1.0, 0.0, 0.0, 0.0},
    {1.0, 0.0, 0.0, 0.0, 0.0},
};
constexpr double kZ5[6][6] = {
    {0.0, 15.0, 0.0, -70.0, 0.0, 63.0}, {1.0, 0.0, -14.0, 0.0, 21.0, 0.0},
    {0.0, -1.0, 0.0, 3.0, 0.0, 0.0}, {-1.0, 0.0, 9.0, 0.0, 0.0, 0.0},
    {0.0, 1.0, 0.0, 0.0, 0.0, 0.0}, {1.0, 0.0, 0.0, 0.0, 0.0, 0.0},
};
constexpr double kZ6[7][7] = {
    {-5.0, 0.0, 105.0, 0.0, -315.0, 0.0, 231.0},
    {0.0, 5.0, 0.0, -30.0, 0.0, 33.0, 0.0},
    {1.0, 0.0, -18.0, 0.0, 33.0, 0.0, 0.0},
    {0.0, -3.0, 0.0, 11.0, 0.0, 0.0, 0.0},
    {-1.0, 0.0, 11.0, 0.0, 0.0, 0.0, 0.0},
    {0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0}, {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
};
constexpr double kZ7[8][8] = {
    {0.0, -35.0, 0.0, 315.0, 0.0, -693.0, 0.0, 429.0},
    {-5.0, 0.0, 135.0, 0.0, -495.0, 0.0, 429.0, 0.0},
    {0.0, 15.0, 0.0, -110.0, 0.0, 143.0, 0.0, 0.0},
    {3.0, 0.0, -66.0, 0.0, 143.0, 0.0, 0.0, 0.0},
    {0.0, -3.0, 0.0, 13.0, 0.0, 0.0, 0.0, 0.0},
    {-1.0, 0.0, 13.0, 0.0, 0.0, 0.0, 0.0, 0.0},
    {0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
    {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
};
constexpr double kZ8[9][9] = {
    {35.0, 0.0, -1260.0, 0.0, 6930.0, 0.0, -12012.0, 0.0, 6435.0},
    {0.0, -35.0, 0.0, 385.0, 0.0, -1001.0, 0.0, 715.0, 0.0},
    {-1.0, 0.0, 33.0, 0.0, -143.0, 0.0, 143.0, 0.0, 0.0},
    {0.0, 3.0, 0.0, -26.0, 0.0, 39.0, 0.0, 0.0, 0.0},
    {1.0, 0.0, -26.0, 0.0, 65.0, 0.0, 0.0, 0.0, 0.0},
    {0.0, -1.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0},
    {-1.0, 0.0, 15.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
    {0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
    {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
};

template <int L>
double z_coefficient(int n1, int n2) {
    if constexpr (L == 1) return kZ1[n1][n2];
    if constexpr (L == 2) return kZ2[n1][n2];
    if constexpr (L == 3) return kZ3[n1][n2];
    if constexpr (L == 4) return kZ4[n1][n2];
    if constexpr (L == 5) return kZ5[n1][n2];
    if constexpr (L == 6) return kZ6[n1][n2];
    if constexpr (L == 7) return kZ7[n1][n2];
    return kZ8[n1][n2];
}

template <typename Scalar>
void complex_product(Scalar a, Scalar b, Scalar& real, Scalar& imag) {
    const Scalar old_real = real;
    real = a * old_real - b * imag;
    imag = a * imag + b * old_real;
}

template <int L, typename Scalar>
void accumulate_s_one(Scalar x, Scalar y, Scalar z, Scalar fn, Scalar* s) {
    int index = L * L - 1;
    Scalar z_power[L + 1] = {1.0};
    for (int n = 1; n <= L; ++n) {
        z_power[n] = z * z_power[n - 1];
    }
    Scalar real = x;
    Scalar imag = y;
    for (int n1 = 0; n1 <= L; ++n1) {
        const int start = (L + n1) % 2 == 0 ? 0 : 1;
        Scalar z_factor = 0.0;
        for (int n2 = start; n2 <= L - n1; n2 += 2) {
            z_factor += z_coefficient<L>(n1, n2) * z_power[n2];
        }
        z_factor *= fn;
        if (n1 == 0) {
            s[index++] += z_factor;
        } else {
            s[index++] += z_factor * real;
            s[index++] += z_factor * imag;
            complex_product(x, y, real, imag);
        }
    }
}

template <int BasisSize>
inline void basis_values_fixed(double cutoff, double distance, double* values) {
    const double fc = 0.5 * std::cos(kPi * distance / cutoff) + 0.5;
    const double a = distance / cutoff - 1.0;
    const double x = 2.0 * a * a - 1.0;
    values[0] = fc;
    if constexpr (BasisSize >= 1) {
        values[1] = 0.5 * (x + 1.0) * fc;
        double previous = 1.0;
        double current = x;
        for (int order = 2; order <= BasisSize; ++order) {
            const double next = 2.0 * x * current - previous;
            previous = current;
            current = next;
            values[order] = 0.5 * (current + 1.0) * fc;
        }
    }
}

inline void accumulate_s_l4(
    double distance, double x, double y, double z, double fn, double* s) {
    const double inverse_distance = 1.0 / distance;
    x *= inverse_distance;
    y *= inverse_distance;
    z *= inverse_distance;
    accumulate_s_one<1>(x, y, z, fn, s);
    accumulate_s_one<2>(x, y, z, fn, s);
    accumulate_s_one<3>(x, y, z, fn, s);
    accumulate_s_one<4>(x, y, z, fn, s);
}

void accumulate_s(int l_max, double distance, double x, double y, double z, double fn, double* s) {
    const double inverse_distance = 1.0 / distance;
    x *= inverse_distance;
    y *= inverse_distance;
    z *= inverse_distance;
    if (l_max >= 1) accumulate_s_one<1>(x, y, z, fn, s);
    if (l_max >= 2) accumulate_s_one<2>(x, y, z, fn, s);
    if (l_max >= 3) accumulate_s_one<3>(x, y, z, fn, s);
    if (l_max >= 4) accumulate_s_one<4>(x, y, z, fn, s);
    if (l_max >= 5) accumulate_s_one<5>(x, y, z, fn, s);
    if (l_max >= 6) accumulate_s_one<6>(x, y, z, fn, s);
    if (l_max >= 7) accumulate_s_one<7>(x, y, z, fn, s);
    if (l_max >= 8) accumulate_s_one<8>(x, y, z, fn, s);
}

template <int L, typename Scalar>
Scalar find_q_one(const Scalar* s) {
    const int start = L * L - 1;
    const int count = 2 * L + 1;
    Scalar q = kC3B[start] * s[start] * s[start];
    for (int k = 1; k < count; ++k) {
        q += 2.0 * kC3B[start + k] * s[start + k] * s[start + k];
    }
    return q;
}

template <typename Scalar>
Scalar q222(const Scalar* s) {
    return kC4B[0] * s[3] * s[3] * s[3]
        + kC4B[1] * s[3] * (s[4] * s[4] + s[5] * s[5])
        + kC4B[2] * s[3] * (s[6] * s[6] + s[7] * s[7])
        + kC4B[3] * s[6] * (s[5] * s[5] - s[4] * s[4])
        + kC4B[4] * s[4] * s[5] * s[7];
}

template <typename Scalar>
Scalar q1111(const Scalar* s) {
    const Scalar s0 = s[0] * s[0];
    const Scalar s12 = s[1] * s[1] + s[2] * s[2];
    return kC5B[0] * s0 * s0 + kC5B[1] * s0 * s12 + kC5B[2] * s12 * s12;
}

template <typename Scalar>
Scalar q112(const Scalar* s) {
    return kC4B2[0] * s[0] * s[0] * s[3]
        + kC4B2[1] * s[0] * (s[1] * s[4] + s[2] * s[5])
        + kC4B2[2] * s[3] * (s[1] * s[1] + s[2] * s[2])
        + kC4B2[3] * s[6] * (s[1] * s[1] - s[2] * s[2])
        + kC4B2[4] * s[1] * s[2] * s[7];
}

template <typename Scalar>
Scalar q123(const Scalar* s) {
    Scalar value = 0.0;
    value += kC4B123[6] * (s[12] * s[2] * s[4] - s[11] * s[2] * s[5]
        + s[1] * s[11] * s[4] + s[1] * s[12] * s[5]);
    value += kC4B123[5] * (s[0] * s[11] * s[6] + s[0] * s[12] * s[7]);
    value += kC4B123[3] * (s[14] * s[2] * s[6] - s[13] * s[2] * s[7]
        + s[1] * s[13] * s[6] + s[1] * s[14] * s[7]);
    value += kC4B123[4] * (s[10] * s[0] * s[5] + s[0] * s[4] * s[9]);
    value += kC4B123[1] * (s[10] * s[2] * s[3] + s[0] * s[3] * s[8]
        + s[1] * s[3] * s[9]);
    value += kC4B123[0] * (s[10] * s[2] * s[6] - s[10] * s[1] * s[7]
        - s[2] * s[7] * s[9] - s[1] * s[6] * s[9]);
    value += kC4B123[2] * (-s[2] * s[5] * s[8] - s[1] * s[4] * s[8]);
    return value;
}

template <typename Scalar>
Scalar q233(const Scalar* s) {
    Scalar value = 0.0;
    value += kC4B233[0] * (s[3] * s[8] * s[8]);
    value += kC4B233[1] * (s[10] * s[10] * s[3] + s[3] * s[9] * s[9]);
    value += kC4B233[2] * (-s[10] * s[10] * s[6] + s[6] * s[9] * s[9]);
    value += kC4B233[3] * (s[4] * s[8] * s[9] + s[10] * s[5] * s[8]);
    value += kC4B233[4] * (-s[13] * s[13] * s[3] - s[14] * s[14] * s[3]);
    value += kC4B233[5] * (-s[14] * s[7] * s[9] - s[13] * s[6] * s[9]
        - s[10] * s[14] * s[6] + s[10] * s[13] * s[7]);
    value += kC4B233[6] * (s[10] * s[7] * s[9]);
    value += kC4B233[7] * (-s[11] * s[6] * s[8] - s[12] * s[7] * s[8]);
    value += kC4B233[8] * (s[11] * s[4] * s[9] + s[12] * s[5] * s[9]
        + s[10] * s[12] * s[4] - s[10] * s[11] * s[5]);
    value += kC4B233[9] * (s[12] * s[14] * s[4] + s[11] * s[14] * s[5]
        + s[13] * s[11] * s[4] - s[13] * s[12] * s[5]);
    return value;
}

template <typename Scalar>
Scalar q134(const Scalar* s) {
    return kC4B134[0] * (-s[10] * s[15] * s[2] - s[1] * s[15] * s[9])
        + kC4B134[1] * (s[0] * s[15] * s[8])
        + kC4B134[2] * (-s[1] * s[13] * s[18] - s[1] * s[14] * s[19]
            - s[2] * s[14] * s[18] + s[2] * s[13] * s[19])
        + kC4B134[3] * (-s[10] * s[18] * s[2] + s[1] * s[10] * s[19]
            + s[1] * s[18] * s[9] + s[2] * s[19] * s[9])
        + kC4B134[4] * (s[1] * s[16] * s[8] + s[2] * s[17] * s[8])
        + kC4B134[5] * (s[0] * s[10] * s[17] + s[0] * s[16] * s[9]
            - s[1] * s[11] * s[16] - s[1] * s[12] * s[17]
            - s[2] * s[12] * s[16] + s[2] * s[11] * s[17])
        + kC4B134[6] * (s[1] * s[13] * s[22] + s[1] * s[14] * s[23]
            - s[2] * s[14] * s[22] + s[2] * s[13] * s[23])
        + kC4B134[7] * (s[0] * s[11] * s[18] + s[0] * s[12] * s[19])
        + kC4B134[8] * (s[0] * s[13] * s[20] + s[0] * s[14] * s[21])
        + kC4B134[9] * (s[1] * s[11] * s[20] + s[1] * s[12] * s[21]
            - s[2] * s[12] * s[20] + s[2] * s[11] * s[21]);
}

struct Dual3 {
    double value = 0.0;
    std::array<double, 3> derivative{};

    Dual3(double value_in = 0.0) : value(value_in) {}
};

Dual3 operator+(const Dual3& a, const Dual3& b) {
    Dual3 result(a.value + b.value);
    for (int d = 0; d < 3; ++d) result.derivative[d] = a.derivative[d] + b.derivative[d];
    return result;
}

Dual3& operator+=(Dual3& a, const Dual3& b) {
    a.value += b.value;
    for (int d = 0; d < 3; ++d) a.derivative[d] += b.derivative[d];
    return a;
}

Dual3& operator*=(Dual3& a, const Dual3& b) {
    const double old_value = a.value;
    for (int d = 0; d < 3; ++d) {
        a.derivative[d] = a.derivative[d] * b.value + old_value * b.derivative[d];
    }
    a.value = old_value * b.value;
    return a;
}

Dual3 operator-(const Dual3& a, const Dual3& b) {
    Dual3 result(a.value - b.value);
    for (int d = 0; d < 3; ++d) result.derivative[d] = a.derivative[d] - b.derivative[d];
    return result;
}

Dual3 operator-(const Dual3& a) {
    Dual3 result(-a.value);
    for (int d = 0; d < 3; ++d) result.derivative[d] = -a.derivative[d];
    return result;
}

Dual3 operator*(const Dual3& a, const Dual3& b) {
    Dual3 result(a.value * b.value);
    for (int d = 0; d < 3; ++d) {
        result.derivative[d] = a.derivative[d] * b.value + a.value * b.derivative[d];
    }
    return result;
}

template <typename Scalar>
Scalar find_q_one_dynamic(int l, const Scalar* s) {
    switch (l) {
    case 1: return find_q_one<1>(s);
    case 2: return find_q_one<2>(s);
    case 3: return find_q_one<3>(s);
    case 4: return find_q_one<4>(s);
    case 5: return find_q_one<5>(s);
    case 6: return find_q_one<6>(s);
    case 7: return find_q_one<7>(s);
    default: return find_q_one<8>(s);
    }
}

int angular_q_channels(
    int l_max, bool has_q_222, bool has_q_1111, bool has_q_112,
    bool has_q_123, bool has_q_233, bool has_q_134,
    const Dual3* s, Dual3* q) {
    int count = 0;
    for (int l = 1; l <= l_max; ++l) q[count++] = find_q_one_dynamic(l, s);
    if (has_q_222) q[count++] = q222(s);
    if (has_q_1111) q[count++] = q1111(s);
    if (has_q_112) q[count++] = q112(s);
    if (has_q_123) q[count++] = q123(s);
    if (has_q_233) q[count++] = q233(s);
    if (has_q_134) q[count++] = q134(s);
    return count;
}

void fill_angular_q(
    int l_max,
    bool has_q_222,
    bool has_q_1111,
    bool has_q_112,
    bool has_q_123,
    bool has_q_233,
    bool has_q_134,
    int n_count,
    int n,
    const double* s,
    double* q) {
    int index = 0;
    if (l_max >= 1) q[index++ * n_count + n] = find_q_one<1>(s);
    if (l_max >= 2) q[index++ * n_count + n] = find_q_one<2>(s);
    if (l_max >= 3) q[index++ * n_count + n] = find_q_one<3>(s);
    if (l_max >= 4) q[index++ * n_count + n] = find_q_one<4>(s);
    if (l_max >= 5) q[index++ * n_count + n] = find_q_one<5>(s);
    if (l_max >= 6) q[index++ * n_count + n] = find_q_one<6>(s);
    if (l_max >= 7) q[index++ * n_count + n] = find_q_one<7>(s);
    if (l_max >= 8) q[index++ * n_count + n] = find_q_one<8>(s);
    if (has_q_222) q[index++ * n_count + n] = q222(s);
    if (has_q_1111) q[index++ * n_count + n] = q1111(s);
    if (has_q_112) q[index++ * n_count + n] = q112(s);
    if (has_q_123) q[index++ * n_count + n] = q123(s);
    if (has_q_233) q[index++ * n_count + n] = q233(s);
    if (has_q_134) q[index++ * n_count + n] = q134(s);
}

inline void fill_angular_q_l4_q222_q1111(
    int n_count, int n, const double* s, double* q) {
    q[n] = find_q_one<1>(s);
    q[n_count + n] = find_q_one<2>(s);
    q[2 * n_count + n] = find_q_one<3>(s);
    q[3 * n_count + n] = find_q_one<4>(s);
    q[4 * n_count + n] = q222(s);
    q[5 * n_count + n] = q1111(s);
}

template <int Count>
inline double dot_basis(const double* coefficients, const double* basis) {
    double value = 0.0;
    for (int k = 0; k < Count; ++k) value += coefficients[k] * basis[k];
    return value;
}

inline double dot_basis_dynamic(
    std::size_t count, const double* coefficients, const double* basis) {
    double value = 0.0;
    for (std::size_t k = 0; k < count; ++k) value += coefficients[k] * basis[k];
    return value;
}

std::vector<std::string> tokenize(const std::string& line) {
    std::string content = line;
    const std::size_t comment = content.find('#');
    if (comment != std::string::npos) content.resize(comment);
    std::istringstream stream(content);
    std::vector<std::string> tokens;
    std::string token;
    while (stream >> token) tokens.push_back(token);
    return tokens;
}

std::vector<std::vector<std::string>> read_model_lines(
    const std::string& path,
    const std::optional<std::string>& content) {
    std::ifstream file;
    std::istringstream memory;
    std::istream* input = nullptr;
    if (content.has_value()) {
        memory.str(*content);
        input = &memory;
    } else {
        file.open(path);
        if (!file) throw std::runtime_error("could not open NEP model: " + path);
        input = &file;
    }
    std::vector<std::vector<std::string>> lines;
    std::string line;
    while (std::getline(*input, line)) {
        auto tokens = tokenize(line);
        if (!tokens.empty()) lines.push_back(std::move(tokens));
    }
    return lines;
}

int parse_int(const std::string& value, const char* what) {
    try {
        std::size_t used = 0;
        const int result = std::stoi(value, &used);
        if (used != value.size()) throw std::invalid_argument("trailing characters");
        return result;
    } catch (...) {
        throw std::invalid_argument(std::string("invalid ") + what + ": " + value);
    }
}

double parse_double(const std::string& value, const char* what) {
    try {
        std::size_t used = 0;
        const double result = std::stod(value, &used);
        if (used != value.size() || !std::isfinite(result)) {
            throw std::invalid_argument("not finite");
        }
        return result;
    } catch (...) {
        throw std::invalid_argument(std::string("invalid ") + what + ": " + value);
    }
}

int atomic_number(const std::string& symbol) {
    static constexpr std::array<std::string_view, 94> elements = {
        "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si",
        "P", "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni",
        "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo",
        "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba",
        "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
        "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po",
        "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu",
    };
    for (std::size_t index = 0; index < elements.size(); ++index) {
        if (elements[index] == symbol) return static_cast<int>(index + 1);
    }
    throw std::invalid_argument("unknown element in NEP model: " + symbol);
}

std::size_t checked_product(std::size_t a, std::size_t b, const char* what) {
    if (b != 0 && a > std::numeric_limits<std::size_t>::max() / b) {
        throw std::overflow_error(std::string("NEP ") + what + " is too large");
    }
    return a * b;
}

} // namespace

struct NepModel {
    std::string path;
    int version = 0;
    int num_types = 0;
    int n_max_radial = 0;
    int n_max_angular = 0;
    int basis_size_radial = 0;
    int basis_size_angular = 0;
    int l_max = 0;
    int num_l = 0;
    bool has_q_222 = false;
    bool has_q_1111 = false;
    bool has_q_112 = false;
    bool has_q_123 = false;
    bool has_q_233 = false;
    bool has_q_134 = false;
    int dimension = 0;
    double radial_cutoff_max = 0.0;
    double angular_cutoff_max = 0.0;
    int hidden_neurons1 = 0;
    int hidden_neurons2 = 0;
    bool zbl_enabled = false;
    double zbl_inner = 0.0;
    double zbl_outer = 0.0;
    std::vector<double> ann_parameters;
    std::vector<std::int32_t> species;
    std::vector<double> radial_cutoff;
    std::vector<double> angular_cutoff;
    std::vector<double> radial_coefficients;
    std::vector<double> angular_coefficients;
    // Hot-loop layouts: coefficients and pair constants are cached in the
    // neighbor-type-major order used by the descriptor kernel.
    std::vector<double> radial_pair_coefficients;
    std::vector<double> angular_pair_coefficients;
    std::vector<double> radial_cutoff_pair;
    std::vector<double> angular_cutoff_pair;
    std::vector<double> scalers;
    std::unordered_map<std::int32_t, int> type_by_species;

    int type_of(std::int32_t z) const {
        const auto found = type_by_species.find(z);
        if (found == type_by_species.end()) {
            throw std::invalid_argument(
                "structure contains an element not present in the NEP model: " + std::to_string(z));
        }
        return found->second;
    }

    double pair_cutoff(const std::vector<double>& cutoffs, int first, int second) const {
        return 0.5 * (cutoffs[static_cast<std::size_t>(first)]
            + cutoffs[static_cast<std::size_t>(second)]);
    }
};

std::shared_ptr<NepModel> load_model(
    const std::string& path,
    const std::string& model_digest,
    const std::optional<std::string>& model_data
) {
    static std::mutex cache_mutex;
    static std::unordered_map<std::string, std::weak_ptr<NepModel>> cache;
    // A resolver-backed construction supplies the exact bytes that are
    // parsed.  Key those bytes directly so a bad/private caller cannot make
    // a different payload reuse a model under a stale digest.  The digest and
    // path branches preserve the legacy native API when no snapshot exists.
    const std::string cache_key = model_data.has_value()
        ? "content:" + *model_data
        : (model_digest.empty() ? "path:" + path : "digest:" + model_digest);
    std::lock_guard<std::mutex> cache_lock(cache_mutex);
    if (const auto found = cache.find(cache_key); found != cache.end()) {
        if (auto cached = found->second.lock()) return cached;
    }
    const auto lines = read_model_lines(path, model_data);
    if (lines.empty() || lines[0].size() < 3 || lines[0][0].rfind("nep", 0) != 0) {
        throw std::invalid_argument("invalid NEP model header");
    }
    auto model = std::make_shared<NepModel>();
    model->path = path;
    model->version = parse_int(lines[0][0].substr(3, 1), "NEP version");
    if (model->version < 3 || model->version > 5) {
        throw std::invalid_argument("unsupported NEP model version");
    }
    const std::string header = lines[0][0];
    if (header.find("spin") != std::string::npos || header.find("charge") != std::string::npos
        || header.find("dipole") != std::string::npos || header.find("polar") != std::string::npos) {
        throw std::invalid_argument("NepCalculator currently supports ordinary NEP descriptors only");
    }
    model->zbl_enabled = header.find("_zbl") != std::string::npos;
    model->num_types = parse_int(lines[0][1], "number of NEP atom types");
    if (model->num_types <= 0 || static_cast<std::size_t>(model->num_types + 2) != lines[0].size()) {
        throw std::invalid_argument("NEP model header has an invalid atom type list");
    }
    model->species.reserve(static_cast<std::size_t>(model->num_types));
    for (int type = 0; type < model->num_types; ++type) {
        const auto z = static_cast<std::int32_t>(atomic_number(lines[0][static_cast<std::size_t>(type + 2)]));
        model->type_by_species.emplace(z, type);
        model->species.push_back(z);
    }

    std::size_t line_index = 1;
    auto next_line = [&]() -> const std::vector<std::string>& {
        if (line_index >= lines.size()) throw std::invalid_argument("unexpected end of NEP model");
        return lines[line_index++];
    };
    auto line = next_line();
    if (!line.empty() && line[0] == "zbl") {
        if (!model->zbl_enabled || (line.size() != 3 && line.size() != 4)) {
            throw std::invalid_argument("invalid NEP ZBL cutoff line");
        }
        model->zbl_inner = parse_double(line[1], "NEP ZBL inner cutoff");
        model->zbl_outer = parse_double(line[2], "NEP ZBL outer cutoff");
        if (model->zbl_inner < 0.0 || model->zbl_outer <= model->zbl_inner) {
            throw std::invalid_argument("NEP ZBL requires 0 <= inner cutoff < outer cutoff");
        }
        if (line.size() == 4) {
            throw std::invalid_argument("typewise NEP ZBL cutoffs are not supported");
        }
        line = next_line();
    } else if (model->zbl_enabled) {
        throw std::invalid_argument("NEP ZBL model is missing its cutoff line");
    }
    while (!line.empty() && line[0].rfind("spin_", 0) == 0) {
        throw std::invalid_argument("spin NEP descriptors are not supported by NepCalculator");
    }
    if (line.empty() || line[0] != "cutoff") throw std::invalid_argument("NEP model is missing cutoff");
    model->radial_cutoff.resize(static_cast<std::size_t>(model->num_types));
    model->angular_cutoff.resize(static_cast<std::size_t>(model->num_types));
    if (line.size() == 5) {
        const double radial = parse_double(line[1], "radial cutoff");
        const double angular = parse_double(line[2], "angular cutoff");
        for (int type = 0; type < model->num_types; ++type) {
            model->radial_cutoff[static_cast<std::size_t>(type)] = radial;
            model->angular_cutoff[static_cast<std::size_t>(type)] = angular;
        }
    } else if (line.size() == static_cast<std::size_t>(2 * model->num_types + 3)) {
        for (int type = 0; type < model->num_types; ++type) {
            model->radial_cutoff[static_cast<std::size_t>(type)] = parse_double(
                line[static_cast<std::size_t>(1 + type * 2)], "radial cutoff");
            model->angular_cutoff[static_cast<std::size_t>(type)] = parse_double(
                line[static_cast<std::size_t>(2 + type * 2)], "angular cutoff");
        }
    } else {
        throw std::invalid_argument("NEP cutoff line has an invalid number of values");
    }
    for (int type = 0; type < model->num_types; ++type) {
        if (model->radial_cutoff[static_cast<std::size_t>(type)] <= 0.0
            || model->angular_cutoff[static_cast<std::size_t>(type)] <= 0.0) {
            throw std::invalid_argument("NEP cutoffs must be positive");
        }
        model->radial_cutoff_max = std::max(model->radial_cutoff_max,
            model->radial_cutoff[static_cast<std::size_t>(type)]);
        model->angular_cutoff_max = std::max(model->angular_cutoff_max,
            model->angular_cutoff[static_cast<std::size_t>(type)]);
    }

    line = next_line();
    if (line.size() != 3 || line[0] != "n_max") throw std::invalid_argument("invalid NEP n_max line");
    model->n_max_radial = parse_int(line[1], "radial n_max");
    model->n_max_angular = parse_int(line[2], "angular n_max");
    line = next_line();
    if (line.size() != 3 || line[0] != "basis_size") {
        throw std::invalid_argument("invalid NEP basis_size line");
    }
    model->basis_size_radial = parse_int(line[1], "radial basis size");
    model->basis_size_angular = parse_int(line[2], "angular basis size");
    line = next_line();
    if (line.size() < 4 || line[0] != "l_max") throw std::invalid_argument("invalid NEP l_max line");
    model->l_max = parse_int(line[1], "l_max");
    if (model->l_max < 0 || model->l_max > 8) throw std::invalid_argument("NEP l_max must be in [0, 8]");
    auto flag = [&](std::size_t index) {
        return index < line.size() && parse_int(line[index], "NEP angular flag") != 0;
    };
    model->has_q_222 = flag(2);
    model->has_q_1111 = flag(3);
    model->has_q_112 = flag(4);
    model->has_q_123 = flag(5);
    model->has_q_233 = flag(6);
    model->has_q_134 = flag(7);
    model->num_l = model->l_max + static_cast<int>(model->has_q_222)
        + static_cast<int>(model->has_q_1111) + static_cast<int>(model->has_q_112)
        + static_cast<int>(model->has_q_123) + static_cast<int>(model->has_q_233)
        + static_cast<int>(model->has_q_134);
    model->dimension = (model->n_max_radial + 1)
        + (model->n_max_angular + 1) * model->num_l;
    if (model->n_max_radial < 0 || model->n_max_angular < 0
        || model->basis_size_radial < 0 || model->basis_size_angular < 0
        || model->dimension <= 0 || model->dimension > 256) {
        throw std::invalid_argument("invalid NEP descriptor dimensions");
    }

    line = next_line();
    if (line.size() != 3 || line[0] != "ANN") throw std::invalid_argument("invalid NEP ANN line");
    const int neurons1 = parse_int(line[1], "NEP first hidden layer size");
    const int neurons2 = parse_int(line[2], "NEP second hidden layer size");
    if (neurons1 <= 0 || neurons2 < 0) throw std::invalid_argument("invalid NEP ANN sizes");
    model->hidden_neurons1 = neurons1;
    model->hidden_neurons2 = neurons2;
    std::size_t ann_parameters = 0;
    if (neurons2 > 0) {
        if (model->version != 4) throw std::invalid_argument("unsupported two-layer NEP model");
        ann_parameters = (static_cast<std::size_t>(model->dimension + 1) * neurons1
            + static_cast<std::size_t>(neurons1 + 2) * neurons2)
            * static_cast<std::size_t>(model->num_types) + 1;
    } else if (model->version == 3) {
        ann_parameters = static_cast<std::size_t>(model->dimension + 2) * neurons1 + 1;
    } else if (model->version == 4) {
        ann_parameters = static_cast<std::size_t>(model->dimension + 2) * neurons1
            * static_cast<std::size_t>(model->num_types) + 1;
    } else {
        ann_parameters = (static_cast<std::size_t>(model->dimension + 2) * neurons1 + 1)
            * static_cast<std::size_t>(model->num_types) + 1;
    }

    std::vector<double> numeric;
    for (; line_index < lines.size(); ++line_index) {
        for (const auto& value : lines[line_index]) numeric.push_back(parse_double(value, "NEP parameter"));
    }
    const std::size_t radial_count = checked_product(
        checked_product(static_cast<std::size_t>(model->n_max_radial + 1),
            static_cast<std::size_t>(model->basis_size_radial + 1), "radial basis"),
        static_cast<std::size_t>(model->num_types * model->num_types), "radial coefficients");
    const std::size_t angular_count = checked_product(
        checked_product(static_cast<std::size_t>(model->n_max_angular + 1),
            static_cast<std::size_t>(model->basis_size_angular + 1), "angular basis"),
        static_cast<std::size_t>(model->num_types * model->num_types), "angular coefficients");
    const std::size_t required = ann_parameters + radial_count + angular_count
        + static_cast<std::size_t>(model->dimension);
    if (numeric.size() < required) throw std::invalid_argument("NEP model parameter block is truncated");
    model->ann_parameters.assign(
        numeric.begin(), numeric.begin() + static_cast<std::ptrdiff_t>(ann_parameters));
    std::size_t cursor = ann_parameters;
    model->radial_coefficients.assign(numeric.begin() + static_cast<std::ptrdiff_t>(cursor),
        numeric.begin() + static_cast<std::ptrdiff_t>(cursor + radial_count));
    cursor += radial_count;
    model->angular_coefficients.assign(numeric.begin() + static_cast<std::ptrdiff_t>(cursor),
        numeric.begin() + static_cast<std::ptrdiff_t>(cursor + angular_count));
    cursor += angular_count;
    model->scalers.assign(numeric.begin() + static_cast<std::ptrdiff_t>(cursor),
        numeric.begin() + static_cast<std::ptrdiff_t>(cursor + static_cast<std::size_t>(model->dimension)));

    const std::size_t type_pairs = static_cast<std::size_t>(model->num_types * model->num_types);
    const std::size_t radial_n = static_cast<std::size_t>(model->n_max_radial + 1);
    const std::size_t angular_n = static_cast<std::size_t>(model->n_max_angular + 1);
    const std::size_t radial_basis = static_cast<std::size_t>(model->basis_size_radial + 1);
    const std::size_t angular_basis = static_cast<std::size_t>(model->basis_size_angular + 1);
    model->radial_cutoff_pair.resize(type_pairs);
    model->angular_cutoff_pair.resize(type_pairs);
    model->radial_pair_coefficients.resize(type_pairs * radial_n * radial_basis);
    model->angular_pair_coefficients.resize(type_pairs * angular_n * angular_basis);
    for (int first = 0; first < model->num_types; ++first) {
        for (int second = 0; second < model->num_types; ++second) {
            const std::size_t pair = static_cast<std::size_t>(first * model->num_types + second);
            model->radial_cutoff_pair[pair] = model->pair_cutoff(
                model->radial_cutoff, first, second);
            model->angular_cutoff_pair[pair] = model->pair_cutoff(
                model->angular_cutoff, first, second);
            for (std::size_t n = 0; n < radial_n; ++n) {
                for (std::size_t k = 0; k < radial_basis; ++k) {
                    model->radial_pair_coefficients[
                        pair * radial_n * radial_basis + n * radial_basis + k] =
                        model->radial_coefficients[(n * radial_basis + k) * type_pairs + pair];
                }
            }
            for (std::size_t n = 0; n < angular_n; ++n) {
                for (std::size_t k = 0; k < angular_basis; ++k) {
                    model->angular_pair_coefficients[
                        pair * angular_n * angular_basis + n * angular_basis + k] =
                        model->angular_coefficients[(n * angular_basis + k) * type_pairs + pair];
                }
            }
        }
    }
    // Expired entries hold only the cache key; sweep them on insert so a
    // long-lived process loading many distinct models does not grow the map.
    for (auto entry = cache.begin(); entry != cache.end();) {
        if (entry->second.expired()) {
            entry = cache.erase(entry);
        } else {
            ++entry;
        }
    }
    cache[cache_key] = model;
    return model;
}

void basis_values(int basis_size, double cutoff, double distance, double* values) {
    const double fc = 0.5 * std::cos(kPi * distance / cutoff) + 0.5;
    const double a = distance / cutoff - 1.0;
    const double x = 2.0 * a * a - 1.0;
    values[0] = fc;
    if (basis_size == 0) return;
    values[1] = 0.5 * (x + 1.0) * fc;
    double previous = 1.0;
    double current = x;
    for (int order = 2; order <= basis_size; ++order) {
        const double next = 2.0 * x * current - previous;
        previous = current;
        current = next;
        values[order] = 0.5 * (current + 1.0) * fc;
    }
}

void basis_values_and_derivative(
    int basis_size, double cutoff, double distance, double* values, double* derivatives) {
    const double inverse_cutoff = 1.0 / cutoff;
    const double angle = kPi * distance * inverse_cutoff;
    const double cutoff_value = 0.5 * std::cos(angle) + 0.5;
    const double cutoff_derivative = -0.5 * std::sin(angle) * kPi * inverse_cutoff;
    const double a = distance * inverse_cutoff - 1.0;
    const double x = 2.0 * a * a - 1.0;
    const double x_derivative = 4.0 * a * inverse_cutoff;
    values[0] = cutoff_value;
    derivatives[0] = cutoff_derivative;
    if (basis_size == 0) return;

    values[1] = 0.5 * (x + 1.0) * cutoff_value;
    derivatives[1] = 0.5 * (x_derivative * cutoff_value
        + (x + 1.0) * cutoff_derivative);
    double previous = 1.0;
    double previous_derivative = 0.0;
    double current = x;
    double current_derivative = x_derivative;
    for (int order = 2; order <= basis_size; ++order) {
        const double next = 2.0 * x * current - previous;
        const double next_derivative = 2.0 * x_derivative * current
            + 2.0 * x * current_derivative - previous_derivative;
        previous = current;
        previous_derivative = current_derivative;
        current = next;
        current_derivative = next_derivative;
        values[order] = 0.5 * (current + 1.0) * cutoff_value;
        derivatives[order] = 0.5 * (current_derivative * cutoff_value
            + (current + 1.0) * cutoff_derivative);
    }
}

void compute_nep(
    const StructureBatchView& batch,
    const NepModel& model,
    int num_threads,
    double* output,
    const std::shared_ptr<ComputeControl>& control,
    std::vector<NeighborGraph>* retained_graphs = nullptr,
    double cutoff_override = 0.0) {
    if (control) {
        control->reset(batch.structures);
    }
    const std::size_t radial_basis = static_cast<std::size_t>(model.basis_size_radial + 1);
    const std::size_t angular_basis = static_cast<std::size_t>(model.basis_size_angular + 1);
    const std::size_t radial_n = static_cast<std::size_t>(model.n_max_radial + 1);
    const std::size_t angular_n = static_cast<std::size_t>(model.n_max_angular + 1);

    // Resolve atomic types once. The old hot loop performed an unordered_map
    // lookup for every center-neighbor edge.
    std::vector<int> atom_types(static_cast<std::size_t>(batch.atoms));
    for (std::int64_t atom = 0; atom < batch.atoms; ++atom) {
        atom_types[static_cast<std::size_t>(atom)] = model.type_of(batch.numbers[atom]);
    }

    auto compute_center = [&](const NeighborGraph& graph,
                              const int* types,
                              std::int64_t center,
                              std::int64_t output_center,
                              std::vector<double>& radial,
                              std::vector<double>& sums,
                              std::vector<double>& angular,
                              std::vector<double>& basis) {
        std::fill(radial.begin(), radial.end(), 0.0);
        std::fill(sums.begin(), sums.end(), 0.0);
        std::fill(angular.begin(), angular.end(), 0.0);
        const int center_type = types[static_cast<std::size_t>(center)];
        const NeighborView neighbors = graph.for_center(center);
        for (std::size_t index = 0; index < neighbors.size; ++index) {
            const int neighbor_type = types[
                static_cast<std::size_t>(neighbors.atoms[index])];
            const std::size_t pair = static_cast<std::size_t>(
                center_type * model.num_types + neighbor_type);
            const double distance2 = neighbors.distance2[index];
            if (distance2 <= 0.0) continue;
            const double distance = std::sqrt(distance2);

            const double radial_cutoff = model.radial_cutoff_pair[pair];
            if (distance < radial_cutoff) {
                if (model.basis_size_radial == 8) {
                    basis_values_fixed<8>(radial_cutoff, distance, basis.data());
                } else {
                    basis_values(model.basis_size_radial, radial_cutoff, distance, basis.data());
                }
                const double* coefficients = model.radial_pair_coefficients.data()
                    + pair * radial_n * radial_basis;
                for (std::size_t n = 0; n < radial_n; ++n) {
                    const double* c_n = coefficients + n * radial_basis;
                    const double value = radial_basis == 9
                        ? dot_basis<9>(c_n, basis.data())
                        : dot_basis_dynamic(radial_basis, c_n, basis.data());
                    radial[n] += value;
                }
            }

            const double angular_cutoff = model.angular_cutoff_pair[pair];
            if (model.l_max > 0 && distance < angular_cutoff) {
                if (model.basis_size_angular == 8) {
                    basis_values_fixed<8>(angular_cutoff, distance, basis.data());
                } else {
                    basis_values(model.basis_size_angular, angular_cutoff, distance, basis.data());
                }
                const double* coefficients = model.angular_pair_coefficients.data()
                    + pair * angular_n * angular_basis;
                const double x = neighbors.displacements[index * 3];
                const double y = neighbors.displacements[index * 3 + 1];
                const double z = neighbors.displacements[index * 3 + 2];
                for (std::size_t n = 0; n < angular_n; ++n) {
                    const double* c_n = coefficients + n * angular_basis;
                    const double value = angular_basis == 9
                        ? dot_basis<9>(c_n, basis.data())
                        : dot_basis_dynamic(angular_basis, c_n, basis.data());
                    if (model.l_max == 4) {
                        accumulate_s_l4(
                            distance, x, y, z, value,
                            sums.data() + n * static_cast<std::size_t>(kNumAngularTerms));
                    } else {
                        accumulate_s(
                            model.l_max, distance, x, y, z, value,
                            sums.data() + n * static_cast<std::size_t>(kNumAngularTerms));
                    }
                }
            }
        }

        for (int n = 0; n <= model.n_max_angular; ++n) {
            const double* s = sums.data() + static_cast<std::size_t>(n) * kNumAngularTerms;
            if (model.l_max == 4 && model.num_l == 6 && model.has_q_222
                && model.has_q_1111 && !model.has_q_112 && !model.has_q_123
                && !model.has_q_233 && !model.has_q_134) {
                fill_angular_q_l4_q222_q1111(
                    model.n_max_angular + 1, n, s, angular.data());
            } else {
                fill_angular_q(
                    model.l_max, model.has_q_222, model.has_q_1111, model.has_q_112,
                    model.has_q_123, model.has_q_233, model.has_q_134, model.n_max_angular + 1,
                    n, s, angular.data());
            }
        }
        double* row = output + output_center * static_cast<std::int64_t>(model.dimension);
        int dimension = 0;
        for (std::size_t n = 0; n < radial_n; ++n) {
            row[dimension] = radial[n] * model.scalers[static_cast<std::size_t>(dimension)];
            ++dimension;
        }
        for (int l = 0; l < model.num_l; ++l) {
            for (std::size_t n = 0; n < angular_n; ++n) {
                const std::size_t source = static_cast<std::size_t>(l) * angular_n + n;
                row[dimension] = angular[source] * model.scalers[static_cast<std::size_t>(dimension)];
                ++dimension;
            }
        }
    };

    if (retained_graphs != nullptr) {
        retained_graphs->resize(static_cast<std::size_t>(batch.structures));
    }
    const std::size_t workspace_basis = std::max(radial_basis, angular_basis);
    const double neighbor_cutoff = std::max({
        model.radial_cutoff_max, model.angular_cutoff_max, cutoff_override});

    // Batch NEP implementations benefit from keeping the neighbor graph next
    // to the descriptor work that consumes it.  The old batch path first built
    // one global graph, copied every per-structure graph into it, and only then
    // launched the descriptor loop.  NEP_CPU/NEPAdapters instead gives each
    // worker a scratch area and completes one structure at a time.  Keep that
    // ownership model here as well; it avoids the global graph merge and keeps
    // the graph-builder's inner loop serial, so OpenMP is not nested.
    auto compute_structure = [&](std::int64_t structure,
                                 std::vector<double>& radial,
                                 std::vector<double>& sums,
                                 std::vector<double>& angular,
                                 std::vector<double>& basis) {
        const std::int64_t begin = batch.offsets[structure];
        const std::int64_t end = batch.offsets[structure + 1];
        const std::int64_t atom_count = end - begin;
        if (atom_count == 0) return;

        std::array<std::int64_t, 2> local_offsets{0, atom_count};
        const StructureBatchView local_batch{
            batch.numbers + begin,
            batch.positions + begin * 3,
            batch.cells + structure * 9,
            batch.pbc + structure * 3,
            local_offsets.data(),
            1,
            atom_count,
        };
        // This path is already inside the worker-level parallel region.  A
        // single-thread graph build prevents nested OpenMP teams while still
        // allowing every worker to build its own structure graph concurrently.
        NeighborGraph graph = build_neighbor_graph(
            local_batch, neighbor_cutoff, control, 1, true, false, false);
        const int* local_types = atom_types.data() + begin;
        for (std::int64_t local = 0; local < atom_count; ++local) {
            compute_center(
                graph, local_types, local, begin + local,
                radial, sums, angular, basis);
        }
        if (retained_graphs != nullptr) {
            (*retained_graphs)[static_cast<std::size_t>(structure)] = std::move(graph);
        }
    };

    // For the common single-thread path, consume each structure immediately.
    // This avoids the batch neighbor builder's local-graph concatenation and
    // keeps the graph and descriptor data in cache for small structures.
    if (num_threads == 1) {
        std::vector<double> radial(radial_n);
        std::vector<double> sums(angular_n * static_cast<std::size_t>(kNumAngularTerms));
        std::vector<double> angular(angular_n * static_cast<std::size_t>(model.num_l));
        std::vector<double> basis(workspace_basis);
        for (std::int64_t structure = 0; structure < batch.structures; ++structure) {
            if (control && control->cancelled()) break;
            compute_structure(structure, radial, sums, angular, basis);
            if (control && !control->cancelled()) {
                control->mark_completed();
            }
        }
        if (control && control->cancelled()) throw CancelledError();
        return;
    }

    if (batch.structures > 1) {
        // Each structure is one independent unit of work in this branch. The
        // shared runner captures worker exceptions and caps the team size so
        // a small batch never spawns idle OpenMP workers.
        detail::run_parallel_matrix_structures(
            batch.structures,
            detail::effective_thread_count(batch.structures, num_threads),
            control,
            [&](std::int64_t structure) {
                std::vector<double> radial(radial_n);
                std::vector<double> sums(angular_n * static_cast<std::size_t>(kNumAngularTerms));
                std::vector<double> angular(angular_n * static_cast<std::size_t>(model.num_l));
                std::vector<double> basis(workspace_basis);
                compute_structure(structure, radial, sums, angular, basis);
                if (control && !control->cancelled()) {
                    control->mark_completed();
                }
            });
        return;
    }

    NeighborGraph graph = build_neighbor_graph(
        batch, neighbor_cutoff, control, num_threads, true, false, false);
#ifdef _OPENMP
    const int workers = num_threads > 0 ? num_threads : omp_get_max_threads();
#pragma omp parallel num_threads(workers)
    {
        std::vector<double> radial(radial_n);
        std::vector<double> sums(angular_n * static_cast<std::size_t>(kNumAngularTerms));
        std::vector<double> angular(angular_n * static_cast<std::size_t>(model.num_l));
        std::vector<double> basis(workspace_basis);
#pragma omp for schedule(static)
        for (std::int64_t center = 0; center < batch.atoms; ++center) {
            if (control && control->cancelled()) continue;
            compute_center(graph, atom_types.data(), center, center, radial, sums, angular, basis);
        }
    }
#else
    std::vector<double> radial(radial_n);
    std::vector<double> sums(angular_n * static_cast<std::size_t>(kNumAngularTerms));
    std::vector<double> angular(angular_n * static_cast<std::size_t>(model.num_l));
    std::vector<double> basis(workspace_basis);
    for (std::int64_t center = 0; center < batch.atoms; ++center) {
        if (control && control->cancelled()) break;
        compute_center(graph, atom_types.data(), center, center, radial, sums, angular, basis);
    }
#endif
    if (retained_graphs != nullptr) {
        (*retained_graphs)[0] = std::move(graph);
    }
    if (control && control->cancelled()) throw CancelledError();
    if (control) {
        control->mark_completed();
    }
}

namespace {

double evaluate_ann(const NepModel& model, int type, const double* q, double* derivative) {
    const int dim = model.dimension;
    const int h1 = model.hidden_neurons1;
    const int h2 = model.hidden_neurons2;
    const std::size_t stride = h2 > 0
        ? static_cast<std::size_t>(dim + 1) * h1 + static_cast<std::size_t>(h1 + 2) * h2
        : static_cast<std::size_t>(dim + 2) * h1 + (model.version == 5 ? 1 : 0);
    const std::size_t base = static_cast<std::size_t>(type) * stride;
    const double* w0 = model.ann_parameters.data() + base;
    const double* b0 = w0 + static_cast<std::size_t>(h1) * dim;
    const double* w1 = b0 + h1;
    std::fill_n(derivative, dim, 0.0);
    double energy = -model.ann_parameters[static_cast<std::size_t>(model.num_types) * stride];
    if (h2 == 0) {
        if (model.version == 5) energy -= w1[h1];
        for (int n = 0; n < h1; ++n) {
            const double* row = w0 + static_cast<std::size_t>(n) * dim;
            double dot = 0.0;
            for (int d = 0; d < dim; ++d) dot += row[d] * q[d];
            const double hidden = std::tanh(dot - b0[n]);
            energy += w1[n] * hidden;
            const double pull = w1[n] * (1.0 - hidden * hidden);
            for (int d = 0; d < dim; ++d) derivative[d] += pull * row[d];
        }
        return energy;
    }

    const double* w1_hidden = w1;
    const double* b1_hidden = w1_hidden + static_cast<std::size_t>(h1) * h2;
    const double* w2 = b1_hidden + h2;
    std::array<double, 120> hidden1{};
    std::array<double, 120> pull1{};
    for (int n = 0; n < h1; ++n) {
        const double* row = w0 + static_cast<std::size_t>(n) * dim;
        double dot = 0.0;
        for (int d = 0; d < dim; ++d) dot += row[d] * q[d];
        hidden1[n] = std::tanh(dot - b0[n]);
    }
    for (int n = 0; n < h2; ++n) {
        const double* row = w1_hidden + static_cast<std::size_t>(n) * h1;
        double dot = 0.0;
        for (int k = 0; k < h1; ++k) dot += row[k] * hidden1[k];
        const double hidden = std::tanh(dot - b1_hidden[n]);
        energy += w2[n] * hidden;
        const double pull = w2[n] * (1.0 - hidden * hidden);
        for (int k = 0; k < h1; ++k) pull1[k] += row[k] * pull;
    }
    for (int n = 0; n < h1; ++n) {
        const double pull = pull1[n] * (1.0 - hidden1[n] * hidden1[n]);
        const double* row = w0 + static_cast<std::size_t>(n) * dim;
        for (int d = 0; d < dim; ++d) derivative[d] += pull * row[d];
    }
    return energy;
}

void accumulate_s_dual(
    int l_max, const Dual3& distance, const Dual3& x, const Dual3& y,
    const Dual3& z, const Dual3& value, Dual3* s) {
    Dual3 inverse(1.0 / distance.value);
    for (int d = 0; d < 3; ++d) {
        inverse.derivative[d] = -distance.derivative[d] / (distance.value * distance.value);
    }
    const Dual3 nx = x * inverse;
    const Dual3 ny = y * inverse;
    const Dual3 nz = z * inverse;
    if (l_max >= 1) accumulate_s_one<1>(nx, ny, nz, value, s);
    if (l_max >= 2) accumulate_s_one<2>(nx, ny, nz, value, s);
    if (l_max >= 3) accumulate_s_one<3>(nx, ny, nz, value, s);
    if (l_max >= 4) accumulate_s_one<4>(nx, ny, nz, value, s);
    if (l_max >= 5) accumulate_s_one<5>(nx, ny, nz, value, s);
    if (l_max >= 6) accumulate_s_one<6>(nx, ny, nz, value, s);
    if (l_max >= 7) accumulate_s_one<7>(nx, ny, nz, value, s);
    if (l_max >= 8) accumulate_s_one<8>(nx, ny, nz, value, s);
}

double zbl_value_and_derivative(
    const NepModel& model, std::int32_t z1, std::int32_t z2,
    double distance, double& derivative) {
    if (!model.zbl_enabled || distance <= 0.0 || distance >= model.zbl_outer) {
        derivative = 0.0;
        return 0.0;
    }
    constexpr double parameters[8] = {
        0.18175, 3.1998, 0.50986, 0.94229,
        0.28022, 0.4029, 0.02817, 0.20162,
    };
    const double a_inv = (std::pow(static_cast<double>(z1), 0.23)
        + std::pow(static_cast<double>(z2), 0.23)) * 2.134563;
    const double charge = 14.399645 * static_cast<double>(z1) * z2;
    double phi = 0.0;
    double phi_prime = 0.0;
    for (int i = 0; i < 4; ++i) {
        const double term = parameters[2 * i] * std::exp(-parameters[2 * i + 1] * distance * a_inv);
        phi += term;
        phi_prime -= parameters[2 * i + 1] * term;
    }
    const double unscreened = charge * phi;
    const double value = unscreened / distance;
    const double value_prime = charge * phi_prime * a_inv / distance
        - unscreened / (distance * distance);
    double cutoff = 0.0;
    double cutoff_prime = 0.0;
    if (distance < model.zbl_inner) {
        cutoff = 1.0;
    } else {
        const double factor = kPi / (model.zbl_outer - model.zbl_inner);
        cutoff = 0.5 * std::cos(factor * (distance - model.zbl_inner)) + 0.5;
        cutoff_prime = -0.5 * std::sin(factor * (distance - model.zbl_inner)) * factor;
    }
    derivative = value_prime * cutoff + value * cutoff_prime;
    return value * cutoff;
}

void compute_nep_prediction(
    const StructureBatchView& batch, const NepModel& model, int num_threads,
    double* energies, double* atom_energies, double* forces,
    const std::shared_ptr<ComputeControl>& control) {
    detail::validate_batch(batch);
    if ((batch.structures > 0 && energies == nullptr)
        || (batch.atoms > 0 && (atom_energies == nullptr || forces == nullptr))) {
        throw std::invalid_argument("NEP prediction output buffers must not be null");
    }
    if (control) control->reset(batch.structures);
    if (batch.structures > 0) std::fill_n(energies, batch.structures, 0.0);
    if (batch.atoms > 0) {
        std::fill_n(atom_energies, batch.atoms, 0.0);
        std::fill_n(forces, batch.atoms * 3, 0.0);
    }
    if (batch.atoms == 0) {
        detail::mark_completed_structures(control, batch.structures);
        return;
    }

    const std::size_t dimension = static_cast<std::size_t>(model.dimension);
    std::vector<double> descriptors(static_cast<std::size_t>(batch.atoms) * dimension);
    std::vector<NeighborGraph> graphs;
    compute_nep(batch, model, num_threads, descriptors.data(), control,
        &graphs, model.zbl_outer);
    if (control) control->reset(batch.structures);
    std::vector<int> types(static_cast<std::size_t>(batch.atoms));
    std::vector<double> ann_gradient(descriptors.size());
    for (std::int64_t atom = 0; atom < batch.atoms; ++atom) {
        types[static_cast<std::size_t>(atom)] = model.type_of(batch.numbers[atom]);
        atom_energies[atom] = evaluate_ann(model, types[static_cast<std::size_t>(atom)],
            descriptors.data() + atom * model.dimension,
            ann_gradient.data() + atom * model.dimension);
    }
    const std::size_t radial_n = static_cast<std::size_t>(model.n_max_radial + 1);
    const std::size_t angular_n = static_cast<std::size_t>(model.n_max_angular + 1);
    const std::size_t radial_basis = static_cast<std::size_t>(model.basis_size_radial + 1);
    const std::size_t angular_basis = static_cast<std::size_t>(model.basis_size_angular + 1);
    std::vector<double> sums(angular_n * kNumAngularTerms);
    std::vector<double> basis(std::max(radial_basis, angular_basis));
    std::vector<double> basis_derivative(basis.size());
    std::vector<double> angular_values;
    std::vector<double> angular_derivatives;

    for (std::int64_t structure = 0; structure < batch.structures; ++structure) {
        detail::check_cancelled(control);
        const std::int64_t structure_begin = batch.offsets[structure];
        const NeighborGraph& graph = graphs[static_cast<std::size_t>(structure)];
        for (std::int64_t center = structure_begin;
             center < batch.offsets[structure + 1]; ++center) {
            const std::int64_t local_center = center - structure_begin;
            const int center_type = types[static_cast<std::size_t>(center)];
            const NeighborView neighbors = graph.for_center(local_center);
            std::fill(sums.begin(), sums.end(), 0.0);
            if (model.l_max > 0) {
                angular_values.resize(neighbors.size * angular_n);
                angular_derivatives.resize(neighbors.size * angular_n);
            }
            // Build the aggregate angular moments once for this center. The
            // per-edge pass below differentiates each contribution against
            // these totals, so work stays linear in the neighbor count.
            for (std::size_t edge = 0; model.l_max > 0 && edge < neighbors.size; ++edge) {
                const double distance2 = neighbors.distance2[edge];
                if (distance2 <= 0.0) continue;
                const double distance = std::sqrt(distance2);
                const std::int64_t neighbor = structure_begin + neighbors.atoms[edge];
                const int neighbor_type = types[static_cast<std::size_t>(neighbor)];
                const std::size_t pair = static_cast<std::size_t>(
                    center_type * model.num_types + neighbor_type);
                const double rc = model.angular_cutoff_pair[pair];
                if (distance >= rc) continue;
                basis_values_and_derivative(
                    model.basis_size_angular, rc, distance, basis.data(), basis_derivative.data());
                const double* coefficients = model.angular_pair_coefficients.data()
                    + pair * angular_n * angular_basis;
                for (std::size_t n = 0; n < angular_n; ++n) {
                    const double* c_n = coefficients + n * angular_basis;
                    const double value = dot_basis_dynamic(angular_basis, c_n, basis.data());
                    const double value_derivative = dot_basis_dynamic(
                        angular_basis, c_n, basis_derivative.data());
                    const std::size_t cache_index = edge * angular_n + n;
                    angular_values[cache_index] = value;
                    angular_derivatives[cache_index] = value_derivative;
                    accumulate_s(model.l_max, distance,
                        neighbors.displacements[edge * 3],
                        neighbors.displacements[edge * 3 + 1],
                        neighbors.displacements[edge * 3 + 2], value,
                        sums.data() + n * kNumAngularTerms);
                }
            }

            const double* center_gradient = ann_gradient.data() + center * model.dimension;
            for (std::size_t edge = 0; edge < neighbors.size; ++edge) {
                const std::int64_t neighbor = structure_begin + neighbors.atoms[edge];
                const int neighbor_type = types[static_cast<std::size_t>(neighbor)];
                const std::size_t pair = static_cast<std::size_t>(
                    center_type * model.num_types + neighbor_type);
                const double distance2 = neighbors.distance2[edge];
                if (distance2 <= 0.0) continue;
                const double distance = std::sqrt(distance2);
                std::array<double, 3> gradient{};

                const double radial_rc = model.radial_cutoff_pair[pair];
                if (distance < radial_rc) {
                    basis_values_and_derivative(model.basis_size_radial, radial_rc,
                        distance, basis.data(), basis_derivative.data());
                    const double* coefficients = model.radial_pair_coefficients.data()
                        + pair * radial_n * radial_basis;
                    for (std::size_t n = 0; n < radial_n; ++n) {
                        const double radial_derivative = dot_basis_dynamic(radial_basis,
                            coefficients + n * radial_basis, basis_derivative.data());
                        const double pull = center_gradient[n] * model.scalers[n];
                        for (int axis = 0; axis < 3; ++axis) {
                            gradient[axis] += pull * radial_derivative
                                * neighbors.displacements[edge * 3 + axis] / distance;
                        }
                    }
                }

                const double angular_rc = model.angular_cutoff_pair[pair];
                if (model.l_max > 0 && distance < angular_rc) {
                    std::array<Dual3, 3> xyz{
                        Dual3(neighbors.displacements[edge * 3]),
                        Dual3(neighbors.displacements[edge * 3 + 1]),
                        Dual3(neighbors.displacements[edge * 3 + 2]),
                    };
                    Dual3 r(distance);
                    for (int axis = 0; axis < 3; ++axis) {
                        xyz[axis].derivative[axis] = 1.0;
                        r.derivative[axis] = xyz[axis].value / distance;
                    }
                    for (std::size_t n = 0; n < angular_n; ++n) {
                        const std::size_t cache_index = edge * angular_n + n;
                        Dual3 value(angular_values[cache_index]);
                        const double value_derivative = angular_derivatives[cache_index];
                        for (int axis = 0; axis < 3; ++axis) {
                            value.derivative[axis] = value_derivative
                                * neighbors.displacements[edge * 3 + axis] / distance;
                        }
                        std::array<Dual3, kNumAngularTerms> edge_sums{};
                        accumulate_s_dual(model.l_max, r, xyz[0], xyz[1], xyz[2],
                            value, edge_sums.data());
                        std::array<Dual3, kNumAngularTerms> total_sums{};
                        const double* sum_values = sums.data() + n * kNumAngularTerms;
                        for (int k = 0; k < kNumAngularTerms; ++k) {
                            total_sums[k].value = sum_values[k];
                            total_sums[k].derivative = edge_sums[k].derivative;
                        }
                        std::array<Dual3, 14> q_derivatives{};
                        const int channels = angular_q_channels(model.l_max,
                            model.has_q_222, model.has_q_1111, model.has_q_112,
                            model.has_q_123, model.has_q_233, model.has_q_134,
                            total_sums.data(), q_derivatives.data());
                        if (channels != model.num_l) {
                            throw std::logic_error("NEP angular channel count mismatch");
                        }
                        for (int channel = 0; channel < channels; ++channel) {
                            const std::size_t feature = radial_n
                                + static_cast<std::size_t>(channel) * angular_n + n;
                            const double pull = center_gradient[feature] * model.scalers[feature];
                            for (int axis = 0; axis < 3; ++axis) {
                                gradient[axis] += pull * q_derivatives[channel].derivative[axis];
                            }
                        }
                    }
                }

                for (int axis = 0; axis < 3; ++axis) {
                    forces[center * 3 + axis] += gradient[axis];
                    forces[neighbor * 3 + axis] -= gradient[axis];
                }

                if (model.zbl_enabled && distance < model.zbl_outer) {
                    double radial_derivative = 0.0;
                    const double value = zbl_value_and_derivative(
                        model, batch.numbers[center], batch.numbers[neighbor],
                        distance, radial_derivative);
                    atom_energies[center] += 0.5 * value;
                    const double scale = 0.5 * radial_derivative / distance;
                    for (int axis = 0; axis < 3; ++axis) {
                        const double force = neighbors.displacements[edge * 3 + axis] * scale;
                        forces[center * 3 + axis] += force;
                        forces[neighbor * 3 + axis] -= force;
                    }
                }
            }
        }
        if (control) control->mark_completed();
    }
    for (std::int64_t structure = 0; structure < batch.structures; ++structure) {
        for (std::int64_t atom = batch.offsets[structure]; atom < batch.offsets[structure + 1]; ++atom) {
            energies[structure] += atom_energies[atom];
        }
    }
    detail::check_cancelled(control);
}

} // namespace

NepCalculator::NepCalculator(NepOptions options)
    : model_(load_model(options.model_path, options.model_digest, options.model_data)),
      num_threads_(options.num_threads) {
    if (num_threads_ < 0) throw std::invalid_argument("NEP num_threads must be non-negative");
}

std::int64_t NepCalculator::feature_count() const noexcept { return model_->dimension; }
const std::vector<std::int32_t>& NepCalculator::species() const noexcept { return model_->species; }
const std::string& NepCalculator::model_path() const noexcept { return model_->path; }
double NepCalculator::radial_cutoff() const noexcept { return model_->radial_cutoff_max; }
double NepCalculator::angular_cutoff() const noexcept { return model_->angular_cutoff_max; }
int NepCalculator::n_max_radial() const noexcept { return model_->n_max_radial; }
int NepCalculator::n_max_angular() const noexcept { return model_->n_max_angular; }
int NepCalculator::l_max() const noexcept { return model_->l_max; }

NepDescriptorParameters NepCalculator::descriptor_parameters() const {
    NepDescriptorParameters result;
    result.version = model_->version;
    result.num_types = model_->num_types;
    result.n_max_radial = model_->n_max_radial;
    result.n_max_angular = model_->n_max_angular;
    result.basis_size_radial = model_->basis_size_radial;
    result.basis_size_angular = model_->basis_size_angular;
    result.l_max = model_->l_max;
    result.num_l = model_->num_l;
    result.has_q_222 = model_->has_q_222;
    result.has_q_1111 = model_->has_q_1111;
    result.has_q_112 = model_->has_q_112;
    result.has_q_123 = model_->has_q_123;
    result.has_q_233 = model_->has_q_233;
    result.has_q_134 = model_->has_q_134;
    result.dimension = model_->dimension;
    result.radial_cutoff_max = model_->radial_cutoff_max;
    result.angular_cutoff_max = model_->angular_cutoff_max;
    result.species = model_->species;
    result.radial_cutoff_pair = model_->radial_cutoff_pair;
    result.angular_cutoff_pair = model_->angular_cutoff_pair;
    result.radial_pair_coefficients = model_->radial_pair_coefficients;
    result.angular_pair_coefficients = model_->angular_pair_coefficients;
    result.scalers = model_->scalers;
    return result;
}

NepPredictionParameters NepCalculator::prediction_parameters() const {
    NepPredictionParameters result;
    static_cast<NepDescriptorParameters&>(result) = descriptor_parameters();
    result.hidden_neurons1 = model_->hidden_neurons1;
    result.hidden_neurons2 = model_->hidden_neurons2;
    result.zbl_enabled = model_->zbl_enabled;
    result.zbl_inner = model_->zbl_inner;
    result.zbl_outer = model_->zbl_outer;
    result.ann_parameters = model_->ann_parameters;
    return result;
}

void NepCalculator::compute(
    const StructureBatchView& batch,
    double* output,
    const std::shared_ptr<ComputeControl>& control
) const {
    assert_open("NEP calculator");
    std::lock_guard<std::mutex> lock(compute_mutex_);
    if (batch.atoms == 0) return;
    compute_nep(batch, *model_, num_threads_, output, control);
}

NepPredictor::NepPredictor(NepOptions options)
    : model_(load_model(options.model_path, options.model_digest, options.model_data)),
      num_threads_(options.num_threads) {
    if (num_threads_ < 0) throw std::invalid_argument("NEP num_threads must be non-negative");
    if (model_->version < 4 || model_->version > 5) {
        throw std::invalid_argument("NEP prediction supports NEP4 and NEP5 models only");
    }
    if (model_->hidden_neurons1 > 120 || model_->hidden_neurons2 > 120) {
        throw std::invalid_argument("NEP prediction supports at most 120 neurons per layer");
    }
}

const std::vector<std::int32_t>& NepPredictor::species() const noexcept {
    return model_->species;
}

const std::string& NepPredictor::model_path() const noexcept {
    return model_->path;
}

void NepPredictor::predict(
    const StructureBatchView& batch,
    double* energy,
    double* atom_energy,
    double* forces,
    const std::shared_ptr<ComputeControl>& control
) const {
    assert_open("NEP predictor");
    std::lock_guard<std::mutex> lock(compute_mutex_);
    compute_nep_prediction(batch, *model_, num_threads_, energy, atom_energy, forces, control);
}

} // namespace mdescriptor
