#include "mdescriptor/cuda/nep.hpp"
#include "mdescriptor/cuda/error.hpp"

#include "dpa4_common.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace mdescriptor::cuda {
namespace {

constexpr int kAtomicNumberCount = 119;
constexpr int kNumAngularTerms = 80;
// GPUMD/NEPAdapters use this float literal in the CUDA cutoff path.  Keeping
// the literal (and its operation order) avoids promoting a mathematically
// equivalent expression into a different float32 rounding path.
constexpr float kPi = 3.1415927f;

// These constants are part of the ordinary NEP descriptor definition.  They
// are the same invariant polynomials used by the CPU parser and NEPAdapters.
__device__ __constant__ const float kC3B[kNumAngularTerms] = {
    0.238732414637843f, 0.119366207318922f, 0.119366207318922f, 0.099471839432435f,
    0.596831036594608f, 0.596831036594608f, 0.149207759148652f, 0.149207759148652f,
    0.139260575205408f, 0.104445431404056f, 0.104445431404056f, 1.044454314040563f,
    1.044454314040563f, 0.174075719006761f, 0.174075719006761f, 0.011190581936149f,
    0.223811638722978f, 0.223811638722978f, 0.111905819361489f, 0.111905819361489f,
    1.566681471060845f, 1.566681471060845f, 0.195835183882606f, 0.195835183882606f,
    0.013677377921960f, 0.102580334414698f, 0.102580334414698f, 2.872249363611549f,
    2.872249363611549f, 0.119677056817148f, 0.119677056817148f, 2.154187022708661f,
    2.154187022708661f, 0.215418702270866f, 0.215418702270866f, 0.004041043476943f,
    0.169723826031592f, 0.169723826031592f, 0.106077391269745f, 0.106077391269745f,
    0.424309565078979f, 0.424309565078979f, 0.127292869523694f, 0.127292869523694f,
    2.800443129521260f, 2.800443129521260f, 0.233370260793438f, 0.233370260793438f,
    0.004662742473395f, 0.004079899664221f, 0.004079899664221f, 0.024479397985326f,
    0.024479397985326f, 0.012239698992663f, 0.012239698992663f, 0.538546755677165f,
    0.538546755677165f, 0.134636688919291f, 0.134636688919291f, 3.500553911901575f,
    3.500553911901575f, 0.250039565135827f, 0.250039565135827f, 0.000082569397966f,
    0.005944996653579f, 0.005944996653579f, 0.104037441437634f, 0.104037441437634f,
    0.762941237209318f, 0.762941237209318f, 0.114441185581398f, 0.114441185581398f,
    5.950941650232678f, 5.950941650232678f, 0.141689086910302f, 0.141689086910302f,
    4.250672607309055f, 4.250672607309055f, 0.265667037956816f, 0.265667037956816f,
};
__device__ __constant__ const float kC4B[5] = {
    -0.007499480826664f, -0.134990654879954f, 0.067495327439977f,
    0.404971964639861f, -0.809943929279723f,
};
__device__ __constant__ const float kC5B[3] = {
    0.026596810706114f, 0.053193621412227f, 0.026596810706114f,
};
__device__ __constant__ const float kC4B2[5] = {
    0.027493550848847f, 0.164961305093080f, -0.013746775424423f,
    0.041240326273270f, 0.082480652546540f,
};
__device__ __constant__ const float kC4B123[7] = {
    -0.008418146349617f, -0.016836292699234f, -0.033672585398469f,
    -0.042090731748086f, -0.067345170796937f, -0.084181463496172f,
    -0.168362926992344f,
};
__device__ __constant__ const float kC4B233[10] = {
    0.008572620635186f, 0.009644198214584f, 0.019288396429168f,
    0.025717861905558f, 0.026789439484956f, 0.032147327381947f,
    0.038576792858337f, 0.128589309527790f, 0.192883964291685f,
    0.321473273819474f,
};
__device__ __constant__ const float kC4B134[10] = {
    0.003645164295772f, 0.004860219061029f, 0.006075273826286f,
    0.018225821478859f, 0.024301095305146f, 0.036451642957719f,
    0.042526916784005f, 0.072903285915437f, 0.085053833568010f,
    0.255161500704030f,
};

__device__ __constant__ const float kZ1[2][2] = {{0.0f, 1.0f}, {1.0f, 0.0f}};
__device__ __constant__ const float kZ2[3][3] = {
    {-1.0f, 0.0f, 3.0f}, {0.0f, 1.0f, 0.0f}, {1.0f, 0.0f, 0.0f},
};
__device__ __constant__ const float kZ3[4][4] = {
    {0.0f, -3.0f, 0.0f, 5.0f}, {-1.0f, 0.0f, 5.0f, 0.0f},
    {0.0f, 1.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f, 0.0f},
};
__device__ __constant__ const float kZ4[5][5] = {
    {3.0f, 0.0f, -30.0f, 0.0f, 35.0f}, {0.0f, -3.0f, 0.0f, 7.0f, 0.0f},
    {-1.0f, 0.0f, 7.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f, 0.0f, 0.0f},
    {1.0f, 0.0f, 0.0f, 0.0f, 0.0f},
};
__device__ __constant__ const float kZ5[6][6] = {
    {0.0f, 15.0f, 0.0f, -70.0f, 0.0f, 63.0f}, {1.0f, 0.0f, -14.0f, 0.0f, 21.0f, 0.0f},
    {0.0f, -1.0f, 0.0f, 3.0f, 0.0f, 0.0f}, {-1.0f, 0.0f, 9.0f, 0.0f, 0.0f, 0.0f},
    {0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
};
__device__ __constant__ const float kZ6[7][7] = {
    {-5.0f, 0.0f, 105.0f, 0.0f, -315.0f, 0.0f, 231.0f},
    {0.0f, 5.0f, 0.0f, -30.0f, 0.0f, 33.0f, 0.0f},
    {1.0f, 0.0f, -18.0f, 0.0f, 33.0f, 0.0f, 0.0f},
    {0.0f, -3.0f, 0.0f, 11.0f, 0.0f, 0.0f, 0.0f},
    {-1.0f, 0.0f, 11.0f, 0.0f, 0.0f, 0.0f, 0.0f},
    {0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
    {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
};
__device__ __constant__ const float kZ7[8][8] = {
    {0.0f, -35.0f, 0.0f, 315.0f, 0.0f, -693.0f, 0.0f, 429.0f},
    {-5.0f, 0.0f, 135.0f, 0.0f, -495.0f, 0.0f, 429.0f, 0.0f},
    {0.0f, 15.0f, 0.0f, -110.0f, 0.0f, 143.0f, 0.0f, 0.0f},
    {3.0f, 0.0f, -66.0f, 0.0f, 143.0f, 0.0f, 0.0f, 0.0f},
    {0.0f, -3.0f, 0.0f, 13.0f, 0.0f, 0.0f, 0.0f, 0.0f},
    {-1.0f, 0.0f, 13.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
    {0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
    {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
};
__device__ __constant__ const float kZ8[9][9] = {
    {35.0f, 0.0f, -1260.0f, 0.0f, 6930.0f, 0.0f, -12012.0f, 0.0f, 6435.0f},
    {0.0f, -35.0f, 0.0f, 385.0f, 0.0f, -1001.0f, 0.0f, 715.0f, 0.0f},
    {-1.0f, 0.0f, 33.0f, 0.0f, -143.0f, 0.0f, 143.0f, 0.0f, 0.0f},
    {0.0f, 3.0f, 0.0f, -26.0f, 0.0f, 39.0f, 0.0f, 0.0f, 0.0f},
    {1.0f, 0.0f, -26.0f, 0.0f, 65.0f, 0.0f, 0.0f, 0.0f, 0.0f},
    {0.0f, -1.0f, 0.0f, 5.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
    {-1.0f, 0.0f, 15.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
    {0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
    {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
};

template <typename Value>
__device__ __forceinline__ void complex_product(
    Value a, Value b, Value& real, Value& imag) {
    const Value old_real = real;
    real = a * old_real - b * imag;
    imag = a * imag + b * old_real;
}

template <int L>
__device__ __forceinline__ float z_coefficient(int n1, int n2) {
    if constexpr (L == 1) return kZ1[n1][n2];
    if constexpr (L == 2) return kZ2[n1][n2];
    if constexpr (L == 3) return kZ3[n1][n2];
    if constexpr (L == 4) return kZ4[n1][n2];
    if constexpr (L == 5) return kZ5[n1][n2];
    if constexpr (L == 6) return kZ6[n1][n2];
    if constexpr (L == 7) return kZ7[n1][n2];
    return kZ8[n1][n2];
}

template <int L, typename Value>
__device__ __forceinline__ void accumulate_s_one(
    Value x, Value y, Value z, Value value, Value* s) {
    int index = L * L - 1;
    Value z_power[L + 1] = {static_cast<Value>(1)};
    for (int power = 1; power <= L; ++power) {
        z_power[power] = z * z_power[power - 1];
    }
    Value real = x;
    Value imag = y;
    for (int n1 = 0; n1 <= L; ++n1) {
        const int start = (L + n1) % 2 == 0 ? 0 : 1;
        Value z_factor = static_cast<Value>(0);
        for (int n2 = start; n2 <= L - n1; n2 += 2) {
            z_factor += z_coefficient<L>(n1, n2) * z_power[n2];
        }
        z_factor *= value;
        if (n1 == 0) {
            s[index++] += z_factor;
        } else {
            s[index++] += z_factor * real;
            s[index++] += z_factor * imag;
            complex_product(x, y, real, imag);
        }
    }
}

// Keep the low-order paths in the same arithmetic form as NEPAdapters.  In
// particular, the reference uses 2*x*y rather than x*y + y*x when building
// the azimuthal powers; that small distinction is visible in the strict
// float32 descriptor comparison.
template <typename Value>
__device__ __forceinline__ void accumulate_s_l1(
    Value x, Value y, Value z, Value value, Value* s) {
    s[0] += z * value;
    s[1] += x * value;
    s[2] += y * value;
}

template <typename Value>
__device__ __forceinline__ void accumulate_s_l2(
    Value x, Value y, Value z, Value value, Value* s) {
    s[3] += (-1.0f + 3.0f * z * z) * value;
    s[4] += z * x * value;
    s[5] += z * y * value;
    s[6] += (x * x - y * y) * value;
    s[7] += (2.0f * x * y) * value;
}

template <typename Value>
__device__ __forceinline__ void accumulate_s_l3(
    Value x, Value y, Value z, Value value, Value* s) {
    const float x2_minus_y2 = x * x - y * y;
    const float two_xy = 2.0f * x * y;
    const float x3_minus_3xy2 = x * x2_minus_y2 - y * two_xy;
    const float three_x2y_minus_y3 = x * two_xy + y * x2_minus_y2;
    s[8] += (-3.0f * z + 5.0f * z * z * z) * value;
    s[9] += (-1.0f + 5.0f * z * z) * x * value;
    s[10] += (-1.0f + 5.0f * z * z) * y * value;
    s[11] += z * x2_minus_y2 * value;
    s[12] += z * two_xy * value;
    s[13] += x3_minus_3xy2 * value;
    s[14] += three_x2y_minus_y3 * value;
}

template <typename Value>
__device__ __forceinline__ void accumulate_s_l4(
    Value x, Value y, Value z, Value value, Value* s) {
    const float z2 = z * z;
    const float x2_minus_y2 = x * x - y * y;
    const float two_xy = 2.0f * x * y;
    const float x3_minus_3xy2 = x * x2_minus_y2 - y * two_xy;
    const float three_x2y_minus_y3 = x * two_xy + y * x2_minus_y2;
    const float x4_minus_6x2y2_plus_y4 =
        x * x3_minus_3xy2 - y * three_x2y_minus_y3;
    const float four_x3y_minus_4xy3 =
        x * three_x2y_minus_y3 + y * x3_minus_3xy2;
    s[15] += (3.0f - 30.0f * z2 + 35.0f * z2 * z2) * value;
    s[16] += (-3.0f * z + 7.0f * z * z2) * x * value;
    s[17] += (-3.0f * z + 7.0f * z * z2) * y * value;
    s[18] += (-1.0f + 7.0f * z2) * x2_minus_y2 * value;
    s[19] += (-1.0f + 7.0f * z2) * two_xy * value;
    s[20] += z * x3_minus_3xy2 * value;
    s[21] += z * three_x2y_minus_y3 * value;
    s[22] += x4_minus_6x2y2_plus_y4 * value;
    s[23] += four_x3y_minus_4xy3 * value;
}

template <typename Value>
__device__ __forceinline__ void accumulate_s(
    int l_max, Value x, Value y, Value z, Value value, Value* s) {
    // The caller has already normalized the displacement using the same
    // float distance used by the NEPAdapters CUDA path.  Renormalizing here
    // would add a second round of float error and make the angular channels
    // needlessly diverge from the CPU/reference implementation.
    if (l_max >= 1) accumulate_s_l1(x, y, z, value, s);
    if (l_max >= 2) accumulate_s_l2(x, y, z, value, s);
    if (l_max >= 3) accumulate_s_l3(x, y, z, value, s);
    if (l_max >= 4) accumulate_s_l4(x, y, z, value, s);
    if (l_max >= 5) accumulate_s_one<5>(x, y, z, value, s);
    if (l_max >= 6) accumulate_s_one<6>(x, y, z, value, s);
    if (l_max >= 7) accumulate_s_one<7>(x, y, z, value, s);
    if (l_max >= 8) accumulate_s_one<8>(x, y, z, value, s);
}

__device__ __forceinline__ float find_q_l1(const float* s) {
    return kC3B[0] * s[0] * s[0]
        + 2.0f * (kC3B[1] * s[1] * s[1] + kC3B[2] * s[2] * s[2]);
}

__device__ __forceinline__ float find_q_l2(const float* s) {
    return kC3B[3] * s[3] * s[3]
        + 2.0f * (kC3B[4] * s[4] * s[4] + kC3B[5] * s[5] * s[5]
            + kC3B[6] * s[6] * s[6] + kC3B[7] * s[7] * s[7]);
}

__device__ __forceinline__ float find_q_l3(const float* s) {
    return kC3B[8] * s[8] * s[8]
        + 2.0f * (kC3B[9] * s[9] * s[9] + kC3B[10] * s[10] * s[10]
            + kC3B[11] * s[11] * s[11] + kC3B[12] * s[12] * s[12]
            + kC3B[13] * s[13] * s[13] + kC3B[14] * s[14] * s[14]);
}

__device__ __forceinline__ float find_q_l4(const float* s) {
    return kC3B[15] * s[15] * s[15]
        + 2.0f * (kC3B[16] * s[16] * s[16] + kC3B[17] * s[17] * s[17]
            + kC3B[18] * s[18] * s[18] + kC3B[19] * s[19] * s[19]
            + kC3B[20] * s[20] * s[20] + kC3B[21] * s[21] * s[21]
            + kC3B[22] * s[22] * s[22] + kC3B[23] * s[23] * s[23]);
}

__device__ __forceinline__ float find_q_one(int angular, const float* s) {
    if (angular == 1) return find_q_l1(s);
    if (angular == 2) return find_q_l2(s);
    if (angular == 3) return find_q_l3(s);
    if (angular == 4) return find_q_l4(s);
    const int start = angular * angular - 1;
    const int count = 2 * angular + 1;
    float result = 0.0f;
    for (int k = 1; k < count; ++k) {
        result += kC3B[start + k] * s[start + k] * s[start + k];
    }
    result *= 2.0f;
    return result + kC3B[start] * s[start] * s[start];
}

template <typename Value>
__device__ __forceinline__ Value q222(const Value* s) {
    return kC4B[0] * s[3] * s[3] * s[3]
        + kC4B[1] * s[3] * (s[4] * s[4] + s[5] * s[5])
        + kC4B[2] * s[3] * (s[6] * s[6] + s[7] * s[7])
        + kC4B[3] * s[6] * (s[5] * s[5] - s[4] * s[4])
        + kC4B[4] * s[4] * s[5] * s[7];
}

template <typename Value>
__device__ __forceinline__ Value q1111(const Value* s) {
    const Value s0 = s[0] * s[0];
    const Value s12 = s[1] * s[1] + s[2] * s[2];
    return kC5B[0] * s0 * s0 + kC5B[1] * s0 * s12 + kC5B[2] * s12 * s12;
}

template <typename Value>
__device__ __forceinline__ Value q112(const Value* s) {
    return kC4B2[0] * s[0] * s[0] * s[3]
        + kC4B2[1] * s[0] * (s[1] * s[4] + s[2] * s[5])
        + kC4B2[2] * s[3] * (s[1] * s[1] + s[2] * s[2])
        + kC4B2[3] * s[6] * (s[1] * s[1] - s[2] * s[2])
        + kC4B2[4] * s[1] * s[2] * s[7];
}

template <typename Value>
__device__ __forceinline__ Value q123(const Value* s) {
    Value value = static_cast<Value>(0.0f);
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

template <typename Value>
__device__ __forceinline__ Value q233(const Value* s) {
    Value value = static_cast<Value>(0.0f);
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

template <typename Value>
__device__ __forceinline__ Value q134(const Value* s) {
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

struct DeviceDual3 {
    float value = 0.0f;
    float derivative[3] = {};

    __device__ DeviceDual3(float input = 0.0f) : value(input) {}
};

__device__ __forceinline__ DeviceDual3 operator+(
    const DeviceDual3& a, const DeviceDual3& b) {
    DeviceDual3 result(a.value + b.value);
    for (int axis = 0; axis < 3; ++axis) {
        result.derivative[axis] = a.derivative[axis] + b.derivative[axis];
    }
    return result;
}

__device__ __forceinline__ DeviceDual3& operator+=(
    DeviceDual3& a, const DeviceDual3& b) {
    a.value += b.value;
    for (int axis = 0; axis < 3; ++axis) {
        a.derivative[axis] += b.derivative[axis];
    }
    return a;
}

__device__ __forceinline__ DeviceDual3& operator*=(
    DeviceDual3& a, const DeviceDual3& b) {
    const float old_value = a.value;
    for (int axis = 0; axis < 3; ++axis) {
        a.derivative[axis] = a.derivative[axis] * b.value
            + old_value * b.derivative[axis];
    }
    a.value = old_value * b.value;
    return a;
}

__device__ __forceinline__ DeviceDual3 operator-(
    const DeviceDual3& a, const DeviceDual3& b) {
    DeviceDual3 result(a.value - b.value);
    for (int axis = 0; axis < 3; ++axis) {
        result.derivative[axis] = a.derivative[axis] - b.derivative[axis];
    }
    return result;
}

__device__ __forceinline__ DeviceDual3 operator-(const DeviceDual3& a) {
    DeviceDual3 result(-a.value);
    for (int axis = 0; axis < 3; ++axis) result.derivative[axis] = -a.derivative[axis];
    return result;
}

__device__ __forceinline__ DeviceDual3 operator*(
    const DeviceDual3& a, const DeviceDual3& b) {
    DeviceDual3 result(a.value * b.value);
    for (int axis = 0; axis < 3; ++axis) {
        result.derivative[axis] = a.derivative[axis] * b.value
            + a.value * b.derivative[axis];
    }
    return result;
}

__device__ __forceinline__ DeviceDual3 dual_cos(const DeviceDual3& input) {
    DeviceDual3 result(cosf(input.value));
    const float scale = -sinf(input.value);
    for (int axis = 0; axis < 3; ++axis) {
        result.derivative[axis] = scale * input.derivative[axis];
    }
    return result;
}

__device__ __forceinline__ DeviceDual3 dual_sqrt(const DeviceDual3& input) {
    DeviceDual3 result(sqrtf(input.value));
    const float scale = 0.5f / result.value;
    for (int axis = 0; axis < 3; ++axis) {
        result.derivative[axis] = scale * input.derivative[axis];
    }
    return result;
}

__device__ __forceinline__ void basis_values_dual(
    int basis_size, float cutoff, const DeviceDual3& distance,
    DeviceDual3* values) {
    const DeviceDual3 fc = 0.5f * dual_cos(distance * (kPi / cutoff)) + 0.5f;
    const DeviceDual3 a = distance * (1.0f / cutoff) - 1.0f;
    const DeviceDual3 x = 2.0f * a * a - 1.0f;
    values[0] = fc;
    if (basis_size == 0) return;
    values[1] = 0.5f * (x + 1.0f) * fc;
    DeviceDual3 previous(1.0f);
    DeviceDual3 current = x;
    for (int order = 2; order <= basis_size; ++order) {
        const DeviceDual3 next = 2.0f * x * current - previous;
        previous = current;
        current = next;
        values[order] = 0.5f * (current + 1.0f) * fc;
    }
}

__device__ __forceinline__ void accumulate_s_dual(
    int l_max, const DeviceDual3& distance,
    const DeviceDual3& x, const DeviceDual3& y, const DeviceDual3& z,
    const DeviceDual3& value, DeviceDual3* s) {
    DeviceDual3 inverse(1.0f / distance.value);
    for (int axis = 0; axis < 3; ++axis) {
        inverse.derivative[axis] = -distance.derivative[axis]
            / (distance.value * distance.value);
    }
    const DeviceDual3 nx = x * inverse;
    const DeviceDual3 ny = y * inverse;
    const DeviceDual3 nz = z * inverse;
    if (l_max >= 1) accumulate_s_one<1>(nx, ny, nz, value, s);
    if (l_max >= 2) accumulate_s_one<2>(nx, ny, nz, value, s);
    if (l_max >= 3) accumulate_s_one<3>(nx, ny, nz, value, s);
    if (l_max >= 4) accumulate_s_one<4>(nx, ny, nz, value, s);
    if (l_max >= 5) accumulate_s_one<5>(nx, ny, nz, value, s);
    if (l_max >= 6) accumulate_s_one<6>(nx, ny, nz, value, s);
    if (l_max >= 7) accumulate_s_one<7>(nx, ny, nz, value, s);
    if (l_max >= 8) accumulate_s_one<8>(nx, ny, nz, value, s);
}

template <typename Value>
__device__ __forceinline__ Value find_q_one_value(int angular, const Value* s) {
    const int start = angular * angular - 1;
    const int count = 2 * angular + 1;
    Value result = kC3B[start] * s[start] * s[start];
    for (int k = 1; k < count; ++k) {
        result += 2.0f * kC3B[start + k] * s[start + k] * s[start + k];
    }
    return result;
}

__device__ __forceinline__ DeviceDual3 angular_q_channel(
    int channel, int l_max,
    bool has_q_222, bool has_q_1111, bool has_q_112,
    bool has_q_123, bool has_q_233, bool has_q_134,
    const DeviceDual3* s) {
    int index = 0;
    for (int angular = 1; angular <= l_max; ++angular, ++index) {
        if (channel == index) return find_q_one_value(angular, s);
    }
    if (has_q_222) {
        if (channel == index) return q222(s);
        ++index;
    }
    if (has_q_1111) {
        if (channel == index) return q1111(s);
        ++index;
    }
    if (has_q_112) {
        if (channel == index) return q112(s);
        ++index;
    }
    if (has_q_123) {
        if (channel == index) return q123(s);
        ++index;
    }
    if (has_q_233) {
        if (channel == index) return q233(s);
        ++index;
    }
    if (has_q_134 && channel == index) return q134(s);
    return DeviceDual3{};
}

__device__ __forceinline__ void basis_values(
    int basis_size, float cutoff, float distance, float* values) {
    const float cutoff_inverse = 1.0f / cutoff;
    const float fc = 0.5f * cosf(kPi * distance * cutoff_inverse) + 0.5f;
    const float x = 2.0f * (distance * cutoff_inverse - 1.0f)
        * (distance * cutoff_inverse - 1.0f) - 1.0f;
    const float half_fc = 0.5f * fc;
    values[0] = fc;
    if (basis_size == 0) {
        return;
    }
    values[1] = (x + 1.0f) * half_fc;
    float previous = 1.0f;
    float current = x;
    for (int order = 2; order <= basis_size; ++order) {
        const float next = 2.0f * x * current - previous;
        previous = current;
        current = next;
        values[order] = (current + 1.0f) * half_fc;
    }
}

__device__ __forceinline__ float dot_basis(
    const float* coefficients, const float* basis, int count) {
    float result = 0.0f;
    for (int index = 0; index < count; ++index) {
        result += coefficients[index] * basis[index];
    }
    return result;
}

__device__ __forceinline__ void write_angular_channels(
    int l_max,
    bool has_q_222,
    bool has_q_1111,
    bool has_q_112,
    bool has_q_123,
    bool has_q_233,
    bool has_q_134,
    int angular_count,
    int n,
    int radial_count,
    int dimension,
    const float* s,
    const float* scalers,
    double* row) {
    int channel = 0;
    // find_q_one dispatches any angular order; model validation caps l_max<=8.
    for (int angular = 1; angular <= l_max; ++angular) {
        const int index = radial_count + channel++ * angular_count + n;
        if (index < dimension) {
            row[index] = static_cast<double>(find_q_one(angular, s))
                * static_cast<double>(scalers[index]);
        }
    }
    if (has_q_222) {
        const int index = radial_count + channel++ * angular_count + n;
        if (index < dimension) row[index] = static_cast<double>(q222(s))
            * static_cast<double>(scalers[index]);
    }
    if (has_q_1111) {
        const int index = radial_count + channel++ * angular_count + n;
        if (index < dimension) row[index] = static_cast<double>(q1111(s))
            * static_cast<double>(scalers[index]);
    }
    if (has_q_112) {
        const int index = radial_count + channel++ * angular_count + n;
        if (index < dimension) row[index] = static_cast<double>(q112(s))
            * static_cast<double>(scalers[index]);
    }
    if (has_q_123) {
        const int index = radial_count + channel++ * angular_count + n;
        if (index < dimension) row[index] = static_cast<double>(q123(s))
            * static_cast<double>(scalers[index]);
    }
    if (has_q_233) {
        const int index = radial_count + channel++ * angular_count + n;
        if (index < dimension) row[index] = static_cast<double>(q233(s))
            * static_cast<double>(scalers[index]);
    }
    if (has_q_134) {
        const int index = radial_count + channel++ * angular_count + n;
        if (index < dimension) row[index] = static_cast<double>(q134(s))
            * static_cast<double>(scalers[index]);
    }
}

constexpr int kNepAngularOrderTile = 3;

__global__ void compute_nep_kernel(
    const std::int32_t* numbers,
    const std::int64_t* graph_offsets,
    const std::int32_t* graph_counts,
    std::int64_t graph_stride,
    const std::int32_t* graph_atoms,
    const double* graph_displacements,
    const std::int32_t* type_lookup,
    int num_types,
    int n_max_radial,
    int n_max_angular,
    int basis_size_radial,
    int basis_size_angular,
    int l_max,
    bool has_q_222,
    bool has_q_1111,
    bool has_q_112,
    bool has_q_123,
    bool has_q_233,
    bool has_q_134,
    int dimension,
    const float* radial_cutoff_pair,
    const float* angular_cutoff_pair,
    const float* radial_pair_coefficients,
    const float* angular_pair_coefficients,
    const float* scalers,
    std::int64_t atoms,
    double* output) {
    const std::int64_t center = static_cast<std::int64_t>(blockIdx.x)
        * blockDim.x + threadIdx.x;
    if (center >= atoms) {
        return;
    }

    const int radial_count = n_max_radial + 1;
    const int angular_count = n_max_angular + 1;
    const int radial_basis_count = basis_size_radial + 1;
    const int angular_basis_count = basis_size_angular + 1;
    const int center_type = numbers[center] >= 0 && numbers[center] < kAtomicNumberCount
        ? type_lookup[numbers[center]] : -1;
    const std::int64_t begin = graph_counts == nullptr
        ? graph_offsets[center] : center * graph_stride;
    const std::int64_t end = graph_counts == nullptr
        ? graph_offsets[center + 1] : begin + graph_counts[center];

    float basis[17];
    float radial[13] = {};
    // NEPAdapters first accumulates the radial basis by contiguous
    // neighbor-type run and only then applies the type-pair coefficients.
    // Reusing this buffer avoids a second per-thread array; it is
    // overwritten by basis_values before the angular pass.
    int radial_run_type = -1;
    for (std::int64_t edge = begin; edge < end; ++edge) {
        const std::int32_t neighbor = graph_atoms[edge];
        const int neighbor_type = numbers[neighbor] >= 0 && numbers[neighbor] < kAtomicNumberCount
            ? type_lookup[numbers[neighbor]] : -1;
        if (center_type < 0 || neighbor_type < 0) {
            continue;
        }
        const int pair = center_type * num_types + neighbor_type;
        const float radial_cutoff = radial_cutoff_pair[pair];
        if (neighbor_type != radial_run_type) {
            if (radial_run_type >= 0) {
                const int run_pair = center_type * num_types + radial_run_type;
                const float* coefficients = radial_pair_coefficients
                    + run_pair * radial_count * radial_basis_count;
                for (int n = 0; n < radial_count; ++n) {
                    radial[n] += dot_basis(
                        coefficients + n * radial_basis_count,
                        basis, radial_basis_count);
                }
            }
            for (int k = 0; k < radial_basis_count; ++k) {
                basis[k] = 0.0f;
            }
            radial_run_type = neighbor_type;
        }
        const float dx = static_cast<float>(graph_displacements[edge * 3 + 0]);
        const float dy = static_cast<float>(graph_displacements[edge * 3 + 1]);
        const float dz = static_cast<float>(graph_displacements[edge * 3 + 2]);
        const float distance = sqrtf(dx * dx + dy * dy + dz * dz);
        if (distance <= 0.0f) {
            continue;
        }
        if (distance >= radial_cutoff) {
            continue;
        }
        const float inverse_cutoff = 1.0f / radial_cutoff;
        const float cutoff_value = 0.5f
            * cosf(kPi * distance * inverse_cutoff) + 0.5f;
        const float x = 2.0f * (distance * inverse_cutoff - 1.0f)
            * (distance * inverse_cutoff - 1.0f) - 1.0f;
        const float half_cutoff = 0.5f * cutoff_value;
        basis[0] += cutoff_value;
        if (basis_size_radial >= 1) {
            basis[1] += (x + 1.0f) * half_cutoff;
            float previous = 1.0f;
            float current = x;
            for (int k = 2; k < radial_basis_count; ++k) {
                const float next = 2.0f * x * current - previous;
                previous = current;
                current = next;
                basis[k] += (current + 1.0f) * half_cutoff;
            }
        }
    }
    if (radial_run_type >= 0) {
        const int run_pair = center_type * num_types + radial_run_type;
        const float* coefficients = radial_pair_coefficients
            + run_pair * radial_count * radial_basis_count;
        for (int n = 0; n < radial_count; ++n) {
            radial[n] += dot_basis(
                coefficients + n * radial_basis_count,
                basis, radial_basis_count);
        }
    }

    double* row = output + center * static_cast<std::int64_t>(dimension);
    for (int n = 0; n < radial_count; ++n) {
        row[n] = static_cast<double>(radial[n]) * static_cast<double>(scalers[n]);
    }

    const bool has_angular = l_max > 0 || has_q_222 || has_q_1111 || has_q_112
        || has_q_123 || has_q_233 || has_q_134;
    if (!has_angular) {
        return;
    }

    // Keep only three angular orders live at a time.  This is the same tile
    // size used by the reference implementation and bounds local state to
    // 3 * 80 floats instead of one [n, l, m] array.
    for (int n_base = 0; n_base < angular_count; n_base += kNepAngularOrderTile) {
        const int active_orders = min(
            kNepAngularOrderTile, angular_count - n_base);
        float s[kNepAngularOrderTile][kNumAngularTerms] = {};

        for (std::int64_t edge = begin; edge < end; ++edge) {
            const std::int32_t neighbor = graph_atoms[edge];
            const int neighbor_type = numbers[neighbor] >= 0 && numbers[neighbor] < kAtomicNumberCount
                ? type_lookup[numbers[neighbor]] : -1;
            if (center_type < 0 || neighbor_type < 0) {
                continue;
            }
            const int pair = center_type * num_types + neighbor_type;
            const float dx = static_cast<float>(graph_displacements[edge * 3 + 0]);
            const float dy = static_cast<float>(graph_displacements[edge * 3 + 1]);
            const float dz = static_cast<float>(graph_displacements[edge * 3 + 2]);
            const float distance = sqrtf(dx * dx + dy * dy + dz * dz);
            if (distance <= 0.0f) {
                continue;
            }
            const float angular_cutoff = angular_cutoff_pair[pair];
            if (distance >= angular_cutoff) {
                continue;
            }
            basis_values(basis_size_angular, angular_cutoff, distance, basis);
            const float* coefficients = angular_pair_coefficients
                + pair * angular_count * angular_basis_count
                + n_base * angular_basis_count;
            const float inverse_distance = 1.0f / distance;
            const float x = dx * inverse_distance;
            const float y = dy * inverse_distance;
            const float z = dz * inverse_distance;
            for (int tile = 0; tile < active_orders; ++tile) {
                const float value = dot_basis(
                    coefficients + tile * angular_basis_count, basis, angular_basis_count);
                accumulate_s(l_max, x, y, z, value, s[tile]);
            }
        }

        for (int tile = 0; tile < active_orders; ++tile) {
            write_angular_channels(
                l_max, has_q_222, has_q_1111, has_q_112, has_q_123, has_q_233,
                has_q_134, angular_count, n_base + tile, radial_count, dimension,
                s[tile], scalers, row);
        }
    }
}

__device__ __forceinline__ void graph_center_range(
    std::int64_t center, const std::int64_t* graph_offsets,
    const std::int32_t* graph_counts, std::int64_t graph_stride,
    std::int64_t& begin, std::int64_t& end) {
    begin = graph_counts == nullptr ? graph_offsets[center] : center * graph_stride;
    end = graph_counts == nullptr ? graph_offsets[center + 1]
        : begin + graph_counts[center];
}

__device__ __forceinline__ void atomic_add_double(double* address, double value) {
#if __CUDA_ARCH__ >= 600
    atomicAdd(address, value);
#else
    auto* bits = reinterpret_cast<unsigned long long int*>(address);
    unsigned long long int old = atomicCAS(bits, 0ULL, 0ULL);
    unsigned long long int assumed = 0;
    do {
        assumed = old;
        old = atomicCAS(bits, assumed,
            __double_as_longlong(value + __longlong_as_double(assumed)));
    } while (assumed != old);
#endif
}

__global__ void evaluate_nep_ann_kernel(
    const std::int32_t* numbers,
    const std::int32_t* type_lookup,
    int num_types,
    int version,
    int dimension,
    int hidden1_count,
    int hidden2_count,
    const double* ann,
    std::int64_t atoms,
    double* descriptor_or_gradient,
    double* atom_energy) {
    const std::int64_t atom = static_cast<std::int64_t>(blockIdx.x)
        * blockDim.x + threadIdx.x;
    if (atom >= atoms) return;
    const int atomic_number = numbers[atom];
    const int type = atomic_number > 0 && atomic_number < kAtomicNumberCount
        ? type_lookup[atomic_number] : -1;
    if (type < 0) {
        atom_energy[atom] = 0.0;
        return;
    }

    const std::size_t stride = hidden2_count > 0
        ? static_cast<std::size_t>(dimension + 1) * hidden1_count
            + static_cast<std::size_t>(hidden1_count + 2) * hidden2_count
        : static_cast<std::size_t>(dimension + 2) * hidden1_count
            + (version == 5 ? 1U : 0U);
    const std::size_t base = static_cast<std::size_t>(type) * stride;
    const double* weights0 = ann + base;
    const double* bias0 = weights0 + static_cast<std::size_t>(hidden1_count) * dimension;
    const double* weights1 = bias0 + hidden1_count;
    double* descriptor = descriptor_or_gradient + atom * dimension;
    double derivative[256] = {};
    double energy = -ann[static_cast<std::size_t>(num_types) * stride];

    if (hidden2_count == 0) {
        if (version == 5) energy -= weights1[hidden1_count];
        for (int neuron = 0; neuron < hidden1_count; ++neuron) {
            const double* row = weights0 + static_cast<std::size_t>(neuron) * dimension;
            double dot = 0.0;
            for (int feature = 0; feature < dimension; ++feature) {
                dot += row[feature] * descriptor[feature];
            }
            const double hidden = tanh(dot - bias0[neuron]);
            const double pull = weights1[neuron] * (1.0 - hidden * hidden);
            energy += weights1[neuron] * hidden;
            for (int feature = 0; feature < dimension; ++feature) {
                derivative[feature] += pull * row[feature];
            }
        }
    } else {
        double hidden1[120] = {};
        double pull1[120] = {};
        const double* weights1_hidden = weights1;
        const double* bias1_hidden = weights1_hidden
            + static_cast<std::size_t>(hidden1_count) * hidden2_count;
        const double* weights2 = bias1_hidden + hidden2_count;
        for (int neuron = 0; neuron < hidden1_count; ++neuron) {
            const double* row = weights0 + static_cast<std::size_t>(neuron) * dimension;
            double dot = 0.0;
            for (int feature = 0; feature < dimension; ++feature) {
                dot += row[feature] * descriptor[feature];
            }
            hidden1[neuron] = tanh(dot - bias0[neuron]);
        }
        for (int neuron = 0; neuron < hidden2_count; ++neuron) {
            const double* row = weights1_hidden + static_cast<std::size_t>(neuron) * hidden1_count;
            double dot = 0.0;
            for (int inner = 0; inner < hidden1_count; ++inner) {
                dot += row[inner] * hidden1[inner];
            }
            const double hidden = tanh(dot - bias1_hidden[neuron]);
            const double pull = weights2[neuron] * (1.0 - hidden * hidden);
            energy += weights2[neuron] * hidden;
            for (int inner = 0; inner < hidden1_count; ++inner) {
                pull1[inner] += row[inner] * pull;
            }
        }
        for (int neuron = 0; neuron < hidden1_count; ++neuron) {
            const double pull = pull1[neuron] * (1.0 - hidden1[neuron] * hidden1[neuron]);
            const double* row = weights0 + static_cast<std::size_t>(neuron) * dimension;
            for (int feature = 0; feature < dimension; ++feature) {
                derivative[feature] += pull * row[feature];
            }
        }
    }

    atom_energy[atom] = energy;
    for (int feature = 0; feature < dimension; ++feature) {
        descriptor[feature] = derivative[feature];
    }
}

__device__ __forceinline__ double zbl_value_and_derivative(
    std::int32_t z1, std::int32_t z2, double inner, double outer,
    double distance, double& derivative) {
    if (distance <= 0.0 || distance >= outer) {
        derivative = 0.0;
        return 0.0;
    }
    constexpr double parameters[8] = {
        0.18175, 3.1998, 0.50986, 0.94229,
        0.28022, 0.4029, 0.02817, 0.20162,
    };
    constexpr double pi = 3.141592653589793238462643383279502884;
    const double a_inv = (pow(static_cast<double>(z1), 0.23)
        + pow(static_cast<double>(z2), 0.23)) * 2.134563;
    const double charge = 14.399645 * static_cast<double>(z1) * z2;
    double phi = 0.0;
    double phi_prime = 0.0;
    for (int index = 0; index < 4; ++index) {
        const double term = parameters[2 * index]
            * exp(-parameters[2 * index + 1] * distance * a_inv);
        phi += term;
        phi_prime -= parameters[2 * index + 1] * term;
    }
    const double value = charge * phi / distance;
    const double value_prime = charge * phi_prime * a_inv / distance
        - charge * phi / (distance * distance);
    double cutoff = 1.0;
    double cutoff_prime = 0.0;
    if (distance >= inner) {
        const double factor = pi / (outer - inner);
        cutoff = 0.5 * cos(factor * (distance - inner)) + 0.5;
        cutoff_prime = -0.5 * sin(factor * (distance - inner)) * factor;
    }
    derivative = value_prime * cutoff + value * cutoff_prime;
    return value * cutoff;
}

__global__ void compute_nep_prediction_forces_kernel(
    const std::int32_t* numbers,
    const std::int64_t* graph_offsets,
    const std::int32_t* graph_counts,
    std::int64_t graph_stride,
    const std::int32_t* graph_atoms,
    const double* graph_displacements,
    const std::int32_t* type_lookup,
    int num_types,
    int n_max_radial,
    int n_max_angular,
    int basis_size_radial,
    int basis_size_angular,
    int l_max,
    bool has_q_222,
    bool has_q_1111,
    bool has_q_112,
    bool has_q_123,
    bool has_q_233,
    bool has_q_134,
    int dimension,
    const float* radial_cutoff_pair,
    const float* angular_cutoff_pair,
    const float* radial_pair_coefficients,
    const float* angular_pair_coefficients,
    const float* scalers,
    int zbl_enabled,
    double zbl_inner,
    double zbl_outer,
    std::int64_t atoms,
    const double* ann_gradient,
    double* atom_energy,
    double* forces) {
    const std::int64_t center = static_cast<std::int64_t>(blockIdx.x)
        * blockDim.x + threadIdx.x;
    if (center >= atoms) return;
    const int center_atomic_number = numbers[center];
    const int center_type = center_atomic_number > 0 && center_atomic_number < kAtomicNumberCount
        ? type_lookup[center_atomic_number] : -1;
    if (center_type < 0) return;

    std::int64_t begin = 0;
    std::int64_t end = 0;
    graph_center_range(center, graph_offsets, graph_counts, graph_stride, begin, end);
    const int radial_count = n_max_radial + 1;
    const int angular_count = n_max_angular + 1;
    const int radial_basis_count = basis_size_radial + 1;
    const int angular_basis_count = basis_size_angular + 1;
    const double* center_gradient = ann_gradient + center * dimension;
    double local_force[3] = {0.0, 0.0, 0.0};

    DeviceDual3 basis[17];
    for (std::int64_t edge = begin; edge < end; ++edge) {
        const std::int32_t neighbor = graph_atoms[edge];
        const int neighbor_atomic_number = numbers[neighbor];
        const int neighbor_type = neighbor_atomic_number > 0 && neighbor_atomic_number < kAtomicNumberCount
            ? type_lookup[neighbor_atomic_number] : -1;
        if (neighbor_type < 0) continue;
        const int pair = center_type * num_types + neighbor_type;
        const float cutoff = radial_cutoff_pair[pair];
        const float dx = static_cast<float>(graph_displacements[edge * 3 + 0]);
        const float dy = static_cast<float>(graph_displacements[edge * 3 + 1]);
        const float dz = static_cast<float>(graph_displacements[edge * 3 + 2]);
        if (dx == 0.0f && dy == 0.0f && dz == 0.0f) continue;
        DeviceDual3 x(dx), y(dy), z(dz);
        x.derivative[0] = 1.0f;
        y.derivative[1] = 1.0f;
        z.derivative[2] = 1.0f;
        const DeviceDual3 distance = dual_sqrt(x * x + y * y + z * z);
        if (distance.value >= cutoff) continue;
        basis_values_dual(basis_size_radial, cutoff, distance, basis);
        const float* coefficients = radial_pair_coefficients
            + pair * radial_count * radial_basis_count;
        double edge_force[3] = {0.0, 0.0, 0.0};
        for (int order = 0; order < radial_count; ++order) {
            DeviceDual3 value;
            for (int basis_index = 0; basis_index < radial_basis_count; ++basis_index) {
                value += coefficients[order * radial_basis_count + basis_index]
                    * basis[basis_index];
            }
            const double pull = center_gradient[order] * scalers[order];
            for (int axis = 0; axis < 3; ++axis) {
                edge_force[axis] += pull * value.derivative[axis];
            }
        }
        for (int axis = 0; axis < 3; ++axis) {
            local_force[axis] += edge_force[axis];
            atomic_add_double(&forces[neighbor * 3 + axis], -edge_force[axis]);
        }
    }

    const int angular_channels = (dimension - radial_count) / angular_count;
    if (angular_channels > 0) {
        float total_s[kNumAngularTerms];
        DeviceDual3 dual_s[kNumAngularTerms];
        DeviceDual3 edge_s[kNumAngularTerms];
        DeviceDual3 dual_basis[17];
        for (int order = 0; order < angular_count; ++order) {
            for (int term = 0; term < kNumAngularTerms; ++term) total_s[term] = 0.0f;
            for (std::int64_t edge = begin; edge < end; ++edge) {
                const std::int32_t neighbor = graph_atoms[edge];
                const int neighbor_atomic_number = numbers[neighbor];
                const int neighbor_type = neighbor_atomic_number > 0 && neighbor_atomic_number < kAtomicNumberCount
                    ? type_lookup[neighbor_atomic_number] : -1;
                if (neighbor_type < 0) continue;
                const int pair = center_type * num_types + neighbor_type;
                const float cutoff = angular_cutoff_pair[pair];
                const float dx = static_cast<float>(graph_displacements[edge * 3 + 0]);
                const float dy = static_cast<float>(graph_displacements[edge * 3 + 1]);
                const float dz = static_cast<float>(graph_displacements[edge * 3 + 2]);
                const float distance = sqrtf(dx * dx + dy * dy + dz * dz);
                if (distance <= 0.0f || distance >= cutoff) continue;
                float radial_basis[17];
                basis_values(basis_size_angular, cutoff, distance, radial_basis);
                const float* coefficients = angular_pair_coefficients
                    + (pair * angular_count + order) * angular_basis_count;
                const float value = dot_basis(coefficients, radial_basis, angular_basis_count);
                const float inverse = 1.0f / distance;
                accumulate_s(l_max, dx * inverse, dy * inverse, dz * inverse,
                    value, total_s);
            }

            for (std::int64_t edge = begin; edge < end; ++edge) {
                const std::int32_t neighbor = graph_atoms[edge];
                const int neighbor_atomic_number = numbers[neighbor];
                const int neighbor_type = neighbor_atomic_number > 0 && neighbor_atomic_number < kAtomicNumberCount
                    ? type_lookup[neighbor_atomic_number] : -1;
                if (neighbor_type < 0) continue;
                const int pair = center_type * num_types + neighbor_type;
                const float cutoff = angular_cutoff_pair[pair];
                const float dx = static_cast<float>(graph_displacements[edge * 3 + 0]);
                const float dy = static_cast<float>(graph_displacements[edge * 3 + 1]);
                const float dz = static_cast<float>(graph_displacements[edge * 3 + 2]);
                if (dx == 0.0f && dy == 0.0f && dz == 0.0f) continue;
                DeviceDual3 x(dx), y(dy), z(dz);
                x.derivative[0] = 1.0f;
                y.derivative[1] = 1.0f;
                z.derivative[2] = 1.0f;
                const DeviceDual3 distance = dual_sqrt(x * x + y * y + z * z);
                if (distance.value >= cutoff) continue;
                basis_values_dual(basis_size_angular, cutoff, distance, dual_basis);
                const float* coefficients = angular_pair_coefficients
                    + (pair * angular_count + order) * angular_basis_count;
                DeviceDual3 value;
                for (int basis_index = 0; basis_index < angular_basis_count; ++basis_index) {
                    value += coefficients[basis_index] * dual_basis[basis_index];
                }
                for (int term = 0; term < kNumAngularTerms; ++term) edge_s[term] = DeviceDual3{};
                accumulate_s_dual(l_max, distance, x, y, z, value, edge_s);
                for (int term = 0; term < kNumAngularTerms; ++term) {
                    dual_s[term] = DeviceDual3(total_s[term]);
                    for (int axis = 0; axis < 3; ++axis) {
                        dual_s[term].derivative[axis] = edge_s[term].derivative[axis];
                    }
                }
                double edge_force[3] = {0.0, 0.0, 0.0};
                for (int channel = 0; channel < angular_channels; ++channel) {
                    const int feature = radial_count + channel * angular_count + order;
                    if (feature >= dimension) continue;
                    const DeviceDual3 q = angular_q_channel(channel, l_max,
                        has_q_222, has_q_1111, has_q_112, has_q_123,
                        has_q_233, has_q_134, dual_s);
                    const double pull = center_gradient[feature] * scalers[feature];
                    for (int axis = 0; axis < 3; ++axis) {
                        edge_force[axis] += pull * q.derivative[axis];
                    }
                }
                for (int axis = 0; axis < 3; ++axis) {
                    local_force[axis] += edge_force[axis];
                    atomic_add_double(&forces[neighbor * 3 + axis], -edge_force[axis]);
                }
            }
        }
    }

    double zbl_energy = 0.0;
    if (zbl_enabled) {
        for (std::int64_t edge = begin; edge < end; ++edge) {
            const std::int32_t neighbor = graph_atoms[edge];
            const int neighbor_atomic_number = numbers[neighbor];
            if (neighbor_atomic_number <= 0) continue;
            const double dx = graph_displacements[edge * 3 + 0];
            const double dy = graph_displacements[edge * 3 + 1];
            const double dz = graph_displacements[edge * 3 + 2];
            const double distance = sqrt(dx * dx + dy * dy + dz * dz);
            if (distance <= 0.0 || distance >= zbl_outer) continue;
            double radial_derivative = 0.0;
            const double value = zbl_value_and_derivative(
                center_atomic_number, neighbor_atomic_number,
                zbl_inner, zbl_outer, distance, radial_derivative);
            zbl_energy += 0.5 * value;
            const double scale = 0.5 * radial_derivative / distance;
            local_force[0] += dx * scale;
            local_force[1] += dy * scale;
            local_force[2] += dz * scale;
            atomic_add_double(&forces[neighbor * 3 + 0], -dx * scale);
            atomic_add_double(&forces[neighbor * 3 + 1], -dy * scale);
            atomic_add_double(&forces[neighbor * 3 + 2], -dz * scale);
        }
    }
    atom_energy[center] += zbl_energy;
    for (int axis = 0; axis < 3; ++axis) {
        atomic_add_double(&forces[center * 3 + axis], local_force[axis]);
    }
}

// A periodic batch is physically expanded on the device so that its neighbor
// order is identical to the ordinary NEP cell-list path.  Reduce the replica
// rows in the same CUDA stream; each output element has one deterministic
// writer and the replica loop follows the expansion order.
__global__ void reduce_expanded_nep_kernel(
    std::int64_t original_atoms,
    int dimension,
    const std::int64_t* expansion_first,
    const std::int64_t* expansion_stride,
    const std::int32_t* expansion_replicas,
    const double* expanded_output,
    double* reduced_output) {
    const std::int64_t element = static_cast<std::int64_t>(blockIdx.x)
        * blockDim.x + threadIdx.x;
    const std::int64_t element_count = original_atoms * static_cast<std::int64_t>(dimension);
    if (element >= element_count) return;
    const std::int64_t atom = element / dimension;
    const int feature = static_cast<int>(element % dimension);
    const std::int32_t replicas = expansion_replicas[atom];
    const std::int64_t first = expansion_first[atom];
    const std::int64_t stride = expansion_stride[atom];
    double sum = 0.0;
    for (std::int32_t replica = 0; replica < replicas; ++replica) {
        const std::int64_t expanded_atom = first
            + static_cast<std::int64_t>(replica) * stride;
        sum += expanded_output[expanded_atom * dimension + feature];
    }
    reduced_output[element] = sum / static_cast<double>(replicas);
}

__global__ void reduce_expanded_nep_prediction_kernel(
    std::int64_t original_atoms,
    const std::int64_t* expansion_first,
    const std::int64_t* expansion_stride,
    const std::int32_t* expansion_replicas,
    const double* expanded_energy,
    const double* expanded_forces,
    double* reduced_energy_and_forces) {
    const std::int64_t element = static_cast<std::int64_t>(blockIdx.x)
        * blockDim.x + threadIdx.x;
    if (element >= original_atoms * 4) return;
    const std::int64_t atom = element / 4;
    const int component = static_cast<int>(element % 4);
    const std::int32_t replicas = expansion_replicas[atom];
    const std::int64_t first = expansion_first[atom];
    const std::int64_t stride = expansion_stride[atom];
    double sum = 0.0;
    for (std::int32_t replica = 0; replica < replicas; ++replica) {
        const std::int64_t expanded_atom = first
            + static_cast<std::int64_t>(replica) * stride;
        sum += component == 0
            ? expanded_energy[expanded_atom]
            : expanded_forces[expanded_atom * 3 + component - 1];
    }
    reduced_energy_and_forces[element] = sum / static_cast<double>(replicas);
}

} // namespace

DeviceNepModel::DeviceNepModel(
    CudaExecutionContext& context,
    const mdescriptor::NepDescriptorParameters& parameters)
    : num_types_(parameters.num_types),
      n_max_radial_(parameters.n_max_radial), n_max_angular_(parameters.n_max_angular),
      basis_size_radial_(parameters.basis_size_radial),
      basis_size_angular_(parameters.basis_size_angular), l_max_(parameters.l_max),
      dimension_(parameters.dimension),
      version_(parameters.version),
      radial_cutoff_max_(parameters.radial_cutoff_max),
      angular_cutoff_max_(parameters.angular_cutoff_max),
      has_q_222_(parameters.has_q_222), has_q_1111_(parameters.has_q_1111),
      has_q_112_(parameters.has_q_112), has_q_123_(parameters.has_q_123),
      has_q_233_(parameters.has_q_233), has_q_134_(parameters.has_q_134),
      host_type_lookup_(kAtomicNumberCount, -1) {
    if (num_types_ <= 0 || dimension_ <= 0 || parameters.species.size() != static_cast<std::size_t>(num_types_)) {
        throw std::invalid_argument("invalid NEP descriptor model dimensions");
    }
    if (n_max_radial_ < 0 || n_max_radial_ > 12 || n_max_angular_ < 0 || n_max_angular_ > 8
        || basis_size_radial_ < 0 || basis_size_radial_ > 16
        || basis_size_angular_ < 0 || basis_size_angular_ > 16
        || l_max_ < 0 || l_max_ > 8
        || parameters.num_l < 0 || parameters.num_l > 14) {
        throw std::invalid_argument("unsupported NEP descriptor model dimensions");
    }
    const std::size_t type_pairs = static_cast<std::size_t>(num_types_) * num_types_;
    const std::size_t radial_count = static_cast<std::size_t>(n_max_radial_ + 1);
    const std::size_t angular_count = static_cast<std::size_t>(n_max_angular_ + 1);
    const std::size_t radial_basis = static_cast<std::size_t>(basis_size_radial_ + 1);
    const std::size_t angular_basis = static_cast<std::size_t>(basis_size_angular_ + 1);
    if (parameters.radial_cutoff_pair.size() != type_pairs
        || parameters.angular_cutoff_pair.size() != type_pairs
        || parameters.radial_pair_coefficients.size() != type_pairs * radial_count * radial_basis
        || parameters.angular_pair_coefficients.size() != type_pairs * angular_count * angular_basis
        || parameters.scalers.size() != static_cast<std::size_t>(dimension_)) {
        throw std::invalid_argument("incomplete NEP descriptor model parameters");
    }
    for (int type = 0; type < num_types_; ++type) {
        const auto atomic_number = parameters.species[static_cast<std::size_t>(type)];
        if (atomic_number <= 0 || atomic_number >= kAtomicNumberCount
            || host_type_lookup_[static_cast<std::size_t>(atomic_number)] >= 0) {
            throw std::invalid_argument("invalid or duplicate NEP model species");
        }
        host_type_lookup_[static_cast<std::size_t>(atomic_number)] = type;
    }

    std::vector<float> radial_cutoff_pair(parameters.radial_cutoff_pair.size());
    std::vector<float> angular_cutoff_pair(parameters.angular_cutoff_pair.size());
    std::vector<float> radial_coefficients(parameters.radial_pair_coefficients.size());
    std::vector<float> angular_coefficients(parameters.angular_pair_coefficients.size());
    std::vector<float> scalers(parameters.scalers.size());
    std::copy(parameters.radial_cutoff_pair.begin(), parameters.radial_cutoff_pair.end(), radial_cutoff_pair.begin());
    std::copy(parameters.angular_cutoff_pair.begin(), parameters.angular_cutoff_pair.end(), angular_cutoff_pair.begin());
    std::copy(parameters.radial_pair_coefficients.begin(), parameters.radial_pair_coefficients.end(), radial_coefficients.begin());
    std::copy(parameters.angular_pair_coefficients.begin(), parameters.angular_pair_coefficients.end(), angular_coefficients.begin());
    std::copy(parameters.scalers.begin(), parameters.scalers.end(), scalers.begin());

    const auto upload = [&](const auto& values, const char* operation) {
        // Synchronous copies: the parser temporaries go out of scope right
        // after construction, so no asynchronous DMA may borrow them.
        return dpa4_common::upload_array<DeviceArray>(
            context, values, operation, "could not select the CUDA device", true);
    };
    type_lookup_ = upload(host_type_lookup_, "could not upload NEP type lookup");
    radial_cutoff_pair_ = upload(radial_cutoff_pair, "could not upload NEP radial cutoffs");
    angular_cutoff_pair_ = upload(angular_cutoff_pair, "could not upload NEP angular cutoffs");
    radial_pair_coefficients_ =
        upload(radial_coefficients, "could not upload NEP radial coefficients");
    angular_pair_coefficients_ =
        upload(angular_coefficients, "could not upload NEP angular coefficients");
    scalers_ = upload(scalers, "could not upload NEP q scalers");
}

DeviceNepModel::DeviceNepModel(
    CudaExecutionContext& context,
    const mdescriptor::NepPredictionParameters& parameters)
    : DeviceNepModel(
        context, static_cast<const mdescriptor::NepDescriptorParameters&>(parameters)) {
    if (parameters.hidden_neurons1 <= 0 || parameters.hidden_neurons1 > 120
        || parameters.hidden_neurons2 < 0 || parameters.hidden_neurons2 > 120
        || parameters.version < 4 || parameters.version > 5
        || parameters.ann_parameters.empty()
        || (parameters.zbl_enabled && parameters.zbl_outer <= parameters.zbl_inner)) {
        throw std::invalid_argument("unsupported NEP prediction model parameters");
    }
    hidden_neurons1_ = parameters.hidden_neurons1;
    hidden_neurons2_ = parameters.hidden_neurons2;
    zbl_enabled_ = parameters.zbl_enabled;
    zbl_inner_ = parameters.zbl_inner;
    zbl_outer_ = parameters.zbl_outer;
    std::vector<double> ann(parameters.ann_parameters.begin(), parameters.ann_parameters.end());
    ann_parameters_ = dpa4_common::upload_array<DeviceArray>(
        context, ann, "could not upload NEP ANN parameters",
        "could not select the CUDA device", true);
}

DeviceNepModel::~DeviceNepModel() noexcept = default;

const std::int32_t* DeviceNepModel::type_lookup() const noexcept {
    return dpa4_common::device_data<std::int32_t>(type_lookup_);
}

const float* DeviceNepModel::radial_cutoff_pair() const noexcept {
    return dpa4_common::device_data<float>(radial_cutoff_pair_);
}

const float* DeviceNepModel::angular_cutoff_pair() const noexcept {
    return dpa4_common::device_data<float>(angular_cutoff_pair_);
}

const float* DeviceNepModel::radial_pair_coefficients() const noexcept {
    return dpa4_common::device_data<float>(radial_pair_coefficients_);
}

const float* DeviceNepModel::angular_pair_coefficients() const noexcept {
    return dpa4_common::device_data<float>(angular_pair_coefficients_);
}

const float* DeviceNepModel::scalers() const noexcept {
    return dpa4_common::device_data<float>(scalers_);
}

const double* DeviceNepModel::ann_parameters() const noexcept {
    return dpa4_common::device_data<double>(ann_parameters_);
}

bool DeviceNepModel::supports_atomic_number(std::int32_t number) const noexcept {
    return number > 0 && number < static_cast<std::int32_t>(host_type_lookup_.size())
        && host_type_lookup_[static_cast<std::size_t>(number)] >= 0;
}

void compute_nep_into(
    CudaExecutionContext& context,
    const DeviceBatch& batch,
    const DeviceNeighborGraph& graph,
    const DeviceNepModel& model,
    double* host_output) {
    if (batch.atoms() <= 0 || model.dimension() <= 0) {
        return;
    }
    if (host_output == nullptr) {
        throw std::invalid_argument("CUDA NEP output destination must not be null");
    }
    const auto dimension = static_cast<std::size_t>(model.dimension());
    const auto expanded_atoms = static_cast<std::size_t>(batch.atoms());
    if (expanded_atoms > std::numeric_limits<std::size_t>::max() / dimension) {
        throw CudaOutOfMemory("CUDA NEP descriptor output is too large");
    }
    const std::size_t expanded_count = expanded_atoms * dimension;
    std::size_t output_count = expanded_count;
    std::size_t reduced_offset = 0;
    std::size_t reduced_count = 0;
    if (batch.expanded()) {
        if (batch.original_atoms() <= 0
            || static_cast<std::size_t>(batch.original_atoms())
                > std::numeric_limits<std::size_t>::max() / dimension) {
            throw CudaOutOfMemory("CUDA NEP reduced output is too large");
        }
        reduced_count = static_cast<std::size_t>(batch.original_atoms()) * dimension;
        if (expanded_count > std::numeric_limits<std::size_t>::max() - reduced_count) {
            throw CudaOutOfMemory("CUDA NEP expanded output is too large");
        }
        reduced_offset = expanded_count;
        output_count += reduced_count;
    }
    double* output = context.output_buffer(output_count);
    constexpr unsigned int block_size = 128;
    const auto blocks = static_cast<unsigned int>(
        (expanded_atoms + block_size - 1) / block_size);
    compute_nep_kernel<<<blocks, block_size, 0, context.stream()>>>(
        batch.numbers(), graph.offsets(),
        graph.slot_major() ? graph.neighbor_counts() : nullptr,
        graph.neighbor_stride(), graph.atoms(), graph.displacements(),
        model.type_lookup(), model.num_types(), model.n_max_radial(), model.n_max_angular(),
        model.basis_size_radial(), model.basis_size_angular(), model.l_max(),
        model.has_q_222(), model.has_q_1111(), model.has_q_112(), model.has_q_123(),
        model.has_q_233(), model.has_q_134(), model.dimension(), model.radial_cutoff_pair(),
        model.angular_cutoff_pair(), model.radial_pair_coefficients(),
        model.angular_pair_coefficients(), model.scalers(), batch.atoms(), output);
    check_cuda(cudaGetLastError(), "CUDA NEP descriptor kernel launch failed");
    if (!batch.expanded()) {
        context.download_output_into(host_output, expanded_count);
        return;
    }

    const auto reduction_blocks = static_cast<unsigned int>(
        (reduced_count + block_size - 1) / block_size);
    reduce_expanded_nep_kernel<<<
        reduction_blocks, block_size, 0, context.stream()>>>(
        batch.original_atoms(), model.dimension(), batch.expansion_first(),
        batch.expansion_stride(), batch.expansion_replicas(), output,
        output + reduced_offset);
    check_cuda(cudaGetLastError(), "CUDA NEP replica reduction kernel launch failed");
    context.download_output_slice_into(reduced_offset, host_output, reduced_count);
}

void predict_nep_into(
    CudaExecutionContext& context,
    const DeviceBatch& batch,
    const DeviceNeighborGraph& graph,
    const DeviceNepModel& model,
    const detail::StructureBatchView& original,
    double* energy,
    double* atom_energy,
    double* forces) {
    if (!model.prediction_enabled()) {
        throw std::invalid_argument("CUDA NEP model was not loaded for prediction");
    }
    if (original.atoms <= 0 || batch.atoms() <= 0) {
        return;
    }
    if (energy == nullptr || atom_energy == nullptr || forces == nullptr
        || original.offsets == nullptr) {
        throw std::invalid_argument("CUDA NEP prediction output buffers must not be null");
    }
    const std::size_t expanded_atoms = static_cast<std::size_t>(batch.atoms());
    const std::size_t original_atoms = static_cast<std::size_t>(original.atoms);
    const std::size_t dimension = static_cast<std::size_t>(model.dimension());
    if (expanded_atoms > std::numeric_limits<std::size_t>::max() / dimension) {
        throw CudaOutOfMemory("CUDA NEP prediction descriptors are too large");
    }
    const std::size_t descriptor_count = expanded_atoms * dimension;
    if (expanded_atoms > (std::numeric_limits<std::size_t>::max() - descriptor_count) / 4) {
        throw CudaOutOfMemory("CUDA NEP prediction output is too large");
    }
    const std::size_t expanded_result_count = expanded_atoms * 4;
    std::size_t output_count = descriptor_count + expanded_result_count;
    std::size_t reduced_offset = 0;
    if (batch.expanded()) {
        if (batch.original_atoms() != original.atoms
            || original_atoms > (std::numeric_limits<std::size_t>::max() - output_count) / 4) {
            throw CudaOutOfMemory("CUDA NEP reduced prediction output is too large");
        }
        reduced_offset = output_count;
        output_count += original_atoms * 4;
    } else if (batch.atoms() != original.atoms) {
        throw std::invalid_argument("CUDA NEP batch does not match the original prediction batch");
    }

    double* output = context.output_buffer(output_count);
    double* descriptors_or_gradient = output;
    double* expanded_energy = output + descriptor_count;
    double* expanded_forces = expanded_energy + expanded_atoms;
    constexpr unsigned int block_size = 128;
    const unsigned int blocks = static_cast<unsigned int>(
        (expanded_atoms + block_size - 1) / block_size);

    compute_nep_kernel<<<blocks, block_size, 0, context.stream()>>>(
        batch.numbers(), graph.offsets(),
        graph.slot_major() ? graph.neighbor_counts() : nullptr,
        graph.neighbor_stride(), graph.atoms(), graph.displacements(),
        model.type_lookup(), model.num_types(), model.n_max_radial(),
        model.n_max_angular(), model.basis_size_radial(), model.basis_size_angular(),
        model.l_max(), model.has_q_222(), model.has_q_1111(), model.has_q_112(),
        model.has_q_123(), model.has_q_233(), model.has_q_134(), model.dimension(),
        model.radial_cutoff_pair(), model.angular_cutoff_pair(),
        model.radial_pair_coefficients(), model.angular_pair_coefficients(),
        model.scalers(), batch.atoms(), descriptors_or_gradient);
    check_cuda(cudaGetLastError(), "CUDA NEP prediction descriptor kernel launch failed");

    evaluate_nep_ann_kernel<<<blocks, block_size, 0, context.stream()>>>(
        batch.numbers(), model.type_lookup(), model.num_types(), model.version(),
        model.dimension(), model.hidden_neurons1(), model.hidden_neurons2(),
        model.ann_parameters(), batch.atoms(), descriptors_or_gradient, expanded_energy);
    check_cuda(cudaGetLastError(), "CUDA NEP ANN prediction kernel launch failed");

    check_cuda(cudaMemsetAsync(expanded_forces, 0,
        expanded_atoms * 3 * sizeof(double), context.stream()),
        "could not initialize CUDA NEP forces");
    compute_nep_prediction_forces_kernel<<<blocks, block_size, 0, context.stream()>>>(
        batch.numbers(), graph.offsets(),
        graph.slot_major() ? graph.neighbor_counts() : nullptr,
        graph.neighbor_stride(), graph.atoms(), graph.displacements(),
        model.type_lookup(), model.num_types(), model.n_max_radial(),
        model.n_max_angular(), model.basis_size_radial(), model.basis_size_angular(),
        model.l_max(), model.has_q_222(), model.has_q_1111(), model.has_q_112(),
        model.has_q_123(), model.has_q_233(), model.has_q_134(), model.dimension(),
        model.radial_cutoff_pair(), model.angular_cutoff_pair(),
        model.radial_pair_coefficients(), model.angular_pair_coefficients(),
        model.scalers(), model.zbl_enabled() ? 1 : 0, model.zbl_inner(), model.zbl_outer(),
        batch.atoms(), descriptors_or_gradient, expanded_energy, expanded_forces);
    check_cuda(cudaGetLastError(), "CUDA NEP prediction force kernel launch failed");

    const std::size_t output_atoms = batch.expanded() ? original_atoms : expanded_atoms;
    const std::size_t result_count = output_atoms * 4;
    std::vector<double> host_result(result_count);
    if (batch.expanded()) {
        double* reduced_output = output + reduced_offset;
        const auto reduction_blocks = static_cast<unsigned int>(
            (result_count + block_size - 1) / block_size);
        reduce_expanded_nep_prediction_kernel<<<
            reduction_blocks, block_size, 0, context.stream()>>>(
            original.atoms, batch.expansion_first(), batch.expansion_stride(),
            batch.expansion_replicas(), expanded_energy, expanded_forces, reduced_output);
        check_cuda(cudaGetLastError(), "CUDA NEP prediction replica reduction failed");
        context.download_output_slice_into(reduced_offset, host_result.data(), result_count);
    } else {
        context.download_output_slice_into(descriptor_count, host_result.data(), result_count);
    }

    for (std::int64_t atom = 0; atom < original.atoms; ++atom) {
        const std::size_t energy_index = batch.expanded()
            ? static_cast<std::size_t>(atom) * 4
            : static_cast<std::size_t>(atom);
        const std::size_t force_index = batch.expanded()
            ? static_cast<std::size_t>(atom) * 4 + 1
            : original_atoms + static_cast<std::size_t>(atom) * 3;
        atom_energy[atom] = host_result[energy_index];
        for (int axis = 0; axis < 3; ++axis) {
            forces[atom * 3 + axis] = host_result[force_index + axis];
        }
    }
    std::fill_n(energy, original.structures, 0.0);
    for (std::int64_t structure = 0; structure < original.structures; ++structure) {
        for (std::int64_t atom = original.offsets[structure];
             atom < original.offsets[structure + 1]; ++atom) {
            energy[structure] += atom_energy[atom];
        }
    }
}

} // namespace mdescriptor::cuda
