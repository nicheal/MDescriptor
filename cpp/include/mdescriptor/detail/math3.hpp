#pragma once

#include <cmath>
#include <stdexcept>

namespace mdescriptor::detail {

// The descriptor kernels use this tiny value type instead of pulling a
// general-purpose linear algebra dependency into the extension.
struct Vec3 {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
};

inline Vec3 operator+(Vec3 left, Vec3 right) {
    return {left.x + right.x, left.y + right.y, left.z + right.z};
}

inline Vec3 operator-(Vec3 left, Vec3 right) {
    return {left.x - right.x, left.y - right.y, left.z - right.z};
}

inline Vec3 operator*(double scale, Vec3 value) {
    return {scale * value.x, scale * value.y, scale * value.z};
}

inline double dot(Vec3 left, Vec3 right) {
    return left.x * right.x + left.y * right.y + left.z * right.z;
}

inline double norm2(Vec3 value) { return dot(value, value); }
inline double norm(Vec3 value) { return std::sqrt(norm2(value)); }

struct Mat3 {
    double a[3][3]{};
};

inline Vec3 row(const Mat3& matrix, int index) {
    return {matrix.a[index][0], matrix.a[index][1], matrix.a[index][2]};
}

inline double determinant(const Mat3& matrix) {
    return matrix.a[0][0] * (matrix.a[1][1] * matrix.a[2][2] - matrix.a[1][2] * matrix.a[2][1])
        - matrix.a[0][1] * (matrix.a[1][0] * matrix.a[2][2] - matrix.a[1][2] * matrix.a[2][0])
        + matrix.a[0][2] * (matrix.a[1][0] * matrix.a[2][1] - matrix.a[1][1] * matrix.a[2][0]);
}

inline Mat3 inverse(const Mat3& matrix) {
    const double determinant_value = determinant(matrix);
    if (!std::isfinite(determinant_value) || std::abs(determinant_value) < 1e-14) {
        throw std::invalid_argument("cell matrix is singular");
    }
    Mat3 result;
    result.a[0][0] = (matrix.a[1][1] * matrix.a[2][2] - matrix.a[1][2] * matrix.a[2][1]) / determinant_value;
    result.a[0][1] = (matrix.a[0][2] * matrix.a[2][1] - matrix.a[0][1] * matrix.a[2][2]) / determinant_value;
    result.a[0][2] = (matrix.a[0][1] * matrix.a[1][2] - matrix.a[0][2] * matrix.a[1][1]) / determinant_value;
    result.a[1][0] = (matrix.a[1][2] * matrix.a[2][0] - matrix.a[1][0] * matrix.a[2][2]) / determinant_value;
    result.a[1][1] = (matrix.a[0][0] * matrix.a[2][2] - matrix.a[0][2] * matrix.a[2][0]) / determinant_value;
    result.a[1][2] = (matrix.a[0][2] * matrix.a[1][0] - matrix.a[0][0] * matrix.a[1][2]) / determinant_value;
    result.a[2][0] = (matrix.a[1][0] * matrix.a[2][1] - matrix.a[1][1] * matrix.a[2][0]) / determinant_value;
    result.a[2][1] = (matrix.a[0][1] * matrix.a[2][0] - matrix.a[0][0] * matrix.a[2][1]) / determinant_value;
    result.a[2][2] = (matrix.a[0][0] * matrix.a[1][1] - matrix.a[0][1] * matrix.a[1][0]) / determinant_value;
    return result;
}

} // namespace mdescriptor::detail
