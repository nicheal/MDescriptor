#pragma once

#include <cstdint>

namespace mdescriptor::detail {

// Keep exactly one orientation for a half neighbor list. Atom indices are
// the primary key; periodic images of the same atom need a deterministic
// signed-shift tie-break because neither index is smaller than the other.
#if defined(__CUDACC__)
#define MDESCRIPTOR_HOST_DEVICE __host__ __device__
#else
#define MDESCRIPTOR_HOST_DEVICE
#endif

MDESCRIPTOR_HOST_DEVICE inline bool keep_half_neighbor(
    std::int64_t center,
    std::int64_t atom,
    std::int32_t shift_x,
    std::int32_t shift_y,
    std::int32_t shift_z) noexcept {
    if (atom != center) {
        return atom > center;
    }
    if (shift_x != 0) {
        return shift_x > 0;
    }
    if (shift_y != 0) {
        return shift_y > 0;
    }
    if (shift_z != 0) {
        return shift_z > 0;
    }
    // The exact self pair is the canonical representative of the zero shift.
    return true;
}

#undef MDESCRIPTOR_HOST_DEVICE

} // namespace mdescriptor::detail
