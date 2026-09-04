#pragma once

#include <cstddef>
#include <string_view>

namespace mdescriptor::cuda {

// The implementation-defined registry owns both names and handlers.  These
// declarations are the small backend seam used by feature discovery and the
// dispatch translation unit; callers never maintain a second ordered list.
inline constexpr std::size_t kExtendedDescriptorCount = 21;

std::size_t extended_descriptor_index(std::string_view name) noexcept;
bool is_extended_descriptor(std::string_view name) noexcept;

} // namespace mdescriptor::cuda
