#include "mdescriptor/descriptor.hpp"

#include <stdexcept>

namespace mdescriptor {

void ComputeControl::reset(std::int64_t total) {
    if (total < 0) {
        throw std::invalid_argument("control total must be non-negative");
    }
    cancelled_.store(false, std::memory_order_release);
    completed_.store(0, std::memory_order_release);
    total_.store(total, std::memory_order_release);
}

void ComputeControl::cancel() noexcept { cancelled_.store(true, std::memory_order_release); }
bool ComputeControl::cancelled() const noexcept { return cancelled_.load(std::memory_order_acquire); }
std::int64_t ComputeControl::completed() const noexcept { return completed_.load(std::memory_order_acquire); }
std::int64_t ComputeControl::total() const noexcept { return total_.load(std::memory_order_acquire); }
void ComputeControl::mark_completed() noexcept { completed_.fetch_add(1, std::memory_order_acq_rel); }

} // namespace mdescriptor
