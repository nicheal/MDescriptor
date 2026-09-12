#pragma once

#include <atomic>
#include <mutex>
#include <stdexcept>
#include <string>

namespace mdescriptor::detail {

// Shared lifecycle state for the calculator classes: a compute mutex plus the
// closed flag, with the release/acquire ordering every backend relies on.
class CalculatorBase {
public:
    void close() noexcept { closed_.store(true, std::memory_order_release); }
    bool closed() const noexcept { return closed_.load(std::memory_order_acquire); }

    // Throws the calculator's closed-state error, e.g.
    // assert_open("SOAP calculator") throws "SOAP calculator is closed".
    void assert_open(const char* name) const {
        if (closed()) {
            throw std::runtime_error(std::string(name) + " is closed");
        }
    }

protected:
    mutable std::mutex compute_mutex_;
    std::atomic<bool> closed_{false};
};

} // namespace mdescriptor::detail
