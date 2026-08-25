#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <stdexcept>

namespace mdescriptor {

class CancelledError : public std::runtime_error {
public:
    CancelledError() : std::runtime_error("descriptor computation cancelled") {}
};

class ComputeControl {
public:
    void reset(std::int64_t total);
    void cancel() noexcept;
    bool cancelled() const noexcept;
    std::int64_t completed() const noexcept;
    std::int64_t total() const noexcept;
    void mark_completed() noexcept;

private:
    std::atomic<bool> cancelled_{false};
    std::atomic<std::int64_t> completed_{0};
    std::atomic<std::int64_t> total_{0};
};

namespace detail {

inline bool cancelled(const std::shared_ptr<ComputeControl>& control) {
    return control && control->cancelled();
}

inline void check_cancelled(const std::shared_ptr<ComputeControl>& control) {
    if (cancelled(control)) {
        throw CancelledError();
    }
}

inline void mark_completed(const std::shared_ptr<ComputeControl>& control) {
    if (control) {
        control->mark_completed();
    }
}

} // namespace detail

} // namespace mdescriptor
