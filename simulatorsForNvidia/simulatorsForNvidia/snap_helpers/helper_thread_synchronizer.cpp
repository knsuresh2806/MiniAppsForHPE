#include "helper_thread_synchronizer.h"

void
helper_thread_synchronizer::main_thread_wait()
{
    std::unique_lock<std::mutex> lock(mutex);
    cv.wait(lock, [this]() { return this->token_holder == main_thread; });
}

void
helper_thread_synchronizer::main_thread_release()
{
    std::lock_guard<std::mutex> lock(mutex);
    if (token_holder != main_thread) {
        throw std::logic_error("main thread tried to release helper_thread_synchronizer without owning");
    }
    token_holder = helper_thread;
    cv.notify_one();
}

void
helper_thread_synchronizer::helper_thread_wait()
{
    std::unique_lock<std::mutex> lock(mutex);
    cv.wait(lock, [this]() { return this->token_holder == helper_thread; });
}

void
helper_thread_synchronizer::helper_thread_release()
{
    std::lock_guard<std::mutex> lock(mutex);
    if (token_holder != helper_thread) {
        throw std::logic_error("helper thread tried to release helper_thread_synchronizer without owning");
    }
    token_holder = main_thread;
    cv.notify_one();
}
