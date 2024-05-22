#ifndef HELPER_THREAD_SYNCHRONIZER_H
#define HELPER_THREAD_SYNCHRONIZER_H

#include <condition_variable>
#include <mutex>

/** 
 * Synchronizes a main thread and helper thread.
 * 
 * Synchronizes a main thread and helper thread by modeling a "token" that is
 * passed between the threads. Whichever thread has the token is safe to access
 * the shared state that this protects. The token starts with the main thread.
 * The operations available to a thread are wait and release, each prefixed with
 * either main_thread_ or helper_thread_. wait acquires the token, blocking if
 * needed until the other thread releases it. release releases the token to the
 * other thread. Calling wait when the token is already with the calling thread
 * is basically a no-op. Calling release when the token is not with the calling
 * thread will throw an exception. This class does not control access to the
 * shared state, which should only be accessed between a wait and release
 * function call.
 */
class helper_thread_synchronizer
{
public:
    helper_thread_synchronizer() = default;
    helper_thread_synchronizer(const helper_thread_synchronizer& other) = delete;
    helper_thread_synchronizer(helper_thread_synchronizer&& other) = default;
    helper_thread_synchronizer& operator=(const helper_thread_synchronizer& other) = delete;
    helper_thread_synchronizer& operator=(helper_thread_synchronizer&& other) = default;
    ~helper_thread_synchronizer() = default;

    void main_thread_wait();
    void main_thread_release();
    void helper_thread_wait();
    void helper_thread_release();

private:
    enum token_holder
    {
        main_thread,
        helper_thread
    } token_holder = main_thread;
    std::condition_variable cv;
    std::mutex mutex;
};

#endif
