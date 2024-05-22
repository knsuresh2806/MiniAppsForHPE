
/**
author: Rahul S. Sampath
 */

#ifndef _ARRAY
#define _ARRAY

#include <cstddef>

template <typename T, size_t N>
struct Array
{
    T v[N];

    T& operator[](size_t pos) { return v[pos]; }

    const T& operator[](size_t pos) const { return v[pos]; }
};

template <typename T, size_t N>
bool
operator<(const Array<T, N>& x, const Array<T, N>& y)
{
    for (size_t i = 0; i < N; ++i) {
        if (x[i] < y[i]) {
            return true;
        }
        if (y[i] < x[i]) {
            return false;
        }
    } //end i
    return false;
}

template <typename T, size_t N>
bool
operator==(const Array<T, N>& x, const Array<T, N>& y)
{
    for (size_t i = 0; i < N; ++i) {
        if (x[i] != y[i]) {
            return false;
        }
    } //end i
    return true;
}

#endif
