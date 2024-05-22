#ifndef VOLUME_INDEX_H
#define VOLUME_INDEX_H

#include <cuda_runtime_api.h>
#define HOST_DEVICE_FN __host__ __device__

struct volume_index
{
    volume_index() = delete; // we have to provide dimension when we create a volume_index
    HOST_DEVICE_FN volume_index(int _n0, int _n1 = 1, int _n2 = 1) : n0(_n0), n1(_n1), n2(_n2) {}
    int n0, n1, n2; // n0 is the fastest dimension, n2 is the slowest dimension
    // int n0_noPad;   // is the fastest dimension without any extension (original value)
    // it is not useful for now, but I (FH) am  leaving for  in case we change mind later
    template <typename T>
    HOST_DEVICE_FN T& operator()(T* data, int i0, int i1) const
    {                                       // 2D
        return data[(int64_t)i1 * n0 + i0]; // int64_t will make the index at 64 bit
    }
    template <typename T>
    HOST_DEVICE_FN T& operator()(T* data, int i0, int i1, int i2) const
    {                                                   // 3D
        return data[((int64_t)i2 * n1 + i1) * n0 + i0]; // int64_t will make the index at 64 bit
    }
    template <typename T>
    HOST_DEVICE_FN T& operator()(T* data, int i0, int i1, int i2, int i3) const
    {                                                               // 4D
        return data[(((int64_t)i3 * n2 + i2) * n1 + i1) * n0 + i0]; // int64_t will make the index at 64 bit
    }
};

#undef HOST_DEVICE_FN

#endif //VOLUME_INDEX_H
