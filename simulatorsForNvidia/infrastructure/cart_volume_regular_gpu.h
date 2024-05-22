#ifndef _CART_VOLUME_REGULAR_GPU
#define _CART_VOLUME_REGULAR_GPU

#include "axis.h"
#include "emsl_error.h"
#include "cart_volume.h"
#include "std_const.h"
#include "emsl_error.h"
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <new>
#include <stdint.h>
#include <limits>
#include "volume_index.h"
#include "cart_volume_regular_host.h"

class cart_volume_regular_host;

class cart_volume_regular_gpu : public cart_volume_regular
{

public:
    using T = float;
    // constructor
    cart_volume_regular_gpu() = delete;                                          // no default, please use axis
    cart_volume_regular_gpu(const cart_volume_regular_gpu&) = delete;            // no copy
    cart_volume_regular_gpu& operator=(const cart_volume_regular_gpu&) = delete; // no copy
    // constructor
    cart_volume_regular_gpu(const axis* ax1_, const axis* ax2_, const axis* ax3_, bool set_to_zero = false);
    virtual ~cart_volume_regular_gpu();
    volume_index vol_idx();

    T* getData();

    size_t getSize() const { return data_size; }
    void copyToData(const T* in);
    void copyAxis(const axis* ax1_, const axis* ax2_, const axis* ax3_);
    bool isAligned();
    void set_constant(float val, bool skipHalos = true, bool skipInterleaving = false);
    void getDims(int& n0, int& n1, int& n2, bool skipHalosAndPadding = false);
    void copyData(cart_volume<float>* vol, bool skipHalos = true, bool skipInterleaving = false);

    // Copy data to/from CPU cart volumes.
    // Note: Skipping halos is expensive, it's better to NOT skip halos.
    void copyFrom(cart_volume_regular_host* from, bool skipHalos = true);
    void copyTo(cart_volume_regular_host* to, bool skipHalos = true);
    void copyFrom(cart_volume_regular_gpu* from, bool skipHalos = true);
    // Asynchronous versions
    void copyFromAsync(cart_volume_regular_host* from, cudaStream_t stream, bool skipHalos = true);
    void copyToAsync(cart_volume_regular_host* to, cudaStream_t stream, bool skipHalos = true);
    void copyFromAsync(cart_volume_regular_gpu* from, cudaStream_t stream, bool skipHalos = true);

private:
    void allocate();
    float2* bMinMax;
    float2* hMinMax;
    uint2* hInfNanCount;
    T* data;
    size_t data_size;
};
#endif //_CART_VOLUME_REGULAR_GPU
