#ifndef KERNEL_UTILS_H
#define KERNEL_UTILS_H

#include "axis.h"
#include "cart_volume.h"
#include "sponge_coef3d.h"
#include "pml_coef3d.h"

#include "cart_volume_regular_gpu.h"
#include "sponge_coef3d_gpu.h"
#include "pml_coef3d_gpu.h"

namespace kernel_utils {
/** Get local index range
     * @param vol: volume
     * @param ixbeg, ixend, iybeg, iyend, izbeg, izend: local index range (inclusive), not incuding halo
     */
inline void
getIndexRange(cart_volume<realtype>* vol_, int& ixbeg, int& ixend, int& iybeg, int& iyend, int& izbeg, int& izend)
{
    auto vol = vol_->as<cart_volume_regular>();
    ixbeg = vol->ax1()->ibeg;
    ixend = vol->ax1()->iend;
    iybeg = vol->ax2()->ibeg;
    iyend = vol->ax2()->iend;
    izbeg = vol->ax3()->ibeg;
    izend = vol->ax3()->iend;
}

inline void
compute_fd_const(double d1, double d2, double d3, realtype& invd1, realtype& invd2, realtype& invd3, realtype& invd12,
                 realtype& invd22, realtype& invd32, realtype& invDXDYBY4, realtype& invDYDZBY4, realtype& invDZDXBY4)
{
    invd1 = 1.0 / d1;
    invd2 = 1.0 / d2;
    invd3 = 1.0 / d3;

    invd12 = 1.0 / (d1 * d1);
    invd22 = 1.0 / (d2 * d2);
    invd32 = 1.0 / (d3 * d3);

    invDXDYBY4 = 1.0 / (4 * d1 * d2);
    invDYDZBY4 = 1.0 / (4 * d2 * d3);
    invDZDXBY4 = 1.0 / (4 * d1 * d3);
}

// The following function overloads ensure that if you pass a float for d1, d2, or d3, you will get a compile-time error
void compute_fd_const(float d1, float d2, float d3, float& invd1, float& invd2, float& invd3, float& invd12,
                      float& invd22, float& invd32, float& invDXDYBY4, float& invDYDZBY4, float& invDZDXBY4) = delete;
void compute_fd_const(float d1, float d2, double d3, float& invd1, float& invd2, float& invd3, float& invd12,
                      float& invd22, float& invd32, float& invDXDYBY4, float& invDYDZBY4, float& invDZDXBY4) = delete;
void compute_fd_const(float d1, double d2, float d3, float& invd1, float& invd2, float& invd3, float& invd12,
                      float& invd22, float& invd32, float& invDXDYBY4, float& invDYDZBY4, float& invDZDXBY4) = delete;
void compute_fd_const(float d1, double d2, double d3, float& invd1, float& invd2, float& invd3, float& invd12,
                      float& invd22, float& invd32, float& invDXDYBY4, float& invDYDZBY4, float& invDZDXBY4) = delete;
void compute_fd_const(double d1, float d2, float d3, float& invd1, float& invd2, float& invd3, float& invd12,
                      float& invd22, float& invd32, float& invDXDYBY4, float& invDYDZBY4, float& invDZDXBY4) = delete;
void compute_fd_const(double d1, float d2, double d3, float& invd1, float& invd2, float& invd3, float& invd12,
                      float& invd22, float& invd32, float& invDXDYBY4, float& invDYDZBY4, float& invDZDXBY4) = delete;
void compute_fd_const(double d1, double d2, float d3, float& invd1, float& invd2, float& invd3, float& invd12,
                      float& invd22, float& invd32, float& invDXDYBY4, float& invDYDZBY4, float& invDZDXBY4) = delete;

}

#endif
