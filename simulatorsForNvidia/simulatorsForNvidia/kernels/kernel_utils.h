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
getSpongeCoef(const sponge_coef3d& sponge_coef, int isub, const realtype*& Ax, const realtype*& Bx, const realtype*& Ay,
              const realtype*& By, const realtype*& Az, const realtype*& Bz)
{
    Ax = sponge_coef.x_A_[isub].data();
    Bx = sponge_coef.x_B_[isub].data();

    Ay = sponge_coef.y_A_[isub].data();
    By = sponge_coef.y_B_[isub].data();

    Az = sponge_coef.z_A_[isub].data();
    Bz = sponge_coef.z_B_[isub].data();
}

inline void
getSpongeCoef(const sponge_coef3d_gpu& sponge_coef, int isub, const float2*& ABx, const float2*& ABy,
              const float2*& ABz)
{
    EMSL_VERIFY(isub < sponge_coef.nsubs)
    ABx = sponge_coef.x_AB_[isub];
    ABy = sponge_coef.y_AB_[isub];
    ABz = sponge_coef.z_AB_[isub];
}

inline void
getPMLCoef(const pml_coef3d& pml_coef, int isub, const realtype*& Ax, const realtype*& Bx, const realtype*& Ay,
           const realtype*& By, const realtype*& Az, const realtype*& Bz)
{
    Ax = pml_coef.x_A_[isub].data();
    Bx = pml_coef.x_B_[isub].data();

    Ay = pml_coef.y_A_[isub].data();
    By = pml_coef.y_B_[isub].data();

    Az = pml_coef.z_A_[isub].data();
    Bz = pml_coef.z_B_[isub].data();
}

inline void
getStaggeredPMLCoef(const pml_coef3d& pml_coef, int isub, const realtype*& Axs, const realtype*& Bxs,
                    const realtype*& Ays, const realtype*& Bys, const realtype*& Azs, const realtype*& Bzs)
{
    Axs = pml_coef.xs_A_[isub].data();
    Bxs = pml_coef.xs_B_[isub].data();

    Ays = pml_coef.ys_A_[isub].data();
    Bys = pml_coef.ys_B_[isub].data();

    Azs = pml_coef.zs_A_[isub].data();
    Bzs = pml_coef.zs_B_[isub].data();
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
