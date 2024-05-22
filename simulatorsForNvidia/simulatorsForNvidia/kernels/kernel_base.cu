#include "kernel_base.h"

#include "cuda_utils.h"
#include "helper_kernels_gpu.h"
#include "cart_volume_regular_gpu.h"
#include "kernel_utils.h"
#include "cart_volume.h"
#include "axis.h"
#include "volume_index.h"
#include "std_const.h"
#include "emsl_error.h"
#include "fd_stencil3d.h"
#include "fd_coef.h"

#include <algorithm>
#include <cmath>
#include <utility>

kernel_base::kernel_base(int sp_order, int bc_opt, int nsub, float dt, double d1, double d2, double d3,
                                   bool update_born, cart_volume<float>**& field1, cart_volume<float>**& field2,
                                   cart_volume<float>**& field3, cart_volume<float>**& field4, cart_volume<float>**& field11,
                                   cart_volume<float>**& field21, cart_volume<float>**& field31, cart_volume<float>**& field41,
                                   const sponge_coef3d_gpu& sponge_coef, std::vector<cart_volume<float>**>& Model8,
                                   std::vector<cart_volume<float>**>& Model9, std::vector<cart_volume<float>**>& Model81,
                                   std::vector<cart_volume<float>**>& Model91, std::vector<float>& wsi,
                                   std::vector<cart_volume<float>**>& Model10)
    : kernel_abstract_base{ sp_order, bc_opt }, sp_order{ sp_order }, nsub{ nsub }, vols_cast_done{ false }, dt{ dt }, d1{ d1 },
      d2{ d2 }, d3{ d3 }, update_born{ update_born }, field1_orig{ field1 }, field2_orig{ field2 }, field3_orig{ field3 }, field4_orig{ field4 },
      field11_orig{ field11 }, field21_orig{ field21 }, field31_orig{ field31 }, field41_orig{ field41 }, sponge_coef{ sponge_coef },
      Model8_orig{ Model8 }, Model9_orig{ Model9 }, Model81_orig{ Model81 }, Model91_orig{ Model91 }, wsi{ wsi }, Model10_orig{ Model10 }
{
    float unused;
    kernel_utils::compute_fd_const(d1, d2, d3, invd1, invd2, invd3, unused, unused, unused, unused, unused, unused);
}

void
kernel_base::ensure_volumes()
{
    if (!vols_cast_done) {
        auto cast_vols = [this](cart_volume<float>** in, std::vector<cart_volume_regular_gpu*>& out) {
            if (in) {
                out.resize(this->nsub, nullptr);
                std::transform(in, in + nsub, out.begin(),
                               [](cart_volume<float>* cv) { return cv ? cv->as<cart_volume_regular_gpu>() : nullptr; });
            }
        };

        cast_vols(field1_orig, field1);
        cast_vols(field2_orig, field2);
        cast_vols(field3_orig, field3);
        cast_vols(field4_orig, field4);
        if (update_born) {
            cast_vols(field11_orig, field11);
            cast_vols(field21_orig, field21);
            cast_vols(field31_orig, field31);
            cast_vols(field41_orig, field41);
        }

        auto cast_vols2 = [this, cast_vols](std::vector<cart_volume<float>**>& in,
                                            std::array<std::vector<cart_volume_regular_gpu*>, 3>& out) {
            EMSL_VERIFY(in.size() == 3);

            for (int i = 0; i < 3; ++i) {
                if (in[i]) {
                    cast_vols(in[i], out[i]);
                }
            }
        };

        cast_vols2(Model8_orig, Model8);
        cast_vols2(Model9_orig, Model9);
        cast_vols2(Model10_orig, Model10);
        if (update_born) {
            cast_vols2(Model81_orig, Model81);
            cast_vols2(Model91_orig, Model91);
        }

        vols_cast_done = true;
    }
}

__global__ __launch_bounds__(128) void kernel_loop2_inner_kernel(float* field1_, float* field2_, const float* field3_,
                                                                       const float* field4_, volume_index idx, int ixbeg,
                                                                       int ixend, int iybeg, int izbeg,
                                                                       const float2* ABx, const float2* ABy,
                                                                       const float2* ABz, const float dt)
{
    int ix = blockDim.x * blockIdx.x + threadIdx.x;
    int iy = blockIdx.y + iybeg;
    int iz = blockIdx.z + izbeg;

    if (ix <= ixend && ix >= ixbeg) {
        float A_yz = min(ABy[iy].x, ABz[iz].x);
        float B_yz = min(ABy[iy].y, ABz[iz].y);

        float A_xyz = min(ABx[ix].x, A_yz);
        float B_xyz = min(ABx[ix].y, B_yz);

        float resx = A_xyz * idx(field1_, ix, iy, iz) + B_xyz * idx(field3_, ix, iy, iz) * dt;
        float resz = A_xyz * idx(field2_, ix, iy, iz) + B_xyz * idx(field4_, ix, iy, iz) * dt;
        idx(field1_, ix, iy, iz) = resx;
        idx(field2_, ix, iy, iz) = resz;
    }
}

void
kernel_base::kernel_loop2_inner(int isub, cudaStream_t stream)
{
    float* field1_ = field1[isub]->getData();
    float* field2_ = field2[isub]->getData();

    float* field3_ = field3[isub]->getData();
    float* field4_ = field4[isub]->getData();

    int ixbeg, ixend, iybeg, iyend, izbeg, izend;
    kernel_utils::getIndexRange(field1[isub], ixbeg, ixend, iybeg, iyend, izbeg, izend);

    const float2 *ABx, *ABy, *ABz;
    kernel_utils::getSpongeCoef(sponge_coef, isub, ABx, ABy, ABz);

    dim3 threads128x1(128, 1, 1);
    dim3 blocks(ixend / threads128x1.x + 1, iyend - iybeg + 1, izend - izbeg + 1);

    volume_index idx = field1[isub]->vol_idx();
    kernel_loop2_inner_kernel<<<blocks, threads128x1, 0, stream>>>(field1_, field2_, field3_, field4_, idx, ixbeg, ixend, iybeg,
                                                                         izbeg, ABx, ABy, ABz, dt);

    if (update_born) {
        float* field11_ = field11[isub]->getData();
        float* field21_ = field21[isub]->getData();

        float* field31_ = field31[isub]->getData();
        float* field41_ = field41[isub]->getData();
        volume_index idx = field11[isub]->vol_idx();
        kernel_loop2_inner_kernel<<<blocks, threads128x1, 0, stream>>>(field11_, field21_, field31_, field41_, idx, ixbeg,
                                                                             ixend, iybeg, izbeg, ABx, ABy, ABz, dt);
    }

    CUDA_CHECK_ERROR(__FILE__, __LINE__); //  __cudaCheckError(__FILE __, __LINE__);
}

__global__ void
kernel_loop4_kernel(float* field1, float* field2, int stride, int izbeg)
{
    // Treating the whole XY plan as a 1D, mapped to CUDA's X dimension.
    int ixy = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.z;
    if (ixy >= stride)
        return;
    size_t index = iz * (size_t)stride + ixy;
    if (iz == izbeg) {
        field1[index] = 0.0f;
        field2[index] = 0.0f;

    } else {
        size_t index_mirror = (2 * izbeg - iz) * (size_t)stride + ixy;
        float resx = -field1[index_mirror];
        float resz = -field2[index_mirror];
        field1[index] = resx;
        field2[index] = resz;
    }
}

void
kernel_base::kernel_loop4_inner(int isub, cudaStream_t stream)
{
    // Total number of elements in a XY plan
    int stride = field1[isub]->ax1()->ntot * field1[isub]->ax2()->ntot;

    // Free surface location
    int izbeg = field1[isub]->ax3()->ibeg;

    // Each XY plan will be processed by the threads X dimension.
    // Launch (izbeg + 1) blocks.z to work on all the Z plans (includes the surface)
    dim3 threads(1024, 1, 1);
    dim3 blocks((stride - 1) / threads.x + 1, 1, izbeg + 1);

    float* field1_ = field1[isub]->getData();
    float* field2_ = field2[isub]->getData();
    kernel_loop4_kernel<<<blocks, threads, 0, stream>>>(field1_, field2_, stride, izbeg);

    if (update_born) {
        float* field11_ = field11[isub]->getData();
        float* field21_ = field21[isub]->getData();
        kernel_loop4_kernel<<<blocks, threads, 0, stream>>>(field11_, field21_, stride, izbeg);
    }
}

// kernel_loop1_inner kernel
// In order to properly align thread 0 with ix=0, we launch more threads in X
// to cover the lower X halo, and then get rid of these threads inside the kernel.
// All the pointers and parameters of size [NMECH] are passed per-value
// with the other kernel arguments as vecparam structures.
template <int nmech>
__global__ void
kernel_loop1_inner_kernel(helper_kernels_gpu::vecparam<const float*, nmech> Model8,
                                  helper_kernels_gpu::vecparam<const float*, nmech> Model9,
                                  helper_kernels_gpu::vecparam<float, nmech> wsi, float* field1, float* field2,
                                  const float* field3, const float* field4, const float2* ABx, const float2* ABy,
                                  const float2* ABz, const float dt, int ixbeg, int ixend, int iybeg, int izbeg,
                                  volume_index vol3d)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y + iybeg;
    int iz = blockIdx.z + izbeg;
    if (ix < ixbeg || ix > ixend)
        return;

    float A_xyz = min(ABx[ix].x, min(ABy[iy].x, ABz[iz].x));
    float B_xyz = min(ABx[ix].y, min(ABy[iy].y, ABz[iz].y));

    float rx = 0.0f;
    float rz = 0.0f;
    for (int imech = 0; imech < nmech; imech++) {
        float omega = wsi.value[imech];
        rx += omega * vol3d(Model8.value[imech], ix, iy, iz);
        rz += omega * vol3d(Model9.value[imech], ix, iy, iz);
    }
    float resx = A_xyz * vol3d(field1, ix, iy, iz) + B_xyz * dt * (vol3d(field3, ix, iy, iz) - rx);
    float resz = A_xyz * vol3d(field2, ix, iy, iz) + B_xyz * dt * (vol3d(field4, ix, iy, iz) - rz);
    vol3d(field1, ix, iy, iz) = resx;
    vol3d(field2, ix, iy, iz) = resz;
}

void
kernel_base::kernel_loop1_inner(int isub, cudaStream_t stream)
{
    // Structures to pass the [NMECH] parameters to the kernel
    helper_kernels_gpu::vecparam<const float*, nmech> Model8ptrs;
    helper_kernels_gpu::vecparam<const float*, nmech> Model9ptrs;
    helper_kernels_gpu::vecparam<float, nmech> wsivals;

    int ixbeg, ixend, iybeg, iyend, izbeg, izend;
    kernel_utils::getIndexRange(field1[isub], ixbeg, ixend, iybeg, iyend, izbeg, izend);
    volume_index vol3d = field1[isub]->vol_idx();

    dim3 threads(128, 1, 1); // Avoid wasting threads if nx is short
    dim3 blocks(ixend / threads.x + 1, (iyend - iybeg + 1), (izend - izbeg + 1));

    const float2 *ABx, *ABy, *ABz;
    kernel_utils::getSpongeCoef(sponge_coef, isub, ABx, ABy, ABz);

    // Populate the NMECH param structures
    for (int imech = 0; imech < nmech; imech++) {
        Model8ptrs.value[imech] = Model8[imech][isub]->getData();
        Model9ptrs.value[imech] = Model9[imech][isub]->getData();
        wsivals.value[imech] = wsi[imech];
    }
    float* field1_ = field1[isub]->getData();
    float* field2_ = field2[isub]->getData();
    const float* field3_ = field3[isub]->getData();
    const float* field4_ = field4[isub]->getData();
    kernel_loop1_inner_kernel<<<blocks, threads, 0, stream>>>(
        Model8ptrs, Model9ptrs, wsivals, field1_, field2_, field3_, field4_, ABx, ABy, ABz, dt, ixbeg, ixend, iybeg, izbeg, vol3d);

    if (update_born) {
        // Update the NMECH param structures
        for (int imech = 0; imech < nmech; imech++) {
            Model8ptrs.value[imech] = Model81[imech][isub]->getData();
            Model9ptrs.value[imech] = Model91[imech][isub]->getData();
        }
        float* field11_ = field11[isub]->getData();
        float* field21_ = field21[isub]->getData();
        const float* field31_ = field31[isub]->getData();
        const float* field41_ = field41[isub]->getData();
        kernel_loop1_inner_kernel<<<blocks, threads, 0, stream>>>(
            Model8ptrs, Model9ptrs, wsivals, field11_, field21_, field31_, field41_, ABx, ABy, ABz, dt, ixbeg, ixend, iybeg, izbeg, vol3d);
    }
}

// kernel_loop3_inner kernel
// In order to properly align thread 0 with ix=0, we launch more threads in X
// to cover the lower X halo, and then get rid of these threads inside the kernel.
// All the pointers and parameters of size [NMECH] are passed per-value
// with the other kernel arguments as vecparam structures.
// Computing both arrays if update_born is true to load Model10 arrays only once.
template <int nmech, bool update_born>
__global__ void
kernel_loop3_inner_kernel(helper_kernels_gpu::vecparam<float*, nmech> Model8,
                                          helper_kernels_gpu::vecparam<float*, nmech> Model9,
                                          helper_kernels_gpu::vecparam<float*, nmech> Model81,
                                          helper_kernels_gpu::vecparam<float*, nmech> Model91,
                                          helper_kernels_gpu::vecparam<const float*, nmech> Model10,
                                          helper_kernels_gpu::vecparam<float, nmech> wsi, const float* field3,
                                          const float* field4, const float* field31, const float* field41, const float dt,
                                          int ixbeg, int ixend, int iybeg, int izbeg, volume_index vol3d)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y + iybeg;
    int iz = blockIdx.z + izbeg;
    if (ix < ixbeg || ix > ixend)
        return;

    float field3val = vol3d(field3, ix, iy, iz);
    float field4val = vol3d(field4, ix, iy, iz);
    float field31val, field41val;
    if (update_born) {
        field31val = vol3d(field31, ix, iy, iz);
        field41val = vol3d(field41, ix, iy, iz);
    }
    for (int imech = 0; imech < nmech; imech++) {
        float omega = wsi.value[imech];
        float alpha = vol3d(Model10.value[imech], ix, iy, iz);

        float mxval = vol3d(Model8.value[imech], ix, iy, iz);
        float mzval = vol3d(Model9.value[imech], ix, iy, iz);
        float resx = mxval + dt * (alpha * field3val - omega * mxval);
        float resz = mzval + dt * (alpha * field4val - omega * mzval);
        float resx1, resz1;
        if (update_born) {
            float mx1val = vol3d(Model81.value[imech], ix, iy, iz);
            float mz1val = vol3d(Model91.value[imech], ix, iy, iz);
            resx1 = mx1val + dt * (alpha * field31val - omega * mx1val);
            resz1 = mz1val + dt * (alpha * field41val - omega * mz1val);
        }
        vol3d(Model8.value[imech], ix, iy, iz) = resx;
        vol3d(Model9.value[imech], ix, iy, iz) = resz;
        if (update_born) {
            vol3d(Model81.value[imech], ix, iy, iz) = resx1;
            vol3d(Model91.value[imech], ix, iy, iz) = resz1;
        }
    }
}

void
kernel_base::kernel_loop3_inner(int isub, cudaStream_t stream)
{
    // Structures to pass the [NMECH] parameters to the kernel
    helper_kernels_gpu::vecparam<float*, nmech> Model8ptrs;
    helper_kernels_gpu::vecparam<float*, nmech> Model9ptrs;
    helper_kernels_gpu::vecparam<float*, nmech> Model81ptrs;
    helper_kernels_gpu::vecparam<float*, nmech> Model91ptrs;
    helper_kernels_gpu::vecparam<const float*, nmech> Model10ptrs;
    helper_kernels_gpu::vecparam<float, nmech> wsivals;

    int ixbeg, ixend, iybeg, iyend, izbeg, izend;
    kernel_utils::getIndexRange(field1[isub], ixbeg, ixend, iybeg, iyend, izbeg, izend);
    volume_index vol3d = field1[isub]->vol_idx();

    dim3 threads(128, 1, 1); // Avoid wasting threads if nx is short
    dim3 blocks(ixend / threads.x + 1, (iyend - iybeg + 1), (izend - izbeg + 1));

    // Populate the NMECH param structures
    for (int imech = 0; imech < nmech; imech++) {
        Model8ptrs.value[imech] = Model8[imech][isub]->getData();
        Model9ptrs.value[imech] = Model9[imech][isub]->getData();
        Model81ptrs.value[imech] = update_born ? Model81[imech][isub]->getData() : nullptr;
        Model91ptrs.value[imech] = update_born ? Model91[imech][isub]->getData() : nullptr;
        Model10ptrs.value[imech] = Model10[imech][isub]->getData();
        wsivals.value[imech] = wsi[imech];
    }
    const float* field3_ = field3[isub]->getData();
    const float* field4_ = field4[isub]->getData();
    const float* field31_ = update_born ? field31[isub]->getData() : nullptr;
    const float* field41_ = update_born ? field41[isub]->getData() : nullptr;

    if (update_born)
        kernel_loop3_inner_kernel<nmech, true>
            <<<blocks, threads, 0, stream>>>(Model8ptrs, Model9ptrs, Model81ptrs, Model91ptrs, Model10ptrs, wsivals, field3_, field4_, field31_,
                                             field41_, dt, ixbeg, ixend, iybeg, izbeg, vol3d);
    else
        kernel_loop3_inner_kernel<nmech, false>
            <<<blocks, threads, 0, stream>>>(Model8ptrs, Model9ptrs, Model81ptrs, Model91ptrs, Model10ptrs, wsivals, field3_, field4_, field31_,
                                             field41_, dt, ixbeg, ixend, iybeg, izbeg, vol3d);
}

// adj_kernel_loop1_inner kernel
// In order to properly align thread 0 with ix=0, we launch more threads in X
// to cover the lower X halo, and then get rid of these threads inside the kernel.
// All the pointers and parameters of size [NMECH] are passed per-value
// with the other kernel arguments as vecparam structures.
//
template <int nmech>
__global__ void
adj_kernel_loop1_inner_kernel(helper_kernels_gpu::vecparam<const float*, nmech> Model8,
                                          helper_kernels_gpu::vecparam<const float*, nmech> Model9,
                                          helper_kernels_gpu::vecparam<const float*, nmech> Model10, float* field1,
                                          float* field2, const float* field3, const float* field4, const float2* ABx,
                                          const float2* ABy, const float2* ABz, const float dt, int ixbeg, int ixend,
                                          int iybeg, int izbeg, volume_index vol3d)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y + iybeg;
    int iz = blockIdx.z + izbeg;
    if (ix < ixbeg || ix > ixend)
        return;

    // Compute correction term
    float rx = 0.0f, rz = 0.0f;
    for (int imech = 0; imech < nmech; imech++) {
        float alpha = vol3d(Model10.value[imech], ix, iy, iz);
        rx += alpha * vol3d(Model8.value[imech], ix, iy, iz);
        rz += alpha * vol3d(Model9.value[imech], ix, iy, iz);
    }

    float A_xyz = min(ABx[ix].x, min(ABy[iy].x, ABz[iz].x));
    float B_xyz = min(ABx[ix].y, min(ABy[iy].y, ABz[iz].y));

    float resx = A_xyz * vol3d(field1, ix, iy, iz) + B_xyz * dt * (vol3d(field3, ix, iy, iz) + rx);
    float resz = A_xyz * vol3d(field2, ix, iy, iz) + B_xyz * dt * (vol3d(field4, ix, iy, iz) + rz);
    vol3d(field1, ix, iy, iz) = resx;
    vol3d(field2, ix, iy, iz) = resz;
}

void
kernel_base::adj_kernel_loop1_inner(int isub, cudaStream_t stream)
{
    // Structures to pass the [NMECH] parameters to the kernel
    helper_kernels_gpu::vecparam<const float*, nmech> Model8ptrs;
    helper_kernels_gpu::vecparam<const float*, nmech> Model9ptrs;
    helper_kernels_gpu::vecparam<const float*, nmech> Model10ptrs;

    int ixbeg, ixend, iybeg, iyend, izbeg, izend;
    kernel_utils::getIndexRange(field1[isub], ixbeg, ixend, iybeg, iyend, izbeg, izend);
    volume_index vol3d = field1[isub]->vol_idx();

    dim3 threads(128, 1, 1); // Avoid wasting threads if nx is short
    dim3 blocks(ixend / threads.x + 1, (iyend - iybeg + 1), (izend - izbeg + 1));

    const float2 *ABx, *ABy, *ABz; // those need to be allocated in gpu. not sure! fatmir
    kernel_utils::getSpongeCoef(sponge_coef, isub, ABx, ABy, ABz);

    // Populate the NMECH param structures
    for (int imech = 0; imech < nmech; imech++) {
        Model8ptrs.value[imech] = Model8[imech][isub]->getData();
        Model9ptrs.value[imech] = Model9[imech][isub]->getData();
        Model10ptrs.value[imech] = Model10[imech][isub]->getData();
    }
    float* field1_ = field1[isub]->getData();
    float* field2_ = field2[isub]->getData();
    const float* field3_ = field3[isub]->getData();
    const float* field4_ = field4[isub]->getData();

    adj_kernel_loop1_inner_kernel<<<blocks, threads, 0, stream>>>(
        Model8ptrs, Model9ptrs, Model10ptrs, field1_, field2_, field3_, field4_, ABx, ABy, ABz, dt, ixbeg, ixend, iybeg, izbeg, vol3d);
}

// adj_kernel_loop3_inner kernel
// In order to properly align thread 0 with ix=0, we launch more threads in X
// to cover the lower X halo, and then get rid of these threads inside the kernel.
// All the pointers and parameters of size [NMECH] are passed per-value
// with the other kernel arguments as vecparam structures.
//
//

template <int nmech>
__global__ void
adj_kernel_loop3_inner_kernel(helper_kernels_gpu::vecparam<float*, nmech> Model8,
                                                  helper_kernels_gpu::vecparam<float*, nmech> Model9,
                                                  helper_kernels_gpu::vecparam<float, nmech> wsi, const float* field3,
                                                  const float* field4, float dt, int ixbeg, int ixend, int iybeg, int izbeg,
                                                  volume_index vol3d)
{

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y + iybeg;
    int iz = blockIdx.z + izbeg;
    if (ix < ixbeg || ix > ixend)
        return;

    float field3val = vol3d(field3, ix, iy, iz);
    float field4val = vol3d(field4, ix, iy, iz);

    // Update memory variables
    for (int imech = 0; imech < nmech; imech++) {
        float omega = wsi.value[imech];
        float mxval = vol3d(Model8.value[imech], ix, iy, iz);
        float mzval = vol3d(Model9.value[imech], ix, iy, iz);
        float resx = mxval + dt * omega * (-field3val - mxval);
        float resz = mzval + dt * omega * (-field4val - mzval);
        vol3d(Model8.value[imech], ix, iy, iz) = resx;
        vol3d(Model9.value[imech], ix, iy, iz) = resz;
    }
}

void
kernel_base::adj_kernel_loop3_inner(int isub, cudaStream_t stream)
{
    // Structures to pass the [NMECH] parameters to the kernel
    helper_kernels_gpu::vecparam<float*, nmech> Model8ptrs;
    helper_kernels_gpu::vecparam<float*, nmech> Model9ptrs;
    helper_kernels_gpu::vecparam<float, nmech> wsivals;

    int ixbeg, ixend, iybeg, iyend, izbeg, izend;
    kernel_utils::getIndexRange(field1[isub], ixbeg, ixend, iybeg, iyend, izbeg, izend);
    volume_index vol3d = field1[isub]->vol_idx();

    dim3 threads(128, 1, 1); // Avoid wasting threads if nx is short
    dim3 blocks(ixend / threads.x + 1, (iyend - iybeg + 1), (izend - izbeg + 1));

    // Populate the NMECH param structures
    for (int imech = 0; imech < nmech; imech++) {
        Model8ptrs.value[imech] = Model8[imech][isub]->getData();
        Model9ptrs.value[imech] = Model9[imech][isub]->getData();
        wsivals.value[imech] = wsi[imech];
    }
    float* field1_ = field1[isub]->getData();
    float* field2_ = field2[isub]->getData();
    const float* field3_ = field3[isub]->getData();
    const float* field4_ = field4[isub]->getData();

    adj_kernel_loop3_inner_kernel<<<blocks, threads, 0, stream>>>(
        Model8ptrs, Model9ptrs, wsivals, field3_, field4_, dt, ixbeg, ixend, iybeg, izbeg, vol3d);
}

__global__ __launch_bounds__(1024) void setVsFromVpEpsDelta_kernel(float coeff, float* Vs, float* Vp, float* Eps,
                                                                   float* Delta, volume_index idx, int ixbeg, int ixend,
                                                                   int iybeg, int iyend, int izbeg, int izend)
{
}

std::pair<float, float>
kernel_base::setVsFromVpEpsDelta(cart_volume<float>* Vs_, cart_volume<float>* Vp_, cart_volume<float>* Epsilon_,
                                      cart_volume<float>* Delta_)
{
}
