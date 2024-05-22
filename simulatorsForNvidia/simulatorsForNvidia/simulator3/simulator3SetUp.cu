#include <math.h>
#include "simulator3SetUp.h"
#include "cart_volume_regular_gpu.h"
#include "file_snap_factory.h"
#include "timer.h"
#include <cuda_pipeline.h>
#include <cuda_awbarrier.h>
#include "utils.h"
#include "helper.h"
#include "coefficients.h"

#define CUCHK(call)                                                 \
{                                                                 \
    cudaError_t err = call;                                         \
    if (cudaSuccess != err)                                         \
    {                                                               \
      fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n", \
              __FILE__, __LINE__, cudaGetErrorString(err));         \
      fflush(stderr);                                               \
      exit(EXIT_FAILURE);                                           \
    }                                                               \
}
// ****************************************************************************
// Constant memory for the coefficients
__constant__ float DF_order8[4];
__constant__ float INT_S_O8[4];

inline void init_constant()
{
    const float df8[4] = {1.01f, 2.01f, 3.01f, 4.01f};
    const float s8[4] = {1.01f, 2.01f, 3.01f, 4.01f};
    CUCHK (cudaMemcpyToSymbol (DF_order8, df8, 4 * sizeof (float)));
    CUCHK (cudaMemcpyToSymbol (INT_S_O8, s8, 4 * sizeof (float)));
}

template <bool SHIFT, int STRIDE>
__device__ inline float drv1_8(float *p)
{
    constexpr int is = SHIFT ? -1 : 0;
    float drv = DF_order8[0] * (p[(is + 1) * STRIDE] - p[is * STRIDE]);
#pragma unroll
    for (int i = 1; i < 4; i++)
        drv += DF_order8[i] * (p[(is + i + 1) * STRIDE] - p[(is - i) * STRIDE]);
    return drv;
}

template <int XS, int YS, int ZS, int TILE_X, int TILE_Y, int TILE_Z>
__launch_bounds__(TILE_X *TILE_Y)
    __global__ void precompute_xz_yz(const float *__restrict__ input,
                                     float *__restrict__ drvxz,
                                     float *__restrict__ drvyz,
                                     int ldimx, int ldimy, int nz)
{
    __shared__ float sm[2][TILE_Y + 8][TILE_X + 8];
    // Partial contributions register queues
    float rqxz[8];
    float rqyz[8];

    const int stride = ldimx * ldimy;

    const int tx = threadIdx.x + 4;
    const int ty = threadIdx.y + 4;
    const int ix = blockIdx.x * TILE_X + tx;
    const int iy = blockIdx.y * TILE_Y + ty;
    const int iz = blockIdx.z * TILE_Z;

    int index = ((iz + 4) * ldimy + iy) * ldimx + ix;
    int index_ld = index - (3 + ZS) * stride;

    for (int i = 0; i < 8; i++)
    {
        rqxz[i] = 0.0f;
        rqyz[i] = 0.0f;
    }

    const int X1 = -3 - XS;
    const int Y1 = -3 - YS;
    const int X2 = 4 - XS;
    const int Y2 = 4 - YS;

    // Preload the first shared memory plan
    __pipeline_memcpy_async(&sm[0][ty][tx], &input[index_ld], sizeof(float));
    if (threadIdx.x < 4)
    {
        __pipeline_memcpy_async(&sm[0][ty][threadIdx.x], &input[index_ld - 4], sizeof(float));
        __pipeline_memcpy_async(&sm[0][ty][tx + TILE_X], &input[index_ld + TILE_X], sizeof(float));
    }
    if (threadIdx.y < 4)
    {
        __pipeline_memcpy_async(&sm[0][threadIdx.y][tx], &input[index_ld - 4 * ldimx], sizeof(float));
        __pipeline_memcpy_async(&sm[0][ty + TILE_Y][tx], &input[index_ld + TILE_Y * ldimx], sizeof(float));
    }
    __pipeline_commit();
    index_ld += stride;

    int k = 1; // Shared memory index flip-flop between 0 and 1. Loaded in 0, now ready to load 1.

    // Prime the partial contributions
#pragma unroll
    for (int i = 0; i < 7; i++)
    {
        // Preload the input data for the next iteration
        __pipeline_memcpy_async(&sm[k][ty][tx], &input[index_ld], sizeof(float));
        if (threadIdx.x < 4)
        {
            __pipeline_memcpy_async(&sm[k][ty][threadIdx.x], &input[index_ld - 4], sizeof(float));
            __pipeline_memcpy_async(&sm[k][ty][tx + TILE_X], &input[index_ld + TILE_X], sizeof(float));
        }
        if (threadIdx.y < 4)
        {
            __pipeline_memcpy_async(&sm[k][threadIdx.y][tx], &input[index_ld - 4 * ldimx], sizeof(float));
            __pipeline_memcpy_async(&sm[k][ty + TILE_Y][tx], &input[index_ld + TILE_Y * ldimx], sizeof(float));
        }
        index_ld += stride;
        __pipeline_commit();

        // Switch to the shared memory for the current iteration
        k ^= 1;

        // Wait for the previous async stage to finish
        __pipeline_wait_prior(1);
        __syncthreads();

        // Update the partial contributions
#pragma unroll
        for (int j = 0; j <= i; j++)
        {
            int icoef = max(3 - i, i - 4);
            rqxz[j] += INT_S_O8[icoef] * (sm[k][ty][tx + X1 + i - j] + sm[k][ty][tx + X2 - i + j]);
            rqyz[j] += INT_S_O8[icoef] * (sm[k][ty + Y1 + i - j][tx] + sm[k][ty + Y2 - i + j][tx]);
        }
        __syncthreads();
    }

    int nzloop = min(TILE_Z, nz - iz);

    // Loop on all the Z block
    for (int zloop = 0; zloop < nzloop; zloop++)
    {
        if (zloop < nzloop - 1)
        {
            // Preload the input data for the next iteration
            __pipeline_memcpy_async(&sm[k][ty][tx], &input[index_ld], sizeof(float));
            if (threadIdx.x < 4)
            {
                __pipeline_memcpy_async(&sm[k][ty][threadIdx.x], &input[index_ld - 4], sizeof(float));
                __pipeline_memcpy_async(&sm[k][ty][tx + TILE_X], &input[index_ld + TILE_X], sizeof(float));
            }
            if (threadIdx.y < 4)
            {
                __pipeline_memcpy_async(&sm[k][threadIdx.y][tx], &input[index_ld - 4 * ldimx], sizeof(float));
                __pipeline_memcpy_async(&sm[k][ty + TILE_Y][tx], &input[index_ld + TILE_Y * ldimx], sizeof(float));
            }
            index_ld += stride;
            __pipeline_commit();
        }

        // Switch to the shared memory for the current iteration
        k ^= 1;
        __pipeline_wait_prior(1);
        __syncthreads();

        // Update the partial contributions
#pragma unroll
        for (int i = 0; i < 8; i++)
        {
            int icoef = max(3 - i, i - 4);
            rqxz[i] += INT_S_O8[icoef] * (sm[k][ty][tx + X1 + i] + sm[k][ty][tx + X2 - i]);
            rqyz[i] += INT_S_O8[icoef] * (sm[k][ty + Y1 + i][tx] + sm[k][ty + Y2 - i][tx]);
        }
        __syncthreads();

        // The oldest partial values are now complete. Store the results
        drvxz[index] = 0.5f * rqxz[0];
        drvyz[index] = 0.5f * rqyz[0];
        index += stride;

        // Rotate the register queues, and zero the newest partial values
        for (int i = 0; i < 7; i++)
        {
            rqxz[i] = rqxz[i + 1];
            rqyz[i] = rqyz[i + 1];
        }
        rqxz[7] = 0.0f;
        rqyz[7] = 0.0f;
    }
}

template <int TILE_X, int TILE_Y>
__device__ inline void loadShared(float *sm, const int &smindex,
                                  const float *data, const int &index, const int &ldimx)
{
    constexpr int smdimx = TILE_X + 8;
    sm[smindex] = data[index];
    if (threadIdx.x < 4)
    {
        sm[smindex - 4] = data[index - 4];
        sm[smindex + TILE_X] = data[index + TILE_X];
    }
    if (threadIdx.y < 4)
    {
        sm[smindex - 4 * smdimx] = data[index - 4 * ldimx];
        sm[smindex + TILE_Y * smdimx] = data[index + TILE_Y * ldimx];
    }
    if (threadIdx.x < 4 && threadIdx.y < 4)
    {
        sm[smindex - 4 * smdimx - 4] = data[index - 4 * ldimx - 4];
        sm[smindex - 4 * smdimx + TILE_X] = data[index - 4 * ldimx + TILE_X];
        sm[smindex + TILE_Y * smdimx - 4] = data[index + TILE_Y * ldimx - 4];
        sm[smindex + TILE_Y * smdimx + TILE_X] = data[index + TILE_Y * ldimx + TILE_X];
    }
}

template <bool XS, bool YS, int STRIDE_Y>
__device__ inline float mixed_XY(float *sm)
{
    constexpr int xs = XS ? -1 : 0;
    constexpr int ys = YS ? -1 : 0;
    float drv = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; i++)
        drv += INT_S_O8[i] * (sm[(ys + i + 1) * STRIDE_Y + xs + i + 1] +
                               sm[(ys + i + 1) * STRIDE_Y + xs - i] +
                               sm[(ys - i) * STRIDE_Y + xs + i + 1] +
                               sm[(ys - i) * STRIDE_Y + xs - i]);

    return (0.5f * drv);
}

__global__ void
init_cijs_kernel(float* M1, float* M2, float* M3, float* M4, float* M5, float* M6, float* M7,
                 float* M8, float* M9, float* M10, float* M11, float* M12, float* M13, float* M14,
                 float* M15, float* M16, float* M17, float* M18, float* M19, float* M20, float* M21, 
                 float* M22, float* d4, float* d5, float* d6, float* d7, float* d8, float* d9,
                 float* zprime, int ixbeg, int ixend, int iybeg, int iyend, int izbeg, int izend, volume_index idx)
{
    int ix = blockDim.x * blockIdx.x + threadIdx.x;
    int iy = blockDim.y * blockIdx.y + threadIdx.y;
    int iz = blockDim.z * blockIdx.z + threadIdx.z;

    /// becasue use 128x1x1 no need check y and z
    if (ix >= ixbeg && ix <= ixend) { //&& iy >= iybeg && iy <= iyend && iz >= izbeg && iz <= izend) {
        idx(M1, ix, iy, iz) = cosf(2.0f * iz + iy + ix);
        idx(M2, ix, iy, iz) = sinf(2.0f * iz + iy + ix);
        idx(M3, ix, iy, iz) = cosf(iz + 2.0f * iy + ix);
        idx(M4, ix, iy, iz) = sinf(iz + 2.0f * iy + ix);
        idx(M5, ix, iy, iz) = cosf(iz + iy + 2.0f * ix);
        idx(M6, ix, iy, iz) = sinf(iz + iy + 2.0f * ix);
        idx(M7, ix, iy, iz) = cosf(3.0f * iz + iy + ix);
        idx(M8, ix, iy, iz) = sinf(3.0f * iz + iy + ix);
        idx(M9, ix, iy, iz) = cosf(iz + 3.0f * iy + ix);
        idx(M10, ix, iy, iz) = sinf(iz + 3.0f * iy + ix);
        idx(M11, ix, iy, iz) = cosf(iz + iy + 3.0f * ix);
        idx(M12, ix, iy, iz) = sinf(iz + iy + 3.0f * ix);
        idx(M13, ix, iy, iz) = cosf(4.0f * iz + iy + ix);
        idx(M14, ix, iy, iz) = sinf(4.0f * iz + iy + ix);
        idx(M15, ix, iy, iz) = cosf(iz + 4.0f * iy + ix);
        idx(M16, ix, iy, iz) = sinf(iz + 4.0f * iy + ix);
        idx(M17, ix, iy, iz) = cosf(iz + iy + 4.0f * ix);
        idx(M18, ix, iy, iz) = sinf(iz + iy + 4.0f * ix);
        idx(M19, ix, iy, iz) = cosf(5.0f * iz + iy + ix);
        idx(M20, ix, iy, iz) = sinf(5.0f * iz + iy + ix);
        idx(M21, ix, iy, iz) = cosf(iz + 5.0f * iy + ix);
        idx(M22, ix, iy, iz) = sinf(iz + 5.0f * iy + ix);
        idx(d4, ix, iy, iz) = cosf(iz + iy + 5.0f * ix);
        idx(d5, ix, iy, iz) = sinf(iz + iy + 5.0f * ix);
        idx(d6, ix, iy, iz) = cosf(6.0f * iz + iy + ix);
        idx(d7, ix, iy, iz) = sinf(6.0f * iz + iy + ix);
        idx(d8, ix, iy, iz) = cosf(iz + 6.0f * iy + ix);
        idx(d9, ix, iy, iz) = sinf(iz + 6.0f * iy + ix);
        zprime[iz] = cosf(6.0f * iz);
    }
}

__global__ void
compute_p_kernel(float* p, float* f4, float* f5, float* f6, float* f7, float* f8, 
                        float* f9, int xtot, int ytot, int ztot, volume_index idx)
{
    int ix = blockDim.x * blockIdx.x + threadIdx.x;
    int iy = blockDim.y * blockIdx.y + threadIdx.y;
    int iz = blockDim.z * blockIdx.z + threadIdx.z;

    /// becasue use 128x1x1 no need check y and z
    if (ix < xtot) { // && iy < ytot && iz < ztot) {
        idx(p, ix, iy, iz) = -(2.0f * idx(f4, ix, iy, iz) + 3.0f * idx(f5, ix, iy, iz) + 4.0f * idx(f6, ix, iy, iz)
                                    + 5.0f * idx(f7, ix, iy, iz) + 6.0f * idx(f8, ix, iy, iz) + 7.0f * idx(f9, ix, iy, iz)) / 27.0f;
    }
}

__global__ void
init_src_kernel(float* f4, float* f5, float* f6, float* f7, float* f8, float* f9, 
                int x, int y, int z, float press, volume_index idx)
{
    idx(f4, x, y, z) = press * 1.0f;
    idx(f5, x, y, z) = press * 2.0f;
    idx(f6, x, y, z) = press * 3.0f;
    idx(f7, x, y, z) = press * 4.0f;
    idx(f8, x, y, z) = press * 5.0f;
    idx(f9, x, y, z) = press * 6.0f;
}

simulator3SetUp::simulator3SetUp(int npoints, int npml, int niter, int xcorr_step, int nzones[3], int ranks[3], int radius_lo, int radius_hi, float bitPerFloat, int fwd_only, MPI_Comm communicator):
    _npoints {npoints},
    _npml {npml},
    _niter {niter},
    scratch_folder {"test_snapshot_area"},
    _xcorr_step {xcorr_step},
    _bitPerFloat {bitPerFloat},
    _fwd_only {fwd_only}
{
    EMSL_VERIFY(nzones[0] >= 1 && nzones[1] >= 1 && nzones[2] >= 1);
    EMSL_VERIFY(ranks[0] >= 1 && ranks[1] >= 1 && ranks[2] >= 1);

    _comm = communicator;
    MPI_Comm_rank(_comm, &_rank);
    r_lo = radius_lo;
    r_hi = radius_hi;
    radius = std::max(radius_lo, radius_hi);

    // Define the number of points in each zone
    for (int i = 0; i < 3; i++) {
        int sum = 0;
        numPts[i] = new int[nzones[i]];
        for (int j = 0; j < nzones[i]; j++) {
            numPts[i][j] = get_n_for_zone(j, nzones[i]);
            sum += numPts[i][j];
        }
        n_total[i] = sum;
    }

    numZonesinX = nzones[0];
    numZonesinY = nzones[1];
    numZonesinZ = nzones[2];

    // Costs are all the same = 1
    costs = new float**[numZonesinZ];
    for (int i = 0; i < numZonesinZ; ++i) {
        costs[i] = new float*[numZonesinY];
        for (int j = 0; j < numZonesinY; ++j) {
            costs[i][j] = new float[numZonesinX];
            for (int k = 0; k < numZonesinX; ++k)
                costs[i][j][k] = 1.0f;
        }
    }

    // Same stencil width for all 3D but allow staggered
    for (int axis = 0; axis < 3; ++axis) {
        for (int sign = 0; sign < 2; ++sign) {
            stencilWidths[axis][sign] = new int**[numZonesinZ];
            int r = sign ? radius_lo : radius_hi;
            for (int z = 0; z < numZonesinZ; ++z) {
                stencilWidths[axis][sign][z] = new int*[numZonesinY];
                for (int y = 0; y < numZonesinY; ++y) {
                    stencilWidths[axis][sign][z][y] = new int[numZonesinX];
                    for (int x = 0; x < numZonesinX; ++x)
                        stencilWidths[axis][sign][z][y][x] = r;
                }
            }
        }
    }
    // Create a decomposition manager
    decompmgr = new decompositionManager3D(nzones, numPts, costs, stencilWidths, ranks, communicator, 0);

    float dx = 1.0f;
    float dy = 1.0f;
    float dz = 1.0f;
    // no halo
    axgiX = new axis(0, dx, _npoints);
    axgiY = new axis(0, dy, _npoints);
    axgiZ = new axis(0, dz, _npoints);
    //Global axis including boundaries
    axgX = new axis(axgiX->o - _npml * axgiX->d, axgiX->e + _npml * axgiX->d, axgiX->d);
    axgY = new axis(axgiY->o - _npml * axgiY->d, axgiY->e + _npml * axgiY->d, axgiY->d);
    axgZ = new axis(axgiZ->o - _npml * axgiZ->d, axgiZ->e + _npml * axgiZ->d, axgiZ->d);

    // Get the  number of subdomains for each dimension
    int nsubs_per_dim[3];
    decompmgr->getSplitNumLocalSubDom(nsubs_per_dim);

    // For each subdomain of each dimension, create an axis
    local_X_axes.resize(nsubs_per_dim[0], nullptr);
    local_X_axes_noHalo.resize(nsubs_per_dim[0], nullptr);
    local_Y_axes.resize(nsubs_per_dim[1], nullptr);
    local_Y_axes_noHalo.resize(nsubs_per_dim[1], nullptr);
    local_Z_axes.resize(nsubs_per_dim[2], nullptr);
    local_Z_axes_noHalo.resize(nsubs_per_dim[2], nullptr);

    for (int isub = 0; isub < nsubs_per_dim[0]; ++isub) // num sub domains in X axis
    {
        int offset = decompmgr->getOffset(isub, 0); // 0 -- X axis
        float origin = axgX->o + (float(offset) * axgX->d);
        int nxloc = decompmgr->getNumPtsSplit(isub, 0);
        local_X_axes[isub] =
            new axis(origin, dx, nxloc, radius, AlignmentElem(AlignMemBytes::CACHELINE, sizeof(float)));
        //for each subdomain of cpu axis without halo.
        local_X_axes_noHalo[isub] = new axis(origin, dx, nxloc);
    }

    for (int isub = 0; isub < nsubs_per_dim[1]; ++isub) // num sub domains in Y axis
    {
        int offset = decompmgr->getOffset(isub, 1); // 1 -- Y axis
        float origin = axgY->o + (float(offset) * axgY->d);
        int nyloc = decompmgr->getNumPtsSplit(isub, 1);
        local_Y_axes[isub] = new axis(origin, dy, nyloc, radius);
        local_Y_axes_noHalo[isub] = new axis(origin, dy, nyloc);
    }

    for (int isub = 0; isub < nsubs_per_dim[2]; ++isub) // num sub domains in Z axis
    {
        int offset = decompmgr->getOffset(isub, 2); // 2 -- Z axis
        float origin = axgZ->o + (float(offset) * axgZ->d);
        int nzloc = decompmgr->getNumPtsSplit(isub, 2);
        local_Z_axes[isub] = new axis(origin, dz, nzloc, radius);
        local_Z_axes_noHalo[isub] = new axis(origin, dz, nzloc);
    }

    // Total number of subvolumes
    nsubs = nsubs_per_dim[0] * nsubs_per_dim[1] * nsubs_per_dim[2];
   
    p_gpu = new cart_volume<float>*[nsubs];
    g1 = new cart_volume<float>*[nsubs];
    g2 = new cart_volume<float>*[nsubs];

    /*****************eTTI cart volumes************/

    f1 = new cart_volume<float>*[nsubs];
    f2 = new cart_volume<float>*[nsubs];
    f3 = new cart_volume<float>*[nsubs];
    f4 = new cart_volume<float>*[nsubs];
    f5 = new cart_volume<float>*[nsubs];
    f6 = new cart_volume<float>*[nsubs];
    f7 = new cart_volume<float>*[nsubs];
    f8 = new cart_volume<float>*[nsubs];
    f9 = new cart_volume<float>*[nsubs];
    M1 = new cart_volume<float>*[nsubs];
    M2 = new cart_volume<float>*[nsubs];
    M3 = new cart_volume<float>*[nsubs];
    M4 = new cart_volume<float>*[nsubs];
    M5 = new cart_volume<float>*[nsubs];
    M6 = new cart_volume<float>*[nsubs];
    M7 = new cart_volume<float>*[nsubs];
    M8 = new cart_volume<float>*[nsubs];
    M9 = new cart_volume<float>*[nsubs];
    M10 = new cart_volume<float>*[nsubs];
    M11 = new cart_volume<float>*[nsubs];
    M12 = new cart_volume<float>*[nsubs];
    M13 = new cart_volume<float>*[nsubs];
    M14 = new cart_volume<float>*[nsubs];
    M15 = new cart_volume<float>*[nsubs];
    M16 = new cart_volume<float>*[nsubs];
    M17 = new cart_volume<float>*[nsubs];
    M18 = new cart_volume<float>*[nsubs];
    M19 = new cart_volume<float>*[nsubs];
    M20 = new cart_volume<float>*[nsubs];
    M21 = new cart_volume<float>*[nsubs];
    M22 = new cart_volume<float>*[nsubs];
    d4 = new cart_volume<float>*[nsubs];
    d5 = new cart_volume<float>*[nsubs];
    d6 = new cart_volume<float>*[nsubs];
    d7 = new cart_volume<float>*[nsubs];
    d8 = new cart_volume<float>*[nsubs];
    d9 = new cart_volume<float>*[nsubs];
    f10 = new cart_volume<float>*[nsubs];
    f11 = new cart_volume<float>*[nsubs];
    f12 = new cart_volume<float>*[nsubs];
    f13 = new cart_volume<float>*[nsubs];
    f14 = new cart_volume<float>*[nsubs];
    f15 = new cart_volume<float>*[nsubs];
    f16 = new cart_volume<float>*[nsubs];
    f17 = new cart_volume<float>*[nsubs];
    f18 = new cart_volume<float>*[nsubs];
    f19 = new cart_volume<float>*[nsubs];
    f20 = new cart_volume<float>*[nsubs];
    f21 = new cart_volume<float>*[nsubs];
    zprime = new float*[nsubs];

    snap_f1.resize(nsubs, nullptr);
    snap_f2.resize(nsubs, nullptr);
    snap_f3.resize(nsubs, nullptr);
    snap_f4.resize(nsubs, nullptr);
    snap_f5.resize(nsubs, nullptr);
    snap_f6.resize(nsubs, nullptr);
    snap_f7.resize(nsubs, nullptr);
    snap_f8.resize(nsubs, nullptr);
    snap_f9.resize(nsubs, nullptr);

    //create and initialize cart vols
    for (int isub = 0; isub < nsubs; isub++) {
        int subid[3];
        decompmgr->getSplitLocalSubDomID(isub, subid);
        axis* ax1 = local_X_axes[subid[0]];
        axis* ax2 = local_Y_axes[subid[1]];
        axis* ax3 = local_Z_axes[subid[2]];

        // creat cart_vols and set 0
        p_gpu[isub] = new cart_volume_regular_gpu(ax1, ax2, ax3, true);
        
        f1[isub] = new cart_volume_regular_gpu(ax1, ax2, ax3, true);
        f2[isub] = new cart_volume_regular_gpu(ax1, ax2, ax3, true);
        f3[isub] = new cart_volume_regular_gpu(ax1, ax2, ax3, true);
        f4[isub] = new cart_volume_regular_gpu(ax1, ax2, ax3, true);
        f5[isub] = new cart_volume_regular_gpu(ax1, ax2, ax3, true);
        f6[isub] = new cart_volume_regular_gpu(ax1, ax2, ax3, true);
        f7[isub] = new cart_volume_regular_gpu(ax1, ax2, ax3, true);
        f8[isub] = new cart_volume_regular_gpu(ax1, ax2, ax3, true);
        f9[isub] = new cart_volume_regular_gpu(ax1, ax2, ax3, true);
        M1[isub] = new cart_volume_regular_gpu(ax1, ax2, ax3, true);
        M2[isub] = new cart_volume_regular_gpu(ax1, ax2, ax3, true);
        M3[isub] = new cart_volume_regular_gpu(ax1, ax2, ax3, true);
        M4[isub] = new cart_volume_regular_gpu(ax1, ax2, ax3, true);
        M5[isub] = new cart_volume_regular_gpu(ax1, ax2, ax3, true);
        M6[isub] = new cart_volume_regular_gpu(ax1, ax2, ax3, true);
        M7[isub] = new cart_volume_regular_gpu(ax1, ax2, ax3, true);
        M8[isub] = new cart_volume_regular_gpu(ax1, ax2, ax3, true);
        M9[isub] = new cart_volume_regular_gpu(ax1, ax2, ax3, true);
        M10[isub] = new cart_volume_regular_gpu(ax1, ax2, ax3, true);
        M11[isub] = new cart_volume_regular_gpu(ax1, ax2, ax3, true);
        M12[isub] = new cart_volume_regular_gpu(ax1, ax2, ax3, true);
        M13[isub] = new cart_volume_regular_gpu(ax1, ax2, ax3, true);
        M14[isub] = new cart_volume_regular_gpu(ax1, ax2, ax3, true);
        M15[isub] = new cart_volume_regular_gpu(ax1, ax2, ax3, true);
        M16[isub] = new cart_volume_regular_gpu(ax1, ax2, ax3, true);
        M17[isub] = new cart_volume_regular_gpu(ax1, ax2, ax3, true);
        M18[isub] = new cart_volume_regular_gpu(ax1, ax2, ax3, true);
        M19[isub] = new cart_volume_regular_gpu(ax1, ax2, ax3, true);
        M20[isub] = new cart_volume_regular_gpu(ax1, ax2, ax3, true);
        M21[isub] = new cart_volume_regular_gpu(ax1, ax2, ax3, true);
        M22[isub] = new cart_volume_regular_gpu(ax1, ax2, ax3, true);
        d4[isub] = new cart_volume_regular_gpu(ax1, ax2, ax3, true);
        d5[isub] = new cart_volume_regular_gpu(ax1, ax2, ax3, true);
        d6[isub] = new cart_volume_regular_gpu(ax1, ax2, ax3, true);
        d7[isub] = new cart_volume_regular_gpu(ax1, ax2, ax3, true);
        d8[isub] = new cart_volume_regular_gpu(ax1, ax2, ax3, true);
        d9[isub] = new cart_volume_regular_gpu(ax1, ax2, ax3, true);
        f10[isub] = new cart_volume_regular_gpu(ax1, ax2, ax3, true);
        f11[isub] = new cart_volume_regular_gpu(ax1, ax2, ax3, true);
        f12[isub] = new cart_volume_regular_gpu(ax1, ax2, ax3, true);
        f13[isub] = new cart_volume_regular_gpu(ax1, ax2, ax3, true);
        f14[isub] = new cart_volume_regular_gpu(ax1, ax2, ax3, true);
        f15[isub] = new cart_volume_regular_gpu(ax1, ax2, ax3, true);
        f16[isub] = new cart_volume_regular_gpu(ax1, ax2, ax3, true);
        f17[isub] = new cart_volume_regular_gpu(ax1, ax2, ax3, true);
        f18[isub] = new cart_volume_regular_gpu(ax1, ax2, ax3, true);
        f19[isub] = new cart_volume_regular_gpu(ax1, ax2, ax3, true);
        f20[isub] = new cart_volume_regular_gpu(ax1, ax2, ax3, true);
        f21[isub] = new cart_volume_regular_gpu(ax1, ax2, ax3, true);
        
        CUCHK(cudaMalloc((void **)&zprime[isub], (ax3->ntot) * sizeof(float)));

        init_cijs(isub);

        // Initialize correlation buffer with snapshots
        axis* ax1_noHalo = local_X_axes_noHalo[subid[0]];
        axis* ax2_noHalo = local_Y_axes_noHalo[subid[1]];
        axis* ax3_noHalo = local_Z_axes_noHalo[subid[2]];

        snap_f1[isub] =  new cart_volume_regular_gpu(ax1_noHalo, ax2_noHalo, ax3_noHalo, true);
        snap_f2[isub] =  new cart_volume_regular_gpu(ax1_noHalo, ax2_noHalo, ax3_noHalo, true);
        snap_f3[isub] =  new cart_volume_regular_gpu(ax1_noHalo, ax2_noHalo, ax3_noHalo, true);
        snap_f4[isub] = new cart_volume_regular_gpu(ax1, ax2, ax3, true);
        snap_f5[isub] = new cart_volume_regular_gpu(ax1, ax2, ax3, true);
        snap_f6[isub] = new cart_volume_regular_gpu(ax1, ax2, ax3, true);
        snap_f7[isub] = new cart_volume_regular_gpu(ax1, ax2, ax3, true);
        snap_f8[isub] = new cart_volume_regular_gpu(ax1, ax2, ax3, true);
        snap_f9[isub] = new cart_volume_regular_gpu(ax1, ax2, ax3, true);
        g1[isub] =  new cart_volume_regular_gpu(ax1_noHalo, ax2_noHalo, ax3_noHalo, true);
        g2[isub] =  new cart_volume_regular_gpu(ax1, ax2, ax3, true);
        corrBuffList.push_back(snap_f1[isub]);
        corrBuffList.push_back(snap_f2[isub]);
        corrBuffList.push_back(snap_f3[isub]);
        corrBuffList.push_back(snap_f4[isub]);
        corrBuffList.push_back(snap_f5[isub]);
        corrBuffList.push_back(snap_f6[isub]);
        corrBuffList.push_back(snap_f7[isub]);
        corrBuffList.push_back(snap_f8[isub]);
        corrBuffList.push_back(snap_f9[isub]);
    }
    int nbuf = corrBuffList.size();
    corrBuff_size.resize(nbuf, 0);

    if (_bitPerFloat > 0 && _bitPerFloat < 32) {
        useZFP_ = true;

        zipped_corr_buff_.resize(nbuf, NULL);
        zfpFields_.resize(nbuf, NULL);

        zfpStream_ = zfp_stream_open(NULL);
        zfp_stream_set_rate(zfpStream_, bitPerFloat, zfp_type_float, 3, 0);
    }
    else 
        useZFP_ = false;

    for (int i = 0; i < nbuf; ++i) {
        cart_volume<realtype>* vol = corrBuffList[i];
        // Use nvalid here, so that the snapshot computational volumes can have alignment padding if desired.
        // The snapshot storage will strip any padding.
        int n1 = vol->as<cart_volume_regular>()->ax1()->nvalid;
        int n2 = vol->as<cart_volume_regular>()->ax2()->nvalid;
        int n3 = vol->as<cart_volume_regular>()->ax3()->nvalid;

        if (!useZFP_) {
            corrBuff_size[i] = n1 * n2 * n3;
        } else {
            zfpFields_[i] = zfp_field_3d(NULL, zfp_type_float, n1, n2, n3);

            zfp_field_set_stride_3d(zfpFields_[i], 1, vol->as<cart_volume_regular>()->ax1()->ntot,
                                    vol->as<cart_volume_regular>()->ax1()->ntot *
                                        vol->as<cart_volume_regular>()->ax2()->ntot);

            long nbyte = zfp_stream_maximum_size(zfpStream_, zfpFields_[i]);

            long nfloat = nbyte / sizeof(realtype);
            if (nfloat * sizeof(realtype) < nbyte)
                nfloat += 1;

            corrBuff_size[i] = nfloat;
            zfp_stream_set_execution(zfpStream_, zfp_exec_cuda);
            CUDA_TRY(cudaMallocHost((void**)&zipped_corr_buff_[i], nfloat * sizeof(realtype)));
        }
    }

    CUDA_CHECK_ERROR(__FILE__, __LINE__);

    //Create cart_volumes for all sf, all subdomains.
    sf.push_back(f4);
    sf.push_back(f5);
    sf.push_back(f6);
    sf.push_back(f7);
    sf.push_back(f8);
    sf.push_back(f9);

    int nf_s = sf.size();

    vf.push_back(f1);
    vf.push_back(f2);
    vf.push_back(f3);
    int nf_v = vf.size();

    vdf.push_back(d4);
    vdf.push_back(d5);
    vdf.push_back(d6);
    vdf.push_back(d7);
    vdf.push_back(d8);
    vdf.push_back(d9);

    int nf_vd = vdf.size();

    //Sub-domain Stencil widths
    for (int axis = 0; axis < 3; ++axis) // 0 -- x; 1 -- y, 2 -- z
    {
        for (int sign = 0; sign < 2; ++sign) // 0 -- neg; 1 -- pos
        {
            subDomStencilWidths_s[axis][sign] = std::vector<std::vector<int>>(nf_s, std::vector<int>(nsubs, 0));
            subDomStencilWidths_v[axis][sign] = std::vector<std::vector<int>>(nf_v, std::vector<int>(nsubs, 0));
            subDomStencilWidths_vd[axis][sign] = std::vector<std::vector<int>>(nf_vd, std::vector<int>(nsubs, 0));
            for (int isub = 0; isub < nsubs; ++isub) {
                subDomStencilWidths_s[axis][sign][0][isub] = radius;
                subDomStencilWidths_s[axis][sign][1][isub] = radius;
                subDomStencilWidths_s[axis][sign][2][isub] = radius;
                subDomStencilWidths_s[axis][sign][3][isub] = radius;
                subDomStencilWidths_s[axis][sign][4][isub] = radius;
                subDomStencilWidths_s[axis][sign][5][isub] = radius;

                subDomStencilWidths_v[axis][sign][0][isub] = radius;
                subDomStencilWidths_v[axis][sign][1][isub] = radius;
                subDomStencilWidths_v[axis][sign][2][isub] = radius;

                subDomStencilWidths_vd[axis][sign][0][isub] = radius;
                subDomStencilWidths_vd[axis][sign][1][isub] = radius;
                subDomStencilWidths_vd[axis][sign][2][isub] = radius;
                subDomStencilWidths_vd[axis][sign][3][isub] = radius;
                subDomStencilWidths_vd[axis][sign][4][isub] = radius;
                subDomStencilWidths_vd[axis][sign][5][isub] = radius;
            } // isub
        }
    }

    //Create the GPU halo manager
    std::vector<std::vector<bool>> incEdges_s(sf.size(), std::vector<bool>(nsubs, true));
    std::vector<std::vector<bool>> incCorners_s(sf.size(), std::vector<bool>(nsubs, true));
    halomgr_s = new haloManager3D_gpu(decompmgr, sf, subDomStencilWidths_s, incEdges_s, incCorners_s);

    std::vector<std::vector<bool>> incEdges_v(vf.size(), std::vector<bool>(nsubs, true));
    std::vector<std::vector<bool>> incCorners_velocity(vf.size(), std::vector<bool>(nsubs, true));
    halomgr_v = new haloManager3D_gpu(decompmgr, vf, subDomStencilWidths_v, incEdges_v, incCorners_velocity);

    std::vector<std::vector<bool>> incEdges_vd(vdf.size(), std::vector<bool>(nsubs, true));
    std::vector<std::vector<bool>> incCorners_vd(vdf.size(), std::vector<bool>(nsubs, true));
    halomgr_vd = new haloManager3D_gpu(decompmgr, vdf, subDomStencilWidths_vd, 
                                                        incEdges_vd, incCorners_vd);


    CUDA_CHECK_ERROR(__FILE__, __LINE__);

    // initKernels();
 
    // CUDA_CHECK_ERROR(__FILE__, __LINE__);

    long nfloats;
    for (int i = 0; i < corrBuff_size.size(); ++i) {
        nfloats += corrBuff_size[i];
    }
    file_snap_p = file_snap_factory::create(_comm, const_cast<char*>(scratch_folder.c_str()),
                                            nfloats, memoryPercent, _niter/xcorr_step, snap_type);
    CUDA_TRY(cudaEventCreate(&writeSnapshotsCompleteEvent_));
    CUDA_TRY(cudaEventCreate(&readSnapshotsCompleteEvent_));
    snapReaderWriter = std::make_unique<rtm_snap_gpu>(corrBuffList, writeSnapshotsCompleteEvent_, readSnapshotsCompleteEvent_, useZFP_);  
}

simulator3SetUp::~simulator3SetUp()
{
    for (int i = 0; i < 3; ++i) {
        delete[] numPts[i];
    }

    if (costs != nullptr) {
        for (int i = 0; i < numZonesinZ; ++i) {
            for (int j = 0; j < numZonesinY; ++j) {
                delete[] costs[i][j];
            }
            delete[] costs[i];
        }
        delete[] costs;
    }

    for (int axis = 0; axis < 3; ++axis) {
        for (int sign = 0; sign < 2; ++sign) {
            if (stencilWidths[axis][sign] != nullptr) {
                for (int z = 0; z < numZonesinZ; ++z) {
                    for (int y = 0; y < numZonesinY; ++y) {
                        delete[] stencilWidths[axis][sign][z][y];
                    }
                    delete[] stencilWidths[axis][sign][z];
                }
                delete[] stencilWidths[axis][sign];
            }
        }
    }

    delete axgiX;
    delete axgiY;
    delete axgiZ;
    delete axgX;
    delete axgY;
    delete axgZ;

    delete decompmgr;
    delete halomgr_s;
    delete halomgr_v;
}


int simulator3SetUp::get_n_for_zone(int izone, int nzones)
{
    if (nzones == 1)
        return _npoints; // 1 zone -> [_npoints]
    else {
        if (nzones == 2) // 2 zones -> [_npoints PML]
            return izone ? _npoints : _npml;
        else // 3+ zones -> [PML _npoints [_npoints...] PML]
            return (izone == 0 || izone == nzones - 1) ? _npml : _npoints;
    }
    EMSL_VERIFY(false); // shouln't be here
    return 0;
}

void simulator3SetUp::init_cijs(int isub)
{
    cart_volume_regular_gpu* M1_gpu = M1[isub]->as<cart_volume_regular_gpu>();

    int ixbeg = M1_gpu->ax1()->ibeg, ixend = M1_gpu->ax1()->iend;
    int iybeg = M1_gpu->ax2()->ibeg, iyend = M1_gpu->ax2()->iend;
    int izbeg = M1_gpu->ax3()->ibeg, izend = M1_gpu->ax3()->iend;

    int xsize = M1_gpu->ax1()->ntot - M1_gpu->ax1()->npad_trailing;

    float* M1data = M1[isub]->as<cart_volume_regular_gpu>()->getData();
    float* M2data = M2[isub]->as<cart_volume_regular_gpu>()->getData();
    float* M3data = M3[isub]->as<cart_volume_regular_gpu>()->getData();
    float* M4data = M4[isub]->as<cart_volume_regular_gpu>()->getData();
    float* M5data = M5[isub]->as<cart_volume_regular_gpu>()->getData();
    float* M6data = M6[isub]->as<cart_volume_regular_gpu>()->getData();
    float* M7data = M7[isub]->as<cart_volume_regular_gpu>()->getData();
    float* M8data = M8[isub]->as<cart_volume_regular_gpu>()->getData();
    float* M9data = M9[isub]->as<cart_volume_regular_gpu>()->getData();
    float* M10data = M10[isub]->as<cart_volume_regular_gpu>()->getData();
    float* M11data = M11[isub]->as<cart_volume_regular_gpu>()->getData();
    float* M12data = M12[isub]->as<cart_volume_regular_gpu>()->getData();
    float* M13data = M13[isub]->as<cart_volume_regular_gpu>()->getData();
    float* M14data = M14[isub]->as<cart_volume_regular_gpu>()->getData();
    float* M15data = M15[isub]->as<cart_volume_regular_gpu>()->getData();
    float* M16data = M16[isub]->as<cart_volume_regular_gpu>()->getData();
    float* M17data = M17[isub]->as<cart_volume_regular_gpu>()->getData();
    float* M18data = M18[isub]->as<cart_volume_regular_gpu>()->getData();
    float* M19data = M19[isub]->as<cart_volume_regular_gpu>()->getData();
    float* M20data = M20[isub]->as<cart_volume_regular_gpu>()->getData();
    float* M21data = M21[isub]->as<cart_volume_regular_gpu>()->getData();
    float* M22data = M22[isub]->as<cart_volume_regular_gpu>()->getData();

    float* d4data = d4[isub]->as<cart_volume_regular_gpu>()->getData();
    float* d5data = d5[isub]->as<cart_volume_regular_gpu>()->getData();
    float* d6data = d6[isub]->as<cart_volume_regular_gpu>()->getData();
    float* d7data = d7[isub]->as<cart_volume_regular_gpu>()->getData();
    float* d8data = d8[isub]->as<cart_volume_regular_gpu>()->getData();
    float* d9data = d9[isub]->as<cart_volume_regular_gpu>()->getData();


    dim3 threads(128, 1, 1);
    dim3 blocks((xsize - 1) / threads.x + 1, (M1_gpu->ax2()->ntot - 1) / threads.y + 1,
                (M1_gpu->ax3()->ntot - 1) / threads.z + 1);

    init_cijs_kernel<<<blocks, threads>>>( M1data,  M2data,  M3data,  M4data,  M5data,  M6data,  M7data,
                  M8data,  M9data,  M10data,  M11data,  M12data,  M13data,  M14data,
                  M15data,  M16data,  M17data,  M18data,  M19data,  M20data,  M21data, 
                  M22data, d4data, d5data, d6data, d7data, d8data, d9data, zprime[isub], ixbeg, ixend,
                  iybeg, iyend, izbeg, izend, M1_gpu->vol_idx());
    CUDA_CHECK_ERROR(__FILE__, __LINE__);
}

template <int TILE_Z = 64>
__launch_bounds__(32 * 16, 1)
    __global__ void vPlainKernel(float *__restrict__ f1,
                                   float *__restrict__ f2,
                                   float *__restrict__ f3,
                                   const float *__restrict__ f4,
                                   const float *__restrict__ f5,
                                   const float *__restrict__ f6,
                                   const float *__restrict__ f7,
                                   const float *__restrict__ f8,
                                   const float *__restrict__ f9,
                                   const float *__restrict__ M22,
                                   const float *__restrict__ zprime,
                                   int ldimx,
                                   int ldimy,
                                   int nz)
{
    const float invd1 = 1.01f;
    const float invd2 = 2.01f;

    __shared__ float shm11[2][16][40]; // SHM11: Center + X halos
    __shared__ float shm13[2][16][40]; // SHM13: Center from register queue + X halos
    __shared__ float shm22[2][24][32]; // SHM22: Center + Y halos
    __shared__ float shm23[2][24][32]; // SHM23: Center from register queue + Y halos
    __shared__ float shm12[2][24][40]; // SHM12: Center + X halos + Y halos

    // Register queues for the Z derivatives
    float rq33[9];
    float rq13[9];
    float rq23[9];
    float m22z;

    const int tx = threadIdx.x + 4;
    const int ty = threadIdx.y + 4;
    const int ix = blockIdx.x * 32 + tx;
    const int iy = blockIdx.y * 16 + ty;
    const int iz = blockIdx.z * TILE_Z;
    const int zblock = min(TILE_Z, nz - iz);
    const int stride = ldimx * ldimy;

    // Index to load the register queues in Z
    int index_z = (iz * ldimy + iy) * ldimx + ix;
    // Index of the point to be computed
    int index = index_z + 4 * stride;

    // Prime the register queues in Z
    for (int i = 0; i < 8; i++)
    {
        rq13[i] = f8[index_z];
        rq23[i] = f9[index_z];
        rq33[i] = f6[index_z];
        index_z += stride;
    }
    m22z = M22[index - stride];

    // Prime the shared memory
    __pipeline_memcpy_async(&shm11[0][threadIdx.y][tx], &f4[index], sizeof(float));      // Center of shm11
    __pipeline_memcpy_async(&shm22[0][ty][threadIdx.x], &f5[index], sizeof(float));      // Center of shm22
    __pipeline_memcpy_async(&shm12[0][ty][tx], &f7[index], sizeof(float));               // Center of shm12

    if (threadIdx.y < 4)
    {
        __pipeline_memcpy_async(&shm23[0][threadIdx.y][threadIdx.x], &f9[index - 4 * ldimx], sizeof(float)); // Y- halos of shm23
        __pipeline_memcpy_async(&shm23[0][ty + 16][threadIdx.x], &f9[index + 16 * ldimx], sizeof(float)); // Y+ halos of shm23
        __pipeline_memcpy_async(&shm22[0][threadIdx.y][threadIdx.x], &f5[index - 4 * ldimx], sizeof(float)); // Y- halos of shm22
        __pipeline_memcpy_async(&shm22[0][ty + 16][threadIdx.x], &f5[index + 16 * ldimx], sizeof(float)); // Y+ halos of shm22
        __pipeline_memcpy_async(&shm12[0][threadIdx.y][tx], &f7[index - 4 * ldimx], sizeof(float)); // Y- halos of shm12
        __pipeline_memcpy_async(&shm12[0][ty + 16][tx], &f7[index + 16 * ldimx], sizeof(float)); // Y+ halos of shm12
    }
    if (threadIdx.x < 4)
    {
        __pipeline_memcpy_async(&shm11[0][threadIdx.y][threadIdx.x], &f4[index - 4], sizeof(float));       // X- halos of shm11
        __pipeline_memcpy_async(&shm11[0][threadIdx.y][tx + 32], &f4[index + 32], sizeof(float)); // X+ halos of shm11
        __pipeline_memcpy_async(&shm13[0][threadIdx.y][threadIdx.x], &f8[index - 4], sizeof(float));       // X- halos of shm13
        __pipeline_memcpy_async(&shm13[0][threadIdx.y][tx + 32], &f8[index + 32], sizeof(float)); // X+ halos of shm13
        __pipeline_memcpy_async(&shm12[0][ty][threadIdx.x], &f7[index - 4], sizeof(float));                // X- halos of shm12
        __pipeline_memcpy_async(&shm12[0][ty][tx + 32], &f7[index + 32], sizeof(float));                   // X+ halos of shm12
    }
    __pipeline_commit();

    int k = 1; // Shared memory index flip-flop between 0 and 1. Loaded in 0, now ready to load 1.

    for (int zloop = 0; zloop < zblock; zloop++)
    {
        // Read new register queue values
        rq13[8] = f8[index_z];
        rq23[8] = f9[index_z];
        rq33[8] = f6[index_z];
        index_z += stride;

        // Async load the shared memory for the next iteration
        __pipeline_memcpy_async(&shm11[k][threadIdx.y][tx], &f4[index + stride], sizeof(float));      // Center of shm11
        __pipeline_memcpy_async(&shm22[k][ty][threadIdx.x], &f5[index + stride], sizeof(float));      // Center of shm22
        __pipeline_memcpy_async(&shm12[k][ty][tx], &f7[index + stride], sizeof(float));               // Center of shm12
        if (threadIdx.y < 4)
        {
            __pipeline_memcpy_async(&shm23[k][threadIdx.y][threadIdx.x], &f9[index + stride - 4 * ldimx], sizeof(float)); // Y- halos of shm23
            __pipeline_memcpy_async(&shm23[k][ty + 16][threadIdx.x], &f9[index + stride + 16 * ldimx], sizeof(float)); // Y+ halos of shm23
            __pipeline_memcpy_async(&shm22[k][threadIdx.y][threadIdx.x], &f5[index + stride - 4 * ldimx], sizeof(float)); // Y- halos of shm22
            __pipeline_memcpy_async(&shm22[k][ty + 16][threadIdx.x], &f5[index + stride + 16 * ldimx], sizeof(float)); // Y+ halos of shm22
            __pipeline_memcpy_async(&shm12[k][threadIdx.y][threadIdx.x], &f7[index + stride - 4 * ldimx], sizeof(float)); // Y- halos of shm12
            __pipeline_memcpy_async(&shm12[k][ty + 16][threadIdx.x], &f7[index + stride + 16 * ldimx], sizeof(float)); // Y+ halos of shm12
        }
        if (threadIdx.x < 4)
        {
            __pipeline_memcpy_async(&shm11[k][threadIdx.y][threadIdx.x], &f4[index + stride - 4], sizeof(float));       // X- halos of shm11
            __pipeline_memcpy_async(&shm11[k][threadIdx.y][tx + 32], &f4[index + stride + 32], sizeof(float)); // X+ halos of shm11
            __pipeline_memcpy_async(&shm13[k][threadIdx.y][threadIdx.x], &f8[index + stride - 4], sizeof(float));       // X- halos of shm13
            __pipeline_memcpy_async(&shm13[k][threadIdx.y][tx + 32], &f8[index + stride + 32], sizeof(float)); // X+ halos of shm13
            __pipeline_memcpy_async(&shm12[k][ty][threadIdx.x], &f7[index + stride - 4], sizeof(float));                // X- halos of shm12
            __pipeline_memcpy_async(&shm12[k][ty][tx + 32], &f7[index + stride + 32], sizeof(float));                   // X+ halos of shm12
        }
        __pipeline_commit();

        // Switch back to the other shared memory
        k ^= 1;

        // Fill the center sections of shared memory with register queues
        shm13[k][threadIdx.y][tx] = rq13[4];
        shm23[k][ty][threadIdx.x] = rq23[4];

        // Wait for the previous stage to complete
        __pipeline_wait_prior(1);
        __syncthreads();

        float d10 = drv1_8<true, 1>(&shm11[k][threadIdx.y][tx]);
        float d11 = drv1_8<true, 32>(&shm22[k][ty][threadIdx.x]);
        float d12 = drv1_8<true, 1>(&rq33[4]);
        float d13 = drv1_8<false, 1>(&shm12[k][ty][tx]);
        float d14 = drv1_8<false, 40>(&shm12[k][ty][tx]);
        float d15 = drv1_8<false, 1>(&rq13[4]);
        float d16 = drv1_8<false, 1>(&shm13[k][threadIdx.y][tx]);
        float d17 = drv1_8<false, 32>(&shm23[k][ty][threadIdx.x]);
        float d18 = drv1_8<false, 1>(&rq23[4]);

        __syncthreads();

        // Rely on the L1 cache to load the M22 model efficiently
        float zpr = zprime[iz + 4 + zloop];
        float res1 = (2.0f * M22[index - 1]) * (d10 * invd1 + d14 * invd2 + d15 * zpr);
        float res2 = (2.0f * M22[index - ldimx]) * (d13 * invd1 + d11 * invd2 + d18 * zpr);
        float res3 = (2.0f * m22z) * (d16 * invd1 + d17 * invd2 + d12 * zpr);
        m22z = M22[index];

        // Using atomics to let the L2 cache do the load-add-store, instead of the SM.
        atomicAdd(f1 + index, res1);
        atomicAdd(f2 + index, res2);
        atomicAdd(f3 + index, res3);
        index += stride;

        // Rotate the register queues
        for (int i = 0; i < 8; i++)
        {
            rq13[i] = rq13[i + 1];
            rq23[i] = rq23[i + 1];
            rq33[i] = rq33[i + 1];
        }
    }
}

template <int TILE_X = 32, int TILE_Y = 32>
__launch_bounds__(TILE_X *TILE_Y, 1)
    __global__ void sPlainKernel(float *__restrict__ f4,
                                      float *__restrict__ f5,
                                      float *__restrict__ f6,
                                      float *__restrict__ f7,
                                      float *__restrict__ f8,
                                      float *__restrict__ f9,
                                      const float *__restrict__ M1,
                                      const float *__restrict__ M2,
                                      const float *__restrict__ M3,
                                      const float *__restrict__ M4,
                                      const float *__restrict__ M5,
                                      const float *__restrict__ M6,
                                      const float *__restrict__ M7,
                                      const float *__restrict__ M8,
                                      const float *__restrict__ M9,
                                      const float *__restrict__ M10,
                                      const float *__restrict__ M11,
                                      const float *__restrict__ M12,
                                      const float *__restrict__ M13,
                                      const float *__restrict__ M14,
                                      const float *__restrict__ M15,
                                      const float *__restrict__ M16,
                                      const float *__restrict__ M17,
                                      const float *__restrict__ M18,
                                      const float *__restrict__ M19,
                                      const float *__restrict__ M20,
                                      const float *__restrict__ M21,
                                      const float *__restrict__ f1_1,
                                      const float *__restrict__ f1_2,
                                      const float *__restrict__ f1_3,
                                      const float *__restrict__ f2_2,
                                      const float *__restrict__ f2_3,
                                      const float *__restrict__ f3_3,
                                      const float *__restrict__ f10,
                                      const float *__restrict__ f11,
                                      const float *__restrict__ f12,
                                      const float *__restrict__ f13,
                                      const float *__restrict__ f14,
                                      const float *__restrict__ f15,
                                      const float *__restrict__ f16,
                                      const float *__restrict__ f17,
                                      const float *__restrict__ f18,
                                      const float *__restrict__ f19,
                                      const float *__restrict__ f20,
                                      const float *__restrict__ f21,
                                      int ldimx, int ldimy)
{
    // Shared memory is (TILE_X + 8) * (TILE_Y + 8), for each of the 6 input arrays
    extern __shared__ float sm_1_1[];
    float *sm_2_2 = sm_1_1 + (TILE_X + 8) * (TILE_Y + 8);
    float *sm_3_3 = sm_2_2 + (TILE_X + 8) * (TILE_Y + 8);
    float *sm_1_2 = sm_3_3 + (TILE_X + 8) * (TILE_Y + 8);
    float *sm_1_3 = sm_1_2 + (TILE_X + 8) * (TILE_Y + 8);
    float *sm_2_3 = sm_1_3 + (TILE_X + 8) * (TILE_Y + 8);
    constexpr int smdimx = TILE_X + 8;
    const int smindex = (threadIdx.y + 4) * smdimx + threadIdx.x + 4; // Natural index within a 2D shared memory plan

    int ix = blockIdx.x * TILE_X + threadIdx.x + 4;
    int iy = blockIdx.y * TILE_Y + threadIdx.y + 4;
    int iz = blockIdx.z + 4;
    int index = (iz * ldimy + iy) * ldimx + ix;

    // Load all the 6 x V arrays in shared memory to compute XY derivatives
    loadShared<TILE_X, TILE_Y>(sm_1_1, smindex, f1_1, index, ldimx);
    loadShared<TILE_X, TILE_Y>(sm_2_2, smindex, f2_2, index, ldimx);
    loadShared<TILE_X, TILE_Y>(sm_3_3, smindex, f3_3, index, ldimx);
    loadShared<TILE_X, TILE_Y>(sm_1_2, smindex, f1_2, index, ldimx);
    loadShared<TILE_X, TILE_Y>(sm_1_3, smindex, f1_3, index, ldimx);
    loadShared<TILE_X, TILE_Y>(sm_2_3, smindex, f2_3, index, ldimx);

    __syncthreads();

    float d19 = sm_1_1[smindex];
    float d20 = sm_2_2[smindex];
    float d21 = sm_3_3[smindex];

    float d22 = sm_1_2[smindex];
    float d24 = sm_1_3[smindex];
    float d26 = sm_2_3[smindex];

    float dv1d2_xx = mixed_XY<false, false, smdimx>(&sm_1_2[smindex]);
    float dv1d3_xx = f14[index];
    float dv2d3_xx = f19[index];

    f4[index] += M1[index] * d19 + M2[index] * d20 + M3[index] * d21 +
                  M6[index] * dv1d2_xx + M5[index] * dv1d3_xx + M4[index] * dv2d3_xx;

    f5[index] += M2[index] * d19 + M7[index] * d20 + M8[index] * d21 +
                  M11[index] * dv1d2_xx + M10[index] * dv1d3_xx + M9[index] * dv2d3_xx;

    f6[index] += M3[index] * d19 + M8[index] * d20 + M12[index] * d21 +
                  M15[index] * dv1d2_xx + M14[index] * dv1d3_xx + M13[index] * dv2d3_xx;

    float dv1d1_xy = mixed_XY<true, true, smdimx>(&sm_1_1[smindex]);
    float dv2d2_xy = mixed_XY<true, true, smdimx>(&sm_2_2[smindex]);
    float dv3d3_xy = mixed_XY<true, true, smdimx>(&sm_3_3[smindex]);
    float dv1d3_xy = f15[index];
    float dv2d3_xy = f18[index];

    f7[index] += M6[index] * dv1d1_xy + M11[index] * dv2d2_xy + M15[index] * dv3d3_xy +
                  M20[index] * dv1d3_xy + M18[index] * dv2d3_xy + M21[index] * d22;

    float dv1d1_xz = f10[index];
    float dv2d2_xz = f16[index];
    float dv3d3_xz = f20[index];
    float dv1d2_xz = f13[index];
    float dv2d3_xz = mixed_XY<true, false, smdimx>(&sm_2_3[smindex]);

    f8[index] += M5[index] * dv1d1_xz + M10[index] * dv2d2_xz + M14[index] * dv3d3_xz +
                  M20[index] * dv1d2_xz + M17[index] * dv2d3_xz + M19[index] * d24;

    float dv1d1_yz = f11[index];
    float dv2d2_yz = f17[index];
    float dv3d3_yz = f21[index];
    float dv1d2_yz = f12[index];
    float dv1d3_yz = mixed_XY<false, true, smdimx>(&sm_1_3[smindex]);

    f9[index] += M4[index] * dv1d1_yz + M9[index] * dv2d2_yz + M13[index] * dv3d3_yz +
                  M18[index] * dv1d2_yz + M17[index] * dv1d3_yz + M16[index] * d26;
}

// ****************************************************************************

void vPlainLoop(float *f1, float *f2, float *f3,
                  float *f4, float *f5, float *f6,
                  float *f7, float *f8, float *f9,
                  float *M22, float *zprime, int nx, int ny, int nz, int ldimx, int ldimy, float &milliseconds_v)
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    constexpr int tile_z = 64;
    dim3 threads(32, 16, 1);
    dim3 blocks((nx + 31) / 32, (ny + 15) / 16, (nz - 1) / tile_z + 1);
    cudaEventRecord(start);
    vPlainKernel<tile_z><<<blocks, threads, 0, 0>>>(f1, f2, f3, f4, f5, f6,
                                                      f7, f8, f9, M22, zprime,
                                                      ldimx, ldimy, nz);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float elapsedTime = 0;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    milliseconds_v += elapsedTime;
}

void sPlainLoop(float *__restrict__ f4,
                     float *__restrict__ f5,
                     float *__restrict__ f6,
                     float *__restrict__ f7,
                     float *__restrict__ f8,
                     float *__restrict__ f9,
                     const float *__restrict__ M1,
                     const float *__restrict__ M2,
                     const float *__restrict__ M3,
                     const float *__restrict__ M4,
                     const float *__restrict__ M5,
                     const float *__restrict__ M6,
                     const float *__restrict__ M7,
                     const float *__restrict__ M8,
                     const float *__restrict__ M9,
                     const float *__restrict__ M10,
                     const float *__restrict__ M11,
                     const float *__restrict__ M12,
                     const float *__restrict__ M13,
                     const float *__restrict__ M14,
                     const float *__restrict__ M15,
                     const float *__restrict__ M16,
                     const float *__restrict__ M17,
                     const float *__restrict__ M18,
                     const float *__restrict__ M19,
                     const float *__restrict__ M20,
                     const float *__restrict__ M21,
                     const float *__restrict__ f1_1,
                     const float *__restrict__ f1_2,
                     const float *__restrict__ f1_3,
                     const float *__restrict__ f2_2,
                     const float *__restrict__ f2_3,
                     const float *__restrict__ f3_3,
                     float *__restrict__ f10,
                     float *__restrict__ f11,
                     float *__restrict__ f12,
                     float *__restrict__ f13,
                     float *__restrict__ f14,
                     float *__restrict__ f15,
                     float *__restrict__ f16,
                     float *__restrict__ f17,
                     float *__restrict__ f18,
                     float *__restrict__ f19,
                     float *__restrict__ f20,
                     float *__restrict__ f21,
                     int nx, int ny, int nz, int ldimx, int ldimy, float &milliseconds_s)
{
    // Compute all the mixed derivatives that involve the Z dimension
    // For small datasets, we could compute them in parallel (different streams)
    {    
        const int tile_x = 32;
        const int tile_y = 16;
        const int tile_z = 64;
        dim3 threads(tile_x, tile_y, 1);
        dim3 blocks((nx - 1) / tile_x + 1, (ny - 1) / tile_y + 1, (nz - 1) / tile_z + 1);
        precompute_xz_yz<1, 1, 1, tile_x, tile_y, tile_z><<<blocks, threads>>>(f1_1, f10, f11, ldimx, ldimy, nz);
        precompute_xz_yz<1, 1, 1, tile_x, tile_y, tile_z><<<blocks, threads>>>(f2_2, f16, f17, ldimx, ldimy, nz);
        precompute_xz_yz<1, 1, 1, tile_x, tile_y, tile_z><<<blocks, threads>>>(f3_3, f20, f21, ldimx, ldimy, nz);
        precompute_xz_yz<0, 0, 1, tile_x, tile_y, tile_z><<<blocks, threads>>>(f1_2, f12, f13, ldimx, ldimy, nz);
        precompute_xz_yz<0, 1, 0, tile_x, tile_y, tile_z><<<blocks, threads>>>(f1_3, f14, f15, ldimx, ldimy, nz);
        precompute_xz_yz<1, 0, 0, tile_x, tile_y, tile_z><<<blocks, threads>>>(f2_3, f18, f19, ldimx, ldimy, nz);
    }

    // Compute the final result, using a bunch of 2D thread blocks
    {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        const int tile_x = 32;
        const int tile_y = 32;
        dim3 threads(tile_x, tile_y, 1);
        dim3 blocks((nx - 1) / tile_x + 1, (ny - 1) / tile_y + 1, nz);
        int sharedmem = (tile_x + 8) * (tile_y + 8) * 6 * sizeof(float);
        CUCHK(cudaFuncSetAttribute(sPlainKernel<tile_x, tile_y>, cudaFuncAttributeMaxDynamicSharedMemorySize, sharedmem));
        cudaEventRecord(start);
        sPlainKernel<tile_x, tile_y><<<blocks, threads, sharedmem, 0>>>(f4, f5, f6, f7, f8, f9,
                                                                             M1, M2, M3, M4, M5, M6, M7, M8, M9, M10, M11,
                                                                             M12, M13, M14, M15, M16, M17, M18, M19, M20, M21,
                                                                             f1_1, f1_2, f1_3, f2_2, f2_3, f3_3,
                                                                             f10, f11, f12, f13, f14, f15,
                                                                             f16, f17, f18, f19, f20, f21,
                                                                             ldimx, ldimy);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float elapsedTime = 0;
        cudaEventElapsedTime(&elapsedTime, start, stop);
        milliseconds_s += elapsedTime;                                                                             
    }
}


template <int TILE_Z>
__launch_bounds__(32 * 16)
    __global__ void vgPlainKernel(float *__restrict__ f1,
                                           float *__restrict__ f2,
                                           float *__restrict__ f3,
                                           float *__restrict__ d4,
                                           float *__restrict__ d5,
                                           float *__restrict__ d6,
                                           float *__restrict__ d7,
                                           float *__restrict__ d8,
                                           float *__restrict__ d9,
                                           float *__restrict__ zprime,
                                           int ldimx,
                                           int ldimy,
                                           int nz)
{
    const float invd1 = 1.01f;
    const float invd2 = 2.02f;
    __shared__ float shm1[2][24][40];
    __shared__ float shm2[2][24][40];
    __shared__ float shm3[2][24][40];

    // Register queues for the Z derivatives
    float rq1[9];
    float rq2[9];
    float rq3[9];

    const int tx = threadIdx.x + 4;
    const int ty = threadIdx.y + 4;
    const int ix = blockIdx.x * 32 + tx;
    const int iy = blockIdx.y * 16 + ty;
    const int iz = blockIdx.z * TILE_Z;
    const int stride = ldimx * ldimy;

    // Index to load the register queues in Z
    int index_z = (iz * ldimy + iy) * ldimx + ix;
    // Index of the point to be computed
    int index = index_z + 4 * stride;

    // Prime the register queues
    for (int i = 0; i < 8; i++)
    {
        rq1[i] = f1[index_z];
        rq2[i] = f2[index_z];
        rq3[i] = f3[index_z];
        index_z += stride;
    }

    // Prime the shared memory (the center is already in the register queue)
    if (threadIdx.x < 4)
    {
        __pipeline_memcpy_async(&shm1[0][ty][threadIdx.x], &f1[index - 4], sizeof(float));
        __pipeline_memcpy_async(&shm2[0][ty][threadIdx.x], &f2[index - 4], sizeof(float));
        __pipeline_memcpy_async(&shm3[0][ty][threadIdx.x], &f3[index - 4], sizeof(float));
        __pipeline_memcpy_async(&shm1[0][ty][tx + 32], &f1[index + 32], sizeof(float));
        __pipeline_memcpy_async(&shm2[0][ty][tx + 32], &f2[index + 32], sizeof(float));
        __pipeline_memcpy_async(&shm3[0][ty][tx + 32], &f3[index + 32], sizeof(float));
    }
    if (threadIdx.y < 4)
    {
        __pipeline_memcpy_async(&shm1[0][threadIdx.y][tx], &f1[index - 4 * ldimx], sizeof(float));
        __pipeline_memcpy_async(&shm2[0][threadIdx.y][tx], &f2[index - 4 * ldimx], sizeof(float));
        __pipeline_memcpy_async(&shm3[0][threadIdx.y][tx], &f3[index - 4 * ldimx], sizeof(float));
        __pipeline_memcpy_async(&shm1[0][ty + 16][tx], &f1[index + 16 * ldimx], sizeof(float));
        __pipeline_memcpy_async(&shm2[0][ty + 16][tx], &f2[index + 16 * ldimx], sizeof(float));
        __pipeline_memcpy_async(&shm3[0][ty + 16][tx], &f3[index + 16 * ldimx], sizeof(float));
    }
    __pipeline_commit();

    int k = 1; // Shared memory index flip-flop between 0 and 1. Loaded in 0, now ready to load 1.

    int nzblock = min(TILE_Z, nz - iz);
    for (int zloop = 0; zloop < nzblock; zloop++)
    {

        // Read new register queue values
        rq1[8] = f1[index_z];
        rq2[8] = f2[index_z];
        rq3[8] = f3[index_z];
        index_z += stride;

        // Async load the shared memory for the next iteration
        if (threadIdx.x < 4)
        {
            __pipeline_memcpy_async(&shm1[k][ty][threadIdx.x], &f1[index + stride - 4], sizeof(float));
            __pipeline_memcpy_async(&shm2[k][ty][threadIdx.x], &f2[index + stride - 4], sizeof(float));
            __pipeline_memcpy_async(&shm3[k][ty][threadIdx.x], &f3[index + stride - 4], sizeof(float));
            __pipeline_memcpy_async(&shm1[k][ty][tx + 32], &f1[index + stride + 32], sizeof(float));
            __pipeline_memcpy_async(&shm2[k][ty][tx + 32], &f2[index + stride + 32], sizeof(float));
            __pipeline_memcpy_async(&shm3[k][ty][tx + 32], &f3[index + stride + 32], sizeof(float));
        }
        if (threadIdx.y < 4)
        {
            __pipeline_memcpy_async(&shm1[k][threadIdx.y][tx], &f1[index + stride - 4 * ldimx], sizeof(float));
            __pipeline_memcpy_async(&shm2[k][threadIdx.y][tx], &f2[index + stride - 4 * ldimx], sizeof(float));
            __pipeline_memcpy_async(&shm3[k][threadIdx.y][tx], &f3[index + stride - 4 * ldimx], sizeof(float));
            __pipeline_memcpy_async(&shm1[k][ty + 16][tx], &f1[index + stride + 16 * ldimx], sizeof(float));
            __pipeline_memcpy_async(&shm2[k][ty + 16][tx], &f2[index + stride + 16 * ldimx], sizeof(float));
            __pipeline_memcpy_async(&shm3[k][ty + 16][tx], &f3[index + stride + 16 * ldimx], sizeof(float));
        }
        __pipeline_commit();

        // Switch back to the other shared memory
        k ^= 1;

        // Fill the center sections of shared memory with register queues
        shm1[k][ty][tx] = rq1[4];
        shm2[k][ty][tx] = rq2[4];
        shm3[k][ty][tx] = rq3[4];

        // Wait for the previous stage to complete
        __pipeline_wait_prior(1);
        __syncthreads();

        float d19 = drv1_8<false, 1>(&shm1[k][ty][tx]);  // X drv on sm1
        float d20 = drv1_8<false, 40>(&shm2[k][ty][tx]); // Y drv on sm2
        float d21 = drv1_8<false, 1>(&rq3[4]);           // Z drv on rq3
        float d22 = drv1_8<true, 40>(&shm1[k][ty][tx]);  // Y drv on sm1
        float d23 = drv1_8<true, 1>(&shm2[k][ty][tx]);   // X drv on sm2
        float d24 = drv1_8<true, 1>(&rq1[4]);            // Z drv on rq1
        float d25 = drv1_8<true, 1>(&shm3[k][ty][tx]);   // X drv on sm3
        float d26 = drv1_8<true, 1>(&rq2[4]);            // Z drv on rq2
        float d27 = drv1_8<true, 40>(&shm3[k][ty][tx]);  // Y drv on sm3

        __syncthreads();

        float zpr = zprime[iz + 4 + zloop];

        d4[index] = invd1 * d19;
        d5[index] = invd2 * d22 + invd1 * d23;
        d6[index] = zpr * d24 + invd1 * d25;
        d7[index] = invd2 * d20;
        d8[index] = zpr * d26 + invd2 * d27;
        d9[index] = zpr * d21;
        index += stride;

        // Rotate the register queues
        for (int i = 0; i < 8; i++)
        {
            rq1[i] = rq1[i + 1];
            rq2[i] = rq2[i + 1];
            rq3[i] = rq3[i + 1];
        }
    }
}

void vgPlainLoop(float *f1, float *f2, float *f3,
                          float *d4, float *d5, float *d6,
                          float *d7, float *d8, float *d9,
                          float *zprime, int nx, int ny, int nz, int ldimx, int ldimy, float &milliseconds_g)
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    constexpr int tile_z = 64;
    dim3 threads(32, 16, 1);
    dim3 blocks((nx + 31) / 32, (ny + 15) / 16, (nz - 1) / tile_z + 1);
    cudaEventRecord(start);
    vgPlainKernel<tile_z><<<blocks, threads, 0, 0>>>(f1, f2, f3,
                                                              d4, d5, d6,
                                                              d7, d8, d9,
                                                              zprime, ldimx, ldimy, nz);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float elapsedTime = 0;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    milliseconds_g += elapsedTime;                                                                 
}

__global__ void
mirror_s_kernel(float* f4, float* f5, float* f6, float* f7, float* f8, float* f9,  int stride, int izbeg)
{
    // Treating the whole XY plan as a 1D, mapped to CUDA's X dimension.
    int ixy = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.z;
    if (ixy >= stride)
        return;
    size_t index = iz * (size_t)stride + ixy;
    if (iz == izbeg) {
        // Set normal stress at free surface to zero
        f4[index] = 0.0f;
        f5[index] = 0.0f;
        f6[index] = 0.0f;
        f7[index] = 0.0f;
        f8[index] = 0.0f;
        f9[index] = 0.0f;
    
    } else {
        // Mirror stress at free surface
        size_t index_mirror = (2 * izbeg - iz) * (size_t)stride + ixy;

        float resf4 = -f4[index_mirror];
        float resf5 = -f5[index_mirror];
        float resf6 = -f6[index_mirror];
        float resf7 = -f7[index_mirror];
        float resf8 = -f8[index_mirror];
        float resf9 = -f9[index_mirror];

        f4[index] =  resf4;
        f5[index] =  resf5;
        f6[index] =  resf6;
        f7[index] =  resf7;
        f8[index] =  resf8;
        f9[index] =  resf9;
    }
}

void mirror_s(int isub, cudaStream_t stream, cart_volume<float>** f4, cart_volume<float>** f5, cart_volume<float>** f6, cart_volume<float>** f7, cart_volume<float>** f8, cart_volume<float>** f9)
{
    // Total number of elements in a XY plan
    int stride = f4[isub]->as<cart_volume_regular_gpu>()->ax1()->ntot * f4[isub]->as<cart_volume_regular_gpu>()->ax2()->ntot;

    // Free surface location
    int izbeg = f4[isub]->as<cart_volume_regular_gpu>()->ax3()->ibeg;

    // Each XY plan will be processed by the threads X dimension.
    // Launch (izbeg + 1) blocks.z to work on all the Z plans (includes the surface)
    dim3 threads(1024, 1, 1);
    dim3 blocks((stride - 1) / threads.x + 1, 1, izbeg + 1);

    float* f4_ = f4[isub]->as<cart_volume_regular_gpu>()->getData();
    float* f5_ = f5[isub]->as<cart_volume_regular_gpu>()->getData();
    float* f6_ = f6[isub]->as<cart_volume_regular_gpu>()->getData();
    float* f7_ = f7[isub]->as<cart_volume_regular_gpu>()->getData();
    float* f8_ = f8[isub]->as<cart_volume_regular_gpu>()->getData();
    float* f9_ = f9[isub]->as<cart_volume_regular_gpu>()->getData();
    mirror_s_kernel<<<blocks, threads, 0, stream>>>(f4_, f5_, f6_, f7_, f8_, f9_, stride, izbeg);
}

__global__ void
mirror_v_kernel(float* f1, float* f2, float* f3, int stride, int izbeg)
{
    // Treating the whole XY plan as a 1D, mapped to CUDA's X dimension.
    int ixy = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.z;
    if (ixy >= stride)
        return;
    size_t index = iz * (size_t)stride + ixy;
    if (iz == izbeg) {
        // Set normal stress at free surface to zero
        f1[index] = 0.0f;
        f2[index] = 0.0f;
        f3[index] = 0.0f;

    } else {
        // Mirror stress at free surface
        size_t index_mirror = (2 * izbeg - iz) * (size_t)stride + ixy;
        float resf1 = -f1[index_mirror];
        float resf2 = -f2[index_mirror];
        float resf3 = -f3[index_mirror];
        f1[index] = resf1;
        f2[index] = resf2;
        f3[index] = resf3;
    }
}

void
mirror_v(int isub, cudaStream_t stream, cart_volume<float>** f1, cart_volume<float>** f2, cart_volume<float>** f3)
{
    // Total number of elements in a XY plan
    int stride = f1[isub]->as<cart_volume_regular_gpu>()->ax1()->ntot * f1[isub]->as<cart_volume_regular_gpu>()->ax2()->ntot;

    // Free surface location
    int izbeg = f1[isub]->as<cart_volume_regular_gpu>()->ax3()->ibeg;

    // Each XY plan will be processed by the threads X dimension.
    // Launch (izbeg + 1) blocks.z to work on all the Z plans (includes the surface)
    dim3 threads(1024, 1, 1);
    dim3 blocks((stride - 1) / threads.x + 1, 1, izbeg + 1);

    float* f1Data = f1[isub]->as<cart_volume_regular_gpu>()->getData();
    float* f2Data = f2[isub]->as<cart_volume_regular_gpu>()->getData();
    float* f3Data = f3[isub]->as<cart_volume_regular_gpu>()->getData();
    mirror_v_kernel<<<blocks, threads, 0, stream>>>(f1Data, f2Data, f3Data, stride, izbeg);
}

__global__ void
update_gv(volume_index idx, volume_index idx_a, float* __restrict__ snap_f22, float* __restrict__ snap_f23,
                   float* __restrict__ g1, float* __restrict__ a_f22, float* __restrict__ a_f23, int ixbeg,
                   int ixend, int iybeg, int iyend, int izbeg, int izend)
{
    int ix = ixbeg + blockIdx.x * blockDim.x + threadIdx.x;
    int iy = iybeg + blockIdx.y * blockDim.y + threadIdx.y;
    int iz = izbeg + blockIdx.z * blockDim.z + threadIdx.z;

    if (ix > ixend || iy > iyend || iz > izend)
        return;

    float vx0 = idx(snap_f22, ix - ixbeg, iy - iybeg, iz - izbeg);
    float vz0 = idx(snap_f23, ix - ixbeg, iy - iybeg, iz - izbeg);

    idx(g1, ix - ixbeg, iy - iybeg, iz - izbeg) +=
        idx_a(a_f22, ix, iy, iz) * vx0 + idx_a(a_f23, ix, iy, iz) * vz0;
}

void
update_g_v(const int isub, cart_volume<float>* snap_f22_vol,
                                                       cart_volume<float>* snap_f23_vol, cart_volume<float>* g1_vol,
                                                       cart_volume<float>* a_f22_vol, cart_volume<float>* a_f23_vol, cart_volume<float>* f4)
{
    int ixbeg, ixend, iybeg, iyend, izbeg, izend;
    kernel_utils::getIndexRange(f4, ixbeg, ixend, iybeg, iyend, izbeg, izend);
    volume_index idx = snap_f22_vol->as<cart_volume_regular_gpu>()->vol_idx();
    volume_index idx_a = a_f22_vol->as<cart_volume_regular_gpu>()->vol_idx();
    auto snap_f22 = snap_f22_vol->as<cart_volume_regular_gpu>()->getData();
    auto snap_f23 = snap_f23_vol->as<cart_volume_regular_gpu>()->getData();
    auto g1 = g1_vol->as<cart_volume_regular_gpu>()->getData();
    auto a_f22 = a_f22_vol->as<cart_volume_regular_gpu>()->getData();
    auto a_f23 = a_f23_vol->as<cart_volume_regular_gpu>()->getData();

    int num_x = ixend - ixbeg + 1;
    int num_y = iyend - iybeg + 1;
    int num_z = izend - izbeg + 1;

    dim3 threads(32, 32, 1);
    dim3 blocks((num_x + threads.x - 1) / threads.x, (num_y + threads.y - 1) / threads.y,
                (num_z + threads.z - 1) / threads.z);

    update_gv<<<blocks, threads, 0>>>(
        idx, idx_a, snap_f22, snap_f23, g1, a_f22, a_f23, ixbeg, ixend, iybeg, iyend, izbeg, izend);
}

template <int ORD>
struct RegBase
{
    float d_f_f24_dx;
    float d_f_f25_dx;
    float d_f_f24_dy;
    float d_f_f25_dy;
    float d_f_f24_dz;
    float d_f_t25_dz;

    float d_a_f24_x;
    float d_a_f25_x;
    float d_a_f24_y;
    float d_a_f25_y;
    float d_a_f24_z;
    float d_a_f25_z;

    float g_rhs;

    float f_f24_dz[ORD + 1];
    float f_f25_dz[ORD + 1];

    //Half register queues
    float g_rhs_xy[ORD / 2 + 1];
    float m18[ORD / 2 + 1];

    float a_f24_z[ORD + 1];
    float a_f25_z[ORD + 1];
};

__device__ __forceinline__ float2
a(float const* __restrict__ f24, float const* __restrict__ f25, float const* __restrict__ m18,
    float const* __restrict__ m1, float const* __restrict__ m2, float const* __restrict__ m3,
    float const* __restrict__ m4, float const* __restrict__ ra, int i)
{
    float f24_ = f24[i];
    float f25_ = f25[i];
    float m18_ = 1.0f / m18[i];
    float m1_ = m1[i];
    float m2_ = m2[i];
    float m3_ = m3[i];
    float m4_ = m4[i];

    float ra_ra = ra[i] * ra[i];
    float lx_zz = (m4_ - m1_) * f24_ - m2_ * f25_;
    float lz_zz = m2_ * f24_ + (m3_ - m4_) * f25_;

    return { m18_ * (m1_ * f24_ + m2_ * f25_ + ra_ra * lx_zz), m18_ * (m4_ * f25_ + ra_ra * lz_zz) };
}

template <int ORD, bool RIGHT>
__device__ __forceinline__ float2
stg(float const* __restrict__ f24, float const* __restrict__ f25, float const* __restrict__ m18,
    float const* __restrict__ m1, float const* __restrict__ m2, float const* __restrict__ m3,
    float const* __restrict__ m4, float const* __restrict__ ra, float h, int i, int s)
{
    constexpr int P = RIGHT ? 1 : 0;
    constexpr int M = RIGHT ? 0 : 1;

    float dx = 0.0f;
    float dz = 0.0f;

#pragma unroll
    for (int r = 0; r < ORD / 2; ++r) {
        auto [xp, zp] = a(f24, f25, m18, m1, m2, m3, m4, ra, i + (r + P) * s);
        auto [xm, zm] = a(f24, f25, m18, m1, m2, m3, m4, ra, i - (r + M) * s);

        dx += stg_d1<ORD>(r) * (xp - xm);
        dz += stg_d1<ORD>(r) * (zp - zm);
    }

    return { dx * h, dz * h };
}

template <int ORD, bool RIGHT>
__device__ __forceinline__ float
stg(float const* __restrict__ t, float h, int i, int s)
{
    constexpr int P = RIGHT ? 1 : 0;
    constexpr int M = RIGHT ? 0 : 1;

    float d = 0.0f;

#pragma unroll
    for (int r = 0; r < ORD / 2; ++r)
        d += stg_d1<ORD>(r) * (t[i + (r + P) * s] - t[i - (r + M) * s]);

    return d * h;
}

template <int ORD, int BX, int BY>
struct smStruct
{
    float f24[BY + ORD][BX + ORD];
    float f25[BY + ORD][BX + ORD];
    float m1[BY + ORD][BX + ORD];
    float m2[BY + ORD][BX + ORD];
    float m3[BY + ORD][BX + ORD];
    float m4[BY + ORD][BX + ORD];
    float m18[BY + ORD][BX + ORD];
    float rx[BY + ORD][BX + ORD];
    float ry[BY + ORD][BX + ORD];
    float rz[BY + ORD][BX + ORD];

    float f_f24_dx[BY][BX + ORD];
    float f_f24_dy[BY + ORD][BX];
    float f_f24_dz[BY][BX];
    float f_f25_dx[BY][BX + ORD];
    float f_f25_dy[BY + ORD][BX];
    float f_f25_dz[BY][BX];

    float a_f24_x[BY][BX + ORD];
    float a_f25_x[BY][BX + ORD];
    float a_f24_y[BY + ORD][BX];
    float a_f25_y[BY + ORD][BX];
    float a_f24_z[BY][BX];
    float a_f25_z[BY][BX];
};

//
template <int ORD, int BX, int BY>
__device__ __forceinline__ void
compute_temp_a_arrays(smStruct<ORD, BX, BY>* sm, bool loader, int txload, int tyload, bool center_x, bool center_y,
                        bool z_only = false)
{

    const int r = ORD / 2;
    if (loader) {
        float4 rx = *reinterpret_cast<float4*>(&sm->rx[tyload][txload]);
        float4 ry = *reinterpret_cast<float4*>(&sm->ry[tyload][txload]);
        float4 rz = *reinterpret_cast<float4*>(&sm->rz[tyload][txload]);
        float4 m18 = *reinterpret_cast<float4*>(&sm->m18[tyload][txload]);
        float4 m1 = *reinterpret_cast<float4*>(&sm->m1[tyload][txload]);
        float4 m2 = *reinterpret_cast<float4*>(&sm->m2[tyload][txload]);
        float4 m3 = *reinterpret_cast<float4*>(&sm->m3[tyload][txload]);
        float4 m4 = *reinterpret_cast<float4*>(&sm->m4[tyload][txload]);
        float4 f24 = *reinterpret_cast<float4*>(&sm->f24[tyload][txload]);
        float4 f25 = *reinterpret_cast<float4*>(&sm->f25[tyload][txload]);

        // Using float4 math
        using namespace helper_kernels_gpu;

        float4 f24_ = f24;
        float4 f25_ = f25;
        float4 m18_ = 1.0f / m18;
        float4 m1_ = m1;
        float4 m2_ = m2;
        float4 m3_ = m3;
        float4 m4_ = m4;

        float4 rx_rx = rx * rx;
        float4 ry_ry = ry * ry;
        float4 rz_rz = rz * rz;
        float4 lx_zz = (m4_ - m1_) * f24_ - m2_ * f25_;
        float4 lz_zz = m2_ * f24_ + (m3_ - m4_) * f25_;

        // write a_f24_x, a_f25_x values (X halo but no Y halo)
        if (center_y) {
            if (!z_only) {
                *reinterpret_cast<float4*>(&sm->a_f24_x[tyload - r][txload]) =
                    m18_ * (m1_ * f24_ + m2_ * f25_ + rx_rx * lx_zz);
                *reinterpret_cast<float4*>(&sm->a_f25_x[tyload - r][txload]) = m18_ * (m4_ * f25_ + rx_rx * lz_zz);
            }
        }
        // Write a_f24_y, a_f25_y values (Y halo but no X halo)
        if (center_x) {
            if (!z_only) {
                *reinterpret_cast<float4*>(&sm->a_f24_y[tyload][txload - r]) =
                    m18_ * (m1_ * f24_ + m2_ * f25_ + ry_ry * lx_zz);
                *reinterpret_cast<float4*>(&sm->a_f25_y[tyload][txload - r]) = m18_ * (m4_ * f25_ + ry_ry * lz_zz);
            }
        }
        // Write a_f24_z, a_f25_z values (no Y halo and no X halo)
        if (center_x && center_y) {
            *reinterpret_cast<float4*>(&sm->a_f24_z[tyload - r][txload - r]) =
                m18_ * (m1_ * f24_ + m2_ * f25_ + rz_rz * lx_zz);
            *reinterpret_cast<float4*>(&sm->a_f25_z[tyload - r][txload - r]) = m18_ * (m4_ * f25_ + rz_rz * lz_zz);
        }
    }
}

template <int ORD, int BX, int BY>
__device__ inline void
compute_z_values(RegBase<ORD>& r, float hz, int i, int g_ig) //i = ORD/2
{
    // Calculate forward second derivatives.
    r.d_f_f24_dz = stg<ORD, true>(r.f_f24_dz, hz, i, 1);
    r.d_f_t25_dz = stg<ORD, true>(r.f_f25_dz, hz, i, 1);

    r.d_a_f24_z = stg<ORD, false>(r.a_f24_z, hz, i, 1);
    r.d_a_f25_z = stg<ORD, false>(r.a_f25_z, hz, i, 1);
}

template <int ORD, int BX, int BY>
__device__ inline void
async_load(smStruct<ORD, BX, BY>* sm, int index_load, bool loader, bool center_x, bool center_y, int txload, int tyload,
           const float* f24, const float* f25, const float* M1, const float* M2, const float* M3, const float* M4,
           const float* Rx, const float* Ry, const float* Rz, const float* m18, const float* f_f24_dx,
           const float* f_f24_dy, const float* f_f24_dz, const float* f_f25_dx, const float* f_f25_dy,
           const float* f_f25_dz)
{
    constexpr int radius = ORD / 2;

    if (loader) {
        __pipeline_memcpy_async(&sm->f24[tyload][txload], f24 + index_load, sizeof(float4));
        __pipeline_memcpy_async(&sm->f25[tyload][txload], f25 + index_load, sizeof(float4));
        __pipeline_memcpy_async(&sm->m1[tyload][txload], M1 + index_load, sizeof(float4));
        __pipeline_memcpy_async(&sm->m2[tyload][txload], M2 + index_load, sizeof(float4));
        __pipeline_memcpy_async(&sm->m3[tyload][txload], M3 + index_load, sizeof(float4));
        __pipeline_memcpy_async(&sm->m4[tyload][txload], M4 + index_load, sizeof(float4));
        __pipeline_memcpy_async(&sm->rx[tyload][txload], Rx + index_load, sizeof(float4));
        __pipeline_memcpy_async(&sm->ry[tyload][txload], Ry + index_load, sizeof(float4));
        __pipeline_memcpy_async(&sm->rz[tyload][txload], Rz + index_load, sizeof(float4));
        __pipeline_memcpy_async(&sm->m18[tyload][txload], m18 + index_load, sizeof(float4));
        if (center_y) {
            __pipeline_memcpy_async(&sm->f_f24_dx[tyload - radius][txload], f_f24_dx + index_load, sizeof(float4));
            __pipeline_memcpy_async(&sm->f_f25_dx[tyload - radius][txload], f_f25_dx + index_load, sizeof(float4));
        }
        if (center_x) {
            __pipeline_memcpy_async(&sm->f_f24_dy[tyload][txload - radius], f_f24_dy + index_load, sizeof(float4));
            __pipeline_memcpy_async(&sm->f_f25_dy[tyload][txload - radius], f_f25_dy + index_load, sizeof(float4));
        }
        if (center_x && center_y) {
            __pipeline_memcpy_async(&sm->f_f24_dz[tyload - radius][txload - radius], f_f24_dz + index_load,
                                    sizeof(float4));
            __pipeline_memcpy_async(&sm->f_f25_dz[tyload - radius][txload - radius], f_f25_dz + index_load,
                                    sizeof(float4));
        }
    }
    __pipeline_commit();
}

template <int ORD, int BX, int BY>
__device__ inline void
populate_register_queues(const smStruct<ORD, BX, BY>* sm, RegBase<ORD>& r, int i, int ty, int tx)
{
    constexpr int radius = ORD / 2;
    r.f_f24_dz[i] = sm->f_f24_dz[ty - radius][tx - radius];
    r.f_f25_dz[i] = sm->f_f25_dz[ty - radius][tx - radius];
    r.a_f24_z[i] = sm->a_f24_z[ty - radius][tx - radius];
    r.a_f25_z[i] = sm->a_f25_z[ty - radius][tx - radius];
}

template <int ORD, int BX, int BY>
__device__ __forceinline__ void
compute_xy_contributions(smStruct<ORD, BX, BY>* sm, float hy, float hx, int iy, int ix, float& d_f_f24_dx,
                         float& d_f_f25_dx, float& d_f_f24_dy, float& d_f_f25_dy, float& d_a_f24_x,
                         float& d_a_f25_x, float& d_a_f24_y, float& d_a_f25_y)
{

    //Our ix and iy indices will start from 0,0 here.  The shared mem arrays we are computing will
    //either have an X halo or a Y halo of size ORD / 2, but not both.  So we shift the
    //appropriate index to ensure that if there are X halos then the x index includes them
    //so we are computing only in the interior region.
    constexpr int radius = ORD / 2;
    int ix_halo = ix + radius;
    int iy_halo = iy + radius;

    //No Y halos
    d_f_f24_dx = stg<ORD, true>(&sm->f_f24_dx[0][0], hx, (iy * (BX + ORD)) + ix_halo, 1);
    d_f_f25_dx = stg<ORD, true>(&sm->f_f25_dx[0][0], hx, (iy * (BX + ORD)) + ix_halo, 1);

    //No X halos
    d_f_f24_dy = stg<ORD, true>(&sm->f_f24_dy[0][0], hy, (iy_halo * (BX)) + ix, BX);
    d_f_f25_dy = stg<ORD, true>(&sm->f_f25_dy[0][0], hy, (iy_halo * (BX)) + ix, BX);

    //No Y halos
    d_a_f24_x = stg<ORD, false>(&sm->a_f24_x[0][0], hx, (iy * (BX + ORD)) + ix_halo, 1);
    d_a_f25_x = stg<ORD, false>(&sm->a_f25_x[0][0], hx, (iy * (BX + ORD)) + ix_halo, 1);

    //No X halos
    d_a_f24_y = stg<ORD, false>(&sm->a_f24_y[0][0], hy, (iy_halo * (BX)) + ix, BX);
    d_a_f25_y = stg<ORD, false>(&sm->a_f25_y[0][0], hy, (iy_halo * (BX)) + ix, BX);
}

template <int ORD>
__device__ inline void
shift_register_queues(RegBase<ORD>& r)
{
#pragma unroll
    for (int i = 1; i < ORD + 1; ++i) {
        r.a_f24_z[i - 1] = r.a_f24_z[i];
        r.a_f25_z[i - 1] = r.a_f25_z[i];

        r.f_f24_dz[i - 1] = r.f_f24_dz[i];
        r.f_f25_dz[i - 1] = r.f_f25_dz[i];
        if (i < ORD / 2 + 1) {
            r.g_rhs_xy[i - 1] = r.g_rhs_xy[i];
            r.m18[i - 1] = r.m18[i];
        }
    }

    r.a_f24_z[ORD] = 0.0f;
    r.a_f25_z[ORD] = 0.0f;

    r.f_f24_dz[ORD] = 0.0f;
    r.f_f25_dz[ORD] = 0.0f;
    r.g_rhs_xy[ORD / 2] = 0.0f;
    r.m18[ORD / 2] = 0.0f;
}

template <int ORD, int BX, int BY, int BZ>
__global__
__launch_bounds__(BX* BY) void compute_g_r(
    float* __restrict__ g, float const* __restrict__ f_f24_dx, float const* __restrict__ f_f25_dx,
    float const* __restrict__ f_f24_dy, float const* __restrict__ f_f25_dy, float const* __restrict__ f_f24_dz,
    float const* __restrict__ f_f25_dz, float const* __restrict__ f24, float const* __restrict__ f25,
    float const* __restrict__ m18, float const* __restrict__ m1, float const* __restrict__ m2,
    float const* __restrict__ m3, float const* __restrict__ m4, float const* __restrict__ rx,
    float const* __restrict__ ry, float const* __restrict__ rz, int nx, int ny, int nz, int ldimx, int ldimy,
    int stride_y, int stride_z, int g_stride_y, int g_stride_z, float hx, float hy, float hz)
{
    RegBase<ORD> r;

    const int ix = blockIdx.x * BX + threadIdx.x;
    const int iy = blockIdx.y * BY + threadIdx.y;
    const int iz = blockIdx.z * BZ + threadIdx.z;

    // Global position of the thread block, using 2D thread blocks
    const int ixblock = blockIdx.x * BX;
    const int iyblock = blockIdx.y * BY;
    const int izblock = blockIdx.z * BZ;
    const int stride = ldimx * ldimy;
    constexpr int radius = ORD / 2;
    const bool active = ix < nx && iy < ny;

    // Dynamic shared memory. Using a non-templated pointer to get the base address,
    // because we can't have a templated extern declaration with different template args,
    // and we need dynamic shared memory to use > 48KB per thread block.
    extern __shared__ char dsmem[];
    // Now we can cast the base pointer to the proper templated struct
    smStruct<ORD, BX, BY>* sm = reinterpret_cast<smStruct<ORD, BX, BY>*>(dsmem);

    // Local thread block shared memory indexes.
    const int lx = threadIdx.x + ORD / 2; // Skip past halo.
    const int ly = threadIdx.y + ORD / 2; // Skip past halo.
    const int lz = threadIdx.z;

    int g_ig = iz * g_stride_z + iy * g_stride_y + ix;

    //----------------------------------
    //remap threads to indices so we can load entire shared mem tile
    //with float4 in one load.

    // Load the shared memory as float4, and remap the threads from (bx, by) into (bx/2, by*2)
    // These reads are offset by -radius from the block offset, to read the halos.
    const int tid = threadIdx.y * BX + threadIdx.x;
    static_assert(BX >= ORD && BY >= ORD, "Incompatible block size for this order");
    const int txload = 4 * (tid % (BX / 2));
    const int tyload = tid / (BX / 2);
    const int ixload = ixblock + txload - radius;
    const int iyload = iyblock + tyload - radius;
    const bool loader = ((BX == ORD || txload < BX + ORD) && (BY == ORD || tyload < BY + ORD)) &&
                        (ixload < nx + radius && iyload < ny + radius);

    int index_load = ((izblock - radius) * ldimy + iyload) * ldimx + ixload;

    // The threads will compute the temp shared arrays with the same float4 thread mapping.
    // We need to identify the threads which are not in the X or Y halos
    const bool center_x = txload >= radius && txload < BX + radius;
    const bool center_y = tyload >= radius && tyload < BY + radius;

    //---------------------------------------------------------------------------------

    async_load(sm, index_load, loader, center_x, center_y, txload, tyload, f24, f25, m1, m2, m3, m4, rx, ry, rz,
               m18, f_f24_dx, f_f24_dy, f_f24_dz, f_f25_dx, f_f25_dy, f_f25_dz);
    index_load += stride;

    // Prime the register queues
    // Loop must be unrolled to allow proper inlining and resolution of FD coefficients.
    int nzloop = min(BZ, nz - izblock);
#pragma unroll
    for (int i = 0; i < ORD + 1; i++) {
        //----------------------------------------------------------------
        //Wait for async load and then sync threads
        __pipeline_wait_prior(0);
        __syncthreads();

        compute_temp_a_arrays<ORD, BX, BY>(sm, loader, txload, tyload, center_x, center_y, i < radius);
        __syncthreads();

        populate_register_queues(sm, r, i, ly, lx);

        // Compute the X Y contributions of the temp arrays once we're out of the Z halo
        if (i >= radius) {
            compute_xy_contributions(sm, hy, hx, threadIdx.y, threadIdx.x, r.d_f_f24_dx, r.d_f_f25_dx,
                                     r.d_f_f24_dy, r.d_f_f25_dy, r.d_a_f24_x, r.d_a_f25_x, r.d_a_f24_y,
                                     r.d_a_f25_y);

            r.g_rhs_xy[i - ORD / 2] =
                sm->a_f24_x[ly - radius][lx] * r.d_f_f24_dx + sm->a_f25_x[ly - radius][lx] * r.d_f_f25_dx +
                sm->a_f24_y[ly][lx - radius] * r.d_f_f24_dy + sm->a_f25_y[ly][lx - radius] * r.d_f_f25_dy +
                r.d_a_f24_x * sm->f_f24_dx[ly - radius][lx] + r.d_a_f25_x * sm->f_f25_dx[ly - radius][lx] +
                r.d_a_f24_y * sm->f_f24_dy[ly][lx - radius] + r.d_a_f25_y * sm->f_f25_dy[ly][lx - radius];
            r.m18[i - ORD / 2] = sm->m18[ly][lx];
            __syncthreads();
        }
        // If this zblock has only 1 plane, we have to skip the last out of bounds load
        if (!(i == ORD && nzloop == 1)) {
            async_load(sm, index_load, loader, center_x, center_y, txload, tyload, f24, f25, m1, m2, m3, m4, rx, ry,
                       rz, m18, f_f24_dx, f_f24_dy, f_f24_dz, f_f25_dx, f_f25_dy, f_f25_dz);
            index_load += stride;
        }
    }

    // Loop on the Z dimension, except the last one
    for (int izloop = 0; izloop < nzloop - 1; izloop++) {

        // Compute the Z values
        compute_z_values<ORD, BX, BY>(r, hz, ORD / 2, g_ig);

        if (active) {
            //write the results
            r.g_rhs = r.g_rhs_xy[0] + r.a_f24_z[ORD / 2] * r.d_f_f24_dz +
                         r.a_f25_z[ORD / 2] * r.d_f_t25_dz + r.d_a_f24_z * r.f_f24_dz[ORD / 2] +
                         r.d_a_f25_z * r.f_f25_dz[ORD / 2];

            atomicAdd_block(&g[g_ig], -r.m18[0] * r.g_rhs);
        }

        shift_register_queues(r);

        //Wait for async load and then sync threads
        __pipeline_wait_prior(0);
        __syncthreads();

        compute_temp_a_arrays<ORD, BX, BY>(sm, loader, txload, tyload, center_x, center_y, izloop > nzloop - radius);
        __syncthreads();

        populate_register_queues(sm, r, ORD, ly, lx);

        if (izloop < nzloop - radius) {
            compute_xy_contributions(sm, hy, hx, threadIdx.y, threadIdx.x, r.d_f_f24_dx, r.d_f_f25_dx,
                                     r.d_f_f24_dy, r.d_f_f25_dy, r.d_a_f24_x, r.d_a_f25_x, r.d_a_f24_y,
                                     r.d_a_f25_y);
            r.g_rhs_xy[ORD / 2] =
                sm->a_f24_x[ly - radius][lx] * r.d_f_f24_dx + sm->a_f25_x[ly - radius][lx] * r.d_f_f25_dx +
                sm->a_f24_y[ly][lx - radius] * r.d_f_f24_dy + sm->a_f25_y[ly][lx - radius] * r.d_f_f25_dy +
                r.d_a_f24_x * sm->f_f24_dx[ly - radius][lx] + r.d_a_f25_x * sm->f_f25_dx[ly - radius][lx] +
                r.d_a_f24_y * sm->f_f24_dy[ly][lx - radius] + r.d_a_f25_y * sm->f_f25_dy[ly][lx - radius];
            r.m18[ORD / 2] = sm->m18[ly][lx];

            __syncthreads();
        }

        if (izloop < nzloop - 2) {
            async_load(sm, index_load, loader, center_x, center_y, txload, tyload, f24, f25, m1, m2, m3, m4, rx, ry,
                       rz, m18, f_f24_dx, f_f24_dy, f_f24_dz, f_f25_dx, f_f25_dy, f_f25_dz);
        }

        index_load += stride;
        g_ig += g_stride_z;
    }

    // Compute the final Z value
    compute_z_values<ORD, BX, BY>(r, hz, ORD / 2, g_ig);

    if (active) {
        //write the results
        r.g_rhs = r.g_rhs_xy[0] + r.a_f24_z[ORD / 2] * r.d_f_f24_dz + r.a_f25_z[ORD / 2] * r.d_f_t25_dz +
                     r.d_a_f24_z * r.f_f24_dz[ORD / 2] + r.d_a_f25_z * r.f_f25_dz[ORD / 2];
        atomicAdd_block(&g[g_ig], -r.m18[0] * r.g_rhs);
    }
}

template <int ORD>
void
launch_r(float* g, float const* f_f24_dx, float const* f_f25_dx, float const* f_f24_dy,
           float const* f_f25_dy, float const* f_f24_dz, float const* f_f25_dz, float const* f24,
           float const* f25, float const* m18, float const* m1, float const* m2, float const* m3, float const* m4,
           float const* rx, float const* ry, float const* rz, int nx, int ny, int nz, int ldimx, int ldimy,
           int stride_y, int stride_z, int g_stride_y, int g_stride_z, float hx, float hy, float hz, bool simple_kernel,
           cudaStream_t stream)
{

    // A block size of 32 x 16 should work well for both 8th order and 16th order.
    // There are register spills for the 16th order kernel with these block sizes,
    // but profiling results showed these were still more efficient than other block sizes.
    constexpr int BX = 32;
    constexpr int BY = 16;
    constexpr int BZ = 64;

    unsigned gx = (nx + BX - 1) / BX;
    unsigned gy = (ny + BY - 1) / BY;
    unsigned gz = (nz + BZ - 1) / BZ;

    constexpr auto KERNEL = compute_g_r<ORD, BX, BY, BZ>;

    size_t shm = sizeof(smStruct<ORD, BX, BY>);
    cudaFuncSetAttribute(KERNEL, cudaFuncAttributeMaxDynamicSharedMemorySize, shm);

    KERNEL<<<dim3{ gx, gy, gz }, dim3{ BX, BY, 1 }, shm, stream>>>(
        g, f_f24_dx, f_f25_dx, f_f24_dy, f_f25_dy, f_f24_dz, f_f25_dz, f24, f25, m18, m1, m2, m3,
        m4, rx, ry, rz, nx, ny, nz, ldimx, ldimy, stride_y, stride_z, g_stride_y, g_stride_z, hx, hy, hz);
    CUDA_CHECK_ERROR(__FILE__, __LINE__);
}

void
launch_update_g_r(float* g, float const* f_f24_dx, float const* f_f25_dx, float const* f_f24_dy,
                               float const* f_f25_dy, float const* f_f24_dz, float const* f_f25_dz,
                               float const* f24, float const* f25, float const* m18, float const* m1,
                               float const* m2, float const* m3, float const* m4, float const* rx, float const* ry,
                               float const* rz, int ixbeg, int ixend, int iybeg, int iyend, int izbeg, int izend,
                               int ldimx, int ldimy, int g_ldimx, int g_ldimy, double dx, double dy, double dz,
                               int order, bool simple_kernel, cudaStream_t stream)
{
    EMSL_VERIFY(order == 8 || order == 12 || order == 16);
    if (ixend < ixbeg || iyend < iybeg || izend < izbeg)
        return;

    //Number of points
    int nx = ixend - ixbeg + 1;
    int ny = iyend - iybeg + 1;
    int nz = izend - izbeg + 1;

    // Input Y and Z strides
    int stride_y = ldimx;
    int stride_z = ldimx * ldimy;

    int g_stride_y = g_ldimx;
    int g_stride_z = g_ldimx * g_ldimy;

    //Shift for cart vol data we are reading/
    int shift = izbeg * stride_z + iybeg * stride_y + ixbeg;
    // g is not shifted.
    f_f24_dx += shift;
    f_f25_dx += shift;
    f_f24_dy += shift;
    f_f25_dy += shift;
    f_f24_dz += shift;
    f_f25_dz += shift;
    f24 += shift;
    f25 += shift;
    m18 += shift;
    m1 += shift;
    m2 += shift;
    m3 += shift;
    m4 += shift;
    rx += shift;
    ry += shift;
    rz += shift;

    float hx, hy, hz, unused;
    kernel_utils::compute_fd_const(dx, dy, dz, hx, hy, hz, unused, unused, unused, unused, unused, unused);

    launch_r<8>(g, f_f24_dx, f_f25_dx, f_f24_dy, f_f25_dy, f_f24_dz, f_f25_dz, f24, f25,
                               m18, m1, m2, m3, m4, rx, ry, rz, nx, ny, nz, ldimx, ldimy, stride_y, stride_z,
                               g_stride_y, g_stride_z, hx, hy, hz, simple_kernel, stream);
}

void
update_g_r_impl(
    const int isub, cart_volume<float>* f_df24dx_vol, cart_volume<float>* f_df24dy_vol,
    cart_volume<float>* f_df24dz_vol, cart_volume<float>* f_df25dx_vol, cart_volume<float>* f_df25dy_vol,
    cart_volume<float>* f_df25dz_vol, cart_volume<float>* g_r_vol, cart_volume<float>* f4, cart_volume<float>* f6,
    cart_volume<float>* M18, cart_volume<float>* M1, cart_volume<float>* M2, cart_volume<float>* M3, cart_volume<float>* M4,
    cart_volume<float>* Rx, cart_volume<float>* Ry, cart_volume<float>* Rz, double d1, double d2, double d3)
{
    // Get the volume dimensions
    int ixbeg = f4->as<cart_volume_regular_gpu>()->ax1()->ibeg;
    int ixend = f4->as<cart_volume_regular_gpu>()->ax1()->iend;
    int iybeg = f4->as<cart_volume_regular_gpu>()->ax2()->ibeg;
    int iyend = f4->as<cart_volume_regular_gpu>()->ax2()->iend;
    int izbeg = f4->as<cart_volume_regular_gpu>()->ax3()->ibeg;
    int izend = f4->as<cart_volume_regular_gpu>()->ax3()->iend;

    // Get the volume leading dimensions
    int ldimx = f4->as<cart_volume_regular_gpu>()->ax1()->ntot;
    int ldimy = f4->as<cart_volume_regular_gpu>()->ax2()->ntot;
    int g_ldimx = g_r_vol->as<cart_volume_regular_gpu>()->ax1()->ntot;
    int g_ldimy = g_r_vol->as<cart_volume_regular_gpu>()->ax2()->ntot;

    float* g = g_r_vol->as<cart_volume_regular_gpu>()->getData();
    float const* f_f24_dx = f_df24dx_vol->as<cart_volume_regular_gpu>()->getData();
    float const* f_f25_dx = f_df25dx_vol->as<cart_volume_regular_gpu>()->getData();
    float const* f_f24_dy = f_df24dy_vol->as<cart_volume_regular_gpu>()->getData();
    float const* f_f25_dy = f_df25dy_vol->as<cart_volume_regular_gpu>()->getData();
    float const* f_f24_dz = f_df24dz_vol->as<cart_volume_regular_gpu>()->getData();
    float const* f_f25_dz = f_df25dz_vol->as<cart_volume_regular_gpu>()->getData();
    int order = 8;
    bool simple_g = false;

    launch_update_g_r(
        g, f_f24_dx, f_f25_dx, f_f24_dy, f_f25_dy, f_f24_dz, f_f25_dz, f4->as<cart_volume_regular_gpu>()->getData(),
        f6->as<cart_volume_regular_gpu>()->getData(), M18->as<cart_volume_regular_gpu>()->getData(),
        M1->as<cart_volume_regular_gpu>()->getData(), M2->as<cart_volume_regular_gpu>()->getData(), M3->as<cart_volume_regular_gpu>()->getData(),
        M4->as<cart_volume_regular_gpu>()->getData(), Rx->as<cart_volume_regular_gpu>()->getData(), Ry->as<cart_volume_regular_gpu>()->getData(),
        Rz->as<cart_volume_regular_gpu>()->getData(), ixbeg, ixend, iybeg, iyend, izbeg, izend, ldimx, ldimy, g_ldimx, g_ldimy, d1, d2, d3, order, simple_g, 0);
}

template <int ORD, int BX, int BY, int BZ>
__global__
__launch_bounds__(BY* BX* BZ) void compute_g_r_simple(
    float* __restrict__ g, float const* __restrict__ f_f24_dx, float const* __restrict__ f_f25_dx,
    float const* __restrict__ f_f24_dy, float const* __restrict__ f_f25_dy, float const* __restrict__ f_f24_dz,
    float const* __restrict__ f_f25_dz, float const* __restrict__ f24, float const* __restrict__ f25,
    float const* __restrict__ m18, float const* __restrict__ m1, float const* __restrict__ m2,
    float const* __restrict__ m3, float const* __restrict__ m4, float const* __restrict__ rx,
    float const* __restrict__ ry, float const* __restrict__ rz, int nx, int ny, int nz, int stride_y, int stride_z,
    int g_stride_y, int g_stride_z, float hx, float hy, float hz)
{
    int const tx = threadIdx.x;
    int const ty = threadIdx.y;
    int const tz = threadIdx.z;
    int const ix = blockIdx.x * BX + tx;
    int const iy = blockIdx.y * BY + ty;
    int const iz = blockIdx.z * BZ + tz;

    if (ix >= nx || iy >= ny || iz >= nz)
        return;

    int const ig = iz * stride_z + iy * stride_y + ix;
    int const g_ig = iz * g_stride_z + iy * g_stride_y + ix;

    auto [a_f24_x, a_f25_x] = a(f24, f25, m18, m1, m2, m3, m4, rx, ig);
    auto [a_f24_y, a_f25_y] = a(f24, f25, m18, m1, m2, m3, m4, ry, ig);
    auto [a_f24_z, a_f25_z] = a(f24, f25, m18, m1, m2, m3, m4, rz, ig);

    float d_f_f24_dx = stg<ORD, true>(f_f24_dx, hx, ig, 1);
    float d_f_f25_dx = stg<ORD, true>(f_f25_dx, hx, ig, 1);
    float d_f_f24_dy = stg<ORD, true>(f_f24_dy, hy, ig, stride_y);
    float d_f_f25_dy = stg<ORD, true>(f_f25_dy, hy, ig, stride_y);
    float d_f_f24_dz = stg<ORD, true>(f_f24_dz, hz, ig, stride_z);
    float d_f_t25_dz = stg<ORD, true>(f_f25_dz, hz, ig, stride_z);

    auto [d_a_f24_x, d_a_f25_x] = stg<ORD, false>(f24, f25, m18, m1, m2, m3, m4, rx, hx, ig, 1);
    auto [d_a_f24_y, d_a_f25_y] = stg<ORD, false>(f24, f25, m18, m1, m2, m3, m4, ry, hy, ig, stride_y);
    auto [d_a_f24_z, d_a_f25_z] = stg<ORD, false>(f24, f25, m18, m1, m2, m3, m4, rz, hz, ig, stride_z);

    float g_rhs = a_f24_x * d_f_f24_dx + a_f25_x * d_f_f25_dx + a_f24_y * d_f_f24_dy +
                     a_f25_y * d_f_f25_dy + a_f24_z * d_f_f24_dz + a_f25_z * d_f_t25_dz +
                     d_a_f24_x * f_f24_dx[ig] + d_a_f25_x * f_f25_dx[ig] + d_a_f24_y * f_f24_dy[ig] +
                     d_a_f25_y * f_f25_dy[ig] + d_a_f24_z * f_f24_dz[ig] + d_a_f25_z * f_f25_dz[ig];

    float g_rhs_xy = a_f24_x * d_f_f24_dx + a_f25_x * d_f_f25_dx + a_f24_y * d_f_f24_dy +
                        a_f25_y * d_f_f25_dy + d_a_f24_x * f_f24_dx[ig] + d_a_f25_x * f_f25_dx[ig] +
                        d_a_f24_y * f_f24_dy[ig] + d_a_f25_y * f_f25_dy[ig];

    atomicAdd_block(&g[g_ig], -m18[ig] * g_rhs);
}

void simulator3SetUp::execute(bool has_src_p, bool has_qp_)
{
    float milliseconds_v = 0;
    float milliseconds_g = 0;
    float milliseconds_s = 0;
    float milliseconds_h = 0;

    float elapsedTime = 0;
    TIMED_BLOCK("Total") {
        float max_pressure = 500.0f;
        TIMED_BLOCK("Forward") {
            if (has_src_p) { // source group in lib/grid/src_group.cpp
                initializeSource(max_pressure);
            }

            for (int iter = 0; iter < _niter; ++iter) {
                cudaEvent_t start_halo, stop_halo;
                cudaEventCreate(&start_halo);
                cudaEventCreate(&stop_halo);            

                cudaEventRecord(start_halo);
                cudaEventSynchronize(start_halo);
                halomgr_s->start_update();
                halomgr_s->finish_update();
                cudaEventRecord(stop_halo);
                cudaEventSynchronize(stop_halo);
                elapsedTime = 0;
                cudaEventElapsedTime(&elapsedTime, start_halo, stop_halo);
                milliseconds_h +=elapsedTime;

                for (int isub = 0; isub < nsubs; ++isub) {
                    float* f1Data = f1[isub]->as<cart_volume_regular_gpu>()->getData();
                    float* f2Data = f2[isub]->as<cart_volume_regular_gpu>()->getData();
                    float* f3Data = f3[isub]->as<cart_volume_regular_gpu>()->getData();
                    float* f4Data = f4[isub]->as<cart_volume_regular_gpu>()->getData();
                    float* f5Data = f5[isub]->as<cart_volume_regular_gpu>()->getData();
                    float* f6Data = f6[isub]->as<cart_volume_regular_gpu>()->getData();
                    float* f7Data = f7[isub]->as<cart_volume_regular_gpu>()->getData();
                    float* f8Data = f8[isub]->as<cart_volume_regular_gpu>()->getData();
                    float* f9Data = f9[isub]->as<cart_volume_regular_gpu>()->getData();
                    float* M22Data = M22[isub]->as<cart_volume_regular_gpu>()->getData();

                    int subid[3];
                    decompmgr->getSplitLocalSubDomID(isub, subid); // Change 0 to isub
                    axis* ax1 = local_X_axes[subid[0]];
                    axis* ax2 = local_Y_axes[subid[1]];
                    axis* ax3 = local_Z_axes[subid[2]];

                    vPlainLoop(f1Data, f2Data, f3Data, f4Data, f5Data, f6Data, f7Data, f8Data, f9Data, 
                                    M22Data, zprime[isub], ax1->n, ax2->n, ax3->n, ax1->ntot, ax2->ntot, milliseconds_v);
                }

                // halo exchange
                cudaEventRecord(start_halo);
                cudaEventSynchronize(start_halo);
                halomgr_v->start_update();            
                halomgr_v->finish_update();
                cudaEventRecord(stop_halo);
                cudaEventSynchronize(stop_halo);
                elapsedTime = 0;
                cudaEventElapsedTime(&elapsedTime, start_halo, stop_halo);
                milliseconds_h +=elapsedTime;                

                for (int isub = 0; isub < nsubs; ++isub) {
                    mirror_s(isub, 0, f4, f5, f6, f7, f8, f9);
                }

                for (int isub = 0; isub < nsubs; ++isub) {
                    mirror_s(isub, 0, f4, f5, f6, f7, f8, f9);
                }

                for (int isub = 0; isub < nsubs; ++isub) {
                    mirror_v(isub, 0, f1, f2, f3);
                }

                for (int isub = 0; isub < nsubs; ++isub) {
                    mirror_v(isub, 0, f1, f2, f3);

                }
            
                for (int isub = 0; isub < nsubs; ++isub) {
                    float* f1Data = f1[isub]->as<cart_volume_regular_gpu>()->getData();
                    float* f2Data = f2[isub]->as<cart_volume_regular_gpu>()->getData();
                    float* f3Data = f3[isub]->as<cart_volume_regular_gpu>()->getData();

                    float *d4Data = d4[isub]->as<cart_volume_regular_gpu>()->getData();
                    float *d5Data = d5[isub]->as<cart_volume_regular_gpu>()->getData();
                    float *d6Data = d6[isub]->as<cart_volume_regular_gpu>()->getData();
                    float *d7Data = d7[isub]->as<cart_volume_regular_gpu>()->getData();
                    float *d8Data = d8[isub]->as<cart_volume_regular_gpu>()->getData();
                    float *d9Data = d9[isub]->as<cart_volume_regular_gpu>()->getData();

                    int subid[3];
                    decompmgr->getSplitLocalSubDomID(isub, subid); // Change 0 to isub
                    axis* ax1 = local_X_axes[subid[0]];
                    axis* ax2 = local_Y_axes[subid[1]];
                    axis* ax3 = local_Z_axes[subid[2]];

                    vgPlainLoop(f1Data, f2Data, f3Data, d4Data, d5Data, d6Data, d7Data, d8Data, d9Data, 
                                            zprime[isub], ax1->n, ax2->n, ax3->n, ax1->ntot, ax2->ntot, milliseconds_g);
                }

                // halo exchange
                cudaEventRecord(start_halo);
                cudaEventSynchronize(start_halo);
                halomgr_vd->start_update();
                halomgr_vd->finish_update();
                cudaEventRecord(stop_halo);
                cudaEventSynchronize(stop_halo);
                elapsedTime = 0;
                cudaEventElapsedTime(&elapsedTime, start_halo, stop_halo);
                milliseconds_h +=elapsedTime;

                for (int isub = 0; isub < nsubs; ++isub) {
                    float* f4Data = f4[isub]->as<cart_volume_regular_gpu>()->getData();
                    float* f5Data = f5[isub]->as<cart_volume_regular_gpu>()->getData();
                    float* f6Data = f6[isub]->as<cart_volume_regular_gpu>()->getData();
                    float* f7Data = f7[isub]->as<cart_volume_regular_gpu>()->getData();
                    float* f8Data = f8[isub]->as<cart_volume_regular_gpu>()->getData();
                    float* f9Data = f9[isub]->as<cart_volume_regular_gpu>()->getData();

                    float *d4Data = d4[isub]->as<cart_volume_regular_gpu>()->getData();
                    float *d5Data = d5[isub]->as<cart_volume_regular_gpu>()->getData();
                    float *d6Data = d6[isub]->as<cart_volume_regular_gpu>()->getData();
                    float *d7Data = d7[isub]->as<cart_volume_regular_gpu>()->getData();
                    float *d8Data = d8[isub]->as<cart_volume_regular_gpu>()->getData();
                    float *d9Data = d9[isub]->as<cart_volume_regular_gpu>()->getData();

                    float* M1Data = M1[isub]->as<cart_volume_regular_gpu>()->getData();
                    float* M2Data = M2[isub]->as<cart_volume_regular_gpu>()->getData();
                    float* M3Data = M3[isub]->as<cart_volume_regular_gpu>()->getData();
                    float* M4Data = M4[isub]->as<cart_volume_regular_gpu>()->getData();
                    float* M5Data = M5[isub]->as<cart_volume_regular_gpu>()->getData();
                    float* M6Data = M6[isub]->as<cart_volume_regular_gpu>()->getData();
                    float* M7Data = M7[isub]->as<cart_volume_regular_gpu>()->getData();
                    float* M8Data = M8[isub]->as<cart_volume_regular_gpu>()->getData();
                    float* M9Data = M9[isub]->as<cart_volume_regular_gpu>()->getData();
                    float* M10Data = M10[isub]->as<cart_volume_regular_gpu>()->getData();
                    float* M11Data = M11[isub]->as<cart_volume_regular_gpu>()->getData();
                    float* M12Data = M12[isub]->as<cart_volume_regular_gpu>()->getData();
                    float* M13Data = M13[isub]->as<cart_volume_regular_gpu>()->getData();
                    float* M14Data = M14[isub]->as<cart_volume_regular_gpu>()->getData();
                    float* M15Data = M15[isub]->as<cart_volume_regular_gpu>()->getData();
                    float* M16Data = M16[isub]->as<cart_volume_regular_gpu>()->getData();
                    float* M17Data = M17[isub]->as<cart_volume_regular_gpu>()->getData();
                    float* M18Data = M18[isub]->as<cart_volume_regular_gpu>()->getData();
                    float* M19Data = M19[isub]->as<cart_volume_regular_gpu>()->getData();
                    float* M20Data = M20[isub]->as<cart_volume_regular_gpu>()->getData();
                    float* M21Data = M21[isub]->as<cart_volume_regular_gpu>()->getData();
                    float* f10Data = f10[isub]->as<cart_volume_regular_gpu>()->getData();
                    float* f11Data = f11[isub]->as<cart_volume_regular_gpu>()->getData();
                    float* f12Data = f12[isub]->as<cart_volume_regular_gpu>()->getData();
                    float* f13Data = f13[isub]->as<cart_volume_regular_gpu>()->getData();
                    float* f14Data = f14[isub]->as<cart_volume_regular_gpu>()->getData();
                    float* f15Data = f15[isub]->as<cart_volume_regular_gpu>()->getData();
                    float* f16Data = f16[isub]->as<cart_volume_regular_gpu>()->getData();
                    float* f17Data = f17[isub]->as<cart_volume_regular_gpu>()->getData();
                    float* f18Data = f18[isub]->as<cart_volume_regular_gpu>()->getData();
                    float* f19Data = f19[isub]->as<cart_volume_regular_gpu>()->getData();
                    float* f20Data = f20[isub]->as<cart_volume_regular_gpu>()->getData();
                    float* f21Data = f21[isub]->as<cart_volume_regular_gpu>()->getData();

                    int subid[3];
                    decompmgr->getSplitLocalSubDomID(isub, subid); // Change 0 to isub
                    axis* ax1 = local_X_axes[subid[0]];
                    axis* ax2 = local_Y_axes[subid[1]];
                    axis* ax3 = local_Z_axes[subid[2]];

                    sPlainLoop(
                            f4Data, f5Data, f6Data, f7Data, f8Data, f9Data,
                            M1Data, M2Data, M3Data, M4Data, M5Data, M6Data,
                            M7Data, M8Data, M9Data, M10Data, M11Data, M12Data,
                            M13Data, M14Data, M15Data, M16Data, M17Data, M18Data,
                            M19Data, M20Data, M21Data,
                            d4Data, d5Data, d6Data, d7Data, d8Data, d9Data,
                            f10Data, f11Data, f12Data, f13Data, f14Data, f15Data, 
                            f16Data, f17Data, f18Data, f19Data, f20Data, f21Data, 
                            ax1->n, ax2->n, ax3->n, ax1->ntot, ax2->ntot, milliseconds_s);
                }

                for (int isub = 0; isub < nsubs; ++isub) {
                    computeP(isub);
                }
                
                if (!_fwd_only) {
                    if (iter == next_snap) {
                        if (!useZFP_) {
                            snapReaderWriter->lock_for_write_uncompressed_fwi_snapshot();
                        } else {
                            snapReaderWriter->lock_for_write_compressed_fwi_snapshot();
                        }
                        bool skip_halo = true;
                        bool no_skip_halo = false;
                        for (int isub = 0; isub < nsubs; ++isub) {
                            // THIS IS NOT COMPUTATIONALLY CORRECT. We need the correct zone id. Taking this approach as we are not worried about correctness.
                            snap_f1[isub]->copyData(f1[isub], skip_halo);
                            snap_f2[isub]->copyData(f2[isub], skip_halo);
                            snap_f3[isub]->copyData(f3[isub], skip_halo);
                        }
                        // halo exchange
                        cudaEventRecord(start_halo);
                        cudaEventSynchronize(start_halo);
                        halomgr_s->start_update();            
                        halomgr_s->finish_update();
                        cudaEventRecord(stop_halo);
                        cudaEventSynchronize(stop_halo);
                        elapsedTime = 0;
                        cudaEventElapsedTime(&elapsedTime, start_halo, stop_halo);
                        milliseconds_h +=elapsedTime;

                        for (int isub = 0; isub < nsubs; ++isub) {
                            snap_f4[isub]->copyData(f4[isub], no_skip_halo);
                            snap_f5[isub]->copyData(f5[isub], no_skip_halo);
                            snap_f6[isub]->copyData(f6[isub], no_skip_halo);
                            snap_f7[isub]->copyData(f7[isub], no_skip_halo);
                            snap_f8[isub]->copyData(f8[isub], no_skip_halo);
                            snap_f9[isub]->copyData(f9[isub], no_skip_halo);
                        }

                        CUDA_TRY(cudaEventRecord(writeSnapshotsCompleteEvent_, 0));
                        if (!useZFP_) {
                            snapReaderWriter->start_write_uncompressed_fwi_snapshot(corrBuffList, corrBuff_size, file_snap_p);
                        } else {
                            snapReaderWriter->start_write_compressed_fwi_snapshot(corrBuffList, corrBuff_size, file_snap_p,
                                                                                zipped_corr_buff_, zfpFields_, zfpStream_);
                        }                    
                        next_snap += _xcorr_step;
                    }
                } // !_fwd_only
            } // end iter
        } // End TIMED_BLOCK ("Forward")

        if (!_fwd_only) {
            if (!useZFP_) {
                snapReaderWriter->start_read_uncompressed_fwi_snapshot(corrBuffList, corrBuff_size, file_snap_p);
            } else {
                snapReaderWriter->start_read_compressed_fwi_snapshot(corrBuffList, corrBuff_size, file_snap_p,
                                                                        zipped_corr_buff_, zfpFields_, zfpStream_);
            }

            TIMED_BLOCK("Adjoint") {
                next_snap -= _xcorr_step;
                for (int iter = _niter - 1; iter >= 0; --iter) {   
                    cudaEvent_t start_halo, stop_halo;
                    cudaEventCreate(&start_halo);
                    cudaEventCreate(&stop_halo);            

                    cudaEventRecord(start_halo);
                    cudaEventSynchronize(start_halo); 
                    halomgr_v->start_update();            
                    halomgr_v->finish_update();
                    cudaEventRecord(stop_halo);
                    cudaEventSynchronize(stop_halo);
                    elapsedTime = 0;
                    cudaEventElapsedTime(&elapsedTime, start_halo, stop_halo);
                    milliseconds_h +=elapsedTime;

                    for (int isub = 0; isub < nsubs; ++isub) {
                        float* f4Data = f4[isub]->as<cart_volume_regular_gpu>()->getData();
                        float* f5Data = f5[isub]->as<cart_volume_regular_gpu>()->getData();
                        float* f6Data = f6[isub]->as<cart_volume_regular_gpu>()->getData();
                        float* f7Data = f7[isub]->as<cart_volume_regular_gpu>()->getData();
                        float* f8Data = f8[isub]->as<cart_volume_regular_gpu>()->getData();
                        float* f9Data = f9[isub]->as<cart_volume_regular_gpu>()->getData();

                        float *d4Data = d4[isub]->as<cart_volume_regular_gpu>()->getData();
                        float *d5Data = d5[isub]->as<cart_volume_regular_gpu>()->getData();
                        float *d6Data = d6[isub]->as<cart_volume_regular_gpu>()->getData();
                        float *d7Data = d7[isub]->as<cart_volume_regular_gpu>()->getData();
                        float *d8Data = d8[isub]->as<cart_volume_regular_gpu>()->getData();
                        float *d9Data = d9[isub]->as<cart_volume_regular_gpu>()->getData();

                        float* M1Data = M1[isub]->as<cart_volume_regular_gpu>()->getData();
                        float* M2Data = M2[isub]->as<cart_volume_regular_gpu>()->getData();
                        float* M3Data = M3[isub]->as<cart_volume_regular_gpu>()->getData();
                        float* M4Data = M4[isub]->as<cart_volume_regular_gpu>()->getData();
                        float* M5Data = M5[isub]->as<cart_volume_regular_gpu>()->getData();
                        float* M6Data = M6[isub]->as<cart_volume_regular_gpu>()->getData();
                        float* M7Data = M7[isub]->as<cart_volume_regular_gpu>()->getData();
                        float* M8Data = M8[isub]->as<cart_volume_regular_gpu>()->getData();
                        float* M9Data = M9[isub]->as<cart_volume_regular_gpu>()->getData();
                        float* M10Data = M10[isub]->as<cart_volume_regular_gpu>()->getData();
                        float* M11Data = M11[isub]->as<cart_volume_regular_gpu>()->getData();
                        float* M12Data = M12[isub]->as<cart_volume_regular_gpu>()->getData();
                        float* M13Data = M13[isub]->as<cart_volume_regular_gpu>()->getData();
                        float* M14Data = M14[isub]->as<cart_volume_regular_gpu>()->getData();
                        float* M15Data = M15[isub]->as<cart_volume_regular_gpu>()->getData();
                        float* M16Data = M16[isub]->as<cart_volume_regular_gpu>()->getData();
                        float* M17Data = M17[isub]->as<cart_volume_regular_gpu>()->getData();
                        float* M18Data = M18[isub]->as<cart_volume_regular_gpu>()->getData();
                        float* M19Data = M19[isub]->as<cart_volume_regular_gpu>()->getData();
                        float* M20Data = M20[isub]->as<cart_volume_regular_gpu>()->getData();
                        float* M21Data = M21[isub]->as<cart_volume_regular_gpu>()->getData();
                        float* f10Data = f10[isub]->as<cart_volume_regular_gpu>()->getData();
                        float* f11Data = f11[isub]->as<cart_volume_regular_gpu>()->getData();
                        float* f12Data = f12[isub]->as<cart_volume_regular_gpu>()->getData();
                        float* f13Data = f13[isub]->as<cart_volume_regular_gpu>()->getData();
                        float* f14Data = f14[isub]->as<cart_volume_regular_gpu>()->getData();
                        float* f15Data = f15[isub]->as<cart_volume_regular_gpu>()->getData();
                        float* f16Data = f16[isub]->as<cart_volume_regular_gpu>()->getData();
                        float* f17Data = f17[isub]->as<cart_volume_regular_gpu>()->getData();
                        float* f18Data = f18[isub]->as<cart_volume_regular_gpu>()->getData();
                        float* f19Data = f19[isub]->as<cart_volume_regular_gpu>()->getData();
                        float* f20Data = f20[isub]->as<cart_volume_regular_gpu>()->getData();
                        float* f21Data = f21[isub]->as<cart_volume_regular_gpu>()->getData();

                        int subid[3];
                        decompmgr->getSplitLocalSubDomID(isub, subid); // Change 0 to isub
                        axis* ax1 = local_X_axes[subid[0]];
                        axis* ax2 = local_Y_axes[subid[1]];
                        axis* ax3 = local_Z_axes[subid[2]];

                        sPlainLoop(
                                f4Data, f5Data, f6Data, f7Data, f8Data, f9Data,
                                M1Data, M2Data, M3Data, M4Data, M5Data, M6Data,
                                M7Data, M8Data, M9Data, M10Data, M11Data, M12Data,
                                M13Data, M14Data, M15Data, M16Data, M17Data, M18Data,
                                M19Data, M20Data, M21Data,
                                d4Data, d5Data, d6Data, d7Data, d8Data, d9Data,
                                f10Data, f11Data, f12Data, f13Data, f14Data, f15Data, 
                                f16Data, f17Data, f18Data, f19Data, f20Data, f21Data, 
                                ax1->n, ax2->n, ax3->n, ax1->ntot, ax2->ntot, milliseconds_s);
                    }

                    cudaEventRecord(start_halo);
                    cudaEventSynchronize(start_halo); 
                    halomgr_s->start_update();
                    halomgr_s->finish_update();
                    cudaEventRecord(stop_halo);
                    cudaEventSynchronize(stop_halo);
                    elapsedTime = 0;
                    cudaEventElapsedTime(&elapsedTime, start_halo, stop_halo);
                    milliseconds_h +=elapsedTime;

                    for (int isub = 0; isub < nsubs; ++isub) {
                        float* f1Data = f1[isub]->as<cart_volume_regular_gpu>()->getData();
                        float* f2Data = f2[isub]->as<cart_volume_regular_gpu>()->getData();
                        float* f3Data = f3[isub]->as<cart_volume_regular_gpu>()->getData();

                        float *d4Data = d4[isub]->as<cart_volume_regular_gpu>()->getData();
                        float *d5Data = d5[isub]->as<cart_volume_regular_gpu>()->getData();
                        float *d6Data = d6[isub]->as<cart_volume_regular_gpu>()->getData();
                        float *d7Data = d7[isub]->as<cart_volume_regular_gpu>()->getData();
                        float *d8Data = d8[isub]->as<cart_volume_regular_gpu>()->getData();
                        float *d9Data = d9[isub]->as<cart_volume_regular_gpu>()->getData();

                        int subid[3];
                        decompmgr->getSplitLocalSubDomID(isub, subid); // Change 0 to isub
                        axis* ax1 = local_X_axes[subid[0]];
                        axis* ax2 = local_Y_axes[subid[1]];
                        axis* ax3 = local_Z_axes[subid[2]];

                        vgPlainLoop(f1Data, f2Data, f3Data, d4Data, d5Data, d6Data, d7Data, d8Data, d9Data, 
                                                zprime[isub], ax1->n, ax2->n, ax3->n, ax1->ntot, ax2->ntot, milliseconds_g);
                    }

                    // halo exchange
                    cudaEventRecord(start_halo);
                    cudaEventSynchronize(start_halo); 
                    halomgr_vd->start_update();
                    halomgr_vd->finish_update();
                    cudaEventRecord(stop_halo);
                    cudaEventSynchronize(stop_halo);
                    elapsedTime = 0;
                    cudaEventElapsedTime(&elapsedTime, start_halo, stop_halo);
                    milliseconds_h +=elapsedTime;

                    for (int isub = 0; isub < nsubs; ++isub) {
                        mirror_s(isub, 0, f4, f5, f6, f7, f8, f9);
                    }

                    for (int isub = 0; isub < nsubs; ++isub) {
                        mirror_s(isub, 0, f4, f5, f6, f7, f8, f9);
                    }

                    for (int isub = 0; isub < nsubs; ++isub) {
                        mirror_v(isub, 0, f1, f2, f3);
                    }

                    for (int isub = 0; isub < nsubs; ++isub) {
                        mirror_v(isub, 0, f1, f2, f3);
                    }
                
                    for (int isub = 0; isub < nsubs; ++isub) {
                        float* f1Data = f1[isub]->as<cart_volume_regular_gpu>()->getData();
                        float* f2Data = f2[isub]->as<cart_volume_regular_gpu>()->getData();
                        float* f3Data = f3[isub]->as<cart_volume_regular_gpu>()->getData();
                        float* f4Data = f4[isub]->as<cart_volume_regular_gpu>()->getData();
                        float* f5Data = f5[isub]->as<cart_volume_regular_gpu>()->getData();
                        float* f6Data = f6[isub]->as<cart_volume_regular_gpu>()->getData();
                        float* f7Data = f7[isub]->as<cart_volume_regular_gpu>()->getData();
                        float* f8Data = f8[isub]->as<cart_volume_regular_gpu>()->getData();
                        float* f9Data = f9[isub]->as<cart_volume_regular_gpu>()->getData();
                        float* M22Data = M22[isub]->as<cart_volume_regular_gpu>()->getData();

                        int subid[3];
                        decompmgr->getSplitLocalSubDomID(isub, subid); // Change 0 to isub
                        axis* ax1 = local_X_axes[subid[0]];
                        axis* ax2 = local_Y_axes[subid[1]];
                        axis* ax3 = local_Z_axes[subid[2]];

                        vPlainLoop(f1Data, f2Data, f3Data, f4Data, f5Data, f6Data, f7Data, f8Data, f9Data, 
                                        M22Data, zprime[isub], ax1->n, ax2->n, ax3->n, ax1->ntot, ax2->ntot, milliseconds_v);
                    }

                    if (iter == next_snap) {
                        if (!useZFP_) {
                            snapReaderWriter->finish_read_uncompressed_fwi_snapshot();
                        } else {
                            snapReaderWriter->finish_read_compressed_fwi_snapshot();
                        }
                    for (int isub = 0; isub < nsubs; ++isub) {
                            update_g_v(isub, snap_f1[isub], snap_f3[isub], g1[isub], f1[isub], f3[isub], f4[isub]);
                        } //isub

                        cudaEventRecord(start_halo);
                        cudaEventSynchronize(start_halo); 
                        halomgr_v->start_update();
                        halomgr_v->finish_update();
                        cudaEventRecord(stop_halo);
                        cudaEventSynchronize(stop_halo);
                        elapsedTime = 0;
                        cudaEventElapsedTime(&elapsedTime, start_halo, stop_halo);
                        milliseconds_h +=elapsedTime;        

                        for (int isub = 0; isub < nsubs; ++isub) {
                            update_g_r_impl(isub, d4[isub], d5[isub], d6[isub], d7[isub], d8[isub], d9[isub], g2[isub],
                                                            f4[isub], f6[isub], M18[isub], M1[isub], M2[isub], M3[isub], M4[isub], M5[isub], M6[isub],
                                                            M7[isub], d1, d2, d3);
                        } // isub
                        CUDA_TRY(cudaEventRecord(readSnapshotsCompleteEvent_, 0));
                        next_snap -= _xcorr_step;
                        if (next_snap >= 0) {
                            if (!useZFP_) {
                                snapReaderWriter->start_read_uncompressed_fwi_snapshot(corrBuffList, corrBuff_size, file_snap_p);
                            } else {
                                snapReaderWriter->start_read_compressed_fwi_snapshot(corrBuffList, corrBuff_size, file_snap_p,
                                                                                        zipped_corr_buff_, zfpFields_, zfpStream_);
                            }
                        }
                    }          
                } // end iter
            } //end TIMED_BLOCK ("Adjoint")
        }
    } // End TIMED_BLOCK ("Total")

    timer locTotal = timer::get_timer("Total");
    timer locForward = timer::get_timer("Forward");
    double locTotalTime = locTotal.get_elapsed();
    double locForwardTime = locForward.get_elapsed();

    double maxTotalTime, maxForwardTime, maxAdjointTime;
    float max_milliseconds_g, max_milliseconds_s, max_milliseconds_v, max_milliseconds_h;
    MPI_Reduce(&locTotalTime, &maxTotalTime, 1, MPI_DOUBLE, MPI_MAX, 0, _comm);
    MPI_Reduce(&locForwardTime, &maxForwardTime, 1, MPI_DOUBLE, MPI_MAX, 0, _comm);
    MPI_Reduce(&milliseconds_v, &max_milliseconds_v, 1, MPI_FLOAT, MPI_MAX, 0, _comm);
    MPI_Reduce(&milliseconds_g, &max_milliseconds_g, 1, MPI_FLOAT, MPI_MAX, 0, _comm);
    MPI_Reduce(&milliseconds_s, &max_milliseconds_s, 1, MPI_FLOAT, MPI_MAX, 0, _comm);
    MPI_Reduce(&milliseconds_h, &max_milliseconds_h, 1, MPI_FLOAT, MPI_MAX, 0, _comm);

    if (!_fwd_only) {
        timer locAdjoint = timer::get_timer("Adjoint");    
        double locAdjointTime = locAdjoint.get_elapsed();
        MPI_Reduce(&locAdjointTime, &maxAdjointTime, 1, MPI_DOUBLE, MPI_MAX, 0, _comm);
    }

    if(_rank == 0) {
        std::cout<<"Total Time: "<<maxTotalTime<<std::endl;
        std::cout<<"Max time for Forward: "<<maxForwardTime<<std::endl;
        if (!_fwd_only)
            std::cout<<"Max time for Adjoint: "<<maxAdjointTime<<std::endl;
        std::cout<<"Max time for velocity: "<<max_milliseconds_v/1000.0f<<std::endl;
        std::cout<<"Max time for velocity gradient: "<<max_milliseconds_g/1000.0f<<std::endl;
        std::cout<<"Max time for stress: "<<max_milliseconds_s/1000.0f<<std::endl;
        std::cout<<"Max time for halo-exchange: "<<max_milliseconds_h/1000.0f<<std::endl;
    }
}

void simulator3SetUp::initializeSource(float p)
{
    //find the cart_vol that has the point position _npoints/2 + 4;
    for (int i = 0; i < nsubs; i++) {
        int splitId[3];
        decompmgr->getSplitLocalSubDomID(i, splitId);
        int stOff[3];
        int endOff[3];
        for (int d = 0; d < 3; ++d) {
            stOff[d] = decompmgr->getOffset(splitId[d], d);
            endOff[d] = stOff[d] + decompmgr->getNumPtsSplit(splitId[d], d) - 1;
        } //end d
        if (stOff[0] <= n_total[0] / 2 && endOff[0] >= n_total[0] / 2 && stOff[1] <= n_total[1] / 2 &&
            endOff[1] >= n_total[1] / 2 && stOff[2] <= n_total[2] / 2 &&
            endOff[2] >= n_total[2] / 2) { // found that point
            int x = n_total[0] / 2 - stOff[0] + radius;
            int y = n_total[1] / 2 - stOff[1] + radius;
            int z = n_total[2] / 2 - stOff[2] + radius;
            init_src_kernel<<<1, 1>>>(f4[i]->as<cart_volume_regular_gpu>()->getData(),
                                      f5[i]->as<cart_volume_regular_gpu>()->getData(),
                                      f6[i]->as<cart_volume_regular_gpu>()->getData(),
                                      f7[i]->as<cart_volume_regular_gpu>()->getData(),
                                      f8[i]->as<cart_volume_regular_gpu>()->getData(),
                                      f9[i]->as<cart_volume_regular_gpu>()->getData(),
                                      x, y, z, p, f4[i]->as<cart_volume_regular_gpu>()->vol_idx());
            CUDA_CHECK_ERROR(__FILE__, __LINE__);
        }
    }
}

void simulator3SetUp::computeP(int isub)
{
    axis* ax1 = p_gpu[isub]->as<cart_volume_regular_gpu>()->ax1();
    axis* ax2 = p_gpu[isub]->as<cart_volume_regular_gpu>()->ax2();
    axis* ax3 = p_gpu[isub]->as<cart_volume_regular_gpu>()->ax3();
    dim3 threads(128, 1, 1);
    dim3 blocks((ax1->ntot - 1) / threads.x + 1, (ax2->ntot - 1) / threads.y + 1, (ax3->ntot - 1) / threads.z + 1);

    compute_p_kernel<<<blocks, threads>>>(
        p_gpu[isub]->as<cart_volume_regular_gpu>()->getData(),
        f4[isub]->as<cart_volume_regular_gpu>()->getData(),
        f5[isub]->as<cart_volume_regular_gpu>()->getData(),
        f6[isub]->as<cart_volume_regular_gpu>()->getData(),
        f7[isub]->as<cart_volume_regular_gpu>()->getData(),
        f8[isub]->as<cart_volume_regular_gpu>()->getData(),
        f9[isub]->as<cart_volume_regular_gpu>()->getData(),
        ax1->ntot, ax2->ntot, ax3->ntot, p_gpu[isub]->as<cart_volume_regular_gpu>()->vol_idx());
    CUDA_CHECK_ERROR(__FILE__, __LINE__);

    CUDA_CHECK_ERROR(__FILE__, __LINE__);
}