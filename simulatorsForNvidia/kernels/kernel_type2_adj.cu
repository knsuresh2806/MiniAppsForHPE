#include <cuda_pipeline.h>
#include "kernel_type2_adj.h"
#include "helper_kernels_gpu.h"
#include "kernel_utils.h"
#include <cassert>

namespace {

// When in interior or sponge (pml_any=false), we need to load the mixed derivatives
// which are calculated in the derivatives kernel
template <int bx, int by, bool pml_any>
struct smpml
{
    // loaded at iz
    float field3_rhs_mixed[by][bx];
    float field4_rhs_mixed[by][bx];
};

template <int bx, int by>
struct smpml<bx, by, true>
{};

// When sponge is active, we can preload the field3/field4 arrays
template <int bx, int by, bool sponge_active>
struct smsponge
{};

template <int bx, int by>
struct smsponge<bx, by, true>
{
    // loaded at iz
    float field3[by][bx];
    float field4[by][bx];
};

template <int order, int bx, int by, bool pml_any, bool sponge_active>
struct smstruct : public smpml<bx, by, pml_any>, public smsponge<bx, by, sponge_active>
{
    // loaded at iz
    float field1_dx[by][bx + order];
    float field2_dx[by][bx + order];
    float field1_dy[by + order][bx];
    float field2_dy[by + order][bx];
    // loaded at iz+4
    float field1_dz[by][bx];
    float field2_dz[by][bx];
};

template <int order, int bx, int by, bool pml_any, bool sponge_active>
__device__ __forceinline__ void
load_shm(smstruct<order, bx, by, pml_any, sponge_active>& sm, float const* field1_dx, float const* field2_dx,
         float const* field1_dy, float const* field2_dy, float const* field1_dz, float const* field2_dz,
         [[maybe_unused]] float const* field3_rhs_mixed, [[maybe_unused]] float const* field4_rhs_mixed,
         [[maybe_unused]] float const* field3, [[maybe_unused]] float const* field4, int index_load_x, int index_load_y,
         [[maybe_unused]] int index_load_nohalo, int index_load_zder, bool loader_x, bool loader_y, bool loader_nohalo,
         int txload, int tyload)
{
    if (loader_x) {
        __pipeline_memcpy_async(&sm.field1_dx[tyload][txload], &field1_dx[index_load_x], sizeof(float4));
        __pipeline_memcpy_async(&sm.field2_dx[tyload][txload], &field2_dx[index_load_x], sizeof(float4));
    }

    if (loader_y) {
        __pipeline_memcpy_async(&sm.field1_dy[tyload][txload], &field1_dy[index_load_y], sizeof(float4));
        __pipeline_memcpy_async(&sm.field2_dy[tyload][txload], &field2_dy[index_load_y], sizeof(float4));
    }

    if (loader_nohalo) {
        __pipeline_memcpy_async(&sm.field1_dz[tyload][txload], &field1_dz[index_load_zder], sizeof(float4));
        __pipeline_memcpy_async(&sm.field2_dz[tyload][txload], &field2_dz[index_load_zder], sizeof(float4));

        if constexpr (!pml_any) {
            __pipeline_memcpy_async(&sm.field3_rhs_mixed[tyload][txload], &field3_rhs_mixed[index_load_nohalo], sizeof(float4));
            __pipeline_memcpy_async(&sm.field4_rhs_mixed[tyload][txload], &field4_rhs_mixed[index_load_nohalo], sizeof(float4));
        }

        if constexpr (sponge_active) {
            __pipeline_memcpy_async(&sm.field3[tyload][txload], &field3[index_load_nohalo], sizeof(float4));
            __pipeline_memcpy_async(&sm.field4[tyload][txload], &field4[index_load_nohalo], sizeof(float4));
        }
    }
    __pipeline_commit();
}

template <FD_COEF_TYPE ctype, int order>
__device__ __forceinline__ float
dfdx_stg_right(float const* f, int stride, float h)
{
    const int radius = order / 2;
    float out = 0.0f;
#pragma unroll
    for (int i = 1; i <= radius; ++i) {
        float c = helper_kernels_gpu::df_coef<order, ctype>(i);
        out += c * (f[i * stride] - f[(-i + 1) * stride]);
    }
    out *= h;
    return out;
}

template <FD_COEF_TYPE ctype, int order, int bx, int by, int bz, bool pml_any, bool sponge_active>
__global__
__launch_bounds__(bx* by) void update_adj_kernel_main_3(
    float* field3, float* field4, float const* field1_dx, float const* field2_dx, float const* field1_dy, float const* field2_dy,
    float const* field1_dz, float const* field2_dz, float const* field3_rhs_mixed, float const* field4_rhs_mixed,
    [[maybe_unused]] float2 const* ab_xx, [[maybe_unused]] float2 const* ab_yy, [[maybe_unused]] float2 const* ab_zz,
    int nx, int ny, int nz, int ixbeg, int ldimx, int ldimy, float dt, float invdx, float invdy, float invdz)
{
    static_assert(!(pml_any && sponge_active), "Can't have both PML and sponge");

    constexpr int radius = order / 2;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int ixblock = blockIdx.x * bx;
    int iyblock = blockIdx.y * by;
    int izblock = blockIdx.z * bz;
    int zstride = ldimx * ldimy;

    // Load the shared memory as float4, and remap the threads from (bx, by) into (bx/2, by*2)
    const int tid = threadIdx.y * bx + threadIdx.x;
    static_assert(bx >= order && by >= order, "Incompatible block size for this order");
    const int txload = 4 * (tid % (bx / 2));
    const int tyload = tid / (bx / 2);

    // For arrays with X halos: start at (ixblock-radius,iyblock,iz) and use the float4 thread mapping.
    // Since the input arrays are offset for y and z halos, but not x, when ixload_x is negative, it will
    // be reading garbage from the previous line. This isn't a "problem" necessarily, because nothing uses
    // that data from shared memory (see active variable), but it wastes some bandwidth, so the loader_x
    // variable should be false if ixload_x is negative.
    const int ixload_x = ixblock + txload - radius;
    const int iyload_x = iyblock + tyload;
    const bool loader_x = (txload < bx + order && tyload < by) && ixload_x >= 0 && (ixload_x < ldimx && iyload_x < ny);
    int index_load_x = (izblock * ldimy + iyload_x) * ldimx + ixload_x;

    // For arrays with Y halos: start at (ixblock,iyblock-radius,iz) and use the float4 thread mapping.
    const int ixload_y = ixblock + txload;
    const int iyload_y = iyblock + tyload - radius;
    const bool loader_y = (txload < bx && tyload < by + order) && (ixload_y < ldimx && iyload_y < ny + radius);
    int index_load_y = (izblock * ldimy + iyload_y) * ldimx + ixload_y;

    // For arrays without halos: start at (ixblock,iyblock,iz) and use the float4 thread mapping.
    // This is also used for loading the z derivatives, since those only use a halo in the Z direction, not X or Y.
    const int ixload_nohalo = ixblock + txload;
    const int iyload_nohalo = iyblock + tyload;
    const int loader_nohalo = (txload < bx && tyload < by) && (ixload_nohalo < ldimx && iyload_nohalo < ny);
    int index_load_nohalo = (izblock * ldimy + iyload_nohalo) * ldimx + ixload_nohalo;
    int index_load_zder = index_load_nohalo + (radius)*zstride;

    static_assert(order == 8 || order == 16,
                  "Radius must be a multiple of 4 due to using float4 for loading shared memory");

    extern __shared__ char sm_[];
    auto sm = reinterpret_cast<smstruct<order, bx, by, pml_any, sponge_active>*>(sm_);

    int ix = ixblock + threadIdx.x;
    int iy = iyblock + threadIdx.y;
    int ixend = ixbeg + nx;
    int active = ix >= ixbeg && ix < ixend && iy < ny;

    // The index of where to write the output
    int index_out = (izblock * ldimy + iy) * ldimx + ix;

    [[maybe_unused]] float sponge_A_xy, sponge_B_xy;
    if constexpr (sponge_active) {
        if (active) {
            sponge_A_xy = min(ab_xx[ix].x, ab_yy[iy].x);
            sponge_B_xy = min(ab_xx[ix].y, ab_yy[iy].y);
        }
    }

    float field1_dz_rq[order] = { 0 };
    float field2_dz_rq[order] = { 0 };

    int index_load_zder_prime = ((izblock - radius + 1) * ldimy + iy) * ldimx + ix;
#pragma unroll
    for (int i = 0; i < order - 1; ++i) {
        field1_dz_rq[i] = field1_dz[index_load_zder_prime];
        field2_dz_rq[i] = field2_dz[index_load_zder_prime];
        index_load_zder_prime += zstride;
    }

    int ism = 0;
    load_shm(sm[ism], field1_dx, field2_dx, field1_dy, field2_dy, field1_dz, field2_dz, field3_rhs_mixed, field4_rhs_mixed, field3, field4, index_load_x,
             index_load_y, index_load_nohalo, index_load_zder, loader_x, loader_y, loader_nohalo, txload, tyload);
    index_load_x += zstride;
    index_load_y += zstride;
    index_load_nohalo += zstride;
    index_load_zder += zstride;
    ism ^= 1;

    int nzloop = min(bz, nz - izblock);
    int iz = 0;

    // The main loop has 1 z plane peeled off at the end (because of async pipeline), but
    // the work is exactly the same, so I've just pulled it into this lambda.  No negative
    // performance impact when I profiled it.
    auto process_z_plane = [&]() {
        field1_dz_rq[order - 1] = sm[ism].field1_dz[ty][tx];
        field2_dz_rq[order - 1] = sm[ism].field2_dz[ty][tx];

        float field1_xx = dfdx_stg_right<ctype, order>(&sm[ism].field1_dx[ty][tx + radius], 1, invdx);
        float field2_xx = dfdx_stg_right<ctype, order>(&sm[ism].field2_dx[ty][tx + radius], 1, invdx);
        float field1_yy = dfdx_stg_right<ctype, order>(&sm[ism].field1_dy[ty + radius][tx], bx, invdy);
        float field2_yy = dfdx_stg_right<ctype, order>(&sm[ism].field2_dy[ty + radius][tx], bx, invdy);
        float field1_zz = dfdx_stg_right<ctype, order>(&field1_dz_rq[radius - 1], 1, invdz);
        float field2_zz = dfdx_stg_right<ctype, order>(&field2_dz_rq[radius - 1], 1, invdz);

        float field3_rhs = field1_xx + field1_yy + field1_zz;
        float field4_rhs = field2_xx + field2_yy + field2_zz;

        if constexpr (!pml_any) {
            field3_rhs += sm[ism].field3_rhs_mixed[ty][tx];
            field4_rhs += sm[ism].field4_rhs_mixed[ty][tx];
        }

        if (active) {
            if constexpr (!sponge_active) {
                atomicAdd(&field3[index_out], field3_rhs * dt);
                atomicAdd(&field4[index_out], field4_rhs * dt);
            } else {
                float sponge_A_xyz = min(sponge_A_xy, ab_zz[iz + izblock].x);
                float sponge_B_xyz = min(sponge_B_xy, ab_zz[iz + izblock].y) * dt;
                field3[index_out] = sponge_A_xyz * sm[ism].field3[ty][tx] + sponge_B_xyz * field3_rhs;
                field4[index_out] = sponge_A_xyz * sm[ism].field4[ty][tx] + sponge_B_xyz * field4_rhs;
            }
        }

        // Rotate the z derivative register queues
#pragma unroll
        for (int i = 1; i < order; ++i) {
            field1_dz_rq[i - 1] = field1_dz_rq[i];
            field2_dz_rq[i - 1] = field2_dz_rq[i];
        }

        index_out += zstride;
        index_load_x += zstride;
        index_load_y += zstride;
        index_load_nohalo += zstride;
        index_load_zder += zstride;
    };

    for (; iz < nzloop - 1; ++iz) {
        __syncthreads(); // Protect sm between iterations
        load_shm(sm[ism], field1_dx, field2_dx, field1_dy, field2_dy, field1_dz, field2_dz, field3_rhs_mixed, field4_rhs_mixed, field3, field4,
                 index_load_x, index_load_y, index_load_nohalo, index_load_zder, loader_x, loader_y, loader_nohalo,
                 txload, tyload);
        ism ^= 1;
        __pipeline_wait_prior(1);
        __syncthreads(); // Wait for pipeline load to complete

        process_z_plane();
    }

    iz = nzloop - 1;

    __pipeline_wait_prior(0);
    ism ^= 1;
    __syncthreads(); // Wait for pipeline load to complete

    process_z_plane();
}

template <int order, bool pml_any, bool sponge_active>
void
launcher(float* field3, float* field4, float const* field1_dx, float const* field2_dx, float const* field1_dy, float const* field2_dy,
         float const* field1_dz, float const* field2_dz, float const* field3_rhs, float const* field4_rhs, float2 const* ab_xx,
         float2 const* ab_yy, float2 const* ab_zz, int nx, int ny, int nz, int ixbeg, int ldimx, int ldimy, float dt,
         float invdx, float invdy, float invdz, cudaStream_t stream)
{
    // These could be tuned, but they seem to do pretty well.
    constexpr int bx = 32;
    constexpr int by = 32;
    constexpr int bz = 64;
    dim3 threads(bx, by, 1);
    // Threads have to cover entire x interior plus the left halo, hence +ixbeg on x blocks
    dim3 blocks((nx + ixbeg - 1) / bx + 1, (ny - 1) / by + 1, (nz - 1) / bz + 1);

    int shmsize = sizeof(smstruct<order, bx, by, pml_any, sponge_active>[2]);
    static bool setshm = false;
    if (!setshm) {
        CUDA_TRY(cudaFuncSetAttribute(
            update_adj_kernel_main_3<_FD_COEF_LEAST_SQUARE, order, bx, by, bz, pml_any, sponge_active>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, shmsize));
        setshm = true;
    }

    update_adj_kernel_main_3<_FD_COEF_LEAST_SQUARE, order, bx, by, bz, pml_any, sponge_active>
        <<<blocks, threads, shmsize, stream>>>(field3, field4, field1_dx, field2_dx, field1_dy, field2_dy, field1_dz, field2_dz, field3_rhs, field4_rhs,
                                               ab_xx, ab_yy, ab_zz, nx, ny, nz, ixbeg, ldimx, ldimy, dt, invdx, invdy,
                                               invdz);
    CUDA_TRY(cudaPeekAtLastError());
}

constexpr decltype(&launcher<8, false, false>) LAUNCHERS[6] = {
    launcher<8, false, false>,  launcher<8, false, true>,  launcher<8, true, false>,
    launcher<16, false, false>, launcher<16, false, true>, launcher<16, true, false>,
};

} // end anonymous namespace

namespace kernel3_gpu_kernels {

void
launch_update_adj_kernel_main_3(float* field3, float* field4, float const* field1_dx, float const* field2_dx,
                                     float const* field1_dy, float const* field2_dy, float const* field1_dz, float const* field2_dz,
                                     float const* field3_rhs, float const* field4_rhs, float2 const* ab_xx, float2 const* ab_yy,
                                     float2 const* ab_zz, int ixbeg, int ixend, int iybeg, int iyend, int izbeg,
                                     int izend, int ldimx, int ldimy, float dt, double dx, double dy, double dz,
                                     bool pmlx, bool pmly, bool pmlz, int order, bool sponge_active,
                                     cudaStream_t stream)
{
    int offset = (izbeg * ldimy + iybeg) * ldimx;
    int nx = (ixend - ixbeg) + 1;
    int ny = (iyend - iybeg) + 1;
    int nz = (izend - izbeg) + 1;

    field3 += offset;
    field4 += offset;
    field1_dx += offset;
    field2_dx += offset;
    field1_dy += offset;
    field2_dy += offset;
    field1_dz += offset;
    field2_dz += offset;
    field3_rhs += offset;
    field4_rhs += offset;
    ab_yy += iybeg;
    ab_zz += izbeg;

    float invdx, invdy, invdz;
    float unused;
    kernel_utils::compute_fd_const(dx, dy, dz, invdx, invdy, invdz, unused, unused, unused, unused, unused, unused);

    bool pml_any = pmlx || pmly || pmlz;
    EMSL_VERIFY(order == 8 || order == 16);
    EMSL_VERIFY(!(sponge_active && pml_any));

    int launcherId = 0;
    if (order == 16) {
        launcherId += 3;
    }
    if (sponge_active) {
        launcherId += 1;
    } else if (pml_any) {
        launcherId += 2;
    }

    LAUNCHERS[launcherId](field3, field4, field1_dx, field2_dx, field1_dy, field2_dy, field1_dz, field2_dz, field3_rhs, field4_rhs, ab_xx, ab_yy, ab_zz,
                          nx, ny, nz, ixbeg, ldimx, ldimy, dt, invdx, invdy, invdz, stream);
}

}
