
#include "cuda_utils.h"
#include "volume_index.h"

__global__ void
update_grad_loop_kernel(volume_index idx, volume_index idx_adj, float* __restrict__ snap_x,
                            float* __restrict__ snap_z, float* __restrict__ grad_Vp, float* __restrict__ adj_x,
                            float* __restrict__ adj_z, int ixbeg, int ixend, int iybeg, int iyend, int izbeg, int izend)
{
    int ix = ixbeg + blockIdx.x * blockDim.x + threadIdx.x;
    int iy = iybeg + blockIdx.y * blockDim.y + threadIdx.y;
    int iz = izbeg + blockIdx.z * blockDim.z + threadIdx.z;

    if (ix > ixend || iy > iyend || iz > izend)
        return;

    float x0 = idx(snap_x, ix - ixbeg, iy - iybeg, iz - izbeg);
    float z0 = idx(snap_z, ix - ixbeg, iy - iybeg, iz - izbeg);

    idx(grad_Vp, ix - ixbeg, iy - iybeg, iz - izbeg) +=
        idx_adj(adj_x, ix, iy, iz) * x0 + idx_adj(adj_z, ix, iy, iz) * z0;
}

void
launch_update_grad_loop_kernel(volume_index idx, volume_index idx_adj, float* __restrict__ snap_x,
                                   float* __restrict__ snap_z, float* __restrict__ grad_Vp, float* __restrict__ adj_x,
                                   float* __restrict__ adj_z, int ixbeg, int ixend, int iybeg, int iyend, int izbeg,
                                   int izend)
{
    int num_x = ixend - ixbeg + 1;
    int num_y = iyend - iybeg + 1;
    int num_z = izend - izbeg + 1;

    dim3 threads(32, 32, 1);
    dim3 blocks((num_x + threads.x - 1) / threads.x, (num_y + threads.y - 1) / threads.y,
                (num_z + threads.z - 1) / threads.z);

    update_grad_loop_kernel<<<blocks, threads, 0>>>(idx, idx_adj, snap_x, snap_z, grad_Vp, adj_x, adj_z, ixbeg,
                                                        ixend, iybeg, iyend, izbeg, izend);
}
