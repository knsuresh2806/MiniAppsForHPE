#include "cart_volume_regular_host.h"
#include "cart_volume_gpu_kernels.h"
#include "axis.h"
#include "volume_index.h"
#include "std_const.h"
#include "emsl_error.h"
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <new>
#include <stdint.h>
#include "cuda_utils.h"

// 3 axis constructor
cart_volume_regular_host ::cart_volume_regular_host(const axis* _ax1_, const axis* _ax2_, const axis* _ax3_,
                                                    bool set_to_zero)
{
    copyAxis(_ax1_, _ax2_, _ax3_);

    allocate_data();

    if (set_to_zero) {
        zero(false, false);
    }

}

// copy constructor with option to leave off halo
cart_volume_regular_host ::cart_volume_regular_host(cart_volume_regular* vol, bool skipHalos, bool skipInterleaving,
                                                    bool set_to_zero)
{
    EMSL_VERIFY(vol != NULL);

    Nx = vol->Nx;
    Ny = vol->Ny;
    Nz = vol->Nz;

    EMSL_VERIFY(vol->ax1() != NULL);
    EMSL_VERIFY(vol->ax2() != NULL);
    EMSL_VERIFY(vol->ax3() != NULL);

    if (skipHalos) {
        _ax1 = new axis(vol->ax1()->o, vol->ax1()->d, vol->ax1()->n, vol->ax1()->alignMnt);
        _ax2 = new axis(vol->ax2()->o, vol->ax2()->d, vol->ax2()->n, vol->ax2()->alignMnt);
        _ax3 = new axis(vol->ax3()->o, vol->ax3()->d, vol->ax3()->n, vol->ax3()->alignMnt);
    } else {
        _ax1 = new axis(vol->ax1());
        _ax2 = new axis(vol->ax2());
        _ax3 = new axis(vol->ax3());
    }
    EMSL_VERIFY(_ax1 != NULL);
    EMSL_VERIFY(_ax2 != NULL);
    EMSL_VERIFY(_ax3 != NULL);

    allocate_data();

    if (set_to_zero) {
        zero(false, false);
    }
} //end copyConstructor - 1

//Copy constructor with axis rotation
cart_volume_regular_host ::cart_volume_regular_host(cart_volume_regular* vol, ROTATE_TYPE type)
{
    EMSL_VERIFY(vol != NULL);

    if (type == _ROTATE_FROM_XYZ_TO_ZXY) {
        Nx = vol->Nz;
        Ny = vol->Nx;
        Nz = vol->Ny;

        EMSL_VERIFY(vol->ax1() != NULL);
        EMSL_VERIFY(vol->ax2() != NULL);
        EMSL_VERIFY(vol->ax3() != NULL);

        _ax1 = new axis(vol->ax3());
        _ax2 = new axis(vol->ax1());
        _ax3 = new axis(vol->ax2());

        EMSL_VERIFY(_ax1 != NULL);
        EMSL_VERIFY(_ax2 != NULL);
        EMSL_VERIFY(_ax3 != NULL);

        allocate_data();
    } else {
        //This type is not implemented
        EMSL_VERIFY(false);
    }
} //end copyConstructor - 2

void
cart_volume_regular_host ::copyAxis(const axis* _ax1_, const axis* _ax2_, const axis* _ax3_)
{
    EMSL_VERIFY(_ax1_ != NULL);
    EMSL_VERIFY(_ax2_ != NULL);
    EMSL_VERIFY(_ax3_ != NULL);

    _ax1 = new axis(_ax1_);
    EMSL_VERIFY(_ax1 != NULL);
    _ax2 = new axis(_ax2_);
    EMSL_VERIFY(_ax2 != NULL);
    _ax3 = new axis(_ax3_);
    EMSL_VERIFY(_ax3 != NULL);

    Nx = (_ax1->iend) - (_ax1->ibeg) + 1;
    Ny = (_ax2->iend) - (_ax2->ibeg) + 1;
    Nz = (_ax3->iend) - (_ax3->ibeg) + 1;
} //end copyAxis

void
cart_volume_regular_host ::allocate_data()
{

    data_size = _ax1->ntot * _ax2->ntot * _ax3->ntot;
    CUDA_TRY(cudaMallocHost((void**)&_data, data_size * sizeof(float)));
    EMSL_VERIFY(_data != NULL); // check the address to make sure allocation is ok

    // construct 3D representation to the 1D data
    _data3d = new (std::nothrow) Float2Ptr[_ax3->ntot];
    EMSL_VERIFY(_data3d != NULL);
    _data3d[0] = new (std::nothrow) FloatPtr[(_ax2->ntot) * (_ax3->ntot)];
    EMSL_VERIFY(_data3d[0] != NULL);
    for (size_t i = 1; i < (_ax3->ntot); ++i) {
        _data3d[i] = _data3d[0] + (i * (_ax2->ntot));
    } //end i
    for (size_t j = 0; j < (_ax3->ntot); ++j) {
        for (size_t i = 0; i < (_ax2->ntot); ++i) {
            _data3d[j][i] = _data + (((j * (_ax2->ntot)) + i) * (_ax1->ntot));
        } //end i
    }     //end j

} //end allocate_data

cart_volume_regular_host ::~cart_volume_regular_host()
{
    if (_data3d) {
        delete[](_data3d[0]);
        delete[] _data3d;
    }
    if (_data)
        cudaFreeHost(_data);
} //end destructor

void
cart_volume_regular_host ::set_constant(float val, bool skipHalos, bool skipInterleaving)
{
    skipInterleaving = false;
    int ntot1, ntot2, ntot3;
    getDims(ntot1, ntot2, ntot3, skipHalos);
    float* vol = data();
    volume_index idx = vol_idx();
    if (skipHalos) {
        const int xbeg = _ax1->ibeg;
        const int xend = _ax1->iend;
        const int ybeg = _ax2->ibeg;
        const int yend = _ax2->iend;
        const int zbeg = _ax3->ibeg;
        const int zend = _ax3->iend;

        for (int iz = zbeg; iz <= zend; ++iz) {
            for (int iy = ybeg; iy <= yend; ++iy) {
#pragma concurrent
                for (int ix = xbeg; ix <= xend; ++ix) {
                    idx(vol, ix, iy, iz) = val;
                } //end ix
            }     //end iy
        }
    } else {
        for (int iz = 0; iz < ntot3; ++iz) {
            for (int iy = 0; iy < ntot2; ++iy) {
#pragma concurrent
                for (int ix = 0; ix < ntot1; ++ix) {
                    idx(vol, ix, iy, iz) = val;
                } //end ix
            }     //end iy
        }
    }
}

// TODO: this copyData() is causing problems after copying to host. found while testing checkpointing.
// fails after the writing first checkpoint
void
cart_volume_regular_host ::copyData(cart_volume<float>* vol_base, bool skipHalos, bool skipInterleaving)
{

    EMSL_VERIFY(vol_base != NULL);
    EMSL_VERIFY(vol_base->is<cart_volume_regular_host>() || vol_base->is<cart_volume_regular_gpu>());

    float* vol;
    int in_ibegx, in_ibegy, in_ibegz;
    if (vol_base->is<cart_volume_regular_host>()) {
        cart_volume_regular_host* volh = vol_base->as<cart_volume_regular_host>();
        if (skipHalos) {
            EMSL_VERIFY(volh->ax1()->n == ax1()->n);
            EMSL_VERIFY(volh->ax2()->n == ax2()->n);
            EMSL_VERIFY(volh->ax3()->n == ax3()->n);
        } else {
            EMSL_VERIFY(volh->ax1()->ntot == ax1()->ntot);
            EMSL_VERIFY(volh->ax2()->ntot == ax2()->ntot);
            EMSL_VERIFY(volh->ax3()->ntot == ax3()->ntot);
            EMSL_VERIFY(volh->ax1()->nvalid == ax1()->nvalid);
        }
        vol = volh->data();
        in_ibegx = volh->ax1()->ibeg;
        in_ibegy = volh->ax2()->ibeg;
        in_ibegz = volh->ax3()->ibeg;
        volume_index idx = vol_idx();
        skipInterleaving = false;
        if (skipHalos) {
            const int xbeg = _ax1->ibeg;
            const int ybeg = _ax2->ibeg;
            const int yend = _ax2->iend;
            const int zbeg = _ax3->ibeg;
            const int zend = _ax3->iend;
            const int xSt = in_ibegx;
            const int yoff = in_ibegy - ybeg;
            const int zoff = in_ibegz - zbeg;
            for (int iz = zbeg; iz <= zend; ++iz) {
                for (int iy = ybeg; iy <= yend; ++iy) {
                    float* buf = &(idx(data(), xbeg, iy, iz));
                    float* volBuf = &(idx(vol, xSt, yoff + iy, zoff + iz));
                    CUDA_TRY(cudaMemcpy(buf, volBuf, (Nx * sizeof(float)), cudaMemcpyDefault));
                } //end iy
            }     //end iz
        } else {
            float* buf = &(idx(data(), 0, 0, 0));
            float* volBuf = &(idx(vol, 0, 0, 0));
            size_t Ntot = ((size_t)(_ax1->ntot)) * (_ax2->ntot) * (_ax3->ntot);
            CUDA_TRY(cudaMemcpy(buf, volBuf, (Ntot * sizeof(float)), cudaMemcpyDefault));
        } // end if skipHalos
    } else if (vol_base->is<cart_volume_regular_gpu>()) {
        copyFrom(vol_base->as<cart_volume_regular_gpu>(), skipHalos);
    }
} //end copyData

void
cart_volume_regular_host ::copyFrom(cart_volume_regular_gpu* from, bool skipHalos)
{
    // Leading dimensions for the GPU volume
    int gpu_ldimx = from->ax1()->ntot;
    int gpu_ldimy = from->ax2()->ntot;
    int gpu_ldimz = from->ax3()->ntot;

    int host_ldimx = ax1()->ntot;
    int host_ldimy = ax2()->ntot;
    int host_ldimz = ax3()->ntot;

    if (!skipHalos) {
        // If the leading dimensions match and we don't skip halos -> direct copy
        if (host_ldimx == gpu_ldimx) {
            EMSL_VERIFY(host_ldimy == gpu_ldimy && host_ldimz == gpu_ldimz);
            CUDA_TRY(cudaMemcpy(&data3d()[0][0][0], from->getData(), gpu_ldimx * gpu_ldimy * gpu_ldimz * sizeof(float),
                                cudaMemcpyDefault));
        }

        // Dimensions don't match but we don't skip halos -> extract GPU data matching CPU data + copy, 2D plans
        else {
            // Allocate a 2D temporary buffer on the device
            float* plan2d;
            size_t size_plan2d = host_ldimx * host_ldimy * sizeof(float);
            CUDA_TRY(cudaMalloc((void**)&plan2d, size_plan2d));

            int host_nx, host_ny, host_nz;
            int gpu_nx, gpu_ny, gpu_nz;
            from->getDims(gpu_nx, gpu_ny, gpu_nz);
            host_nx = ax1()->ntot;
            host_ny = ax2()->ntot;
            host_nz = ax3()->ntot;
            EMSL_VERIFY(host_nx == gpu_nx && host_ny == gpu_ny && host_nz == gpu_nz);
            dim3 threads(128, 1, 1);
            dim3 blocks((gpu_nx - 1) / threads.x + 1, gpu_ny, 1);
            for (int iz = 0; iz < gpu_nz; iz++) {
                float* gpu_ptr = from->getData() + iz * (off_t)(gpu_ldimx * gpu_ldimy);
                float* host_ptr = &data3d()[iz][0][0];
                // Call a packing kernel on the device
                pack_2d<<<blocks, threads>>>(plan2d, gpu_ptr, gpu_nx, host_ldimx, gpu_ldimx);
                // Pageable memory copy, and kernel in the default stream
                CUDA_TRY(cudaMemcpy(host_ptr, plan2d, size_plan2d, cudaMemcpyDefault));
            }
            CUDA_TRY(cudaFree(plan2d));
        }
    }

    // Skipping halos -> copy GPU data directly to CPU, then unpack on CPU, 2D plans
    else {
        float* plan2d;
        size_t size_plan2d = gpu_ldimx * gpu_ldimy * sizeof(float);
        CUDA_TRY(cudaMallocHost((void**)&plan2d, size_plan2d));

        int gpu_nx = from->ax1()->n;
        int gpu_ny = from->ax2()->n;
        int gpu_nz = from->ax3()->n;
        int host_nx = ax1()->n;
        int host_ny = ax2()->n;
        int host_nz = ax3()->n;
        int host_firstiz = ax3()->nghost;
        int gpu_firstiz = from->ax3()->nghost;
        EMSL_VERIFY(host_nx == gpu_nx && host_ny == gpu_ny && host_nz == gpu_nz);

        for (int iz = 0; iz < host_nz; iz++) {
            // Copy the whole 2D plan from the GPU back to CPU pinned memory
            float* gpu_ptr = from->getData() + (gpu_firstiz + iz) * (off_t)(gpu_ldimx * gpu_ldimy);
            CUDA_TRY(cudaMemcpy(plan2d, gpu_ptr, size_plan2d, cudaMemcpyDefault));

            // Loops on Y and X to extract data
            int host_iz = host_firstiz + iz;
            for (int iy = 0; iy < host_ny; iy++) {
                int host_iy = ax2()->nghost + iy;
                int gpu_iy = from->ax2()->nghost + iy;
                memcpy(&data3d()[host_iz][host_iy][ax1()->nghost], &plan2d[gpu_iy * gpu_ldimx + from->ax1()->nghost],
                       host_nx * sizeof(float));
            }
        }
        CUDA_TRY(cudaFreeHost(plan2d));
    }
}

volume_index
cart_volume_regular_host ::vol_idx()
{

    size_t n0 = _ax1->ntot;
    size_t n1 = _ax2->ntot;
    size_t n2 = _ax3->ntot;
    return volume_index(n0, n1, n2);
}

void
cart_volume_regular_host::getDims(int& n0, int& n1, int& n2, bool skipHalosAndPadding)
{
    if (skipHalosAndPadding) {
        n0 = _ax1->iend - _ax1->ibeg + 1;
        n1 = _ax2->iend - _ax2->ibeg + 1;
        n2 = _ax3->iend - _ax3->ibeg + 1;
    } else {
        n0 = _ax1->ntot - _ax1->npad_trailing;
        n1 = _ax2->ntot;
        n2 = _ax3->ntot;
    }
}
