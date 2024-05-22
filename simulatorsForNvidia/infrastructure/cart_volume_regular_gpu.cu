
#include "cart_volume_regular_gpu.h"
#include "cart_volume_gpu_kernels.h"

#include "cuda_utils.h"

class cart_volume_regular_gpu;
//not
// constuctor from 3 axis and bool to set data to zero or not
//
cart_volume_regular_gpu::cart_volume_regular_gpu(const axis* ax1_, const axis* _ax2_, const axis* _ax3_,
                                                 bool set_to_zero)
    : data(nullptr), bMinMax(nullptr), hMinMax(nullptr), hInfNanCount(nullptr)

{

    copyAxis(ax1_, _ax2_, _ax3_);

    data_size = _ax1->ntot * _ax2->ntot * _ax3->ntot;

    allocate();

    if (set_to_zero) // set the array to 0
        CUDA_TRY(cudaMemset((void*)(this->data), 0, data_size * sizeof(T)));
}

//
// destructor : free
cart_volume_regular_gpu::~cart_volume_regular_gpu()
{
    if (data)
        CUDA_TRY(cudaFree(data));

    if (bMinMax)
        CUDA_TRY(cudaFree(bMinMax));

    if (hMinMax)
        CUDA_TRY(cudaFreeHost(hMinMax));

    if (hInfNanCount)
        CUDA_TRY(cudaFreeHost(hInfNanCount));
}

volume_index
cart_volume_regular_gpu::vol_idx()
{

    int n0 = _ax1->ntot;
    int n1 = _ax2->ntot;
    int n2 = _ax3->ntot;
    return volume_index(n0, n1, n2);
}

//
// check and copy three axis in this class
//

void
cart_volume_regular_gpu::copyAxis(const axis* _ax1_, const axis* _ax2_, const axis* _ax3_)
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
}

bool
cart_volume_regular_gpu::isAligned()
{
    return _ax1->alignMnt.alignValue != AlignMemBytes::NOEXTENSION;
}

void
cart_volume_regular_gpu::allocate()
{
    CUDA_TRY(cudaMalloc((void**)&(this->data), (data_size) * sizeof(T)));
}

void
cart_volume_regular_gpu::set_constant(float val, bool skipHalos, bool skipInterleaving)
{
    size_t stride = (size_t)_ax2->ntot * _ax1->ntot;

    int n1, n2, n3;
    getDims(n1, n2, n3, skipHalos);

    if (isAligned()) {
        dim3 threads32x32(32, 32, 1);
        dim3 blocks((_ax1->ntot - _ax1->npad_trailing) / (4 * threads32x32.x) + 1, (n2 - 1) / threads32x32.y + 1, n3);

        if (skipHalos) {
            size_t offset = (size_t(_ax3->ibeg) * _ax2->ntot + size_t(_ax2->ibeg)) * _ax1->ntot;
            blocks.x = (_ax1->nghost + n1 - 1) / (4 * threads32x32.x) + 1; // modify the blocks.x to a new size now
            set<<<blocks, threads32x32, 0>>>(data + offset, _ax1->nghost, n1, n2, _ax1->ntot, stride, val);
            cudaDeviceSynchronize();

        } else {
            set<<<blocks, threads32x32, 0>>>(data, 0, n1, n2, _ax1->ntot, stride, val);
            cudaDeviceSynchronize();
        }
    } else {
        dim3 threads128x1(128, 1, 1);
        dim3 blocks((n1 - 1) / (threads128x1.x) + 1, n2, n3);

        size_t offset = 0;
        if (skipHalos) {
            offset = (size_t(_ax3->ibeg) * _ax2->ntot + size_t(_ax2->ibeg)) * _ax1->ntot + _ax1->ibeg;
        }
        set_unaligned<<<blocks, threads128x1, 0>>>(data + offset, n1, n2, _ax1->ntot, stride, val);
        cudaDeviceSynchronize();
    }

    CUDA_CHECK_ERROR(__FILE__, __LINE__);
}

cart_volume_regular_gpu::T*
cart_volume_regular_gpu::getData()
{
    return data;
}

//
// compute the inner/outer dimensions of the cube based on skipHalosAndPadding
//
void
cart_volume_regular_gpu::getDims(int& n0, int& n1, int& n2, bool skipHalosAndPadding)
{
    if (skipHalosAndPadding) {
        n0 = _ax1->iend - _ax1->ibeg + 1;
        n1 = _ax2->iend - _ax2->ibeg + 1;
        n2 = _ax3->iend - _ax3->ibeg + 1;
    }

    else {
        n0 = _ax1->ntot - _ax1->npad_trailing;
        n1 = _ax2->ntot - _ax2->npad_trailing;
        n2 = _ax3->ntot - _ax3->npad_trailing;
    }
}

void
cart_volume_regular_gpu ::copyData(cart_volume<float>* vol, bool skipHalos, bool skipInterleaving)
{
    skipInterleaving = false;
    if (vol->is<cart_volume_regular_host>()) {
        copyFrom(vol->as<cart_volume_regular_host>(), skipHalos);
    } else if (vol->is<cart_volume_regular_gpu>()) {
        copyFrom(vol->as<cart_volume_regular_gpu>(), skipHalos);
    } else {
        EMSL_ERROR("unknown type of cart_volume in cart_volume_regular_gpu::copyData()");
    }
}

void
cart_volume_regular_gpu::copyFrom(cart_volume_regular_gpu* from, bool skipHalos)
{
    copyFromAsync(from, 0, skipHalos);
    CUDA_TRY(cudaStreamSynchronize(0));
}

void
cart_volume_regular_gpu::copyFromAsync(cart_volume_regular_gpu* from, cudaStream_t stream, bool skipHalos)
{
    volume_index idx = vol_idx();
    volume_index idx_from = from->vol_idx();
    if (skipHalos) {
        // Ensure the inner dimensions (no halo) are the same
        EMSL_VERIFY(from->ax1()->n == ax1()->n);
        EMSL_VERIFY(from->ax2()->n == ax2()->n);
        EMSL_VERIFY(from->ax3()->n == ax3()->n);

        const int xbeg_to = _ax1->ibeg;
        const int ybeg_to = _ax2->ibeg;
        const int zbeg_to = _ax3->ibeg;
        const int xdim_to = _ax1->ntot;
        const int ydim_to = _ax2->ntot;
        const int xbeg_from = from->ax1()->ibeg;
        const int ybeg_from = from->ax2()->ibeg;
        const int zbeg_from = from->ax3()->ibeg;
        const int xdim_from = from->ax1()->ntot;
        const int ydim_from = from->ax2()->ntot;
        const int nx = _ax1->n;
        const int ny = _ax2->n;
        const int nz = _ax3->n;
        float* destPtr = &idx(getData(), xbeg_to, ybeg_to, zbeg_to);
        float* fromPtr = &idx_from(from->getData(), xbeg_from, ybeg_from, zbeg_from);

        dim3 threads(128, 8, 1);
        dim3 blocks((nx - 1) / threads.x + 1, (ny - 1) / threads.y + 1, (nz - 1) / threads.z + 1);

        copy_and_restride_data<<<blocks, threads, 0, stream>>>(destPtr, fromPtr, nx, ny, nz, xdim_to, ydim_to,
                                                               xdim_from, ydim_from);
        CUDA_TRY(cudaPeekAtLastError());
    } else if (from->ax1()->nallocated_elements != ax1()->nallocated_elements) {
        // include halos, but the volumes have different padding.
        // Ensure the outer dimensions (including halo, but not padding) are the same
        EMSL_VERIFY(from->ax1()->nvalid == ax1()->nvalid);
        EMSL_VERIFY(from->ax2()->nvalid == ax2()->nvalid);
        EMSL_VERIFY(from->ax3()->nvalid == ax3()->nvalid);

        const int xbeg_to = 0;
        const int ybeg_to = 0;
        const int zbeg_to = 0;
        const int xdim_to = _ax1->nallocated_elements;
        const int ydim_to = _ax2->nallocated_elements;
        const int xbeg_from = 0;
        const int ybeg_from = 0;
        const int zbeg_from = 0;
        const int xdim_from = from->ax1()->nallocated_elements;
        const int ydim_from = from->ax2()->nallocated_elements;
        const int nx = _ax1->nvalid;
        const int ny = _ax2->nvalid;
        const int nz = _ax3->nvalid;
        float* destPtr = getData();
        float* fromPtr = from->getData();
        dim3 threads(128, 8, 1);
        dim3 blocks((nx - 1) / threads.x + 1, (ny - 1) / threads.y + 1, (nz - 1) / threads.z + 1);

        copy_and_restride_data<<<blocks, threads, 0, stream>>>(destPtr, fromPtr, nx, ny, nz, xdim_to, ydim_to,
                                                               xdim_from, ydim_from);
        CUDA_TRY(cudaPeekAtLastError());
    } else {
        // Ensure the volumes are the exact same size (including halo) and padding
        EMSL_VERIFY(from->ax1()->nvalid == ax1()->nvalid);
        EMSL_VERIFY(from->ax2()->nvalid == ax2()->nvalid);
        EMSL_VERIFY(from->ax3()->nvalid == ax3()->nvalid);
        EMSL_VERIFY(from->ax1()->nallocated_elements == ax1()->nallocated_elements);
        EMSL_VERIFY(from->ax2()->nallocated_elements == ax2()->nallocated_elements);
        EMSL_VERIFY(from->ax3()->nallocated_elements == ax3()->nallocated_elements);

        float* destPtr = &(idx(getData(), 0, 0, 0));
        float* srcPtr = &(idx(from->getData(), 0, 0, 0));
        size_t Ntot = ((size_t)(_ax1->ntot)) * (_ax2->ntot) * (_ax3->ntot);
        CUDA_TRY(cudaMemcpyAsync(destPtr, srcPtr, (Ntot * sizeof(float)), cudaMemcpyDeviceToDevice, stream));
    }
}

void
cart_volume_regular_gpu::copyTo(cart_volume_regular_host* to, bool skipHalos)
{
    copyToAsync(to, 0, skipHalos);
    CUDA_TRY(cudaStreamSynchronize(0));
}

void
cart_volume_regular_gpu::copyToAsync(cart_volume_regular_host* to, cudaStream_t stream, bool skipHalos)
{
    // Ensure interior dimensions are the same, because it probably never makes sense to copy between
    // volumes with different interior dimensions.
    EMSL_VERIFY(ax1()->n == to->ax1()->n);
    EMSL_VERIFY(ax2()->n == to->ax2()->n);
    EMSL_VERIFY(ax3()->n == to->ax3()->n);

    int gpu_ldimx = ax1()->nallocated_elements;
    int gpu_ldimy = ax2()->nallocated_elements;
    int gpu_ldimz = ax3()->nallocated_elements;

    int host_ldimx = to->ax1()->nallocated_elements;
    int host_ldimy = to->ax2()->nallocated_elements;
    int host_ldimz = to->ax3()->nallocated_elements;

    // If the leading dimensions match and we are including halos -> direct copy
    if (!skipHalos && host_ldimx == gpu_ldimx) {
        EMSL_VERIFY(host_ldimy == gpu_ldimy && host_ldimz == gpu_ldimz);
        CUDA_TRY(cudaMemcpyAsync(to->data(), getData(), gpu_ldimx * gpu_ldimy * gpu_ldimz * sizeof(float),
                                 cudaMemcpyDefault, stream));
    }
    // Otherwise copy using the "copy_and_restride" kernel.  We can use it because we're using host pinned memory
    else {
        int host_nx, host_ny, host_nz, host_firstiz, host_firstiy, host_firstix;
        int gpu_nx, gpu_ny, gpu_nz, gpu_firstiz, gpu_firstiy, gpu_firstix;

        getDims(gpu_nx, gpu_ny, gpu_nz, skipHalos);
        to->getDims(host_nx, host_ny, host_nz, skipHalos);
        EMSL_VERIFY(host_nx == gpu_nx && host_ny == gpu_ny && host_nz == gpu_nz);
        if (skipHalos) {
            host_firstix = to->ax1()->ibeg;
            host_firstiy = to->ax2()->ibeg;
            host_firstiz = to->ax3()->ibeg;
            gpu_firstix = ax1()->ibeg;
            gpu_firstiy = ax2()->ibeg;
            gpu_firstiz = ax3()->ibeg;
        } else {
            host_firstix = 0;
            host_firstiy = 0;
            host_firstiz = 0;
            gpu_firstix = 0;
            gpu_firstiy = 0;
            gpu_firstiz = 0;
        }

        dim3 threads(128, 8, 1);
        dim3 blocks((gpu_nx - 1) / threads.x + 1, (gpu_ny - 1) / threads.y + 1, (gpu_nz - 1) / threads.z + 1);

        auto hostIdx = to->vol_idx();
        auto gpuIdx = vol_idx();

        copy_and_restride_data<<<blocks, threads, 0, stream>>>(
            &hostIdx(to->data(), host_firstix, host_firstiy, host_firstiz),
            &gpuIdx(getData(), gpu_firstix, gpu_firstiy, gpu_firstiz), gpu_nx, gpu_ny, gpu_nz, host_ldimx, host_ldimy,
            gpu_ldimx, gpu_ldimy);
        CUDA_TRY(cudaPeekAtLastError());
    }
}

void
cart_volume_regular_gpu::copyFrom(cart_volume_regular_host* from, bool skipHalos)
{
    copyFromAsync(from, 0, skipHalos);
    CUDA_TRY(cudaStreamSynchronize(0));
}

void
cart_volume_regular_gpu::copyFromAsync(cart_volume_regular_host* from, cudaStream_t stream, bool skipHalos)
{
    // Ensure interior dimensions are the same, because it probably never makes sense to copy between
    // volumes with different interior dimensions.
    EMSL_VERIFY(ax1()->n == from->ax1()->n);
    EMSL_VERIFY(ax2()->n == from->ax2()->n);
    EMSL_VERIFY(ax3()->n == from->ax3()->n);

    int gpu_ldimx = ax1()->nallocated_elements;
    int gpu_ldimy = ax2()->nallocated_elements;
    int gpu_ldimz = ax3()->nallocated_elements;

    int host_ldimx = from->ax1()->nallocated_elements;
    int host_ldimy = from->ax2()->nallocated_elements;
    int host_ldimz = from->ax3()->nallocated_elements;

    // If the leading dimensions match and we are including halos -> direct copy
    if (!skipHalos && host_ldimx == gpu_ldimx) {
        EMSL_VERIFY(host_ldimy == gpu_ldimy && host_ldimz == gpu_ldimz);
        CUDA_TRY(cudaMemcpyAsync(getData(), from->data(), gpu_ldimx * gpu_ldimy * gpu_ldimz * sizeof(float),
                                 cudaMemcpyDefault, stream));
    }
    // Otherwise copy using the "copy_and_restride" kernel.  We can use it because we're using host pinned memory
    else {
        int host_nx, host_ny, host_nz, host_firstiz, host_firstiy, host_firstix;
        int gpu_nx, gpu_ny, gpu_nz, gpu_firstiz, gpu_firstiy, gpu_firstix;

        getDims(gpu_nx, gpu_ny, gpu_nz, skipHalos);
        from->getDims(host_nx, host_ny, host_nz, skipHalos);
        EMSL_VERIFY(host_nx == gpu_nx && host_ny == gpu_ny && host_nz == gpu_nz);
        if (skipHalos) {
            host_firstix = from->ax1()->ibeg;
            host_firstiy = from->ax2()->ibeg;
            host_firstiz = from->ax3()->ibeg;
            gpu_firstix = ax1()->ibeg;
            gpu_firstiy = ax2()->ibeg;
            gpu_firstiz = ax3()->ibeg;
        } else {
            host_firstix = 0;
            host_firstiy = 0;
            host_firstiz = 0;
            gpu_firstix = 0;
            gpu_firstiy = 0;
            gpu_firstiz = 0;
        }

        dim3 threads(128, 8, 1);
        dim3 blocks((gpu_nx - 1) / threads.x + 1, (gpu_ny - 1) / threads.y + 1, (gpu_nz - 1) / threads.z + 1);

        auto hostIdx = from->vol_idx();
        auto gpuIdx = vol_idx();

        copy_and_restride_data<<<blocks, threads, 0, stream>>>(
            &gpuIdx(getData(), gpu_firstix, gpu_firstiy, gpu_firstiz),
            &hostIdx(from->data(), host_firstix, host_firstiy, host_firstiz), gpu_nx, gpu_ny, gpu_nz, gpu_ldimx,
            gpu_ldimy, host_ldimx, host_ldimy);
        CUDA_TRY(cudaPeekAtLastError());
    }
}
