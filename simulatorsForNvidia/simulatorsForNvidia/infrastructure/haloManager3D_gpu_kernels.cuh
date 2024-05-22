/*
   GPU Halo exchange manager kernels. 
   Author: G. Thomas-Collignon
*/

#include "haloManager3D_gpu.h"

// ****************************************************************************
// Kernel to swap the halos in the X dimension, in both directions (2-way)
// Copies radius_hi values from lo_vol to hi_vol, and copies radius_lo values from hi_vol to lo_vol.
// The pointers must point to the first value to be exchanged.
// The number of threads in X must be max(radius_lo, radius_hi)
template <typename T>
__global__ void
copy_2way_X(T* lo_vol, T* hi_vol, int radius_lo, int radius_hi, int nyswap, int lo_ldimx, int lo_ldimy, int hi_ldimx,
            int hi_ldimy)
{
    int ix = threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z;
    if (iy >= nyswap)
        return;
    int lo_index = (iz * lo_ldimy + iy) * lo_ldimx + ix;
    int hi_index = (iz * hi_ldimy + iy) * hi_ldimx + ix;
    // Copy radius_lo values from lo_vol-> hi_vol
    if (ix < radius_lo)
        hi_vol[hi_index] = lo_vol[lo_index];
    // Copy radius_hi values from hi_vol-> lo_vol
    if (ix < radius_hi)
        lo_vol[lo_index + radius_lo] = hi_vol[hi_index + radius_lo];
}

// ************************************
// Launch a 2-way copy of X halos betwen adjacent local cart_volumes
// Halos are skipped in Y and Z.
void
haloManager3D_gpu::exchange_halos_X(cart_volume<T>* lo_vol, cart_volume<T>* hi_vol, int radius_lo, int radius_hi,
                                   cudaStream_t stream)
{
    cart_volume_regular_gpu* lo_vol_gpu = lo_vol->as<cart_volume_regular_gpu>();
    cart_volume_regular_gpu* hi_vol_gpu = hi_vol->as<cart_volume_regular_gpu>();
    // Leading dimensions for the volumes
    int lo_ldimx = lo_vol_gpu->ax1()->ntot;
    int lo_ldimy = lo_vol_gpu->ax2()->ntot;
    int hi_ldimx = hi_vol_gpu->ax1()->ntot;
    int hi_ldimy = hi_vol_gpu->ax2()->ntot;
    // Compute offsets of the first point to be echanged.
    int lo_offset =
        (lo_vol_gpu->ax3()->ibeg * lo_ldimy + lo_vol_gpu->ax2()->ibeg) * lo_ldimx + lo_vol_gpu->ax1()->iend - radius_lo + 1;
    int hi_offset =
        (hi_vol_gpu->ax3()->ibeg * hi_ldimy + hi_vol_gpu->ax2()->ibeg) * hi_ldimx + hi_vol_gpu->ax1()->ibeg - radius_lo;
    // The number of values to exhange in Y and Z should match between the 2 volumes
    int unused, nyswap, nzswap, nyswap_hi, nzswap_hi;
    lo_vol_gpu->getDims(unused, nyswap, nzswap, true); // Skipping halos in  Y and Z
    hi_vol_gpu->getDims(unused, nyswap_hi, nzswap_hi, true);
    EMSL_VERIFY(nyswap == nyswap_hi && nzswap == nzswap_hi);
    EMSL_VERIFY(max(radius_lo, radius_hi) != 0);
    // Launch the 2-way kernel
    dim3 threads(max(radius_lo, radius_hi), 32, 1);
    dim3 blocks(1, (nyswap - 1) / threads.y + 1, nzswap);
    copy_2way_X<T><<<blocks, threads, 0, stream>>>(lo_vol_gpu->getData() + lo_offset, hi_vol_gpu->getData() + hi_offset,
                                                   radius_lo, radius_hi, nyswap, lo_ldimx, lo_ldimy, hi_ldimx,
                                                   hi_ldimy);
}

// ****************************************************************************
// Kernel to swap the halos in the Y dimension, in both directions (2-way)
// Copies radius_hi values from lo_vol to hi_vol, and copies radius_lo values from hi_vol to lo_vol.
// The pointers must point to the first value to be exchanged.
// The number of threads in Y must be max(radius_lo, radius_hi)
template <typename T>
__global__ void
copy_2way_Y(T* lo_vol, T* hi_vol, int radius_lo, int radius_hi, int nxswap, int lo_ldimx, int lo_ldimy, int hi_ldimx,
            int hi_ldimy)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = threadIdx.y;
    int iz = blockIdx.z;
    if (ix >= nxswap)
        return;

    int lo_index = (iz * lo_ldimy + iy) * lo_ldimx + ix;
    int hi_index = (iz * hi_ldimy + iy) * hi_ldimx + ix;
    // Copy radius_lo values from lo_vol-> hi_vol
    if (iy < radius_lo)
        hi_vol[hi_index] = lo_vol[lo_index];
    lo_index += radius_lo * lo_ldimx;
    hi_index += radius_lo * hi_ldimx;
    // Copy radius_hi values from hi_vol-> lo_vol
    if (iy < radius_hi)
        lo_vol[lo_index] = hi_vol[hi_index];
}

// ************************************
// Launch a 2-way copy of Y halos betwen adjacent local cart_volumes
// Halos are skipped in Z, but included in the X dimension.
void
haloManager3D_gpu::exchange_halos_Y(cart_volume<T>* lo_vol, cart_volume<T>* hi_vol, int radius_lo, int radius_hi,
                                   cudaStream_t stream)
{
    cart_volume_regular_gpu* lo_vol_gpu = lo_vol->as<cart_volume_regular_gpu>();
    cart_volume_regular_gpu* hi_vol_gpu = hi_vol->as<cart_volume_regular_gpu>();
    // Leading dimensions for the volumes
    int lo_ldimx = lo_vol_gpu->ax1()->ntot;
    int lo_ldimy = lo_vol_gpu->ax2()->ntot;
    int hi_ldimx = hi_vol_gpu->ax1()->ntot;
    int hi_ldimy = hi_vol_gpu->ax2()->ntot;
    // Compute offsets of the first point to be echanged. Include the X halos.
    int lo_offset = (lo_vol_gpu->ax3()->ibeg * lo_ldimy + lo_vol_gpu->ax2()->iend - radius_lo + 1) * lo_ldimx;
    int hi_offset = (hi_vol_gpu->ax3()->ibeg * hi_ldimy + hi_vol_gpu->ax2()->ibeg - radius_lo) * hi_ldimx;
    // The number of values to exhange in Y and Z should match between the 2 volumes
    int unused, nxswap, nzswap, nxswap_hi, nzswap_hi;
    lo_vol_gpu->getDims(nxswap, unused, unused, false);
    lo_vol_gpu->getDims(unused, unused, nzswap, true);
    hi_vol_gpu->getDims(nxswap_hi, unused, unused, false);
    hi_vol_gpu->getDims(unused, unused, nzswap_hi, true);
    EMSL_VERIFY(nxswap == nxswap_hi && nzswap == nzswap_hi);
    EMSL_VERIFY(max(radius_lo, radius_hi) != 0);
    // Launch the 2-way kernel
    dim3 threads(32, max(radius_lo, radius_hi), 1);
    dim3 blocks((nxswap - 1) / threads.x + 1, 1, nzswap);
    copy_2way_Y<T><<<blocks, threads, 0, stream>>>(lo_vol_gpu->getData() + lo_offset, hi_vol_gpu->getData() + hi_offset,
                                                   radius_lo, radius_hi, nxswap, lo_ldimx, lo_ldimy, hi_ldimx,
                                                   hi_ldimy);
}

// ****************************************************************************
// Kernel to swap the halos in the Z dimension, in both directions (2-way)
// Copies radius_hi values from lo_vol to hi_vol, and copies radius_lo values from hi_vol to lo_vol.
// The pointers must point to the first value to be exchanged.
// The number of threads in Z must be max(radius_lo, radius_hi)
template <typename T>
__global__ void
copy_2way_Z(T* lo_vol, T* hi_vol, int radius_lo, int radius_hi, int nxswap, int lo_ldimx, int lo_ldimy, int hi_ldimx,
            int hi_ldimy)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y;
    int iz = threadIdx.z;
    if (ix >= nxswap)
        return;

    int lo_index = (iz * lo_ldimy + iy) * lo_ldimx + ix;
    int hi_index = (iz * hi_ldimy + iy) * hi_ldimx + ix;
    // Copy radius_lo values from lo_vol-> hi_vol
    if (iz < radius_lo)
        hi_vol[hi_index] = lo_vol[lo_index];
    lo_index += radius_lo * lo_ldimx * lo_ldimy;
    hi_index += radius_lo * hi_ldimx * hi_ldimy;
    // Copy radius_hi values from hi_vol-> lo_vol
    if (iz < radius_hi)
        lo_vol[lo_index] = hi_vol[hi_index];
}

// ************************************
// Launch a 2-way copy of Y halos betwen adjacent local cart_volumes
// Halos are skipped in Z, but included in the X dimension.
void
haloManager3D_gpu::exchange_halos_Z(cart_volume<T>* lo_vol, cart_volume<T>* hi_vol, int radius_lo, int radius_hi,
                                   cudaStream_t stream)
{
    cart_volume_regular_gpu* lo_vol_gpu = lo_vol->as<cart_volume_regular_gpu>();
    cart_volume_regular_gpu* hi_vol_gpu = hi_vol->as<cart_volume_regular_gpu>();
    // Leading dimensions for the volumes
    int lo_ldimx = lo_vol_gpu->ax1()->ntot;
    int lo_ldimy = lo_vol_gpu->ax2()->ntot;
    int hi_ldimx = hi_vol_gpu->ax1()->ntot;
    int hi_ldimy = hi_vol_gpu->ax2()->ntot;
    // Compute offsets of the first point to be echanged. Include the X and Y halos
    int lo_offset = (lo_vol_gpu->ax3()->iend - radius_lo + 1) * lo_ldimy * lo_ldimx;
    int hi_offset = (hi_vol_gpu->ax3()->ibeg - radius_lo) * hi_ldimy * hi_ldimx;
    // The number of values to exhange in Y and Z should match between the 2 volumes
    int unused, nxswap, nyswap, nxswap_hi, nyswap_hi;
    lo_vol_gpu->getDims(nxswap, nyswap, unused, false);
    hi_vol_gpu->getDims(nxswap_hi, nyswap_hi, unused, false);
    EMSL_VERIFY(nxswap == nxswap_hi && nyswap == nyswap_hi);
    EMSL_VERIFY(max(radius_lo, radius_hi) != 0);
    // Launch the 2-way kernel
    dim3 threads(32, 1, max(radius_lo, radius_hi));
    dim3 blocks((nxswap - 1) / threads.x + 1, nyswap, 1);
    copy_2way_Z<T><<<blocks, threads, 0, stream>>>(lo_vol_gpu->getData() + lo_offset, hi_vol_gpu->getData() + hi_offset,
                                                   radius_lo, radius_hi, nxswap, lo_ldimx, lo_ldimy, hi_ldimx,
                                                   hi_ldimy);
}

// ****************************************************************************
// Kernel to pack or unpack the halos in the X dimension
// It copies radius points in X from one buffer to the other

template <typename T>
__global__ void
pack_unpack_X(const T* src, T* dst, int nyswap, int src_ldimx, int src_ldimy, int dst_ldimx, int dst_ldimy)
{
    int ix = threadIdx.x; // radius threads in X
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int iz = blockIdx.z;
    if (iy >= nyswap)
        return;
    int src_index = (iz * src_ldimy + iy) * src_ldimx + ix;
    int dst_index = (iz * dst_ldimy + iy) * dst_ldimx + ix;
    dst[dst_index] = src[src_index];
}

// Copy the X halos between the communication buffer and the volume (pack or unpack, lo or hi)
// The halos in Y and Z are not included in this exchange.
template <bool lo, bool pack>
void
haloManager3D_gpu::copy_halos_X(cart_volume<T>* vol, T* combuf, int radius, cudaStream_t stream)
{
    cart_volume_regular_gpu* vol_gpu = vol->as<cart_volume_regular_gpu>();
    // Leading dimensions
    int ldimx = vol_gpu->ax1()->ntot;
    int ldimy = vol_gpu->ax2()->ntot;
    // Compute offset of the first point to be packed or unpacked
    int offset = (vol_gpu->ax3()->ibeg * ldimy + vol_gpu->ax2()->ibeg) * ldimx;
    if (lo) {
        offset += vol_gpu->ax1()->ibeg;
        if (!pack)
            offset -= radius;
    } else {
        offset += vol_gpu->ax1()->iend + 1;
        if (pack)
            offset -= radius;
    }
    T* volptr = vol_gpu->getData() + offset;

    // The number of values to exhange in Y and Z (skip halos)
    int unused, nyswap, nzswap;
    vol_gpu->getDims(unused, nyswap, nzswap, true);

    EMSL_VERIFY(radius != 0);

    // Launch the kernel
    dim3 threads(radius, 32, 1);
    dim3 blocks(1, (nyswap - 1) / threads.y + 1, nzswap);
    if (pack)
        pack_unpack_X<T><<<blocks, threads, 0, stream>>>(volptr, combuf, nyswap, ldimx, ldimy, radius, nyswap);
    else
        pack_unpack_X<T><<<blocks, threads, 0, stream>>>(combuf, volptr, nyswap, radius, nyswap, ldimx, ldimy);
}

// Copy the X halos (lo or hi) from the cart volume to a remote buffer
// The halos in Y and Z are not included in this exchange.
template <bool lo>
void
haloManager3D_gpu::copy_halos_X(cart_volume<T>* vol, T* dest, int dest_ldimx, int dest_ldimy, int radius, cudaStream_t stream)
{
    cart_volume_regular_gpu* vol_gpu = vol->as<cart_volume_regular_gpu>();
    // Leading dimensions of the source volume
    int ldimx = vol_gpu->ax1()->ntot;
    int ldimy = vol_gpu->ax2()->ntot;
    // Compute offset of the first point to be read
    int offset = (vol_gpu->ax3()->ibeg * ldimy + vol_gpu->ax2()->ibeg) * ldimx +
        (lo ? vol_gpu->ax1()->ibeg : (vol_gpu->ax1()->iend - radius + 1));
    T* volptr = vol_gpu->getData() + offset;

    // The number of values to exhange in Y and Z (skip halos)
    int unused, nyswap, nzswap;
    vol_gpu->getDims(unused, nyswap, nzswap, true);

    EMSL_VERIFY(radius != 0);

    // Launch the kernel
    dim3 threads(radius, 32, 1);
    dim3 blocks(1, (nyswap - 1) / threads.y + 1, nzswap);
    pack_unpack_X<T><<<blocks, threads, 0, stream>>>(volptr, dest, nyswap, ldimx, ldimy, dest_ldimx, dest_ldimy);
}

// Return the size of the communication buffer matching the code above
int
haloManager3D_gpu::get_size_combuf_X(cart_volume<T>* vol, int radius)
{
    cart_volume_regular_gpu* vol_gpu = vol->as<cart_volume_regular_gpu>();
    // The number of values to exhange in Y and Z (skip halos)
    int unused, nyswap, nzswap;
    vol_gpu->getDims(unused, nyswap, nzswap, true);
    return (radius * nyswap * nzswap);
}


// ****************************************************************************
// Kernel to pack or unpack the halos in the Y dimension
// It copies radius points in Y from one buffer to the other

template <typename T>
__global__ void
pack_unpack_Y(const T* src, T* dst, int nxswap, int src_ldimx, int src_ldimy, int dst_ldimx, int dst_ldimy)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = threadIdx.y; // radius threads in Y
    int iz = blockIdx.z;
    if (ix >= nxswap)
        return;
    int src_index = (iz * src_ldimy + iy) * src_ldimx + ix;
    int dst_index = (iz * dst_ldimy + iy) * dst_ldimx + ix;
    dst[dst_index] = src[src_index];
}

// Copy the Y halos between the communication buffer and the volume (pack or unpack, lo or hi)
// The halos in Z are not included in this exchange, but the halos in X are also copied.
template <bool lo, bool pack>
void
haloManager3D_gpu::copy_halos_Y(cart_volume<T>* vol, T* combuf, int radius, cudaStream_t stream)
{
    cart_volume_regular_gpu* vol_gpu = vol->as<cart_volume_regular_gpu>();
    // Leading dimensions
    int ldimx = vol_gpu->ax1()->ntot;
    int ldimy = vol_gpu->ax2()->ntot;
    // Compute offset of the first point to be packed or unpacked
    int offset = vol_gpu->ax3()->ibeg * ldimy * ldimx;
    if (lo) {
        offset += vol_gpu->ax2()->ibeg * ldimx;
        if (!pack)
            offset -= radius * ldimx;
    } else {
        offset += (vol_gpu->ax2()->iend + 1) * ldimx;
        if (pack)
            offset -= radius * ldimx;
    }
    T* volptr = vol_gpu->getData() + offset;

    // The number of values to exhange in X (include halos) and Z (skip halos)
    int unused, nxswap, nzswap;
    vol_gpu->getDims(nxswap, unused, unused, false);
    vol_gpu->getDims(unused, unused, nzswap, true);

    EMSL_VERIFY(radius != 0);

    // Launch the kernel
    dim3 threads(32, radius, 1);
    dim3 blocks((nxswap - 1) / threads.x + 1, 1, nzswap);
    if (pack)
        pack_unpack_Y<T><<<blocks, threads, 0, stream>>>(volptr, combuf, nxswap, ldimx, ldimy, nxswap, radius);
    else
        pack_unpack_Y<T><<<blocks, threads, 0, stream>>>(combuf, volptr, nxswap, nxswap, radius, ldimx, ldimy);
}

// Copy the Y halos (lo or hi) from the cart volume to a remote buffer
// The halos in Z are not included in this exchange, but the halos in X are also copied.
template <bool lo>
void
haloManager3D_gpu::copy_halos_Y(cart_volume<T>* vol, T* dest, int dest_ldimx, int dest_ldimy, int radius, cudaStream_t stream)
{
    cart_volume_regular_gpu* vol_gpu = vol->as<cart_volume_regular_gpu>();
    // Leading dimensions
    int ldimx = vol_gpu->ax1()->ntot;
    int ldimy = vol_gpu->ax2()->ntot;
    // Compute offset of the first point to be read
    int offset = vol_gpu->ax3()->ibeg * ldimy * ldimx + ldimx *
        (lo ? vol_gpu->ax2()->ibeg : (vol_gpu->ax2()->iend - radius + 1));
    T* volptr = vol_gpu->getData() + offset;

    // The number of values to exhange in X (include halos) and Z (skip halos)
    int unused, nxswap, nzswap;
    vol_gpu->getDims(nxswap, unused, unused, false);
    vol_gpu->getDims(unused, unused, nzswap, true);

    EMSL_VERIFY(radius != 0);

    // Launch the kernel
    dim3 threads(32, radius, 1);
    dim3 blocks((nxswap - 1) / threads.x + 1, 1, nzswap);
    pack_unpack_Y<T><<<blocks, threads, 0, stream>>>(volptr, dest, nxswap, ldimx, ldimy, dest_ldimx, dest_ldimy);
}

// Return the size of the communication buffer matching the code above
int
haloManager3D_gpu::get_size_combuf_Y(cart_volume<T>* vol, int radius)
{
    cart_volume_regular_gpu* vol_gpu = vol->as<cart_volume_regular_gpu>();
    // The number of values to exhange in X (include halos) and Z (skip halos)
    int unused, nxswap, nzswap;
    vol_gpu->getDims(nxswap, unused, unused, false);
    vol_gpu->getDims(unused, unused, nzswap, true);
    return (radius * nxswap * nzswap);
}

// ****************************************************************************
// Kernel to pack or unpack the halos in the Z dimension
// It copies radius points in Z from one buffer to the other

template <typename T>
__global__ void
pack_unpack_Z(const T* src, T* dst, int nxswap, int src_ldimx, int src_ldimy, int dst_ldimx, int dst_ldimy)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y;
    int iz = blockIdx.z;
    if (ix >= nxswap)
        return;
    int src_index = (iz * src_ldimy + iy) * src_ldimx + ix;
    int dst_index = (iz * dst_ldimy + iy) * dst_ldimx + ix;
    dst[dst_index] = src[src_index];
}

// Copy the Z halos between the communication buffer and the volume (pack or unpack, lo or hi)
// The halos in X and Y are included.
template <bool lo, bool pack>
void
haloManager3D_gpu::copy_halos_Z(cart_volume<T>* vol, T* combuf, int radius, cudaStream_t stream)
{
    cart_volume_regular_gpu* vol_gpu = vol->as<cart_volume_regular_gpu>();
    // Leading dimensions
    int ldimx = vol_gpu->ax1()->ntot;
    int ldimy = vol_gpu->ax2()->ntot;
    // Compute offset of the first point to be packed or unpacked
    int offset;
    if (lo) {
        offset = vol_gpu->ax3()->ibeg * ldimx * ldimy;
        if (!pack)
            offset -= radius * ldimx * ldimy;
    } else {
        offset = (vol_gpu->ax3()->iend + 1) * ldimx * ldimy;
        if (pack)
            offset -= radius * ldimx * ldimy;
    }
    T* volptr = vol_gpu->getData() + offset;

    // The number of values to exhange in X and Y (with halos)
    int unused, nxswap, nyswap;
    vol_gpu->getDims(nxswap, nyswap, unused, false);

    EMSL_VERIFY(radius != 0);

    // Launch the kernel
    dim3 threads(nxswap > 32 ? 128 : 32, 1, 1);
    dim3 blocks((nxswap - 1) / threads.x + 1, nyswap, radius);
    if (pack)
        pack_unpack_Z<T><<<blocks, threads, 0, stream>>>(volptr, combuf, nxswap, ldimx, ldimy, nxswap, nyswap);
    else
        pack_unpack_Z<T><<<blocks, threads, 0, stream>>>(combuf, volptr, nxswap, nxswap, nyswap, ldimx, ldimy);
}

// Copy the Z halos (lo or hi) from the cart volume to a remote buffer
// The halos in X and Y are included.
template <bool lo>
void
haloManager3D_gpu::copy_halos_Z(cart_volume<T>* vol, T* dest, int dest_ldimx, int dest_ldimy, int radius, cudaStream_t stream)
{
    cart_volume_regular_gpu* vol_gpu = vol->as<cart_volume_regular_gpu>();
    // Leading dimensions
    int ldimx = vol_gpu->ax1()->ntot;
    int ldimy = vol_gpu->ax2()->ntot;
    // Compute offset of the first point to be read
    int offset = (ldimx * ldimy) * (lo ? vol_gpu->ax3()->ibeg : (vol_gpu->ax3()->iend - radius + 1));

    T* volptr = vol_gpu->getData() + offset;

    // The number of values to exhange in X and Y (with halos)
    int unused, nxswap, nyswap;
    vol_gpu->getDims(nxswap, nyswap, unused, false);

    EMSL_VERIFY(radius != 0);

    // Launch the kernel
    dim3 threads(nxswap > 32 ? 128 : 32, 1, 1);
    dim3 blocks((nxswap - 1) / threads.x + 1, nyswap, radius);
    pack_unpack_Z<T><<<blocks, threads, 0, stream>>>(volptr, dest, nxswap, ldimx, ldimy, dest_ldimx, dest_ldimy);
}

// Return the size of the communication buffer matching the code above
int
haloManager3D_gpu::get_size_combuf_Z(cart_volume<T>* vol, int radius)
{
    cart_volume_regular_gpu* vol_gpu = vol->as<cart_volume_regular_gpu>();
    // The number of values to exhange in X and Y (with halos)
    int unused, nxswap, nyswap;
    vol_gpu->getDims(nxswap, nyswap, unused, false);
    return (radius * nxswap * nyswap);
}

// ****************************************************************************
