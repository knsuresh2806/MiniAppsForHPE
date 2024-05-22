#ifndef PML_COEF3D_GPU_H
#define PML_COEF3D_GPU_H

#include <vector>
#include <array>
#include "pml_coef3d.h"
#include "emsl_error.h"
#include "cuda_utils.h"

class pml_coef3d_gpu
{
public:
    pml_coef3d_gpu() : nsubs(0) {}

    explicit pml_coef3d_gpu(const pml_coef3d& pmlCPU) : nsubs(0) { setCoeffs(pmlCPU); }

    /* Disable copying to prevent copying coefficient arrays */
    pml_coef3d_gpu& operator=(const pml_coef3d_gpu&) = delete;

    virtual ~pml_coef3d_gpu()
    {

        for (auto p : x_AB_)
            if (p)
                CUDA_TRY(cudaFree(p));
        for (auto p : y_AB_)
            if (p)
                CUDA_TRY(cudaFree(p));
        for (auto p : z_AB_)
            if (p)
                CUDA_TRY(cudaFree(p));
        for (auto p : xs_AB_)
            if (p)
                CUDA_TRY(cudaFree(p));
        for (auto p : ys_AB_)
            if (p)
                CUDA_TRY(cudaFree(p));
        for (auto p : zs_AB_)
            if (p)
                CUDA_TRY(cudaFree(p));
    }

    //
    // set coeffs for the case of constructor without arguments
    //
    void setCoeffs(const pml_coef3d& pmlCPU)
    {

        EMSL_VERIFY(nsubs == 0); // we should not set if the coeffs have been set by the constructor

        nsubs = pmlCPU.x_A_.size();
        EMSL_VERIFY(nsubs > 0);
        x_AB_.resize(nsubs);
        y_AB_.resize(nsubs);
        z_AB_.resize(nsubs);
        xs_AB_.resize(nsubs);
        ys_AB_.resize(nsubs);
        zs_AB_.resize(nsubs);
        copyCoeffs(pmlCPU);
    }

    std::array<float2 const*, 6> getPMLCoef(int isub) const
    {
        EMSL_VERIFY(isub >= 0 && isub < x_AB_.size());

        return { x_AB_[isub] == nullptr ? nullptr : x_AB_[isub] + PAD,
                 y_AB_[isub] == nullptr ? nullptr : y_AB_[isub] + PAD,
                 z_AB_[isub] == nullptr ? nullptr : z_AB_[isub] + PAD,
                 xs_AB_[isub] == nullptr ? nullptr : xs_AB_[isub] + PAD,
                 ys_AB_[isub] == nullptr ? nullptr : ys_AB_[isub] + PAD,
                 zs_AB_[isub] == nullptr ? nullptr : zs_AB_[isub] + PAD };
    }

private:
    // Padding so Cuda kernels could read pre/past end (full thread block).
    static constexpr int PAD = 32;

    int nsubs;
    void copyCoeffs(const pml_coef3d& pmlCPU)
    { // allocate the gpu coeffs and copy from cpu to gpu

        //
        // verify that the vectors have the same size
        //
        EMSL_VERIFY(nsubs == pmlCPU.x_A_.size());
        EMSL_VERIFY(nsubs == pmlCPU.x_B_.size());
        EMSL_VERIFY(nsubs == pmlCPU.y_A_.size());
        EMSL_VERIFY(nsubs == pmlCPU.y_B_.size());
        EMSL_VERIFY(nsubs == pmlCPU.z_A_.size());
        EMSL_VERIFY(nsubs == pmlCPU.z_B_.size());

        EMSL_VERIFY(nsubs == pmlCPU.xs_A_.size());
        EMSL_VERIFY(nsubs == pmlCPU.xs_B_.size());
        EMSL_VERIFY(nsubs == pmlCPU.ys_A_.size());
        EMSL_VERIFY(nsubs == pmlCPU.ys_B_.size());
        EMSL_VERIFY(nsubs == pmlCPU.zs_A_.size());
        EMSL_VERIFY(nsubs == pmlCPU.zs_B_.size());
        //
        // allocate the memory in gpu and upload from cpu arrays.
        // x_A_ and x_B_ are stored together as float2, the same is applied to other arrays
        //
        for (int i = 0; i < nsubs; ++i) {
            x_AB_[i] = allocate_load(pmlCPU.x_A_[i], pmlCPU.x_B_[i]);
            y_AB_[i] = allocate_load(pmlCPU.y_A_[i], pmlCPU.y_B_[i]);
            z_AB_[i] = allocate_load(pmlCPU.z_A_[i], pmlCPU.z_B_[i]);
            xs_AB_[i] = allocate_load(pmlCPU.xs_A_[i], pmlCPU.xs_B_[i]);
            ys_AB_[i] = allocate_load(pmlCPU.ys_A_[i], pmlCPU.ys_B_[i]);
            zs_AB_[i] = allocate_load(pmlCPU.zs_A_[i], pmlCPU.zs_B_[i]);
        }
    }

    float2* allocate_load(const std::vector<float>& a,
                          const std::vector<float>& b) // allocate the memory in gpu and memcpy the coeffs as float2
    {
        float2* result = nullptr;
        EMSL_VERIFY(a.size() == b.size());
        if (a.size() == 0) // nothing to allocate
            return result;

        auto gpu_size = a.size() + PAD * 2;

        std::vector<float2> tmp(gpu_size, { 0.0f, 0.0f });
        for (uint i = 0; i < a.size(); ++i) {
            tmp[PAD + i].x = a[i];
            tmp[PAD + i].y = b[i];
        }
        CUDA_TRY(cudaMalloc((void**)&result, gpu_size * sizeof(float2)));

        CUDA_TRY(cudaMemcpy((void*)result, (void*)tmp.data(), gpu_size * sizeof(float2), cudaMemcpyHostToDevice));
        return result;
    }

    // PML coefficients on non-staggered mesh locations (field1, tyy, field2)
    std::vector<float2*> x_AB_, y_AB_, z_AB_; // .x for A and .y for B

    // PML coefficients on staggered mesh locations (field3, vy, field4)
    std::vector<float2*> xs_AB_, ys_AB_, zs_AB_;
};
#endif
