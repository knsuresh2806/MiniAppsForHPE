#ifndef SPONGE_COEF3D_GPU_H
#define SPONGE_COEF3D_GPU_H

#include "sponge_coef3d.h"
#include <array>
#include <vector>
#include "emsl_error.h"
#include "cuda_utils.h"

class sponge_coef3d_gpu
{
public:
    sponge_coef3d_gpu() : nsubs(0), active_coefs(false) {}

    explicit sponge_coef3d_gpu(const sponge_coef3d& spongeCpu) : sponge_coef3d_gpu() { setCoeffs(spongeCpu); }

    virtual ~sponge_coef3d_gpu()
    {
        for (int i = 0; i < nsubs; ++i) {
            CUDA_TRY(cudaFree(x_AB_[i]));
            CUDA_TRY(cudaFree(y_AB_[i]));
            CUDA_TRY(cudaFree(z_AB_[i]));
        }
        if (nsubs) {
            free(x_AB_);
            free(y_AB_);
            free(z_AB_);
        }
    }

    /* Disable copying to prevent copying coefficient arrays */
    sponge_coef3d_gpu(const sponge_coef3d_gpu&) = delete;
    sponge_coef3d_gpu& operator=(const sponge_coef3d_gpu&) = delete;

    //
    // set coeffs for the case of constructor without arguments
    //
    void setCoeffs(const sponge_coef3d& spongeCpu)
    {
        EMSL_VERIFY(nsubs == 0); // we should not set if the coeffs have been set previously
        nsubs = spongeCpu.y_A_.size();
        EMSL_VERIFY(nsubs > 0);

        x_AB_ = (float2**)malloc(nsubs * sizeof(float2*));
        EMSL_VERIFY(x_AB_ != nullptr);
        for (int i = 0; i < nsubs; ++i)
            x_AB_[i] = allocate_load(spongeCpu.x_A_[i], spongeCpu.x_B_[i]);

        y_AB_ = (float2**)malloc(nsubs * sizeof(float2*));
        EMSL_VERIFY(y_AB_ != nullptr);
        for (int i = 0; i < nsubs; ++i)
            y_AB_[i] = allocate_load(spongeCpu.y_A_[i], spongeCpu.y_B_[i]);

        z_AB_ = (float2**)malloc(nsubs * sizeof(float2*));
        EMSL_VERIFY(z_AB_ != nullptr);
        for (int i = 0; i < nsubs; ++i)
            z_AB_[i] = allocate_load(spongeCpu.z_A_[i], spongeCpu.z_B_[i]);
    }

    bool use_sponge() const { return active_coefs; }

    int nsubs;
    float2** x_AB_;
    float2** y_AB_;
    float2** z_AB_;

    std::array<float2 const*, 3> getSpongeCoef(int isub) const
    {
        EMSL_VERIFY(isub >= 0 && isub < nsubs)
        return { x_AB_[isub], y_AB_[isub], z_AB_[isub] };
    }

private:
    // active_coefs is true if at least one coefficient is not 1.0
    bool active_coefs;

    float2* allocate_load(const std::vector<float>& a, const std::vector<float>& b)
    {
        if (a.empty())
            return nullptr;

        std::vector<float2> ab(a.size());
        for (std::size_t i = 0; i < a.size(); ++i) {
            if (a[i] != 1.0f || b[i] != 1.0f)
                active_coefs = true;
            ab[i] = { a[i], b[i] };
        }

        void* result = nullptr;
        CUDA_TRY(cudaMalloc(&result, ab.size() * sizeof(float2)));
        CUDA_TRY(cudaMemcpy(result, (void*)ab.data(), ab.size() * sizeof(float2), cudaMemcpyHostToDevice));
        return static_cast<float2*>(result);
    }
};
#endif
