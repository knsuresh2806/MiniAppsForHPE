#ifndef KERNEL_BASE_H
#define KERNEL_BASE_H

#include <array>

#include "kernel_abstract_base.h"
#include "sponge_coef3d.h"
#include "sponge_coef3d_gpu.h"
#include "volume_index.h"

class cart_volume_regular_gpu;

class kernel_base : public kernel_abstract_base
{
public:
    kernel_base(int sp_order, int bc_opt, int nsub, float dt, double dx, double dy, double dz, bool use_born,
                     cart_volume<float>**& field1, cart_volume<float>**& field2, cart_volume<float>**& field3,
                     cart_volume<float>**& field4, cart_volume<float>**& field11, cart_volume<float>**& field21,
                     cart_volume<float>**& field31, cart_volume<float>**& field41, const sponge_coef3d_gpu& sponge_coef,
                     std::vector<cart_volume<float>**>& Model8, std::vector<cart_volume<float>**>& Model9,
                     std::vector<cart_volume<float>**>& Model81, std::vector<cart_volume<float>**>& Model91,
                     std::vector<float>& wsi, std::vector<cart_volume<float>**>& Model10);

    static std::pair<float, float> setVsFromVpEpsDelta(cart_volume<float>* Vs_, cart_volume<float>* Vp_,
                                                       cart_volume<float>* Epsilon_, cart_volume<float>* Delta_);

    static constexpr int nmech = 3;

private:
    // Override methods calling the GPU-specific methods with the default stream(0)
    void ensure_volumes() override;
    void kernel_loop2_inner(int isub) override { kernel_loop2_inner(isub, 0); }
    void kernel_loop1_inner(int isub) override { kernel_loop1_inner(isub, 0); }
    void kernel_loop3_inner(int isub) override { kernel_loop3_inner(isub, 0); };
    void adj_kernel_loop1_inner(int isub) override { adj_kernel_loop1_inner(isub, 0); };
    void adj_kernel_loop3_inner(int isub) override
    {
        adj_kernel_loop3_inner(isub, 0);
    };
    void kernel_loop4_inner(int isub) override { kernel_loop4_inner(isub, 0); };

    // GPU-only method with a stream parameter
    void kernel_loop2_inner(int isub, cudaStream_t stream);
    void kernel_loop1_inner(int isub, cudaStream_t stream);
    void kernel_loop3_inner(int isub, cudaStream_t stream);
    void adj_kernel_loop1_inner(int isub, cudaStream_t stream);
    void adj_kernel_loop3_inner(int isub, cudaStream_t stream);
    void kernel_loop4_inner(int isub, cudaStream_t stream);

    // template <typename T>
    // void kernel_loop4_kernel(int isub);

    bool vols_cast_done;

    int nsub;
    int sp_order;
    float dt;
    double d1, d2, d3;
    bool update_born;

    float invd1;
    float invd2;
    float invd3;

    // References to the cart_volume<float> arrays on the fdsim class.  This lets us receive the cart_volume<float>s from
    // fdsim in the kernel constructor so that we don't need to pass them in every function call.  They
    // are references back to fdsim data members because the cart_volume<float> arrays aren't initialized yet when
    // the kernel is constructed.
    cart_volume<float>**& field1_orig;
    cart_volume<float>**& field2_orig;
    cart_volume<float>**& field3_orig;
    cart_volume<float>**& field4_orig;
    cart_volume<float>**& field11_orig;
    cart_volume<float>**& field21_orig;
    cart_volume<float>**& field31_orig;
    cart_volume<float>**& field41_orig;

    std::vector<cart_volume<float>**>& Model8_orig;
    std::vector<cart_volume<float>**>& Model9_orig;
    std::vector<cart_volume<float>**>& Model81_orig;
    std::vector<cart_volume<float>**>& Model91_orig;
    std::vector<cart_volume<float>**>& Model10_orig;

    std::vector<cart_volume_regular_gpu*> field1;
    std::vector<cart_volume_regular_gpu*> field2;
    std::vector<cart_volume_regular_gpu*> field3;
    std::vector<cart_volume_regular_gpu*> field4;
    std::vector<cart_volume_regular_gpu*> field11;
    std::vector<cart_volume_regular_gpu*> field21;
    std::vector<cart_volume_regular_gpu*> field31;
    std::vector<cart_volume_regular_gpu*> field41;

    std::array<std::vector<cart_volume_regular_gpu*>, nmech> Model8;
    std::array<std::vector<cart_volume_regular_gpu*>, nmech> Model9;
    std::array<std::vector<cart_volume_regular_gpu*>, nmech> Model81;
    std::array<std::vector<cart_volume_regular_gpu*>, nmech> Model91;
    std::array<std::vector<cart_volume_regular_gpu*>, nmech> Model10;

    const sponge_coef3d_gpu& sponge_coef;
    std::vector<float>& wsi;
};

#endif
