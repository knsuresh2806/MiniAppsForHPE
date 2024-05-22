#ifndef KERNEL_TYPE1_GPU_H
#define KERNEL_TYPE1_GPU_H

#include "kernel_type1_base.h"
#include "pml_coef3d_gpu.h"
#include "sponge_coef3d_gpu.h"
#include "cart_volume_regular_gpu.h"

class kernel_type1_gpu : public kernel_type1_base
{
public:
    // TODO: Do we need temp volumes field3_rhs and field4_rhs? Maybe for PML? Otherwise, do we get rid of them?
    kernel_type1_gpu(int bc_opt, int order, int nsub, double d1, double d2, double d3, float dt,
                         cart_volume<float>**& field1, cart_volume<float>**& field2, cart_volume<float>**& field3,
                         cart_volume<float>**& field4, cart_volume<float>**& field3_rhs, cart_volume<float>**& field4_rhs,
                         cart_volume<float>**& model1_, cart_volume<float>**& model2_, cart_volume<float>**& model3_,
                         cart_volume<float>**& model4_, cart_volume<float>**& model5_, cart_volume<float>**& model6_,
                         cart_volume<float>**& model7_, pml_volumes_cg3d& pml, const pml_coef3d_gpu& pml_coef,
                         const sponge_coef3d_gpu& sponge_coef);

private:
    void ensure_volumes() override;
    void fwd_main_loop_2_inner(int isub, bool pml_x, bool pml_y, bool pml_z, bool pml_any) override;
    void adj_main_loop_2_inner(int isub, bool pml_x, bool pml_y, bool pml_z, bool pml_any) override;
    void update_grad_loop_inner(const int isub, cart_volume<float>* snap_x_vol, cart_volume<float>* snap_z_vol,
                                    cart_volume<float>* grad_Vp_vol, cart_volume<float>* adj_x_vol,
                                    cart_volume<float>* adj_z_vol) override;


    bool vols_cast_done;

    int nsub;
    int bc_opt;
    int order;

    double d1, d2, d3;
    float dt;

    cart_volume<float>**& field1_orig;
    cart_volume<float>**& field2_orig;
    cart_volume<float>**& field3_orig;
    cart_volume<float>**& field4_orig;
    cart_volume<float>**& field3_rhs_orig;
    cart_volume<float>**& field4_rhs_orig;
    cart_volume<float>**& model1_orig;
    cart_volume<float>**& model2_orig;
    cart_volume<float>**& model3_orig;
    cart_volume<float>**& model4_orig;
    cart_volume<float>**& model5_orig;
    cart_volume<float>**& model6_orig;
    cart_volume<float>**& model7_orig;

    std::vector<cart_volume_regular_gpu*> field1;
    std::vector<cart_volume_regular_gpu*> field2;
    std::vector<cart_volume_regular_gpu*> field3;
    std::vector<cart_volume_regular_gpu*> field4;
    std::vector<cart_volume_regular_gpu*> field3_rhs;
    std::vector<cart_volume_regular_gpu*> field4_rhs;
    std::vector<cart_volume_regular_gpu*> model1;
    std::vector<cart_volume_regular_gpu*> model2;
    std::vector<cart_volume_regular_gpu*> model3;
    std::vector<cart_volume_regular_gpu*> model4;
    std::vector<cart_volume_regular_gpu*> model5;
    std::vector<cart_volume_regular_gpu*> model6;
    std::vector<cart_volume_regular_gpu*> model7;
    pml_volumes_cg3d& pml;
    const pml_coef3d_gpu& pml_coef;
    const sponge_coef3d_gpu& sponge_coef;
};
#endif // KERNEL_TYPE1_GPU_H
