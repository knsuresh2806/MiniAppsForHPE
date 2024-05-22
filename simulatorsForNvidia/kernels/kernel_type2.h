#ifndef KERNEL_TYPE2_H
#define KERNEL_TYPE2_H

#include <vector>

#include "kernel_type2_base.h"
#include "pml_coef3d_gpu.h"
#include "sponge_coef3d_gpu.h"

class cart_volume_regular_gpu;
class pml_volumes_cg3d;
class pml_coef3d_gpu;
class sponge_coef3d_gpu;

class kernel_type2 : public kernel_type2_base
{
public:
    kernel_type2(int bc_opt, int order, int nsub, double d1, double d2, double d3, float dt,
                             cart_volume<float>**& field1, cart_volume<float>**& field2, cart_volume<float>**& field3,
                             cart_volume<float>**& field4, cart_volume<float>**& field3_rhs, cart_volume<float>**& field4_rhs,
                             cart_volume<float>**& model1_, cart_volume<float>**& model2_, cart_volume<float>**& model3_,
                             cart_volume<float>**& model4_, cart_volume<float>**& Model11_, cart_volume<float>**& model5_,
                             cart_volume<float>**& model6_, cart_volume<float>**& model7_, cart_volume<float>**& dfield1dx,
                             cart_volume<float>**& dfield1dy, cart_volume<float>**& dfield1dz, cart_volume<float>**& dfield2dx,
                             cart_volume<float>**& dfield2dy, cart_volume<float>**& dfield2dz,
                             std::vector<cart_volume<float>*>& snap_field1, std::vector<cart_volume<float>*>& snap_field2,
                             pml_volumes_cg3d& pml, const pml_coef3d_gpu& pml_coef,
                             const sponge_coef3d_gpu& sponge_coef, const bool simple_gradient = false);

    static void init_cijs(int nsubdom, cart_volume<float>** Model11_, cart_volume<float>** model1_,
                          cart_volume<float>** model2_, cart_volume<float>** model3_, cart_volume<float>** model4_,
                          cart_volume<float>** model5_, cart_volume<float>** model6_, cart_volume<float>** model7_);

private:
    /** Ensure that any volumes that were passed to the constructor have been cast to the appropriate type */
    void ensure_volumes() override;
    void fwd_main_loop_impl(int isub, bool pml_x, bool pml_y, bool pml_z, bool pml_any) override;
    void fwd_main_loop_2_impl(int isub, bool pml_x, bool pml_y, bool pml_z, bool pml_any) override;
    void adj_main_loop_impl(int isub, bool pml_x, bool pml_y, bool pml_z, bool pml_any) override;
    void adj_main_loop_2_impl(int isub, bool pml_x, bool pml_y, bool pml_z, bool pml_any) override;
    void kernel_loop4_derivative_impl(int isub) override;
    void snaps_loop_impl(int isub) override;
    void compute_kernel2_derivatives_optimized(int isub, bool pml_x, bool pml_y, bool pml_z, bool pml_any,
                                              cart_volume_regular_gpu* field1_cart_, cart_volume_regular_gpu* field2_cart_);
    void update_grad_loop_1_impl(const int isub, cart_volume<float>* snap_field3_vol, cart_volume<float>* snap_field4_vol,
                                      cart_volume<float>* grad_Vp_vol, cart_volume<float>* adj_x_vol,
                                      cart_volume<float>* adj_z_vol) override;
    void update_grad_loop_2_impl(const int isub, cart_volume<float>* fwd_field1_xin_vol,
                                       cart_volume<float>* fwd_field1_yin_vol, cart_volume<float>* fwd_field1_zin_vol,
                                       cart_volume<float>* fwd_field2_xin_vol, cart_volume<float>* fwd_field2_yin_vol,
                                       cart_volume<float>* fwd_field2_zin_vol, cart_volume<float>* grad_Rho_vol) override;

    void swap_pml_volumes(int isub);

    bool vols_cast_done;
    bool simple_gradient;

    int bc_opt;
    int nsub;
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
    cart_volume<float>**& Model11_orig;
    cart_volume<float>**& dfield1dx_orig;
    cart_volume<float>**& dfield1dy_orig;
    cart_volume<float>**& dfield1dz_orig;
    cart_volume<float>**& dfield2dx_orig;
    cart_volume<float>**& dfield2dy_orig;
    cart_volume<float>**& dfield2dz_orig;
    std::vector<cart_volume<float>*>& snap_field1_orig;
    std::vector<cart_volume<float>*>& snap_field2_orig;

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
    std::vector<cart_volume_regular_gpu*> Model11;
    std::vector<cart_volume_regular_gpu*> dfield1dx;
    std::vector<cart_volume_regular_gpu*> dfield1dy;
    std::vector<cart_volume_regular_gpu*> dfield1dz;
    std::vector<cart_volume_regular_gpu*> dfield2dx;
    std::vector<cart_volume_regular_gpu*> dfield2dy;
    std::vector<cart_volume_regular_gpu*> dfield2dz;
    std::vector<cart_volume_regular_gpu*> snap_field1;
    std::vector<cart_volume_regular_gpu*> snap_field2;

    pml_volumes_cg3d& pml;
    const pml_coef3d_gpu& pml_coef;
    const sponge_coef3d_gpu& sponge_coef;
};

#endif