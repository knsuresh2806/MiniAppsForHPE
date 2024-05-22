#include <algorithm>
#include <vector>

#include "kernel_utils.h"
#include "kernel_abstract_base.h"
#include "kernel_type1_gpu.h"
#include "kernel_type1_gpu_fwd.h"
#include "kernel_type1_gpu_adj.h"
#include "kernel_type1_gpu_grad.h"
#include "cart_volume_regular.h"
#include "cart_volume_regular_gpu.h"
#include "sim_const.h"

kernel_type1_gpu::kernel_type1_gpu(
    int bc_opt, int order, int nsub, double d1, double d2, double d3, float dt, cart_volume<float>**& field1,
    cart_volume<float>**& field2, cart_volume<float>**& field3, cart_volume<float>**& field4, cart_volume<float>**& field3_rhs,
    cart_volume<float>**& field4_rhs, cart_volume<float>**& model1_, cart_volume<float>**& model2_, cart_volume<float>**& model3_,
    cart_volume<float>**& model4_, cart_volume<float>**& model5_, cart_volume<float>**& model6_, cart_volume<float>**& model7_,
    pml_volumes_cg3d& pml, const pml_coef3d_gpu& pml_coef, const sponge_coef3d_gpu& sponge_coef)
    : vols_cast_done{ false }, bc_opt{ bc_opt }, order{ order }, nsub{ nsub }, d1{ d1 }, d2{ d2 }, d3{ d3 }, dt{ dt },
      field1_orig{ field1 }, field2_orig{ field2 }, field3_orig{ field3 }, field4_orig{ field4 }, field3_rhs_orig{ field3_rhs },
      field4_rhs_orig{ field4_rhs }, model1_orig{ model1_ }, model2_orig{ model2_ }, model3_orig{ model3_ }, model4_orig{ model4_ }, model5_orig{ model5_ },
      model6_orig{ model6_ }, model7_orig{ model7_ }, pml{ pml }, pml_coef{ pml_coef }, sponge_coef{ sponge_coef }
{}

void
kernel_type1_gpu::ensure_volumes()
{
    if (!vols_cast_done) {
        auto cast_vols = [this](cart_volume<float>** in, std::vector<cart_volume_regular_gpu*>& out) {
            if (in) {
                out.resize(this->nsub, nullptr);
                std::transform(in, in + nsub, out.begin(), [](cart_volume<float>* cv) -> cart_volume_regular_gpu* {
                    if (!cv) {
                        return nullptr;
                    }
                    auto gpu = cv->as<cart_volume_regular_gpu>();
                    EMSL_VERIFY(gpu->isAligned());
                    return gpu;
                });
            }
        };

        cast_vols(field1_orig, field1);
        cast_vols(field2_orig, field2);
        cast_vols(field3_orig, field3);
        cast_vols(field4_orig, field4);
        cast_vols(field3_rhs_orig, field3_rhs);
        cast_vols(field4_rhs_orig, field4_rhs);
        cast_vols(model1_orig, model1);
        cast_vols(model2_orig, model2);
        cast_vols(model3_orig, model3);
        cast_vols(model4_orig, model4);
        cast_vols(model5_orig, model5);
        cast_vols(model6_orig, model6);
        cast_vols(model7_orig, model7);

        if (bc_opt != _BDY_SPONGE) {
            auto check_vols = [this](cart_volume<float>** in) {
                if (in) {
                    std::for_each(in, in + nsub, [](cart_volume<float>* cv) {
                        if (cv) {
                            auto gpu = cv->as<cart_volume_regular_gpu>();
                            EMSL_VERIFY(gpu->isAligned());
                        }
                    });
                }
            };

            check_vols(pml.field1_x_);
            check_vols(pml.field2_x_);
            check_vols(pml.rxx_x_);
            check_vols(pml.rzz_x_);
            check_vols(pml.field1_xx_);
            check_vols(pml.field2_xx_);
            check_vols(pml.field1_y_);
            check_vols(pml.field2_y_);
            check_vols(pml.rxx_y_);
            check_vols(pml.rzz_y_);
            check_vols(pml.field1_yy_);
            check_vols(pml.field2_yy_);
            check_vols(pml.field1_z_);
            check_vols(pml.field2_z_);
            check_vols(pml.rxx_z_);
            check_vols(pml.rzz_z_);
            check_vols(pml.field1_zz_);
            check_vols(pml.field2_zz_);
        }

        vols_cast_done = true;
    }
}

void
kernel_type1_gpu::fwd_main_loop_2_inner(int isub, bool pml_x, bool pml_y, bool pml_z, bool pml_any)
{
    // Get the volume dimensions
    int ixbeg = field1[isub]->ax1()->ibeg;
    int ixend = field1[isub]->ax1()->iend;
    int iybeg = field1[isub]->ax2()->ibeg;
    int iyend = field1[isub]->ax2()->iend;
    int izbeg = field1[isub]->ax3()->ibeg;
    int izend = field1[isub]->ax3()->iend;

    // Get the volume leading dimensions
    int ldimx = field1[isub]->ax1()->ntot;
    int ldimy = field1[isub]->ax2()->ntot;
    int ldimz = field1[isub]->ax3()->ntot;

    const float2 *ABx, *ABy, *ABz;
    kernel_utils::getSpongeCoef(sponge_coef, isub, ABx, ABy, ABz);
    bool use_sponge = sponge_coef.use_sponge();

    fwd_main_loop_2_1_inner_gpu::launch(
        field3[isub]->getData(), field4[isub]->getData(), field1[isub]->getData(), field2[isub]->getData(), model1[isub]->getData(),
        model2[isub]->getData(), model3[isub]->getData(), model4[isub]->getData(), model5[isub]->getData(), model6[isub]->getData(),
        model7[isub]->getData(), ABx, ABy, ABz, ixbeg, ixend, iybeg, iyend, izbeg, izend, ldimx, ldimy, ldimz, ldimx,
        ldimy, ldimz, dt, d1, d2, d3, order, 0, false, use_sponge);
}

void
kernel_type1_gpu::adj_main_loop_2_inner(int isub, bool pml_x, bool pml_y, bool pml_z, bool pml_any)
{
    // Get the volume dimensions
    int ixbeg = field1[isub]->ax1()->ibeg;
    int ixend = field1[isub]->ax1()->iend;
    int iybeg = field1[isub]->ax2()->ibeg;
    int iyend = field1[isub]->ax2()->iend;
    int izbeg = field1[isub]->ax3()->ibeg;
    int izend = field1[isub]->ax3()->iend;

    // Get the volume leading dimensions
    int ldimx = field1[isub]->ax1()->ntot;
    int ldimy = field1[isub]->ax2()->ntot;
    int ldimz = field1[isub]->ax3()->ntot;

    const float2 *ABx, *ABy, *ABz;
    kernel_utils::getSpongeCoef(sponge_coef, isub, ABx, ABy, ABz);
    bool use_sponge = sponge_coef.use_sponge();

    adj_main_loop_2_inner_gpu::launch(
        field3[isub]->getData(), field4[isub]->getData(), field1[isub]->getData(), field2[isub]->getData(), model1[isub]->getData(),
        model2[isub]->getData(), model3[isub]->getData(), model4[isub]->getData(), model5[isub]->getData(), model6[isub]->getData(),
        model7[isub]->getData(), ABx, ABy, ABz, ixbeg, ixend, iybeg, iyend, izbeg, izend, ldimx, ldimy, ldimz, dt, d1,
        d2, d3, order, 0, use_sponge);
}

void
kernel_type1_gpu::update_grad_loop_inner(const int isub, cart_volume<float>* snap_x_vol,
                                                 cart_volume<float>* snap_z_vol, cart_volume<float>* grad_Vp_vol,
                                                 cart_volume<float>* adj_x_vol, cart_volume<float>* adj_z_vol)
{
    int ixbeg, ixend, iybeg, iyend, izbeg, izend;
    kernel_utils::getIndexRange(adj_x_vol->as<cart_volume_regular_gpu>(), ixbeg, ixend, iybeg, iyend, izbeg, izend);

    volume_index idx = snap_x_vol->as<cart_volume_regular_gpu>()->vol_idx();
    volume_index idx_adj = adj_x_vol->as<cart_volume_regular_gpu>()->vol_idx();
    auto snap_x = snap_x_vol->as<cart_volume_regular_gpu>()->getData();
    auto snap_z = snap_z_vol->as<cart_volume_regular_gpu>()->getData();
    auto grad_Vp = grad_Vp_vol->as<cart_volume_regular_gpu>()->getData();
    auto adj_x = adj_x_vol->as<cart_volume_regular_gpu>()->getData();
    auto adj_z = adj_z_vol->as<cart_volume_regular_gpu>()->getData();

    launch_update_grad_loop_kernel(idx, idx_adj, snap_x, snap_z, grad_Vp, adj_x, adj_z, ixbeg, ixend, iybeg, iyend,
                                       izbeg, izend);
}

