#include "kernel_type2.h"

#include <algorithm>
#include <vector>

#include "kernel_utils.h"

#include "cart_volume_regular_gpu.h"
#include "pml_volumes_cg3d.h"
#include "pml_coef3d_gpu.h"
#include "sponge_coef3d_gpu.h"
#include "kernel_type2_small.h"
#include "axis.h"
#include "std_const.h"
#include "sim_const.h"
#include "kernel_type2_grad.h"
#include "kernel_type2_grad_snap.h"

#include "kernel_utils.h"
#include "kernel_type2_adj.h"
#include "kernel_type2_fwd.h"
#include "kernel_type2_fwd_drv.h"


kernel_type2::kernel_type2(
    int bc_opt, int order, int nsub, double d1, double d2, double d3, float dt, cart_volume<float>**& field1,
    cart_volume<float>**& field2, cart_volume<float>**& field3, cart_volume<float>**& field4, cart_volume<float>**& field3_rhs,
    cart_volume<float>**& field4_rhs, cart_volume<float>**& model1_, cart_volume<float>**& model2_, cart_volume<float>**& model3_,
    cart_volume<float>**& model4_, cart_volume<float>**& Model11_, cart_volume<float>**& model5_, cart_volume<float>**& model6_,
    cart_volume<float>**& model7_, cart_volume<float>**& dfield1dx, cart_volume<float>**& dfield1dy, cart_volume<float>**& dfield1dz,
    cart_volume<float>**& dfield2dx, cart_volume<float>**& dfield2dy, cart_volume<float>**& dfield2dz,
    std::vector<cart_volume<float>*>& snap_field1, std::vector<cart_volume<float>*>& snap_field2, pml_volumes_cg3d& pml,
    const pml_coef3d_gpu& pml_coef, const sponge_coef3d_gpu& sponge_coef, const bool simple_gradient)
    : vols_cast_done{ false }, bc_opt{ bc_opt }, order{ order }, nsub{ nsub }, d1{ d1 }, d2{ d2 }, d3{ d3 }, dt{ dt },
      field1_orig{ field1 }, field2_orig{ field2 }, field3_orig{ field3 }, field4_orig{ field4 }, field3_rhs_orig{ field3_rhs }, field4_rhs_orig{ field4_rhs },
      model1_orig{ model1_ }, model2_orig{ model2_ }, model3_orig{ model3_ }, model4_orig{ model4_ }, Model11_orig{ Model11_ }, model5_orig{ model5_ },
      dfield1dx_orig{ dfield1dx }, dfield1dy_orig{ dfield1dy }, dfield1dz_orig{ dfield1dz }, dfield2dx_orig{ dfield2dx }, dfield2dy_orig{ dfield2dy },
      dfield2dz_orig{ dfield2dz }, snap_field1_orig{ snap_field1 }, snap_field2_orig{ snap_field2 }, model6_orig{ model6_ }, model7_orig{ model7_ },
      pml{ pml }, pml_coef{ pml_coef }, sponge_coef{ sponge_coef }, simple_gradient{ simple_gradient }
{}



void
kernel_type2::ensure_volumes()
{
    if (!vols_cast_done) {
        auto cast_vols = [this](cart_volume<float>** in, std::vector<cart_volume_regular_gpu*>& out,
                                bool requireAligned = true) {
            if (in) {
                out.resize(this->nsub, nullptr);
                std::transform(in, in + nsub, out.begin(),
                               [requireAligned](cart_volume<float>* cv) -> cart_volume_regular_gpu* {
                                   if (!cv) {
                                       return nullptr;
                                   }
                                   auto gpu = cv->as<cart_volume_regular_gpu>();
                                   if (requireAligned) {
                                       EMSL_VERIFY(gpu->isAligned());
                                   }
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
        cast_vols(Model11_orig, Model11);
        cast_vols(dfield1dx_orig, dfield1dx);
        cast_vols(dfield1dy_orig, dfield1dy);
        cast_vols(dfield1dz_orig, dfield1dz);
        cast_vols(dfield2dx_orig, dfield2dx);
        cast_vols(dfield2dy_orig, dfield2dy);
        cast_vols(dfield2dz_orig, dfield2dz);
        cast_vols(snap_field1_orig.data(), snap_field1, false);
        cast_vols(snap_field2_orig.data(), snap_field2, false);

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

namespace {
__global__ __launch_bounds__(1024) void init_cijs_rho_kernel(volume_index vol3d, float* __restrict__ Model11,
                                                             float* __restrict__ model1, float* __restrict__ model2,
                                                             float* __restrict__ model3, float* __restrict__ model4,
                                                             float* __restrict__ model5, float* __restrict__ model6,
                                                             float* __restrict__ model7, int ixbeg, int iybeg, int izbeg,
                                                             int ixend, int iyend, int izend)
{
}
}

void
kernel_type2::init_cijs(int nsubdom, cart_volume<float>** Model11_, cart_volume<float>** model1_,
                                    cart_volume<float>** model2_, cart_volume<float>** model3_, cart_volume<float>** model4_,
                                    cart_volume<float>** model5_, cart_volume<float>** model6_, cart_volume<float>** model7_)
{
}

void
kernel_type2::compute_kernel2_derivatives_optimized(int isub, bool pml_x, bool pml_y, bool pml_z,
                                                               bool pml_any, cart_volume_regular_gpu* field1_cart_,
                                                               cart_volume_regular_gpu* field2_cart_)
{
    // Get the volume dimensions
    int ixbeg = field1_cart_->ax1()->ibeg;
    int ixend = field1_cart_->ax1()->iend;
    int iybeg = field1_cart_->ax2()->ibeg;
    int iyend = field1_cart_->ax2()->iend;
    int izbeg = field1_cart_->ax3()->ibeg;
    int izend = field1_cart_->ax3()->iend;

    // Get the volume leading dimensions
    int ldimx = field1_cart_->ax1()->ntot;
    int ldimy = field1_cart_->ax2()->ntot;

    float unused;
    float invdx, invdy, invdz;
    kernel_utils::compute_fd_const(d1, d2, d3, invdx, invdy, invdz, unused, unused, unused, unused, unused, unused);
    kernel3_gpu_kernels::launch_fwd_kernel2_drv(
        field1[isub]->getData(), field2[isub]->getData(), Model11[isub]->getData(), dfield1dx[isub]->getData(),
        dfield1dy[isub]->getData(), dfield1dz[isub]->getData(), dfield2dx[isub]->getData(), dfield2dy[isub]->getData(),
        dfield2dz[isub]->getData(), ixbeg, ixend, iybeg, iyend, izbeg, izend, invdx, invdy, invdz, ldimx, ldimy, order,
        0);
}

void
kernel_type2::fwd_main_loop_impl(int isub, bool pml_x, bool pml_y, bool pml_z,
                                                              bool pml_any)
{
    cart_volume_regular_gpu* field1_cart_ = field1[isub]->as<cart_volume_regular_gpu>();
    cart_volume_regular_gpu* field2_cart_ = field2[isub]->as<cart_volume_regular_gpu>();
    compute_kernel2_derivatives_optimized(isub, pml_x, pml_y, pml_z, pml_any, field1_cart_, field2_cart_);
}

void
kernel_type2::fwd_main_loop_2_impl(int isub, bool pml_x, bool pml_y, bool pml_z, bool pml_any)
{
    // Get the volume dimensions
    int ixbeg = field3[isub]->ax1()->ibeg;
    int ixend = field3[isub]->ax1()->iend;
    int iybeg = field3[isub]->ax2()->ibeg;
    int iyend = field3[isub]->ax2()->iend;
    int izbeg = field3[isub]->ax3()->ibeg;
    int izend = field3[isub]->ax3()->iend;

    // Get the volume leading dimensions
    int ldimx = field3[isub]->ax1()->ntot;
    int ldimy = field3[isub]->ax2()->ntot;

    float2 const* pml_ab_xx = pml_any ? pml_coef.getPMLCoef(isub)[0] : nullptr;
    float2 const* pml_ab_yy = pml_any ? pml_coef.getPMLCoef(isub)[1] : nullptr;
    float2 const* pml_ab_zz = pml_any ? pml_coef.getPMLCoef(isub)[2] : nullptr;
    float2 const* sponge_ab_xx = !pml_any ? sponge_coef.getSpongeCoef(isub)[0] : nullptr;
    float2 const* sponge_ab_yy = !pml_any ? sponge_coef.getSpongeCoef(isub)[1] : nullptr;
    float2 const* sponge_ab_zz = !pml_any ? sponge_coef.getSpongeCoef(isub)[2] : nullptr;

    auto ptr = [this](cart_volume<float>** pml_volumes_cg3d::*p, int isub) -> float* {
        return (bc_opt != _BDY_SPONGE) && (pml.*p)[isub] ? (pml.*p)[isub]->as<cart_volume_regular_gpu>()->getData()
                                                         : nullptr;
    };

    bool sponge_active = (bc_opt == _BDY_SPONGE && sponge_coef.use_sponge());

    launch_update_rho_fwd_main_loop_2_1(
        field3[isub]->getData(), field4[isub]->getData(), dfield1dx[isub]->getData(), dfield2dx[isub]->getData(),
        dfield1dy[isub]->getData(), dfield2dy[isub]->getData(), dfield1dz[isub]->getData(), dfield2dz[isub]->getData(),
        field1[isub]->getData(), field2[isub]->getData(), Model11[isub]->getData(), model1[isub]->getData(), model2[isub]->getData(),
        model3[isub]->getData(), model4[isub]->getData(), model5[isub]->getData(), model6[isub]->getData(), model7[isub]->getData(),
        ptr(&pml_volumes_cg3d::field1_xx_, isub), ptr(&pml_volumes_cg3d::field2_xx_, isub),
        ptr(&pml_volumes_cg3d::field1_yy_, isub), ptr(&pml_volumes_cg3d::field2_yy_, isub),
        ptr(&pml_volumes_cg3d::field1_zz_, isub), ptr(&pml_volumes_cg3d::field2_zz_, isub), pml_ab_xx, pml_ab_yy, pml_ab_zz,
        sponge_ab_xx, sponge_ab_yy, sponge_ab_zz, ixbeg, ixend, iybeg, iyend, izbeg, izend, ldimx, ldimy, dt, d1, d2,
        d3, sponge_active, order, pml_x, pml_y, pml_z, 0);
}

void
kernel_type2::adj_main_loop_impl(int isub, bool pml_x, bool pml_y, bool pml_z,
                                                              bool pml_any)
{
    cart_volume_regular_gpu* field1_cart_ = field1[isub]->as<cart_volume_regular_gpu>();
    cart_volume_regular_gpu* field2_cart_ = field2[isub]->as<cart_volume_regular_gpu>();
    // Get the volume dimensions
    int ixbeg = field1_cart_->ax1()->ibeg;
    int ixend = field1_cart_->ax1()->iend;
    int iybeg = field1_cart_->ax2()->ibeg;
    int iyend = field1_cart_->ax2()->iend;
    int izbeg = field1_cart_->ax3()->ibeg;
    int izend = field1_cart_->ax3()->iend;

    // Get the volume leading dimensions
    int ldimx = field1_cart_->ax1()->nallocated_elements;
    int ldimy = field1_cart_->ax2()->nallocated_elements;

    float unused;
    float invdx, invdy, invdz, invdxx, invdyy, invdzz, invdxy, invdyz, invdzx;
    kernel_utils::compute_fd_const(d1, d2, d3, invdx, invdy, invdz, invdxx, invdyy, invdzz, invdxy, invdyz, invdzx);

    adj_kernel3_derivatives::launch_adj_kernel2_drv(
        dfield1dx[isub]->getData(), dfield2dx[isub]->getData(), dfield1dy[isub]->getData(), dfield2dy[isub]->getData(),
        dfield1dz[isub]->getData(), dfield2dz[isub]->getData(), field3_rhs[isub]->getData(), field4_rhs[isub]->getData(),
        field1[isub]->getData(), field2[isub]->getData(), Model11[isub]->getData(), model1[isub]->getData(),
        model2[isub]->getData(), model3[isub]->getData(), model4[isub]->getData(), model5[isub]->getData(),
        model6[isub]->getData(), model7[isub]->getData(), ixbeg, ixend, iybeg, iyend, izbeg, izend, invdxx, invdyy,
        invdzz, invdxy, invdyz, invdzx, ldimx, ldimy, order, dt, invdx, invdy, invdz, 0);
}

void
kernel_type2::adj_main_loop_2_impl(int isub, bool pml_x, bool pml_y, bool pml_z, bool pml_any)
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

    float2 const* ab_xx = sponge_coef.getSpongeCoef(isub)[0];
    float2 const* ab_yy = sponge_coef.getSpongeCoef(isub)[1];
    float2 const* ab_zz = sponge_coef.getSpongeCoef(isub)[2];
    bool sponge_active = sponge_coef.use_sponge();

    kernel3_gpu_kernels::launch_update_adj_kernel_main_3(
        field3[isub]->getData(), field4[isub]->getData(), dfield1dx[isub]->getData(), dfield2dx[isub]->getData(),
        dfield1dy[isub]->getData(), dfield2dy[isub]->getData(), dfield1dz[isub]->getData(), dfield2dz[isub]->getData(),
        field3_rhs[isub]->getData(), field4_rhs[isub]->getData(), ab_xx, ab_yy, ab_zz, ixbeg, ixend, iybeg, iyend, izbeg, izend,
        ldimx, ldimy, dt, d1, d2, d3, pml_x, pml_y, pml_z, order, sponge_active, 0);
}

void
kernel_type2::kernel_loop4_derivative_impl(int isub)
{
    int nxtot = field1[isub]->as<cart_volume_regular_gpu>()->ax1()->ntot;
    int nytot = field1[isub]->as<cart_volume_regular_gpu>()->ax2()->ntot;
    int izbeg = field1[isub]->as<cart_volume_regular_gpu>()->ax3()->ibeg;

    dim3 threads(32, 32, 1);
    dim3 blocks((nxtot + threads.x - 1) / threads.x, (nytot + threads.y - 1) / threads.y, izbeg + 1);

    kernel3_gpu_kernels::kernel_loop4_derivatives<<<blocks, threads>>>(
        dfield1dx[isub]->as<cart_volume_regular_gpu>()->getData(), dfield1dy[isub]->as<cart_volume_regular_gpu>()->getData(),
        dfield1dz[isub]->as<cart_volume_regular_gpu>()->getData(), dfield2dx[isub]->as<cart_volume_regular_gpu>()->getData(),
        dfield2dy[isub]->as<cart_volume_regular_gpu>()->getData(), dfield2dz[isub]->as<cart_volume_regular_gpu>()->getData(),
        nxtot, nytot, izbeg);
}

void
kernel_type2::snaps_loop_impl(int isub)
{
    // Use the "simple" derivative calculation (same as fwd interior kernel).
    // Once we have an optimized version, we can consider switching to that.
    cart_volume_regular_gpu* snap_field1_gpu_ = snap_field1[isub]->as<cart_volume_regular_gpu>();
    cart_volume_regular_gpu* snap_field2_gpu_ = snap_field2[isub]->as<cart_volume_regular_gpu>();
    compute_kernel2_derivatives_optimized(isub, false, false, false, false, snap_field1_gpu_, snap_field2_gpu_);
}

void
kernel_type2::update_grad_loop_1_impl(const int isub, cart_volume<float>* snap_field3_vol,
                                                       cart_volume<float>* snap_field4_vol, cart_volume<float>* grad_Vp_vol,
                                                       cart_volume<float>* adj_field3_vol, cart_volume<float>* adj_field4_vol)
{
    int ixbeg, ixend, iybeg, iyend, izbeg, izend;
    kernel_utils::getIndexRange(field1[isub]->as<cart_volume_regular_gpu>(), ixbeg, ixend, iybeg, iyend, izbeg, izend);
    volume_index idx = snap_field3_vol->as<cart_volume_regular_gpu>()->vol_idx();
    volume_index idx_adj = adj_field3_vol->as<cart_volume_regular_gpu>()->vol_idx();
    auto snap_field3 = snap_field3_vol->as<cart_volume_regular_gpu>()->getData();
    auto snap_field4 = snap_field4_vol->as<cart_volume_regular_gpu>()->getData();
    auto grad_Vp = grad_Vp_vol->as<cart_volume_regular_gpu>()->getData();
    auto adj_field3 = adj_field3_vol->as<cart_volume_regular_gpu>()->getData();
    auto adj_field4 = adj_field4_vol->as<cart_volume_regular_gpu>()->getData();

    int num_x = ixend - ixbeg + 1;
    int num_y = iyend - iybeg + 1;
    int num_z = izend - izbeg + 1;

    dim3 threads(32, 32, 1);
    dim3 blocks((num_x + threads.x - 1) / threads.x, (num_y + threads.y - 1) / threads.y,
                (num_z + threads.z - 1) / threads.z);

    kernel3_gpu_kernels::update_Vp_gradient<<<blocks, threads, 0>>>(
        idx, idx_adj, snap_field3, snap_field4, grad_Vp, adj_field3, adj_field4, ixbeg, ixend, iybeg, iyend, izbeg, izend);
}

void
kernel_type2::update_grad_loop_2_impl(
    const int isub, cart_volume<float>* fwd_dfield1dx_vol, cart_volume<float>* fwd_dfield1dy_vol,
    cart_volume<float>* fwd_dfield1dz_vol, cart_volume<float>* fwd_dfield2dx_vol, cart_volume<float>* fwd_dfield2dy_vol,
    cart_volume<float>* fwd_dfield2dz_vol, cart_volume<float>* grad_Rho_vol)
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
    int grad_ldimx = grad_Rho_vol->as<cart_volume_regular_gpu>()->ax1()->ntot;
    int grad_ldimy = grad_Rho_vol->as<cart_volume_regular_gpu>()->ax2()->ntot;

    float* grad = grad_Rho_vol->as<cart_volume_regular_gpu>()->getData();
    float const* fwd_field1_dx = fwd_dfield1dx_vol->as<cart_volume_regular_gpu>()->getData();
    float const* fwd_field2_dx = fwd_dfield2dx_vol->as<cart_volume_regular_gpu>()->getData();
    float const* fwd_field1_dy = fwd_dfield1dy_vol->as<cart_volume_regular_gpu>()->getData();
    float const* fwd_field2_dy = fwd_dfield2dy_vol->as<cart_volume_regular_gpu>()->getData();
    float const* fwd_field1_dz = fwd_dfield1dz_vol->as<cart_volume_regular_gpu>()->getData();
    float const* fwd_field2_dz = fwd_dfield2dz_vol->as<cart_volume_regular_gpu>()->getData();

    launch_update_rho_gradient_rho(
        grad, fwd_field1_dx, fwd_field2_dx, fwd_field1_dy, fwd_field2_dy, fwd_field1_dz, fwd_field2_dz, field1[isub]->getData(),
        field2[isub]->getData(), Model11[isub]->getData(), model1[isub]->getData(), model2[isub]->getData(), model3[isub]->getData(),
        model4[isub]->getData(), model5[isub]->getData(), model6[isub]->getData(), model7[isub]->getData(), ixbeg, ixend, iybeg, iyend,
        izbeg, izend, ldimx, ldimy, grad_ldimx, grad_ldimy, d1, d2, d3, order, simple_gradient, 0);
}

void
kernel_type2::swap_pml_volumes(int isub)
{
}
