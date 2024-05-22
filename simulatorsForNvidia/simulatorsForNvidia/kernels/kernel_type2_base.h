#ifndef KERNEL_TYPE2_BASE_H
#define KERNEL_TYPE2_BASE_H

#include "pml_volumes_cg3d.h"
#include "pml_coef3d.h"
#include "sponge_coef3d.h"

template <class T>
class cart_volume;

class kernel_type2_base
{
public:
    virtual ~kernel_type2_base() = default;

    void fwd_main_loop(int isub, bool pml_x, bool pml_y, bool pml_z, bool pml_any);
    void fwd_main_loop_2(int isub, bool pml_x, bool pml_y, bool pml_z, bool pml_any);
    void adj_main_loop(int isub, bool pml_x, bool pml_y, bool pml_z, bool pml_any);
    void adj_main_loop_2(int isub, bool pml_x, bool pml_y, bool pml_z, bool pml_any);
    void kernel_loop4_derivative(int isub);
    void snaps_loop(int isub);

    /** Updates the Vp gradient */
    void update_grad_loop_1(const int isub, cart_volume<realtype>* snap_x_vol, cart_volume<realtype>* snap_z_vol,
                                 cart_volume<realtype>* grad_Vp_vol, cart_volume<realtype>* adj_x_vol,
                                 cart_volume<realtype>* adj_z_vol);
    void update_grad_loop_2(const int isub, cart_volume<realtype>* fwd_field1_xin_vol,
                                  cart_volume<realtype>* fwd_field1_yin_vol, cart_volume<realtype>* fwd_field1_zin_vol,
                                  cart_volume<realtype>* fwd_field2_xin_vol, cart_volume<realtype>* fwd_field2_yin_vol,
                                  cart_volume<realtype>* fwd_field2_zin_vol, cart_volume<realtype>* grad_Rho_vol);

private:
    virtual void ensure_volumes() = 0;
    virtual void fwd_main_loop_impl(int isub, bool pml_x, bool pml_y, bool pml_z, bool pml_any) = 0;
    virtual void fwd_main_loop_2_impl(int isub, bool pml_x, bool pml_y, bool pml_z, bool pml_any) = 0;
    virtual void adj_main_loop_impl(int isub, bool pml_x, bool pml_y, bool pml_z, bool pml_any) = 0;
    virtual void adj_main_loop_2_impl(int isub, bool pml_x, bool pml_y, bool pml_z, bool pml_any) = 0;
    virtual void kernel_loop4_derivative_impl(int isub) = 0;
    virtual void snaps_loop_impl(int isub) = 0;
    virtual void update_grad_loop_1_impl(const int isub, cart_volume<realtype>* snap_field3_vol,
                                              cart_volume<realtype>* snap_field4_vol, cart_volume<realtype>* grad_Vp_vol,
                                              cart_volume<realtype>* adj_field3_vol, cart_volume<realtype>* adj_field4_vol) = 0;
    virtual void update_grad_loop_2_impl(const int isub, cart_volume<realtype>* fwd_field1_xin_vol,
                                               cart_volume<realtype>* fwd_field1_yin_vol,
                                               cart_volume<realtype>* fwd_field1_zin_vol,
                                               cart_volume<realtype>* fwd_field2_xin_vol,
                                               cart_volume<realtype>* fwd_field2_yin_vol,
                                               cart_volume<realtype>* fwd_field2_zin_vol,
                                               cart_volume<realtype>* grad_Rho_vol) = 0;
};

inline void
kernel_type2_base::fwd_main_loop(int isub, bool pml_x, bool pml_y, bool pml_z, bool pml_any)
{
    ensure_volumes();
    fwd_main_loop_impl(isub, pml_x, pml_y, pml_z, pml_any);
}

inline void
kernel_type2_base::fwd_main_loop_2(int isub, bool pml_x, bool pml_y, bool pml_z, bool pml_any)
{
    ensure_volumes();
    fwd_main_loop_2_impl(isub, pml_x, pml_y, pml_z, pml_any);
}

inline void
kernel_type2_base::adj_main_loop(int isub, bool pml_x, bool pml_y, bool pml_z, bool pml_any)
{
    ensure_volumes();
    adj_main_loop_impl(isub, pml_x, pml_y, pml_z, pml_any);
}

inline void
kernel_type2_base::adj_main_loop_2(int isub, bool pml_x, bool pml_y, bool pml_z, bool pml_any)
{
    ensure_volumes();
    adj_main_loop_2_impl(isub, pml_x, pml_y, pml_z, pml_any);
}

inline void
kernel_type2_base::kernel_loop4_derivative(int isub)
{
    ensure_volumes();
    kernel_loop4_derivative_impl(isub);
}

inline void
kernel_type2_base::snaps_loop(int isub)
{
    ensure_volumes();
    snaps_loop_impl(isub);
}

inline void
kernel_type2_base::update_grad_loop_1(const int isub, cart_volume<realtype>* snap_field3_vol,
                                          cart_volume<realtype>* snap_field4_vol, cart_volume<realtype>* grad_Vp_vol,
                                          cart_volume<realtype>* adj_field3_vol, cart_volume<realtype>* adj_field4_vol)
{
    ensure_volumes();
    update_grad_loop_1_impl(isub, snap_field3_vol, snap_field4_vol, grad_Vp_vol, adj_field3_vol, adj_field4_vol);
}


inline void
kernel_type2_base::update_grad_loop_2(const int isub, cart_volume<realtype>* fwd_field1_xin_vol,
                                           cart_volume<realtype>* fwd_field1_yin_vol,
                                           cart_volume<realtype>* fwd_field1_zin_vol,
                                           cart_volume<realtype>* fwd_field2_xin_vol,
                                           cart_volume<realtype>* fwd_field2_yin_vol,
                                           cart_volume<realtype>* fwd_field2_zin_vol, cart_volume<realtype>* grad_Rho_vol)
{
    ensure_volumes();
    update_grad_loop_2_impl(isub, fwd_field1_xin_vol, fwd_field1_yin_vol, fwd_field1_zin_vol, fwd_field2_xin_vol,
                                  fwd_field2_yin_vol, fwd_field2_zin_vol, grad_Rho_vol);
}

#endif
