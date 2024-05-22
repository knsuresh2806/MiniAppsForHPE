#ifndef KERNEL_TYPE1_BASE_H
#define KERNEL_TYPE1_BASE_H

#include "pml_volumes_cg3d.h"
#include "pml_coef3d.h"
#include "sponge_coef3d.h"
template <class T>
class cart_volume;
class kernel_abstract_base;

class kernel_type1_base
{
public:
    virtual ~kernel_type1_base() = default;

    void fwd_main_loop_2(int isub, bool pml_x, bool pml_y, bool pml_z, bool pml_any);
    void adj_main_loop_2(int isub, bool pml_x, bool pml_y, bool pml_z, bool pml_any);
    void update_grad_loop(const int isub, cart_volume<realtype>* snap_x_vol, cart_volume<realtype>* snap_z_vol,
                              cart_volume<realtype>* grad_Vp_vol, cart_volume<realtype>* adj_x_vol,
                              cart_volume<realtype>* adj_z_vol);

private:
    /** Ensure that any volumes that were passed to the constructor have been cast to the appropriate type */
    virtual void ensure_volumes() = 0;
    virtual void fwd_main_loop_2_inner(int isub, bool pml_x, bool pml_y, bool pml_z, bool pml_any) = 0;
    virtual void adj_main_loop_2_inner(int isub, bool pml_x, bool pml_y, bool pml_z, bool pml_any) = 0;
    virtual void update_grad_loop_inner(const int isub, cart_volume<realtype>* snap_x_vol,
                                            cart_volume<realtype>* snap_z_vol, cart_volume<realtype>* grad_Vp_vol,
                                            cart_volume<realtype>* adj_x_vol, cart_volume<realtype>* adj_z_vol) = 0;
};

inline void
kernel_type1_base::fwd_main_loop_2(int isub, bool pml_x, bool pml_y, bool pml_z, bool pml_any)
{
    ensure_volumes();
    fwd_main_loop_2_inner(isub, pml_x, pml_y, pml_z, pml_any);
}

inline void
kernel_type1_base::adj_main_loop_2(int isub, bool pml_x, bool pml_y, bool pml_z, bool pml_any)
{
    ensure_volumes();
    adj_main_loop_2_inner(isub, pml_x, pml_y, pml_z, pml_any);
}

inline void
kernel_type1_base::update_grad_loop(const int isub, cart_volume<realtype>* snap_x_vol,
                                       cart_volume<realtype>* snap_z_vol, cart_volume<realtype>* grad_Vp_vol,
                                       cart_volume<realtype>* adj_x_vol, cart_volume<realtype>* adj_z_vol)
{
    ensure_volumes();
    update_grad_loop_inner(isub, snap_x_vol, snap_z_vol, grad_Vp_vol, adj_x_vol, adj_z_vol);
}

#endif
