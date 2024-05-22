#ifndef PML_VOLUMES_CG3D_H
#define PML_VOLUMES_CG3D_H

#include "std_const.h"

template <class T>
class cart_volume;

struct pml_volumes_cg3d
{
    // PML auxiliary variable (needed by CPU & GPU kernels)
    // For CPML: field1_x_ corrects du/dx, and field1_xx_ corrects du/dxx
    cart_volume<realtype>**field1_x_, **field1_xx_, **field1_y_, **field1_yy_, **field1_z_, **field1_zz_;
    cart_volume<realtype>**field2_x_, **field2_xx_, **field2_y_, **field2_yy_, **field2_z_, **field2_zz_;

    cart_volume<realtype>**field11_x_, **field11_xx_, **field11_y_, **field11_yy_, **field11_z_, **field11_zz_;
    cart_volume<realtype>**field21_x_, **field21_xx_, **field21_y_, **field21_yy_, **field21_z_, **field21_zz_;

    // Saves modified 2nd derivatives (only needed by CPU kernels)
    cart_volume<realtype>**field1_xx_out_, **field1_yy_out_, **field1_zz_out_;
    cart_volume<realtype>**field2_xx_out_, **field2_yy_out_, **field2_zz_out_;

    cart_volume<realtype>**field11_xx_out_, **field11_yy_out_, **field11_zz_out_;
    cart_volume<realtype>**field21_xx_out_, **field21_yy_out_, **field21_zz_out_;

    // Inputs for 2nd derivatives (only needed for adjoint CPU & GPU kernels)
    cart_volume<realtype>**field1_xx_in_, **field1_yy_in_, **field1_zz_in_;
    cart_volume<realtype>**field2_xx_in_, **field2_yy_in_, **field2_zz_in_;

    cart_volume<realtype>**field11_xx_in_, **field11_yy_in_, **field11_zz_in_;
    cart_volume<realtype>**field21_xx_in_, **field21_yy_in_, **field21_zz_in_;

    // Only needed by GPU kernels
    cart_volume<realtype>**rxx_x_, **rxx_y_, **rxx_z_;
    cart_volume<realtype>**rzz_x_, **rzz_y_, **rzz_z_;
};

#endif
