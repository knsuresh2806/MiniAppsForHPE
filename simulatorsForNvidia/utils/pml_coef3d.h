#ifndef PML_COEF3D_H
#define PML_COEF3D_H

#include <vector>
#include "std_const.h"

class pml_coef3d
{
public:
    pml_coef3d() = default;

    /* Disable copying to prevent copying coefficient arrays */
    pml_coef3d(const pml_coef3d&) = delete;
    pml_coef3d& operator=(const pml_coef3d&) = delete;

    // PML coefficients on non-staggered mesh locations (field1, tyy, field2)
    std::vector<std::vector<realtype>> x_A_, x_B_;
    std::vector<std::vector<realtype>> y_A_, y_B_;
    std::vector<std::vector<realtype>> z_A_, z_B_;

    // PML coefficients on staggered mesh locations (field3, vy, field4)
    std::vector<std::vector<realtype>> xs_A_, xs_B_;
    std::vector<std::vector<realtype>> ys_A_, ys_B_;
    std::vector<std::vector<realtype>> zs_A_, zs_B_;
};

#endif
