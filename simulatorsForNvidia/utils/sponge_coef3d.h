#ifndef SPONGE_COEF3D_H
#define SPONGE_COEF3D_H

#include <vector>

class sponge_coef3d
{
public:
    sponge_coef3d() = default;

    /* Disable copying to prevent copying coefficient arrays */
    sponge_coef3d(const sponge_coef3d&) = delete;
    sponge_coef3d& operator=(const sponge_coef3d&) = delete;

    // Sponge layer coefficients, on non-staggered mesh
    // Note: we still use these coefficents even on staggered mesh
    std::vector<std::vector<realtype>> x_A_, x_B_;
    std::vector<std::vector<realtype>> y_A_, y_B_;
    std::vector<std::vector<realtype>> z_A_, z_B_;
};
#endif
