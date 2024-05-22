// Auto-generated 3D finite difference stencils
// Author: Hong Zhao

#ifndef _FD_STENCIL3D_H
#define _FD_STENCIL3D_H

//Staggered 1st derivative, output at i+0.5
#define DFDX_order4(a, ix, iy, iz)                                                                                     \
    invd1*(DF_order4_1 * (a[iz][iy][ix + 1] - a[iz][iy][ix]) + DF_order4_2 * (a[iz][iy][ix + 2] - a[iz][iy][ix - 1]))

#define DFDY_order4(a, ix, iy, iz)                                                                                     \
    invd2*(DF_order4_1 * (a[iz][iy + 1][ix] - a[iz][iy][ix]) + DF_order4_2 * (a[iz][iy + 2][ix] - a[iz][iy - 1][ix]))

#define DFDZ_order4(a, ix, iy, iz)                                                                                     \
    invd3*(DF_order4_1 * (a[iz + 1][iy][ix] - a[iz][iy][ix]) + DF_order4_2 * (a[iz + 2][iy][ix] - a[iz - 1][iy][ix]))

#define DFDX_order6(a, ix, iy, iz)                                                                                     \
    invd1*(DF_order6_1 * (a[iz][iy][ix + 1] - a[iz][iy][ix]) + DF_order6_2 * (a[iz][iy][ix + 2] - a[iz][iy][ix - 1]) + \
           DF_order6_3 * (a[iz][iy][ix + 3] - a[iz][iy][ix - 2]))

#define DFDY_order6(a, ix, iy, iz)                                                                                     \
    invd2*(DF_order6_1 * (a[iz][iy + 1][ix] - a[iz][iy][ix]) + DF_order6_2 * (a[iz][iy + 2][ix] - a[iz][iy - 1][ix]) + \
           DF_order6_3 * (a[iz][iy + 3][ix] - a[iz][iy - 2][ix]))

#define DFDZ_order6(a, ix, iy, iz)                                                                                     \
    invd3*(DF_order6_1 * (a[iz + 1][iy][ix] - a[iz][iy][ix]) + DF_order6_2 * (a[iz + 2][iy][ix] - a[iz - 1][iy][ix]) + \
           DF_order6_3 * (a[iz + 3][iy][ix] - a[iz - 2][iy][ix]))

#define DFDX_order8(a, ix, iy, iz)                                                                                     \
    invd1*(DF_order8_1 * (a[iz][iy][ix + 1] - a[iz][iy][ix]) + DF_order8_2 * (a[iz][iy][ix + 2] - a[iz][iy][ix - 1]) + \
           DF_order8_3 * (a[iz][iy][ix + 3] - a[iz][iy][ix - 2]) +                                                     \
           DF_order8_4 * (a[iz][iy][ix + 4] - a[iz][iy][ix - 3]))

#define DFDY_order8(a, ix, iy, iz)                                                                                     \
    invd2*(DF_order8_1 * (a[iz][iy + 1][ix] - a[iz][iy][ix]) + DF_order8_2 * (a[iz][iy + 2][ix] - a[iz][iy - 1][ix]) + \
           DF_order8_3 * (a[iz][iy + 3][ix] - a[iz][iy - 2][ix]) +                                                     \
           DF_order8_4 * (a[iz][iy + 4][ix] - a[iz][iy - 3][ix]))

#define DFDZ_order8(a, ix, iy, iz)                                                                                     \
    invd3*(DF_order8_1 * (a[iz + 1][iy][ix] - a[iz][iy][ix]) + DF_order8_2 * (a[iz + 2][iy][ix] - a[iz - 1][iy][ix]) + \
           DF_order8_3 * (a[iz + 3][iy][ix] - a[iz - 2][iy][ix]) +                                                     \
           DF_order8_4 * (a[iz + 4][iy][ix] - a[iz - 3][iy][ix]))

#define DFDX_order12(a, ix, iy, iz)                                                                                    \
    invd1*(DF_order12_1 * (a[iz][iy][ix + 1] - a[iz][iy][ix]) +                                                        \
           DF_order12_2 * (a[iz][iy][ix + 2] - a[iz][iy][ix - 1]) +                                                    \
           DF_order12_3 * (a[iz][iy][ix + 3] - a[iz][iy][ix - 2]) +                                                    \
           DF_order12_4 * (a[iz][iy][ix + 4] - a[iz][iy][ix - 3]) +                                                    \
           DF_order12_5 * (a[iz][iy][ix + 5] - a[iz][iy][ix - 4]) +                                                    \
           DF_order12_6 * (a[iz][iy][ix + 6] - a[iz][iy][ix - 5]))

#define DFDY_order12(a, ix, iy, iz)                                                                                    \
    invd2*(DF_order12_1 * (a[iz][iy + 1][ix] - a[iz][iy][ix]) +                                                        \
           DF_order12_2 * (a[iz][iy + 2][ix] - a[iz][iy - 1][ix]) +                                                    \
           DF_order12_3 * (a[iz][iy + 3][ix] - a[iz][iy - 2][ix]) +                                                    \
           DF_order12_4 * (a[iz][iy + 4][ix] - a[iz][iy - 3][ix]) +                                                    \
           DF_order12_5 * (a[iz][iy + 5][ix] - a[iz][iy - 4][ix]) +                                                    \
           DF_order12_6 * (a[iz][iy + 6][ix] - a[iz][iy - 5][ix]))

#define DFDZ_order12(a, ix, iy, iz)                                                                                    \
    invd3*(DF_order12_1 * (a[iz + 1][iy][ix] - a[iz][iy][ix]) +                                                        \
           DF_order12_2 * (a[iz + 2][iy][ix] - a[iz - 1][iy][ix]) +                                                    \
           DF_order12_3 * (a[iz + 3][iy][ix] - a[iz - 2][iy][ix]) +                                                    \
           DF_order12_4 * (a[iz + 4][iy][ix] - a[iz - 3][iy][ix]) +                                                    \
           DF_order12_5 * (a[iz + 5][iy][ix] - a[iz - 4][iy][ix]) +                                                    \
           DF_order12_6 * (a[iz + 6][iy][ix] - a[iz - 5][iy][ix]))

#define DFDX_order16(a, ix, iy, iz)                                                                                    \
    invd1*(DF_order16_1 * (a[iz][iy][ix + 1] - a[iz][iy][ix]) +                                                        \
           DF_order16_2 * (a[iz][iy][ix + 2] - a[iz][iy][ix - 1]) +                                                    \
           DF_order16_3 * (a[iz][iy][ix + 3] - a[iz][iy][ix - 2]) +                                                    \
           DF_order16_4 * (a[iz][iy][ix + 4] - a[iz][iy][ix - 3]) +                                                    \
           DF_order16_5 * (a[iz][iy][ix + 5] - a[iz][iy][ix - 4]) +                                                    \
           DF_order16_6 * (a[iz][iy][ix + 6] - a[iz][iy][ix - 5]) +                                                    \
           DF_order16_7 * (a[iz][iy][ix + 7] - a[iz][iy][ix - 6]) +                                                    \
           DF_order16_8 * (a[iz][iy][ix + 8] - a[iz][iy][ix - 7]))

#define DFDY_order16(a, ix, iy, iz)                                                                                    \
    invd2*(DF_order16_1 * (a[iz][iy + 1][ix] - a[iz][iy][ix]) +                                                        \
           DF_order16_2 * (a[iz][iy + 2][ix] - a[iz][iy - 1][ix]) +                                                    \
           DF_order16_3 * (a[iz][iy + 3][ix] - a[iz][iy - 2][ix]) +                                                    \
           DF_order16_4 * (a[iz][iy + 4][ix] - a[iz][iy - 3][ix]) +                                                    \
           DF_order16_5 * (a[iz][iy + 5][ix] - a[iz][iy - 4][ix]) +                                                    \
           DF_order16_6 * (a[iz][iy + 6][ix] - a[iz][iy - 5][ix]) +                                                    \
           DF_order16_7 * (a[iz][iy + 7][ix] - a[iz][iy - 6][ix]) +                                                    \
           DF_order16_8 * (a[iz][iy + 8][ix] - a[iz][iy - 7][ix]))

#define DFDZ_order16(a, ix, iy, iz)                                                                                    \
    invd3*(DF_order16_1 * (a[iz + 1][iy][ix] - a[iz][iy][ix]) +                                                        \
           DF_order16_2 * (a[iz + 2][iy][ix] - a[iz - 1][iy][ix]) +                                                    \
           DF_order16_3 * (a[iz + 3][iy][ix] - a[iz - 2][iy][ix]) +                                                    \
           DF_order16_4 * (a[iz + 4][iy][ix] - a[iz - 3][iy][ix]) +                                                    \
           DF_order16_5 * (a[iz + 5][iy][ix] - a[iz - 4][iy][ix]) +                                                    \
           DF_order16_6 * (a[iz + 6][iy][ix] - a[iz - 5][iy][ix]) +                                                    \
           DF_order16_7 * (a[iz + 7][iy][ix] - a[iz - 6][iy][ix]) +                                                    \
           DF_order16_8 * (a[iz + 8][iy][ix] - a[iz - 7][iy][ix]))

//central 1st order derivative
#define DFDXC_order8(a, ix, iy, iz)                                                                                    \
    invd1*(DFC_order8_1 * (a[iz][iy][ix + 1] - a[iz][iy][ix - 1]) +                                                    \
           DFC_order8_2 * (a[iz][iy][ix + 2] - a[iz][iy][ix - 2]) +                                                    \
           DFC_order8_3 * (a[iz][iy][ix + 3] - a[iz][iy][ix - 3]) +                                                    \
           DFC_order8_4 * (a[iz][iy][ix + 4] - a[iz][iy][ix - 4]))

#define DFDYC_order8(a, ix, iy, iz)                                                                                    \
    invd2*(DFC_order8_1 * (a[iz][iy + 1][ix] - a[iz][iy - 1][ix]) +                                                    \
           DFC_order8_2 * (a[iz][iy + 2][ix] - a[iz][iy - 2][ix]) +                                                    \
           DFC_order8_3 * (a[iz][iy + 3][ix] - a[iz][iy - 3][ix]) +                                                    \
           DFC_order8_4 * (a[iz][iy + 4][ix] - a[iz][iy - 4][ix]))

#define DFDZC_order8(a, ix, iy, iz)                                                                                    \
    invd3*(DFC_order8_1 * (a[iz + 1][iy][ix] - a[iz - 1][iy][ix]) +                                                    \
           DFC_order8_2 * (a[iz + 2][iy][ix] - a[iz - 2][iy][ix]) +                                                    \
           DFC_order8_3 * (a[iz + 3][iy][ix] - a[iz - 3][iy][ix]) +                                                    \
           DFC_order8_4 * (a[iz + 4][iy][ix] - a[iz - 4][iy][ix]))

#define DFDXC_order16(a, ix, iy, iz)                                                                                   \
    invd1*(DFC_order16_1 * (a[iz][iy][ix + 1] - a[iz][iy][ix - 1]) +                                                   \
           DFC_order16_2 * (a[iz][iy][ix + 2] - a[iz][iy][ix - 2]) +                                                   \
           DFC_order16_3 * (a[iz][iy][ix + 3] - a[iz][iy][ix - 3]) +                                                   \
           DFC_order16_4 * (a[iz][iy][ix + 4] - a[iz][iy][ix - 4]) +                                                   \
           DFC_order16_5 * (a[iz][iy][ix + 5] - a[iz][iy][ix - 5]) +                                                   \
           DFC_order16_6 * (a[iz][iy][ix + 6] - a[iz][iy][ix - 6]) +                                                   \
           DFC_order16_7 * (a[iz][iy][ix + 7] - a[iz][iy][ix - 7]) +                                                   \
           DFC_order16_8 * (a[iz][iy][ix + 8] - a[iz][iy][ix - 8]))

#define DFDYC_order16(a, ix, iy, iz)                                                                                   \
    invd2*(DFC_order16_1 * (a[iz][iy + 1][ix] - a[iz][iy - 1][ix]) +                                                   \
           DFC_order16_2 * (a[iz][iy + 2][ix] - a[iz][iy - 2][ix]) +                                                   \
           DFC_order16_3 * (a[iz][iy + 3][ix] - a[iz][iy - 3][ix]) +                                                   \
           DFC_order16_4 * (a[iz][iy + 4][ix] - a[iz][iy - 4][ix]) +                                                   \
           DFC_order16_5 * (a[iz][iy + 5][ix] - a[iz][iy - 5][ix]) +                                                   \
           DFC_order16_6 * (a[iz][iy + 6][ix] - a[iz][iy - 6][ix]) +                                                   \
           DFC_order16_7 * (a[iz][iy + 7][ix] - a[iz][iy - 7][ix]) +                                                   \
           DFC_order16_8 * (a[iz][iy + 8][ix] - a[iz][iy - 8][ix]))

#define DFDZC_order16(a, ix, iy, iz)                                                                                   \
    invd3*(DFC_order16_1 * (a[iz + 1][iy][ix] - a[iz - 1][iy][ix]) +                                                   \
           DFC_order16_2 * (a[iz + 2][iy][ix] - a[iz - 2][iy][ix]) +                                                   \
           DFC_order16_3 * (a[iz + 3][iy][ix] - a[iz - 3][iy][ix]) +                                                   \
           DFC_order16_4 * (a[iz + 4][iy][ix] - a[iz - 4][iy][ix]) +                                                   \
           DFC_order16_5 * (a[iz + 5][iy][ix] - a[iz - 5][iy][ix]) +                                                   \
           DFC_order16_6 * (a[iz + 6][iy][ix] - a[iz - 6][iy][ix]) +                                                   \
           DFC_order16_7 * (a[iz + 7][iy][ix] - a[iz - 7][iy][ix]) +                                                   \
           DFC_order16_8 * (a[iz + 8][iy][ix] - a[iz - 8][iy][ix]))
//Non-staggered 2nd derivative
#define D2FDX2_order8(a, ix, iy, iz)                                                                                   \
    invd12*(DF2_order8_0 * a[iz][iy][ix] + DF2_order8_1 * (a[iz][iy][ix + 1] + a[iz][iy][ix - 1]) +                    \
            DF2_order8_2 * (a[iz][iy][ix + 2] + a[iz][iy][ix - 2]) +                                                   \
            DF2_order8_3 * (a[iz][iy][ix + 3] + a[iz][iy][ix - 3]) +                                                   \
            DF2_order8_4 * (a[iz][iy][ix + 4] + a[iz][iy][ix - 4]))

#define D2FDY2_order8(a, ix, iy, iz)                                                                                   \
    invd22*(DF2_order8_0 * a[iz][iy][ix] + DF2_order8_1 * (a[iz][iy + 1][ix] + a[iz][iy - 1][ix]) +                    \
            DF2_order8_2 * (a[iz][iy + 2][ix] + a[iz][iy - 2][ix]) +                                                   \
            DF2_order8_3 * (a[iz][iy + 3][ix] + a[iz][iy - 3][ix]) +                                                   \
            DF2_order8_4 * (a[iz][iy + 4][ix] + a[iz][iy - 4][ix]))

#define D2FDZ2_order8(a, ix, iy, iz)                                                                                   \
    invd32*(DF2_order8_0 * a[iz][iy][ix] + DF2_order8_1 * (a[iz + 1][iy][ix] + a[iz - 1][iy][ix]) +                    \
            DF2_order8_2 * (a[iz + 2][iy][ix] + a[iz - 2][iy][ix]) +                                                   \
            DF2_order8_3 * (a[iz + 3][iy][ix] + a[iz - 3][iy][ix]) +                                                   \
            DF2_order8_4 * (a[iz + 4][iy][ix] + a[iz - 4][iy][ix]))

#define D2FDYDZ_order8(a, ix, iy, iz)                                                                                  \
    invDYDZBY4*(DF2_order8_1 *                                                                                         \
                    (a[iz + 1][iy + 1][ix] + a[iz - 1][iy - 1][ix] - a[iz - 1][iy + 1][ix] - a[iz + 1][iy - 1][ix]) +  \
                DF2_order8_2 *                                                                                         \
                    (a[iz + 2][iy + 2][ix] + a[iz - 2][iy - 2][ix] - a[iz - 2][iy + 2][ix] - a[iz + 2][iy - 2][ix]) +  \
                DF2_order8_3 *                                                                                         \
                    (a[iz + 3][iy + 3][ix] + a[iz - 3][iy - 3][ix] - a[iz - 3][iy + 3][ix] - a[iz + 3][iy - 3][ix]) +  \
                DF2_order8_4 *                                                                                         \
                    (a[iz + 4][iy + 4][ix] + a[iz - 4][iy - 4][ix] - a[iz - 4][iy + 4][ix] - a[iz + 4][iy - 4][ix]))

//The inner paranthesis for X derivatives is necessary to get binary match between data-block and non-data-block versions.

#define D2FDZDX_order8(a, ix, iy, iz)                                                                                  \
    invDZDXBY4*(                                                                                                       \
        DF2_order8_1 *                                                                                                 \
            ((a[iz + 1][iy][ix + 1] - a[iz - 1][iy][ix + 1]) + (a[iz - 1][iy][ix - 1] - a[iz + 1][iy][ix - 1])) +      \
        DF2_order8_2 *                                                                                                 \
            ((a[iz + 2][iy][ix + 2] - a[iz - 2][iy][ix + 2]) + (a[iz - 2][iy][ix - 2] - a[iz + 2][iy][ix - 2])) +      \
        DF2_order8_3 *                                                                                                 \
            ((a[iz + 3][iy][ix + 3] - a[iz - 3][iy][ix + 3]) + (a[iz - 3][iy][ix - 3] - a[iz + 3][iy][ix - 3])) +      \
        DF2_order8_4 *                                                                                                 \
            ((a[iz + 4][iy][ix + 4] - a[iz - 4][iy][ix + 4]) + (a[iz - 4][iy][ix - 4] - a[iz + 4][iy][ix - 4])))

#define D2FDXDY_order8(a, ix, iy, iz)                                                                                  \
    invDXDYBY4*(                                                                                                       \
        DF2_order8_1 *                                                                                                 \
            ((a[iz][iy + 1][ix + 1] - a[iz][iy - 1][ix + 1]) + (a[iz][iy - 1][ix - 1] - a[iz][iy + 1][ix - 1])) +      \
        DF2_order8_2 *                                                                                                 \
            ((a[iz][iy + 2][ix + 2] - a[iz][iy - 2][ix + 2]) + (a[iz][iy - 2][ix - 2] - a[iz][iy + 2][ix - 2])) +      \
        DF2_order8_3 *                                                                                                 \
            ((a[iz][iy + 3][ix + 3] - a[iz][iy - 3][ix + 3]) + (a[iz][iy - 3][ix - 3] - a[iz][iy + 3][ix - 3])) +      \
        DF2_order8_4 *                                                                                                 \
            ((a[iz][iy + 4][ix + 4] - a[iz][iy - 4][ix + 4]) + (a[iz][iy - 4][ix - 4] - a[iz][iy + 4][ix - 4])))

#define D2FDX2_order12(a, ix, iy, iz)                                                                                  \
    invd12*(DF2_order12_0 * a[iz][iy][ix] + DF2_order12_1 * (a[iz][iy][ix + 1] + a[iz][iy][ix - 1]) +                  \
            DF2_order12_2 * (a[iz][iy][ix + 2] + a[iz][iy][ix - 2]) +                                                  \
            DF2_order12_3 * (a[iz][iy][ix + 3] + a[iz][iy][ix - 3]) +                                                  \
            DF2_order12_4 * (a[iz][iy][ix + 4] + a[iz][iy][ix - 4]) +                                                  \
            DF2_order12_5 * (a[iz][iy][ix + 5] + a[iz][iy][ix - 5]) +                                                  \
            DF2_order12_6 * (a[iz][iy][ix + 6] + a[iz][iy][ix - 6]))

#define D2FDY2_order12(a, ix, iy, iz)                                                                                  \
    invd22*(DF2_order12_0 * a[iz][iy][ix] + DF2_order12_1 * (a[iz][iy + 1][ix] + a[iz][iy - 1][ix]) +                  \
            DF2_order12_2 * (a[iz][iy + 2][ix] + a[iz][iy - 2][ix]) +                                                  \
            DF2_order12_3 * (a[iz][iy + 3][ix] + a[iz][iy - 3][ix]) +                                                  \
            DF2_order12_4 * (a[iz][iy + 4][ix] + a[iz][iy - 4][ix]) +                                                  \
            DF2_order12_5 * (a[iz][iy + 5][ix] + a[iz][iy - 5][ix]) +                                                  \
            DF2_order12_6 * (a[iz][iy + 6][ix] + a[iz][iy - 6][ix]))

#define D2FDZ2_order12(a, ix, iy, iz)                                                                                  \
    invd32*(DF2_order12_0 * a[iz][iy][ix] + DF2_order12_1 * (a[iz + 1][iy][ix] + a[iz - 1][iy][ix]) +                  \
            DF2_order12_2 * (a[iz + 2][iy][ix] + a[iz - 2][iy][ix]) +                                                  \
            DF2_order12_3 * (a[iz + 3][iy][ix] + a[iz - 3][iy][ix]) +                                                  \
            DF2_order12_4 * (a[iz + 4][iy][ix] + a[iz - 4][iy][ix]) +                                                  \
            DF2_order12_5 * (a[iz + 5][iy][ix] + a[iz - 5][iy][ix]) +                                                  \
            DF2_order12_6 * (a[iz + 6][iy][ix] + a[iz - 6][iy][ix]))

#define D2FDXDY_order12(a, ix, iy, iz)                                                                                 \
    invDXDYBY4*(DF2_order12_1 *                                                                                        \
                    (a[iz][iy + 1][ix + 1] + a[iz][iy - 1][ix - 1] - a[iz][iy + 1][ix - 1] - a[iz][iy - 1][ix + 1]) +  \
                DF2_order12_2 *                                                                                        \
                    (a[iz][iy + 2][ix + 2] + a[iz][iy - 2][ix - 2] - a[iz][iy + 2][ix - 2] - a[iz][iy - 2][ix + 2]) +  \
                DF2_order12_3 *                                                                                        \
                    (a[iz][iy + 3][ix + 3] + a[iz][iy - 3][ix - 3] - a[iz][iy + 3][ix - 3] - a[iz][iy - 3][ix + 3]) +  \
                DF2_order12_4 *                                                                                        \
                    (a[iz][iy + 4][ix + 4] + a[iz][iy - 4][ix - 4] - a[iz][iy + 4][ix - 4] - a[iz][iy - 4][ix + 4]) +  \
                DF2_order12_5 *                                                                                        \
                    (a[iz][iy + 5][ix + 5] + a[iz][iy - 5][ix - 5] - a[iz][iy + 5][ix - 5] - a[iz][iy - 5][ix + 5]) +  \
                DF2_order12_6 *                                                                                        \
                    (a[iz][iy + 6][ix + 6] + a[iz][iy - 6][ix - 6] - a[iz][iy + 6][ix - 6] - a[iz][iy - 6][ix + 6]))

#define D2FDYDZ_order12(a, ix, iy, iz)                                                                                 \
    invDYDZBY4*(DF2_order12_1 *                                                                                        \
                    (a[iz + 1][iy + 1][ix] + a[iz - 1][iy - 1][ix] - a[iz + 1][iy - 1][ix] - a[iz - 1][iy + 1][ix]) +  \
                DF2_order12_2 *                                                                                        \
                    (a[iz + 2][iy + 2][ix] + a[iz - 2][iy - 2][ix] - a[iz + 2][iy - 2][ix] - a[iz - 2][iy + 2][ix]) +  \
                DF2_order12_3 *                                                                                        \
                    (a[iz + 3][iy + 3][ix] + a[iz - 3][iy - 3][ix] - a[iz + 3][iy - 3][ix] - a[iz - 3][iy + 3][ix]) +  \
                DF2_order12_4 *                                                                                        \
                    (a[iz + 4][iy + 4][ix] + a[iz - 4][iy - 4][ix] - a[iz + 4][iy - 4][ix] - a[iz - 4][iy + 4][ix]) +  \
                DF2_order12_5 *                                                                                        \
                    (a[iz + 5][iy + 5][ix] + a[iz - 5][iy - 5][ix] - a[iz + 5][iy - 5][ix] - a[iz - 5][iy + 5][ix]) +  \
                DF2_order12_6 *                                                                                        \
                    (a[iz + 6][iy + 6][ix] + a[iz - 6][iy - 6][ix] - a[iz + 6][iy - 6][ix] - a[iz - 6][iy + 6][ix]))

#define D2FDZDX_order12(a, ix, iy, iz)                                                                                 \
    invDZDXBY4*(DF2_order12_1 *                                                                                        \
                    (a[iz + 1][iy][ix + 1] + a[iz - 1][iy][ix - 1] - a[iz + 1][iy][ix - 1] - a[iz - 1][iy][ix + 1]) +  \
                DF2_order12_2 *                                                                                        \
                    (a[iz + 2][iy][ix + 2] + a[iz - 2][iy][ix - 2] - a[iz + 2][iy][ix - 2] - a[iz - 2][iy][ix + 2]) +  \
                DF2_order12_3 *                                                                                        \
                    (a[iz + 3][iy][ix + 3] + a[iz - 3][iy][ix - 3] - a[iz + 3][iy][ix - 3] - a[iz - 3][iy][ix + 3]) +  \
                DF2_order12_4 *                                                                                        \
                    (a[iz + 4][iy][ix + 4] + a[iz - 4][iy][ix - 4] - a[iz + 4][iy][ix - 4] - a[iz - 4][iy][ix + 4]) +  \
                DF2_order12_5 *                                                                                        \
                    (a[iz + 5][iy][ix + 5] + a[iz - 5][iy][ix - 5] - a[iz + 5][iy][ix - 5] - a[iz - 5][iy][ix + 5]) +  \
                DF2_order12_6 *                                                                                        \
                    (a[iz + 6][iy][ix + 6] + a[iz - 6][iy][ix - 6] - a[iz + 6][iy][ix - 6] - a[iz - 6][iy][ix + 6]))

#define D2FDX2_order16(a, ix, iy, iz)                                                                                  \
    invd12*(DF2_order16_0 * a[iz][iy][ix] + DF2_order16_1 * (a[iz][iy][ix + 1] + a[iz][iy][ix - 1]) +                  \
            DF2_order16_2 * (a[iz][iy][ix + 2] + a[iz][iy][ix - 2]) +                                                  \
            DF2_order16_3 * (a[iz][iy][ix + 3] + a[iz][iy][ix - 3]) +                                                  \
            DF2_order16_4 * (a[iz][iy][ix + 4] + a[iz][iy][ix - 4]) +                                                  \
            DF2_order16_5 * (a[iz][iy][ix + 5] + a[iz][iy][ix - 5]) +                                                  \
            DF2_order16_6 * (a[iz][iy][ix + 6] + a[iz][iy][ix - 6]) +                                                  \
            DF2_order16_7 * (a[iz][iy][ix + 7] + a[iz][iy][ix - 7]) +                                                  \
            DF2_order16_8 * (a[iz][iy][ix + 8] + a[iz][iy][ix - 8]))

#define D2FDY2_order16(a, ix, iy, iz)                                                                                  \
    invd22*(DF2_order16_0 * a[iz][iy][ix] + DF2_order16_1 * (a[iz][iy + 1][ix] + a[iz][iy - 1][ix]) +                  \
            DF2_order16_2 * (a[iz][iy + 2][ix] + a[iz][iy - 2][ix]) +                                                  \
            DF2_order16_3 * (a[iz][iy + 3][ix] + a[iz][iy - 3][ix]) +                                                  \
            DF2_order16_4 * (a[iz][iy + 4][ix] + a[iz][iy - 4][ix]) +                                                  \
            DF2_order16_5 * (a[iz][iy + 5][ix] + a[iz][iy - 5][ix]) +                                                  \
            DF2_order16_6 * (a[iz][iy + 6][ix] + a[iz][iy - 6][ix]) +                                                  \
            DF2_order16_7 * (a[iz][iy + 7][ix] + a[iz][iy - 7][ix]) +                                                  \
            DF2_order16_8 * (a[iz][iy + 8][ix] + a[iz][iy - 8][ix]))

#define D2FDZ2_order16(a, ix, iy, iz)                                                                                  \
    invd32*(DF2_order16_0 * a[iz][iy][ix] + DF2_order16_1 * (a[iz + 1][iy][ix] + a[iz - 1][iy][ix]) +                  \
            DF2_order16_2 * (a[iz + 2][iy][ix] + a[iz - 2][iy][ix]) +                                                  \
            DF2_order16_3 * (a[iz + 3][iy][ix] + a[iz - 3][iy][ix]) +                                                  \
            DF2_order16_4 * (a[iz + 4][iy][ix] + a[iz - 4][iy][ix]) +                                                  \
            DF2_order16_5 * (a[iz + 5][iy][ix] + a[iz - 5][iy][ix]) +                                                  \
            DF2_order16_6 * (a[iz + 6][iy][ix] + a[iz - 6][iy][ix]) +                                                  \
            DF2_order16_7 * (a[iz + 7][iy][ix] + a[iz - 7][iy][ix]) +                                                  \
            DF2_order16_8 * (a[iz + 8][iy][ix] + a[iz - 8][iy][ix]))

#define D2FDXDY_order16(a, ix, iy, iz)                                                                                 \
    invDXDYBY4*(DF2_order16_1 *                                                                                        \
                    (a[iz][iy + 1][ix + 1] + a[iz][iy - 1][ix - 1] - a[iz][iy + 1][ix - 1] - a[iz][iy - 1][ix + 1]) +  \
                DF2_order16_2 *                                                                                        \
                    (a[iz][iy + 2][ix + 2] + a[iz][iy - 2][ix - 2] - a[iz][iy + 2][ix - 2] - a[iz][iy - 2][ix + 2]) +  \
                DF2_order16_3 *                                                                                        \
                    (a[iz][iy + 3][ix + 3] + a[iz][iy - 3][ix - 3] - a[iz][iy + 3][ix - 3] - a[iz][iy - 3][ix + 3]) +  \
                DF2_order16_4 *                                                                                        \
                    (a[iz][iy + 4][ix + 4] + a[iz][iy - 4][ix - 4] - a[iz][iy + 4][ix - 4] - a[iz][iy - 4][ix + 4]) +  \
                DF2_order16_5 *                                                                                        \
                    (a[iz][iy + 5][ix + 5] + a[iz][iy - 5][ix - 5] - a[iz][iy + 5][ix - 5] - a[iz][iy - 5][ix + 5]) +  \
                DF2_order16_6 *                                                                                        \
                    (a[iz][iy + 6][ix + 6] + a[iz][iy - 6][ix - 6] - a[iz][iy + 6][ix - 6] - a[iz][iy - 6][ix + 6]) +  \
                DF2_order16_7 *                                                                                        \
                    (a[iz][iy + 7][ix + 7] + a[iz][iy - 7][ix - 7] - a[iz][iy + 7][ix - 7] - a[iz][iy - 7][ix + 7]) +  \
                DF2_order16_8 *                                                                                        \
                    (a[iz][iy + 8][ix + 8] + a[iz][iy - 8][ix - 8] - a[iz][iy + 8][ix - 8] - a[iz][iy - 8][ix + 8]))

#define D2FDYDZ_order16(a, ix, iy, iz)                                                                                 \
    invDYDZBY4*(DF2_order16_1 *                                                                                        \
                    (a[iz + 1][iy + 1][ix] + a[iz - 1][iy - 1][ix] - a[iz + 1][iy - 1][ix] - a[iz - 1][iy + 1][ix]) +  \
                DF2_order16_2 *                                                                                        \
                    (a[iz + 2][iy + 2][ix] + a[iz - 2][iy - 2][ix] - a[iz + 2][iy - 2][ix] - a[iz - 2][iy + 2][ix]) +  \
                DF2_order16_3 *                                                                                        \
                    (a[iz + 3][iy + 3][ix] + a[iz - 3][iy - 3][ix] - a[iz + 3][iy - 3][ix] - a[iz - 3][iy + 3][ix]) +  \
                DF2_order16_4 *                                                                                        \
                    (a[iz + 4][iy + 4][ix] + a[iz - 4][iy - 4][ix] - a[iz + 4][iy - 4][ix] - a[iz - 4][iy + 4][ix]) +  \
                DF2_order16_5 *                                                                                        \
                    (a[iz + 5][iy + 5][ix] + a[iz - 5][iy - 5][ix] - a[iz + 5][iy - 5][ix] - a[iz - 5][iy + 5][ix]) +  \
                DF2_order16_6 *                                                                                        \
                    (a[iz + 6][iy + 6][ix] + a[iz - 6][iy - 6][ix] - a[iz + 6][iy - 6][ix] - a[iz - 6][iy + 6][ix]) +  \
                DF2_order16_7 *                                                                                        \
                    (a[iz + 7][iy + 7][ix] + a[iz - 7][iy - 7][ix] - a[iz + 7][iy - 7][ix] - a[iz - 7][iy + 7][ix]) +  \
                DF2_order16_8 *                                                                                        \
                    (a[iz + 8][iy + 8][ix] + a[iz - 8][iy - 8][ix] - a[iz + 8][iy - 8][ix] - a[iz - 8][iy + 8][ix]))

#define D2FDZDX_order16(a, ix, iy, iz)                                                                                 \
    invDZDXBY4*(DF2_order16_1 *                                                                                        \
                    (a[iz + 1][iy][ix + 1] + a[iz - 1][iy][ix - 1] - a[iz + 1][iy][ix - 1] - a[iz - 1][iy][ix + 1]) +  \
                DF2_order16_2 *                                                                                        \
                    (a[iz + 2][iy][ix + 2] + a[iz - 2][iy][ix - 2] - a[iz + 2][iy][ix - 2] - a[iz - 2][iy][ix + 2]) +  \
                DF2_order16_3 *                                                                                        \
                    (a[iz + 3][iy][ix + 3] + a[iz - 3][iy][ix - 3] - a[iz + 3][iy][ix - 3] - a[iz - 3][iy][ix + 3]) +  \
                DF2_order16_4 *                                                                                        \
                    (a[iz + 4][iy][ix + 4] + a[iz - 4][iy][ix - 4] - a[iz + 4][iy][ix - 4] - a[iz - 4][iy][ix + 4]) +  \
                DF2_order16_5 *                                                                                        \
                    (a[iz + 5][iy][ix + 5] + a[iz - 5][iy][ix - 5] - a[iz + 5][iy][ix - 5] - a[iz - 5][iy][ix + 5]) +  \
                DF2_order16_6 *                                                                                        \
                    (a[iz + 6][iy][ix + 6] + a[iz - 6][iy][ix - 6] - a[iz + 6][iy][ix - 6] - a[iz - 6][iy][ix + 6]) +  \
                DF2_order16_7 *                                                                                        \
                    (a[iz + 7][iy][ix + 7] + a[iz - 7][iy][ix - 7] - a[iz + 7][iy][ix - 7] - a[iz - 7][iy][ix + 7]) +  \
                DF2_order16_8 *                                                                                        \
                    (a[iz + 8][iy][ix + 8] + a[iz - 8][iy][ix - 8] - a[iz + 8][iy][ix - 8] - a[iz - 8][iy][ix + 8]))

#endif //_FD_STENCIL3D_H
