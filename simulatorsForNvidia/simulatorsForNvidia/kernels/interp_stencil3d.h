// Autogenerated 3D midpoint interpolation stencils
// Author: Hong Zhao

#ifndef INTERP_STENCIL3D_H_
#define INTERP_STENCIL3D_H_

// Interpolation along 1 direction, output at i+0.5
#define INTERP_X_order4(a, ix, iy, iz)                                                                                 \
    (INTERP_order4_0 * (a[iz][iy][ix + 1] + a[iz][iy][ix]) + INTERP_order4_1 * (a[iz][iy][ix + 2] + a[iz][iy][ix - 1]))

#define INTERP_Y_order4(a, ix, iy, iz)                                                                                 \
    (INTERP_order4_0 * (a[iz][iy + 1][ix] + a[iz][iy][ix]) + INTERP_order4_1 * (a[iz][iy + 2][ix] + a[iz][iy - 1][ix]))

#define INTERP_Z_order4(a, ix, iy, iz)                                                                                 \
    (INTERP_order4_0 * (a[iz + 1][iy][ix] + a[iz][iy][ix]) + INTERP_order4_1 * (a[iz + 2][iy][ix] + a[iz - 1][iy][ix]))

#define INTERP_X_order8(a, ix, iy, iz)                                                                                 \
    (INTERP_order8_0 * (a[iz][iy][ix + 1] + a[iz][iy][ix]) +                                                           \
     INTERP_order8_1 * (a[iz][iy][ix + 2] + a[iz][iy][ix - 1]) +                                                       \
     INTERP_order8_2 * (a[iz][iy][ix + 3] + a[iz][iy][ix - 2]) +                                                       \
     INTERP_order8_3 * (a[iz][iy][ix + 4] + a[iz][iy][ix - 3]))

#define INTERP_Y_order8(a, ix, iy, iz)                                                                                 \
    (INTERP_order8_0 * (a[iz][iy + 1][ix] + a[iz][iy][ix]) +                                                           \
     INTERP_order8_1 * (a[iz][iy + 2][ix] + a[iz][iy - 1][ix]) +                                                       \
     INTERP_order8_2 * (a[iz][iy + 3][ix] + a[iz][iy - 2][ix]) +                                                       \
     INTERP_order8_3 * (a[iz][iy + 4][ix] + a[iz][iy - 3][ix]))

#define INTERP_Z_order8(a, ix, iy, iz)                                                                                 \
    (INTERP_order8_0 * (a[iz + 1][iy][ix] + a[iz][iy][ix]) +                                                           \
     INTERP_order8_1 * (a[iz + 2][iy][ix] + a[iz - 1][iy][ix]) +                                                       \
     INTERP_order8_2 * (a[iz + 3][iy][ix] + a[iz - 2][iy][ix]) +                                                       \
     INTERP_order8_3 * (a[iz + 4][iy][ix] + a[iz - 3][iy][ix]))

// Interpolation along 2 directions, output at (i+0.5, j+0.5)
#define INTERP_XY_order4(a, ix, iy, iz)                                                                                \
    0.5 * (INTERP_order4_0 * (a[iz][iy + 1][ix + 1] + a[iz][iy][ix] + a[iz][iy][ix + 1] + a[iz][iy + 1][ix]) +         \
           INTERP_order4_1 *                                                                                           \
               (a[iz][iy + 2][ix + 2] + a[iz][iy - 1][ix - 1] + a[iz][iy - 1][ix + 2] + a[iz][iy + 2][ix - 1]))

#define INTERP_XZ_order4(a, ix, iy, iz)                                                                                \
    0.5 * (INTERP_order4_0 * (a[iz + 1][iy][ix + 1] + a[iz][iy][ix] + a[iz][iy][ix + 1] + a[iz + 1][iy][ix]) +         \
           INTERP_order4_1 *                                                                                           \
               (a[iz + 2][iy][ix + 2] + a[iz - 1][iy][ix - 1] + a[iz - 1][iy][ix + 2] + a[iz + 2][iy][ix - 1]))

#define INTERP_YZ_order4(a, ix, iy, iz)                                                                                \
    0.5 * (INTERP_order4_0 * (a[iz + 1][iy + 1][ix] + a[iz][iy][ix] + a[iz][iy + 1][ix] + a[iz + 1][iy][ix]) +         \
           INTERP_order4_1 *                                                                                           \
               (a[iz + 2][iy + 2][ix] + a[iz - 1][iy - 1][ix] + a[iz - 1][iy + 2][ix] + a[iz + 2][iy - 1][ix]))

#define INTERP_XY_order8(a, ix, iy, iz)                                                                                \
    0.5 * (INTERP_order8_0 * (a[iz][iy + 1][ix + 1] + a[iz][iy][ix] + a[iz][iy][ix + 1] + a[iz][iy + 1][ix]) +         \
           INTERP_order8_1 *                                                                                           \
               (a[iz][iy + 2][ix + 2] + a[iz][iy - 1][ix - 1] + a[iz][iy - 1][ix + 2] + a[iz][iy + 2][ix - 1]) +       \
           INTERP_order8_2 *                                                                                           \
               (a[iz][iy + 3][ix + 3] + a[iz][iy - 2][ix - 2] + a[iz][iy - 2][ix + 3] + a[iz][iy + 3][ix - 2]) +       \
           INTERP_order8_3 *                                                                                           \
               (a[iz][iy + 4][ix + 4] + a[iz][iy - 3][ix - 3] + a[iz][iy - 3][ix + 4] + a[iz][iy + 4][ix - 3]))

#define INTERP_XZ_order8(a, ix, iy, iz)                                                                                \
    0.5 * (INTERP_order8_0 * (a[iz + 1][iy][ix + 1] + a[iz][iy][ix] + a[iz][iy][ix + 1] + a[iz + 1][iy][ix]) +         \
           INTERP_order8_1 *                                                                                           \
               (a[iz + 2][iy][ix + 2] + a[iz - 1][iy][ix - 1] + a[iz - 1][iy][ix + 2] + a[iz + 2][iy][ix - 1]) +       \
           INTERP_order8_2 *                                                                                           \
               (a[iz + 3][iy][ix + 3] + a[iz - 2][iy][ix - 2] + a[iz - 2][iy][ix + 3] + a[iz + 3][iy][ix - 2]) +       \
           INTERP_order8_3 *                                                                                           \
               (a[iz + 4][iy][ix + 4] + a[iz - 3][iy][ix - 3] + a[iz - 3][iy][ix + 4] + a[iz + 4][iy][ix - 3]))

#define INTERP_YZ_order8(a, ix, iy, iz)                                                                                \
    0.5 * (INTERP_order8_0 * (a[iz + 1][iy + 1][ix] + a[iz][iy][ix] + a[iz][iy + 1][ix] + a[iz + 1][iy][ix]) +         \
           INTERP_order8_1 *                                                                                           \
               (a[iz + 2][iy + 2][ix] + a[iz - 1][iy - 1][ix] + a[iz - 1][iy + 2][ix] + a[iz + 2][iy - 1][ix]) +       \
           INTERP_order8_2 *                                                                                           \
               (a[iz + 3][iy + 3][ix] + a[iz - 2][iy - 2][ix] + a[iz - 2][iy + 3][ix] + a[iz + 3][iy - 2][ix]) +       \
           INTERP_order8_3 *                                                                                           \
               (a[iz + 4][iy + 4][ix] + a[iz - 3][iy - 3][ix] + a[iz - 3][iy + 4][ix] + a[iz + 4][iy - 3][ix]))

#endif //INTERP_STENCIL3D_H_
