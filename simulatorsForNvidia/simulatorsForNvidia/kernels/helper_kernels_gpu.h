#ifndef HELPER_KERNELS_GPU_H
#define HELPER_KERNELS_GPU_H

#include "volume_index.h"
#include "fd_coef.h"
namespace helper_kernels_gpu {
// These host / device functions will return the FD coefficients.
// These are meant to be inlined, in fully unrolled loops, or performance will suffer
// Calling the non-specialized method means this order is not implemented

// ****************************************************************************
// First derivatives
template <int order, FD_COEF_TYPE coef_type>
__host__ __device__ inline float df_coef(int i);

// Taylor 8th order
template <>
__host__ __device__ inline float
df_coef<8, _FD_COEF_TAYLOR>(int i)
{
    float coefs[] = { 0.0f, TAYLOR::DF_order8_1, TAYLOR::DF_order8_2, TAYLOR::DF_order8_3, TAYLOR::DF_order8_4 };
    return coefs[i];
}
// LS 8th and 16th order
template <>
__host__ __device__ inline float
df_coef<8, _FD_COEF_LEAST_SQUARE>(int i)
{
    float coefs[] = { 0.0f, LEAST_SQUARE::DF_order8_1, LEAST_SQUARE::DF_order8_2, LEAST_SQUARE::DF_order8_3,
                      LEAST_SQUARE::DF_order8_4 };
    return coefs[i];
}
template <>
__host__ __device__ inline float
df_coef<16, _FD_COEF_LEAST_SQUARE>(int i)
{
    float coefs[] = { 0.0f,
                      LEAST_SQUARE::DF_order16_1,
                      LEAST_SQUARE::DF_order16_2,
                      LEAST_SQUARE::DF_order16_3,
                      LEAST_SQUARE::DF_order16_4,
                      LEAST_SQUARE::DF_order16_5,
                      LEAST_SQUARE::DF_order16_6,
                      LEAST_SQUARE::DF_order16_7,
                      LEAST_SQUARE::DF_order16_8 };
    return coefs[i];
}
// Taylor PML 8th order
template <>
__host__ __device__ inline float
df_coef<8, _FD_COEF_TAYLOR_PML>(int i)
{
    float coefs[] = { 0.0f, TAYLOR_PML::DF_order8_1, TAYLOR_PML::DF_order8_2, TAYLOR_PML::DF_order8_3,
                      TAYLOR_PML::DF_order8_4 };
    return coefs[i];
}

// ****************************************************************************
// Second derivatives
template <int order, FD_COEF_TYPE coef_type>
__host__ __device__ inline float df2_coef(int i);

// Taylor 8th and 16th order
template <>
__host__ __device__ inline float
df2_coef<8, _FD_COEF_TAYLOR>(int i)
{
    float coefs[] = { TAYLOR::DF2_order8_0, TAYLOR::DF2_order8_1, TAYLOR::DF2_order8_2, TAYLOR::DF2_order8_3,
                      TAYLOR::DF2_order8_4 };
    return coefs[i];
}
template <>
__host__ __device__ inline float
df2_coef<16, _FD_COEF_TAYLOR>(int i)
{
    float coefs[] = { TAYLOR::DF2_order16_0, TAYLOR::DF2_order16_1, TAYLOR::DF2_order16_2,
                      TAYLOR::DF2_order16_3, TAYLOR::DF2_order16_4, TAYLOR::DF2_order16_5,
                      TAYLOR::DF2_order16_6, TAYLOR::DF2_order16_7, TAYLOR::DF2_order16_8 };
    return coefs[i];
}
// LS 8th and 16th order
template <>
__host__ __device__ inline float
df2_coef<8, _FD_COEF_LEAST_SQUARE>(int i)
{
    float coefs[] = { LEAST_SQUARE::DF2_order8_0, LEAST_SQUARE::DF2_order8_1, LEAST_SQUARE::DF2_order8_2,
                      LEAST_SQUARE::DF2_order8_3, LEAST_SQUARE::DF2_order8_4 };
    return coefs[i];
}
template <>
__host__ __device__ inline float
df2_coef<16, _FD_COEF_LEAST_SQUARE>(int i)
{
    float coefs[] = { LEAST_SQUARE::DF2_order16_0, LEAST_SQUARE::DF2_order16_1, LEAST_SQUARE::DF2_order16_2,
                      LEAST_SQUARE::DF2_order16_3, LEAST_SQUARE::DF2_order16_4, LEAST_SQUARE::DF2_order16_5,
                      LEAST_SQUARE::DF2_order16_6, LEAST_SQUARE::DF2_order16_7, LEAST_SQUARE::DF2_order16_8 };
    return coefs[i];
}
// Taylor PML 8th and 16th order
template <>
__host__ __device__ inline float
df2_coef<8, _FD_COEF_TAYLOR_PML>(int i)
{
    float coefs[] = { TAYLOR_PML::DF2_order8_0, TAYLOR_PML::DF2_order8_1, TAYLOR_PML::DF2_order8_2,
                      TAYLOR_PML::DF2_order8_3, TAYLOR_PML::DF2_order8_4 };
    return coefs[i];
}
template <>
__host__ __device__ inline float
df2_coef<16, _FD_COEF_TAYLOR_PML>(int i)
{
    float coefs[] = { TAYLOR_PML::DF2_order16_0, TAYLOR_PML::DF2_order16_1, TAYLOR_PML::DF2_order16_2,
                      TAYLOR_PML::DF2_order16_3, TAYLOR_PML::DF2_order16_4, TAYLOR_PML::DF2_order16_5,
                      TAYLOR_PML::DF2_order16_6, TAYLOR_PML::DF2_order16_7, TAYLOR_PML::DF2_order16_8 };
    return coefs[i];
}

// ****************************************************************************
// Derivatives operating on a 3D volume
template <int order, FD_COEF_TYPE ctype, typename T>
__host__ __device__ inline T
drv11(const T* ptr, volume_index vol, int i1, int i2, int i3, float invd11)
{
    T tmp = df2_coef<order, ctype>(0) * vol(ptr, i1, i2, i3);
    for (int i = 1; i <= order / 2; i++)
        tmp += df2_coef<order, ctype>(i) * (vol(ptr, i1 - i, i2, i3) + vol(ptr, i1 + i, i2, i3));
    return invd11 * tmp;
}

template <int order, FD_COEF_TYPE ctype, typename T>
__host__ __device__ inline T
drv22(const T* ptr, volume_index vol, int i1, int i2, int i3, float invd22)
{
    T tmp = df2_coef<order, ctype>(0) * vol(ptr, i1, i2, i3);
    for (int i = 1; i <= order / 2; i++)
        tmp += df2_coef<order, ctype>(i) * (vol(ptr, i1, i2 - i, i3) + vol(ptr, i1, i2 + i, i3));
    return invd22 * tmp;
}

template <int order, FD_COEF_TYPE ctype, typename T>
__host__ __device__ inline T
drv33(const T* ptr, volume_index vol, int i1, int i2, int i3, float invd33)
{
    T tmp = df2_coef<order, ctype>(0) * vol(ptr, i1, i2, i3);
    for (int i = 1; i <= order / 2; i++)
        tmp += df2_coef<order, ctype>(i) * (vol(ptr, i1, i2, i3 - i) + vol(ptr, i1, i2, i3 + i));
    return invd33 * tmp;
}

template <int order, FD_COEF_TYPE ctype, typename T>
__host__ __device__ inline T
drv12(const T* ptr, volume_index vol, int i1, int i2, int i3, float invd12)
{
    T tmp = df2_coef<order, ctype>(1) * ((vol(ptr, i1 + 1, i2 + 1, i3) - vol(ptr, i1 + 1, i2 - 1, i3)) +
                                         (vol(ptr, i1 - 1, i2 - 1, i3) - vol(ptr, i1 - 1, i2 + 1, i3)));
    for (int i = 2; i <= order / 2; i++)
        tmp += df2_coef<order, ctype>(i) * ((vol(ptr, i1 + i, i2 + i, i3) - vol(ptr, i1 + i, i2 - i, i3)) +
                                            (vol(ptr, i1 - i, i2 - i, i3) - vol(ptr, i1 - i, i2 + i, i3)));
    return invd12 * tmp;
}

template <int order, FD_COEF_TYPE ctype, typename T>
__host__ __device__ inline T
drv23(const T* ptr, volume_index vol, int i1, int i2, int i3, float invd23)
{
    T tmp = df2_coef<order, ctype>(1) * ((vol(ptr, i1, i2 + 1, i3 + 1) - vol(ptr, i1, i2 - 1, i3 + 1)) +
                                         (vol(ptr, i1, i2 - 1, i3 - 1) - vol(ptr, i1, i2 + 1, i3 - 1)));
    for (int i = 2; i <= order / 2; i++)
        tmp += df2_coef<order, ctype>(i) * ((vol(ptr, i1, i2 + i, i3 + i) - vol(ptr, i1, i2 - i, i3 + i)) +
                                            (vol(ptr, i1, i2 - i, i3 - i) - vol(ptr, i1, i2 + i, i3 - i)));
    return invd23 * tmp;
}

template <int order, FD_COEF_TYPE ctype, typename T>
__host__ __device__ inline T
drv31(const T* ptr, volume_index vol, int i1, int i2, int i3, float invd31)
{
    T tmp = df2_coef<order, ctype>(1) * ((vol(ptr, i1 + 1, i2, i3 + 1) - vol(ptr, i1 - 1, i2, i3 + 1)) +
                                         (vol(ptr, i1 - 1, i2, i3 - 1) - vol(ptr, i1 + 1, i2, i3 - 1)));
    for (int i = 2; i <= order / 2; i++)
        tmp += df2_coef<order, ctype>(i) * ((vol(ptr, i1, i2 + i, i3 + i) - vol(ptr, i1, i2 - i, i3 + i)) +
                                            (vol(ptr, i1, i2 - i, i3 - i) - vol(ptr, i1, i2 + i, i3 - i)));
    return invd31 * tmp;
}

// drfield3X_cg_simulator2_adj computes
// field3 = drfield3X (model1 * field1 + model2 * field2 + rx * rx * ((-model1 + model4) * field1 - model2 * field2)
// field4 = drfield3X (model4 * field2 + rx * rx * (model2 * field1 + (-model4 + model3) * field2))
template <int order, FD_COEF_TYPE ctype, typename T>
__host__ __device__ inline void
drfield3X_cg_simulator2_adj(const T* field1, const T* field2, const T* rx, const T* model1, const T* model2, const T* model3, const T* model4,
                 volume_index vol, int i1, int i2, int i3, float invdx, T& field3, T& field4)
{
    T rxx = vol(rx, i1, i2, i3) * vol(rx, i1, i2, i3);
    T tmpx = df2_coef<order, ctype>(0) *
             (vol(model1, i1, i2, i3) * vol(field1, i1, i2, i3) + vol(model2, i1, i2, i3) * vol(field2, i1, i2, i3) +
              rxx * ((vol(model4, i1, i2, i3) - vol(model1, i1, i2, i3)) * vol(field1, i1, i2, i3) -
                     vol(model2, i1, i2, i3) * vol(field2, i1, i2, i3)));
    T tmpz = df2_coef<order, ctype>(0) * (vol(model4, i1, i2, i3) * vol(field2, i1, i2, i3) +
                                          rxx * (vol(model2, i1, i2, i3) * vol(field1, i1, i2, i3) +
                                                 (vol(model3, i1, i2, i3) - vol(model4, i1, i2, i3)) * vol(field2, i1, i2, i3)));
    for (int i = 1; i <= order / 2; i++) {
        T rxxm = vol(rx, i1 - i, i2, i3) * vol(rx, i1 - i, i2, i3);
        T rxxp = vol(rx, i1 + i, i2, i3) * vol(rx, i1 + i, i2, i3);
        tmpx +=
            df2_coef<order, ctype>(i) *
            (vol(model1, i1 - i, i2, i3) * vol(field1, i1 - i, i2, i3) + vol(model2, i1 - i, i2, i3) * vol(field2, i1 - i, i2, i3) +
             rxxm * ((vol(model4, i1 - i, i2, i3) - vol(model1, i1 - i, i2, i3)) * vol(field1, i1 - i, i2, i3) -
                     vol(model2, i1 - i, i2, i3) * vol(field2, i1 - i, i2, i3)) +
             vol(model1, i1 + i, i2, i3) * vol(field1, i1 + i, i2, i3) + vol(model2, i1 + i, i2, i3) * vol(field2, i1 + i, i2, i3) +
             rxxp * ((vol(model4, i1 + i, i2, i3) - vol(model1, i1 + i, i2, i3)) * vol(field1, i1 + i, i2, i3) -
                     vol(model2, i1 + i, i2, i3) * vol(field2, i1 + i, i2, i3)));
        tmpz += df2_coef<order, ctype>(i) *
                (vol(model4, i1 - i, i2, i3) * vol(field2, i1 - i, i2, i3) +
                 rxxm * (vol(model2, i1 - i, i2, i3) * vol(field1, i1 - i, i2, i3) +
                         (vol(model3, i1 - i, i2, i3) - vol(model4, i1 - i, i2, i3)) * vol(field2, i1 - i, i2, i3)) +
                 vol(model4, i1 + i, i2, i3) * vol(field2, i1 + i, i2, i3) +
                 rxxp * (vol(model2, i1 + i, i2, i3) * vol(field1, i1 + i, i2, i3) +
                         (vol(model3, i1 + i, i2, i3) - vol(model4, i1 + i, i2, i3)) * vol(field2, i1 + i, i2, i3)));
    }
    field3 = tmpx * invdx;
    field4 = tmpz * invdx;
}

// drvYY_cg_simulator2_adj computes
// field3 = drvYY (model1 * field1 + model2 * field2 + ry * ry * ((-model1 + model4) * field1 - model2 * field2)
// field4 = drvYY (model4 * field2 + ry * ry * (model2 * field1 + (-model4 + model3) * field2))
template <int order, FD_COEF_TYPE ctype, typename T>
__host__ __device__ inline void
drvYY_cg_simulator2_adj(const T* field1, const T* field2, const T* ry, const T* model1, const T* model2, const T* model3, const T* model4,
                 volume_index vol, int i1, int i2, int i3, float invdy, T& field3, T& field4)
{
    T ryy = vol(ry, i1, i2, i3) * vol(ry, i1, i2, i3);
    T tmpx = df2_coef<order, ctype>(0) *
             (vol(model1, i1, i2, i3) * vol(field1, i1, i2, i3) + vol(model2, i1, i2, i3) * vol(field2, i1, i2, i3) +
              ryy * ((vol(model4, i1, i2, i3) - vol(model1, i1, i2, i3)) * vol(field1, i1, i2, i3) -
                     vol(model2, i1, i2, i3) * vol(field2, i1, i2, i3)));
    T tmpz = df2_coef<order, ctype>(0) * (vol(model4, i1, i2, i3) * vol(field2, i1, i2, i3) +
                                          ryy * (vol(model2, i1, i2, i3) * vol(field1, i1, i2, i3) +
                                                 (vol(model3, i1, i2, i3) - vol(model4, i1, i2, i3)) * vol(field2, i1, i2, i3)));
    for (int i = 1; i <= order / 2; i++) {
        T ryym = vol(ry, i1, i2 - i, i3) * vol(ry, i1, i2 - i, i3);
        T ryyp = vol(ry, i1, i2 + i, i3) * vol(ry, i1, i2 + i, i3);
        tmpx +=
            df2_coef<order, ctype>(i) *
            (vol(model1, i1, i2 - i, i3) * vol(field1, i1, i2 - i, i3) + vol(model2, i1, i2 - i, i3) * vol(field2, i1, i2 - i, i3) +
             ryym * ((vol(model4, i1, i2 - i, i3) - vol(model1, i1, i2 - i, i3)) * vol(field1, i1, i2 - i, i3) -
                     vol(model2, i1, i2 - i, i3) * vol(field2, i1, i2 - i, i3)) +
             vol(model1, i1, i2 + i, i3) * vol(field1, i1, i2 + i, i3) + vol(model2, i1, i2 + i, i3) * vol(field2, i1, i2 + i, i3) +
             ryyp * ((vol(model4, i1, i2 + i, i3) - vol(model1, i1, i2 + i, i3)) * vol(field1, i1, i2 + i, i3) -
                     vol(model2, i1, i2 + i, i3) * vol(field2, i1, i2 + i, i3)));
        tmpz += df2_coef<order, ctype>(i) *
                (vol(model4, i1, i2 - i, i3) * vol(field2, i1, i2 - i, i3) +
                 ryym * (vol(model2, i1, i2 - i, i3) * vol(field1, i1, i2 - i, i3) +
                         (vol(model3, i1, i2 - i, i3) - vol(model4, i1, i2 - i, i3)) * vol(field2, i1, i2 - i, i3)) +
                 vol(model4, i1, i2 + i, i3) * vol(field2, i1, i2 + i, i3) +
                 ryyp * (vol(model2, i1, i2 + i, i3) * vol(field1, i1, i2 + i, i3) +
                         (vol(model3, i1, i2 + i, i3) - vol(model4, i1, i2 + i, i3)) * vol(field2, i1, i2 + i, i3)));
    }
    field3 = tmpx * invdy;
    field4 = tmpz * invdy;
}

// drfield4Z_cg_simulator2_adj computes
// field3 = drfield4Z (model1 * field1 + model2 * field2 + rz * rz * ((-model1 + model4) * field1 - model2 * field2)
// field4 = drfield4Z (model4 * field2 + rz * rz * (model2 * field1 + (-model4 + model3) * field2))
template <int order, FD_COEF_TYPE ctype, typename T>
__host__ __device__ inline void
drfield4Z_cg_simulator2_adj(const T* field1, const T* field2, const T* rz, const T* model1, const T* model2, const T* model3, const T* model4,
                 volume_index vol, int i1, int i2, int i3, float invdz, T& field3, T& field4)
{
    T rzz = vol(rz, i1, i2, i3) * vol(rz, i1, i2, i3);
    T tmpx = df2_coef<order, ctype>(0) *
             (vol(model1, i1, i2, i3) * vol(field1, i1, i2, i3) + vol(model2, i1, i2, i3) * vol(field2, i1, i2, i3) +
              rzz * ((vol(model4, i1, i2, i3) - vol(model1, i1, i2, i3)) * vol(field1, i1, i2, i3) -
                     vol(model2, i1, i2, i3) * vol(field2, i1, i2, i3)));
    T tmpz = df2_coef<order, ctype>(0) * (vol(model4, i1, i2, i3) * vol(field2, i1, i2, i3) +
                                          rzz * (vol(model2, i1, i2, i3) * vol(field1, i1, i2, i3) +
                                                 (vol(model3, i1, i2, i3) - vol(model4, i1, i2, i3)) * vol(field2, i1, i2, i3)));
    for (int i = 1; i <= order / 2; i++) {
        T rzzm = vol(rz, i1, i2, i3 - i) * vol(rz, i1, i2, i3 - i);
        T rzzp = vol(rz, i1, i2, i3 + i) * vol(rz, i1, i2, i3 + i);
        tmpx +=
            df2_coef<order, ctype>(i) *
            (vol(model1, i1, i2, i3 - i) * vol(field1, i1, i2, i3 - i) + vol(model2, i1, i2, i3 - i) * vol(field2, i1, i2, i3 - i) +
             rzzm * ((vol(model4, i1, i2, i3 - i) - vol(model1, i1, i2, i3 - i)) * vol(field1, i1, i2, i3 - i) -
                     vol(model2, i1, i2, i3 - i) * vol(field2, i1, i2, i3 - i)) +
             vol(model1, i1, i2, i3 + i) * vol(field1, i1, i2, i3 + i) + vol(model2, i1, i2, i3 + i) * vol(field2, i1, i2, i3 + i) +
             rzzp * ((vol(model4, i1, i2, i3 + i) - vol(model1, i1, i2, i3 + i)) * vol(field1, i1, i2, i3 + i) -
                     vol(model2, i1, i2, i3 + i) * vol(field2, i1, i2, i3 + i)));
        tmpz += df2_coef<order, ctype>(i) *
                (vol(model4, i1, i2, i3 - i) * vol(field2, i1, i2, i3 - i) +
                 rzzm * (vol(model2, i1, i2, i3 - i) * vol(field1, i1, i2, i3 - i) +
                         (vol(model3, i1, i2, i3 - i) - vol(model4, i1, i2, i3 - i)) * vol(field2, i1, i2, i3 - i)) +
                 vol(model4, i1, i2, i3 + i) * vol(field2, i1, i2, i3 + i) +
                 rzzp * (vol(model2, i1, i2, i3 + i) * vol(field1, i1, i2, i3 + i) +
                         (vol(model3, i1, i2, i3 + i) - vol(model4, i1, i2, i3 + i)) * vol(field2, i1, i2, i3 + i)));
    }
    field3 = tmpx * invdz;
    field4 = tmpz * invdz;
}

// drfield3Y_cg_simulator2_adj computes
// field3 = drfield3Y (2 * rx * ry * ((-model1 + model4) * field1 - model2 * field2)
// field4 = drfield3Y (2 * rx * ry * (model2 * field1 + (-model4 + model3) * field2))
template <int order, FD_COEF_TYPE ctype, typename T>
__host__ __device__ inline void
drfield3Y_cg_simulator2_adj(const T* field1, const T* field2, const T* rx, const T* ry, const T* model1, const T* model2, const T* model3,
                 const T* model4, volume_index vol, int i1, int i2, int i3, float invdxy, T& field3, T& field4)
{
    T tmpx = 0.0f;
    T tmpz = 0.0f;
    for (int i = 1; i <= order / 2; i++) {
        T rxymm = 2.0f * vol(rx, i1 - i, i2 - i, i3) * vol(ry, i1 - i, i2 - i, i3);
        T rxymp = 2.0f * vol(rx, i1 - i, i2 + i, i3) * vol(ry, i1 - i, i2 + i, i3);
        T rxypm = 2.0f * vol(rx, i1 + i, i2 - i, i3) * vol(ry, i1 + i, i2 - i, i3);
        T rxypp = 2.0f * vol(rx, i1 + i, i2 + i, i3) * vol(ry, i1 + i, i2 + i, i3);
        tmpx += df2_coef<order, ctype>(i) *
                (rxymm * ((vol(model4, i1 - i, i2 - i, i3) - vol(model1, i1 - i, i2 - i, i3)) * vol(field1, i1 - i, i2 - i, i3) -
                          vol(model2, i1 - i, i2 - i, i3) * vol(field2, i1 - i, i2 - i, i3)) -
                 rxymp * ((vol(model4, i1 - i, i2 + i, i3) - vol(model1, i1 - i, i2 + i, i3)) * vol(field1, i1 - i, i2 + i, i3) -
                          vol(model2, i1 - i, i2 + i, i3) * vol(field2, i1 - i, i2 + i, i3)) -
                 rxypm * ((vol(model4, i1 + i, i2 - i, i3) - vol(model1, i1 + i, i2 - i, i3)) * vol(field1, i1 + i, i2 - i, i3) -
                          vol(model2, i1 + i, i2 - i, i3) * vol(field2, i1 + i, i2 - i, i3)) +
                 rxypp * ((vol(model4, i1 + i, i2 + i, i3) - vol(model1, i1 + i, i2 + i, i3)) * vol(field1, i1 + i, i2 + i, i3) -
                          vol(model2, i1 + i, i2 + i, i3) * vol(field2, i1 + i, i2 + i, i3)));
        tmpz +=
            df2_coef<order, ctype>(i) *
            (rxymm * (vol(model2, i1 - i, i2 - i, i3) * vol(field1, i1 - i, i2 - i, i3) +
                      (vol(model3, i1 - i, i2 - i, i3) - vol(model4, i1 - i, i2 - i, i3)) * vol(field2, i1 - i, i2 - i, i3)) -
             rxymp * (vol(model2, i1 - i, i2 + i, i3) * vol(field1, i1 - i, i2 + i, i3) +
                      (vol(model3, i1 - i, i2 + i, i3) - vol(model4, i1 - i, i2 + i, i3)) * vol(field2, i1 - i, i2 + i, i3)) -
             rxypm * (vol(model2, i1 + i, i2 - i, i3) * vol(field1, i1 + i, i2 - i, i3) +
                      (vol(model3, i1 + i, i2 - i, i3) - vol(model4, i1 + i, i2 - i, i3)) * vol(field2, i1 + i, i2 - i, i3)) +
             rxypp * (vol(model2, i1 + i, i2 + i, i3) * vol(field1, i1 + i, i2 + i, i3) +
                      (vol(model3, i1 + i, i2 + i, i3) - vol(model4, i1 + i, i2 + i, i3)) * vol(field2, i1 + i, i2 + i, i3)));
    }
    field3 = tmpx * invdxy;
    field4 = tmpz * invdxy;
}

// drvYZ_cg_simulator2_adj computes
// field3 = drvYZ (2 * ry * rz * ((-model1 + model4) * field1 - model2 * field2)
// field4 = drvYZ (2 * ry * rz * (model2 * field1 + (-model4 + model3) * field2))
template <int order, FD_COEF_TYPE ctype, typename T>
__host__ __device__ inline void
drvYZ_cg_simulator2_adj(const T* field1, const T* field2, const T* ry, const T* rz, const T* model1, const T* model2, const T* model3,
                 const T* model4, volume_index vol, int i1, int i2, int i3, float invdyz, T& field3, T& field4)
{
    T tmpx = 0.0f;
    T tmpz = 0.0f;
    for (int i = 1; i <= order / 2; i++) {
        T ryzmm = 2.0f * vol(ry, i1, i2 - i, i3 - i) * vol(rz, i1, i2 - i, i3 - i);
        T ryzmp = 2.0f * vol(ry, i1, i2 - i, i3 + i) * vol(rz, i1, i2 - i, i3 + i);
        T ryzpm = 2.0f * vol(ry, i1, i2 + i, i3 - i) * vol(rz, i1, i2 + i, i3 - i);
        T ryzpp = 2.0f * vol(ry, i1, i2 + i, i3 + i) * vol(rz, i1, i2 + i, i3 + i);
        tmpx += df2_coef<order, ctype>(i) *
                (ryzmm * ((vol(model4, i1, i2 - i, i3 - i) - vol(model1, i1, i2 - i, i3 - i)) * vol(field1, i1, i2 - i, i3 - i) -
                          vol(model2, i1, i2 - i, i3 - i) * vol(field2, i1, i2 - i, i3 - i)) -
                 ryzmp * ((vol(model4, i1, i2 - i, i3 + i) - vol(model1, i1, i2 - i, i3 + i)) * vol(field1, i1, i2 - i, i3 + i) -
                          vol(model2, i1, i2 - i, i3 + i) * vol(field2, i1, i2 - i, i3 + i)) -
                 ryzpm * ((vol(model4, i1, i2 + i, i3 - i) - vol(model1, i1, i2 + i, i3 - i)) * vol(field1, i1, i2 + i, i3 - i) -
                          vol(model2, i1, i2 + i, i3 - i) * vol(field2, i1, i2 + i, i3 - i)) +
                 ryzpp * ((vol(model4, i1, i2 + i, i3 + i) - vol(model1, i1, i2 + i, i3 + i)) * vol(field1, i1, i2 + i, i3 + i) -
                          vol(model2, i1, i2 + i, i3 + i) * vol(field2, i1, i2 + i, i3 + i)));
        tmpz +=
            df2_coef<order, ctype>(i) *
            (ryzmm * (vol(model2, i1, i2 - i, i3 - i) * vol(field1, i1, i2 - i, i3 - i) +
                      (vol(model3, i1, i2 - i, i3 - i) - vol(model4, i1, i2 - i, i3 - i)) * vol(field2, i1, i2 - i, i3 - i)) -
             ryzmp * (vol(model2, i1, i2 - i, i3 + i) * vol(field1, i1, i2 - i, i3 + i) +
                      (vol(model3, i1, i2 - i, i3 + i) - vol(model4, i1, i2 - i, i3 + i)) * vol(field2, i1, i2 - i, i3 + i)) -
             ryzpm * (vol(model2, i1, i2 + i, i3 - i) * vol(field1, i1, i2 + i, i3 - i) +
                      (vol(model3, i1, i2 + i, i3 - i) - vol(model4, i1, i2 + i, i3 - i)) * vol(field2, i1, i2 + i, i3 - i)) +
             ryzpp * (vol(model2, i1, i2 + i, i3 + i) * vol(field1, i1, i2 + i, i3 + i) +
                      (vol(model3, i1, i2 + i, i3 + i) - vol(model4, i1, i2 + i, i3 + i)) * vol(field2, i1, i2 + i, i3 + i)));
    }
    field3 = tmpx * invdyz;
    field4 = tmpz * invdyz;
}

// drfield3Z_cg_simulator2_adj computes
// field3 = drfield3Z (2 * rx * rz * ((-model1 + model4) * field1 - model2 * field2)
// field4 = drfield3Z (2 * rx * rz * (model2 * field1 + (-model4 + model3) * field2))
template <int order, FD_COEF_TYPE ctype, typename T>
__host__ __device__ inline void
drfield3Z_cg_simulator2_adj(const T* field1, const T* field2, const T* rx, const T* rz, const T* model1, const T* model2, const T* model3,
                 const T* model4, volume_index vol, int i1, int i2, int i3, float invdzx, T& field3, T& field4)
{
    T tmpx = 0.0f;
    T tmpz = 0.0f;
    for (int i = 1; i <= order / 2; i++) {
        T rxzmm = 2.0f * vol(rx, i1 - i, i2, i3 - i) * vol(rz, i1 - i, i2, i3 - i);
        T rxzmp = 2.0f * vol(rx, i1 - i, i2, i3 + i) * vol(rz, i1 - i, i2, i3 + i);
        T rxzpm = 2.0f * vol(rx, i1 + i, i2, i3 - i) * vol(rz, i1 + i, i2, i3 - i);
        T rxzpp = 2.0f * vol(rx, i1 + i, i2, i3 + i) * vol(rz, i1 + i, i2, i3 + i);
        tmpx += df2_coef<order, ctype>(i) *
                (rxzmm * ((vol(model4, i1 - i, i2, i3 - i) - vol(model1, i1 - i, i2, i3 - i)) * vol(field1, i1 - i, i2, i3 - i) -
                          vol(model2, i1 - i, i2, i3 - i) * vol(field2, i1 - i, i2, i3 - i)) -
                 rxzmp * ((vol(model4, i1 - i, i2, i3 + i) - vol(model1, i1 - i, i2, i3 + i)) * vol(field1, i1 - i, i2, i3 + i) -
                          vol(model2, i1 - i, i2, i3 + i) * vol(field2, i1 - i, i2, i3 + i)) -
                 rxzpm * ((vol(model4, i1 + i, i2, i3 - i) - vol(model1, i1 + i, i2, i3 - i)) * vol(field1, i1 + i, i2, i3 - i) -
                          vol(model2, i1 + i, i2, i3 - i) * vol(field2, i1 + i, i2, i3 - i)) +
                 rxzpp * ((vol(model4, i1 + i, i2, i3 + i) - vol(model1, i1 + i, i2, i3 + i)) * vol(field1, i1 + i, i2, i3 + i) -
                          vol(model2, i1 + i, i2, i3 + i) * vol(field2, i1 + i, i2, i3 + i)));
        tmpz +=
            df2_coef<order, ctype>(i) *
            (rxzmm * (vol(model2, i1 - i, i2, i3 - i) * vol(field1, i1 - i, i2, i3 - i) +
                      (vol(model3, i1 - i, i2, i3 - i) - vol(model4, i1 - i, i2, i3 - i)) * vol(field2, i1 - i, i2, i3 - i)) -
             rxzmp * (vol(model2, i1 - i, i2, i3 + i) * vol(field1, i1 - i, i2, i3 + i) +
                      (vol(model3, i1 - i, i2, i3 + i) - vol(model4, i1 - i, i2, i3 + i)) * vol(field2, i1 - i, i2, i3 + i)) -
             rxzpm * (vol(model2, i1 + i, i2, i3 - i) * vol(field1, i1 + i, i2, i3 - i) +
                      (vol(model3, i1 + i, i2, i3 - i) - vol(model4, i1 + i, i2, i3 - i)) * vol(field2, i1 + i, i2, i3 - i)) +
             rxzpp * (vol(model2, i1 + i, i2, i3 + i) * vol(field1, i1 + i, i2, i3 + i) +
                      (vol(model3, i1 + i, i2, i3 + i) - vol(model4, i1 + i, i2, i3 + i)) * vol(field2, i1 + i, i2, i3 + i)));
    }
    field3 = tmpx * invdzx;
    field4 = tmpz * invdzx;
}

// ****************************************************************************
// Basic vector math for float2 and float4 (TODO: different file?)

// Float2
__host__ __device__ inline float2
operator+(const float2& a, const float2& b)
{
    return make_float2(a.x + b.x, a.y + b.y);
}
__host__ __device__ inline float2
operator-(const float2& a, const float2& b)
{
    return make_float2(a.x - b.x, a.y - b.y);
}
__host__ __device__ inline float2
operator-(const float2& a)
{
    return make_float2(-a.x, -a.y);
}
__host__ __device__ inline float2
operator*(const float2& a, const float2& b)
{
    return make_float2(a.x * b.x, a.y * b.y);
}
__host__ __device__ inline float2
operator*(const float2& a, const float& b)
{
    return make_float2(a.x * b, a.y * b);
}
__host__ __device__ inline float2
operator*(const float& a, const float2& b)
{
    return b * a;
}
__host__ __device__ inline float2
operator/(const float2& a, const float2& b)
{
    return make_float2(a.x / b.x, a.y / b.y);
}
__host__ __device__ inline float2
operator/(const float2& a, const float& b)
{
    return make_float2(a.x / b, a.y / b);
}
__host__ __device__ inline float2
operator/(const float& a, const float2& b)
{
    return make_float2(a / b.x, a / b.y);
}
__host__ __device__ inline void
operator+=(float2& a, const float2& b)
{
    a.x += b.x;
    a.y += b.y;
}
__host__ __device__ inline void
operator-=(float2& a, const float2& b)
{
    a.x -= b.x;
    a.y -= b.y;
}
__host__ __device__ inline void
operator*=(float2& a, const float2& b)
{
    a.x *= b.x;
    a.y *= b.y;
}
__host__ __device__ inline void
operator*=(float2& a, const float& b)
{
    a.x *= b;
    a.y *= b;
}

// Float4
__host__ __device__ inline float4
operator+(const float4& a, const float4& b)
{
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}
__host__ __device__ inline float4
operator-(const float4& a, const float4& b)
{
    return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}
__host__ __device__ inline float4
operator-(const float4& a)
{
    return make_float4(-a.x, -a.y, -a.z, -a.w);
}
__host__ __device__ inline float4
operator*(const float4& a, const float4& b)
{
    return make_float4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}
__host__ __device__ inline float4
operator*(const float4& a, const float& b)
{
    return make_float4(a.x * b, a.y * b, a.z * b, a.w * b);
}
__host__ __device__ inline float4
operator*(const float& a, const float4& b)
{
    return b * a;
}
__host__ __device__ inline float4
operator/(const float4& a, const float4& b)
{
    return make_float4(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w);
}
__host__ __device__ inline float4
operator/(const float4& a, const float& b)
{
    return make_float4(a.x / b, a.y / b, a.z / b, a.w / b);
}
__host__ __device__ inline float4
operator/(const float& a, const float4& b)
{
    return make_float4(a / b.x, a / b.y, a / b.z, a / b.w);
}

__host__ __device__ inline void
operator+=(float4& a, const float4& b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
}
__host__ __device__ inline void
operator-=(float4& a, const float4& b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    a.w -= b.w;
}
__host__ __device__ inline void
operator*=(float4& a, const float4& b)
{
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
    a.w *= b.w;
}
__host__ __device__ inline void
operator*=(float4& a, const float& b)
{
    a.x *= b;
    a.y *= b;
    a.z *= b;
    a.w *= b;
}

// Simple structure that holds N values (N relatively small and known at compile time).
// This is useful to pass small 1D arrays as parameters to a kernel, as we can't pass
// 1D arrays "per value" to a kernel, but we can pass a structure "per value".
template <typename T, int n>
struct vecparam
{
    T value[n];
};

} // end of namespace helper_kernels_gpu

#endif //HELPER_KERNELS_GPU_H
