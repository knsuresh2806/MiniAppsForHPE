#ifndef HELPER_KERNELS_GPU_H
#define HELPER_KERNELS_GPU_H

namespace helper_kernels_gpu {
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
