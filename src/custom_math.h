#pragma once

#include <vector_types.h>

#ifdef __CUDACC__
#define HOST_OR_DEVICE __forceinline__ __device__
#define MAKE_FLOAT(DIMS, ...) make_float##DIMS(__VA_ARGS__)
#define SQRT(X) sqrt(X)
#define TANF(X) tanf(X)
#else
#include <cmath>
#define HOST_OR_DEVICE inline
#define MAKE_FLOAT(DIMS, ...) float##DIMS { __VA_ARGS__ }
#define SQRT(X) std::sqrt(X)
#define TANF(X) std::tanf(X)
#endif

#define M_PI 3.1415926535897932385

static HOST_OR_DEVICE float3 operator+(const float3& a, const float3& b)
{
    return MAKE_FLOAT(3, a.x + b.x, a.y + b.y, a.z + b.z);
}

static HOST_OR_DEVICE float3 operator-(const float3& a, const float3& b)
{
    return MAKE_FLOAT(3, a.x - b.x, a.y - b.y, a.z - b.z);
}

static HOST_OR_DEVICE float2 operator+(const float2& a, const float2& b)
{
    return MAKE_FLOAT(2, a.x + b.x, a.y + b.y);
}

static HOST_OR_DEVICE float3 operator*(float s, const float3& v)
{
    return MAKE_FLOAT(3, s * v.x, s * v.y, s * v.z);
}

static HOST_OR_DEVICE float2 operator*(float s, const float2& v)
{
    return MAKE_FLOAT(2, s * v.x, s * v.y);
}

static HOST_OR_DEVICE float length(float3 v)
{
    return SQRT(v.x * v.x + v.y * v.y + v.z * v.z);
}

static HOST_OR_DEVICE float3 normalize(float3 v)
{
    float n = length(v);
    return MAKE_FLOAT(3, v.x / n, v.y / n, v.z / n);
}

static HOST_OR_DEVICE float3 cross(float3 a, float3 b)
{
    return MAKE_FLOAT(3,
                      a.y * b.z - a.z * b.y,
                      a.z * b.x - a.x * b.z,
                      a.x * b.y - a.y * b.x
    );
}
