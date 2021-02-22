#pragma clang diagnostic push
#pragma ide diagnostic ignored "OCUnusedGlobalDeclarationInspection"
#pragma ide diagnostic ignored "bugprone-reserved-identifier"

#include <optix.h>
#include "shader_types.h"

// no name mangling
extern "C" __constant__ params_t params;

//*****************************************************************************
// HELPER FUNCTIONS

static __forceinline__ __device__ float3 operator+(const float3& a, const float3& b)
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

static __forceinline__ __device__ float2 operator+(const float2& a, const float2& b)
{
    return make_float2(a.x + b.x, a.y + b.y);
}

static __forceinline__ __device__ float3 operator*(float s, const float3& v)
{
    return make_float3(s * v.x, s * v.y, s * v.z);
}

static __forceinline__ __device__ float2 operator*(float s, const float2& v)
{
    return make_float2(s * v.x, s * v.y);
}

static __forceinline__ __device__ float3 normalize(float3 v)
{
    float n = sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
    return make_float3(v.x / n, v.y / n, v.z / n);
}

static __forceinline__ __device__ void set_payload(float4 p)
{
    // Set the payload, but change [0, 1] to [0, 255]
    optixSetPayload_0((unsigned int) (255 * p.x));
    optixSetPayload_1((unsigned int) (255 * p.y));
    optixSetPayload_2((unsigned int) (255 * p.z));
    // optixSetPayload_3(p.w); alpha is always 255, doesn't need to be on the payload
}

static __forceinline__ __device__ void set_payload(uchar4 p)
{
    // Set the payload, the input is already in [0, 255]
    optixSetPayload_0(p.x);
    optixSetPayload_1(p.y);
    optixSetPayload_2(p.z);
    // optixSetPayload_3(p.w); alpha is always 255, doesn't need to be on the payload
}

static __device__ void compute_ray(uint3 idx, uint3 dims, float3& origin_out, float3& dir_out)
{
    // Send a ray out from the camera, depending on the position in the launch grid
    // coords should be transformed from [0, 1] to [-1, 1]
    float2 coords = make_float2((float) idx.x / (float) dims.x, (float) idx.y / (float) dims.y);
    coords = (2.0f * coords) + make_float2(-1.0f, -1.0f);

    dir_out = normalize((coords.x * params.camera_u) + (coords.y * params.camera_v) + params.camera_w);
    origin_out = params.camera_position;
}


//*****************************************************************************
// SHADER FUNCTIONS

extern "C" __global__ void __raygen__rg()
{
    // Find ray origin and direction
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dims = optixGetLaunchDimensions();
    float3 ray_origin;
    float3 ray_dir;
    compute_ray(idx, dims, ray_origin, ray_dir);

    // Trace the ray
    unsigned int p0, p1, p2;
    optixTrace(
            params.handle,
            ray_origin,
            ray_dir,
            0.0f,
            1e16f,
            0.0f,
            OptixVisibilityMask(255),
            OPTIX_RAY_FLAG_NONE,
            0, 1, 0, // SBT
            p0, p1, p2                 // payload
    );

    // The payload at the end of the ray's path will be color, use it at full opacity
    params.image[idx.y * params.image_width + idx.x] = make_uchar4(p0, p1, p2, 255);
}

extern "C" __global__ void __miss__ms()
{
    auto data = reinterpret_cast<miss_data_t*>(optixGetSbtDataPointer());
    set_payload(data->background);
}

extern "C" __global__ void __closesthit__ch()
{
    const float2 barycentrics = optixGetTriangleBarycentrics();
    const float4 payload = make_float4(barycentrics.x, barycentrics.y, 1.0f, 1.0f);
    set_payload(payload);
}

#pragma clang diagnostic pop
