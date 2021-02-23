#pragma clang diagnostic push
#pragma ide diagnostic ignored "OCUnusedGlobalDeclarationInspection"
#pragma ide diagnostic ignored "bugprone-reserved-identifier"

#include <optix.h>
#include "shader_types.h"
#include "custom_math.h"

// no name mangling
extern "C" __constant__ params_t params;

//*****************************************************************************
// HELPER FUNCTIONS

static __forceinline__ __device__ void set_payload(float4 p)
{
    // Set the payload, but change [0, 1] to [0, 255]
    optixSetPayload_0(static_cast<unsigned int>(255 * p.x));
    optixSetPayload_1(static_cast<unsigned int>(255 * p.y));
    optixSetPayload_2(static_cast<unsigned int>(255 * p.z));
    optixSetPayload_3(static_cast<unsigned int>(255 * p.w));
}

static __forceinline__ __device__ void set_payload(uchar4 p)
{
    // Set the payload, the input is already in [0, 255]
    optixSetPayload_0(p.x);
    optixSetPayload_1(p.y);
    optixSetPayload_2(p.z);
    optixSetPayload_3(p.w);
}

static __forceinline__ __device__ void compute_ray(uint3 idx, uint3 dims, float3& origin_out, float3& dir_out)
{
    // Send a ray out from the camera, depending on the position in the launch grid
    // coords should be transformed from [0, 1] to [-1, 1]
    float2 coords = make_float2((float) idx.x / (float) dims.x, (float) idx.y / (float) dims.y);
    coords = (2.0f * coords) + make_float2(-1.0f, -1.0f);

    dir_out = normalize((coords.x * params.camera.u) + (coords.y * params.camera.v) + params.camera.w);
    origin_out = params.camera.position;
}


//*****************************************************************************
// SHADER FUNCTIONS

extern "C" __global__ void __raygen__rg()
{
    // Find ray origin and direction
    float3 ray_origin, ray_dir;
    const auto idx = optixGetLaunchIndex();
    const auto dims = optixGetLaunchDimensions();
    compute_ray(idx, dims, ray_origin, ray_dir);

    // Trace the ray
    unsigned int p0, p1, p2, p3;
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
            p0, p1, p2, p3                 // payload
    );

    // The payload at the end of the ray's path will be color, use it at full opacity
    params.image[idx.y * params.image_width + idx.x] = make_uchar4(p0, p1, p2, p3);
}

extern "C" __global__ void __miss__ms()
{
    auto data = reinterpret_cast<miss_data_t*>(optixGetSbtDataPointer());
    set_payload(data->background);
}

extern "C" __global__ void __closesthit__ch()
{
    auto data = reinterpret_cast<hitgroup_data_t*>(optixGetSbtDataPointer());
    auto id = optixGetPrimitiveIndex();
    auto& idx = data->indices[id];

    // Calculate normal for this face
    auto& A = data->vertices[idx.x];
    auto& B = data->vertices[idx.y];
    auto& C = data->vertices[idx.z];
    const float3& N = normalize(cross(B - A, C - A));

    // Calculate radiance
    auto rayDir = optixGetWorldRayDirection();
    float weight = 0.2f + 0.8f * fabsf(dot(rayDir, N));
    float3 radiance = weight * make_float3(
            static_cast<float>(data->color.x) / 255.0f,
            static_cast<float>(data->color.y) / 255.0f,
            static_cast<float>(data->color.z) / 255.0f
    );

    auto payload = make_float4(radiance.x, radiance.y, radiance.z, 1.0f);
    set_payload(payload);
}

#pragma clang diagnostic pop
