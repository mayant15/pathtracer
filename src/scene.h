#pragma once

#include "custom_math.h"
#include "shader_types.h"

#include <vector>
#include <string>

struct camera_t
{
    float3 position { 0.0f, 0.0f, -2.0f };
    float3 look_at { 0.0f, 0.0f, 0.0f };
    float fov = 35;
    float aspect_ratio = 1024.0f / 728.0f;

    camera_t() = default;

    void set_params(params_t& params) const
    {
        float3 w = look_at - position;
        float l = length(w) * TANF(0.5f * fov * M_PI / 180.0f);

        float3 u = normalize(cross(w, { 0.0f, 1.0f, 0.0f }));
        float3 v = normalize(cross(u, w));

        // Horizontal half-extent
        u = l * u;

        // Vertical half-extent
        v = l * aspect_ratio * v;

        params.camera.position = position;
        params.camera.u = u;
        params.camera.v = v;
        params.camera.w = w;

//        params.camera_u = { 1.10456955f, 0.0f, 0.0f };
//        params.camera_v = { 0.0f, 0.828427136f, 0.0f };
//        params.camera_w = { 0.0f, 0.0f, -2.0f };
    }
};

struct scene_t
{
    std::vector<float3> vertices;
    camera_t camera;

    explicit scene_t(const std::string& path);
};
