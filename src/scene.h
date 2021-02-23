#pragma once

#include "custom_math.h"
#include "shader_types.h"

#include <vector>
#include <string>

struct camera_t
{
    float3 position { 3.0f, 3.0f, 4.0f };
    float3 look_at { 0.0f, 0.0f, 0.0f };
    float fov = 45;
    float aspect_ratio = 1024.0f / 728.0f;

    camera_t() = default;

    void set_params(params_t& params) const
    {
        float3 w = look_at - position;
        float l = length(w) * TANF(fov * M_PI / 180.0f);

        float3 u = normalize(cross(w, { 0.0f, 1.0f, 0.0f }));
        float3 v = normalize(cross(u, w));

        // Horizontal half-extent
        u = l * u;

        // Vertical half-extent
        v = (l / aspect_ratio) * v;

        params.camera.position = position;
        params.camera.u = u;
        params.camera.v = v;
        params.camera.w = w;
    }
};

struct scene_t
{
    std::vector<float> vertices;
//    std::vector<unsigned int> indices;
    camera_t camera;

    explicit scene_t(const std::string& path)
    {
        // Just a triangle for now
        vertices = {
                // positions
                -1.0f, 1.0f, -1.0f,
                -1.0f, -1.0f, -1.0f,
                1.0f, -1.0f, -1.0f,
                1.0f, -1.0f, -1.0f,
                1.0f, 1.0f, -1.0f,
                -1.0f, 1.0f, -1.0f,

                -1.0f, -1.0f, 1.0f,
                -1.0f, -1.0f, -1.0f,
                -1.0f, 1.0f, -1.0f,
                -1.0f, 1.0f, -1.0f,
                -1.0f, 1.0f, 1.0f,
                -1.0f, -1.0f, 1.0f,

                1.0f, -1.0f, -1.0f,
                1.0f, -1.0f, 1.0f,
                1.0f, 1.0f, 1.0f,
                1.0f, 1.0f, 1.0f,
                1.0f, 1.0f, -1.0f,
                1.0f, -1.0f, -1.0f,

                -1.0f, -1.0f, 1.0f,
                -1.0f, 1.0f, 1.0f,
                1.0f, 1.0f, 1.0f,
                1.0f, 1.0f, 1.0f,
                1.0f, -1.0f, 1.0f,
                -1.0f, -1.0f, 1.0f,

                -1.0f, 1.0f, -1.0f,
                1.0f, 1.0f, -1.0f,
                1.0f, 1.0f, 1.0f,
                1.0f, 1.0f, 1.0f,
                -1.0f, 1.0f, 1.0f,
                -1.0f, 1.0f, -1.0f,

                -1.0f, -1.0f, -1.0f,
                -1.0f, -1.0f, 1.0f,
                1.0f, -1.0f, -1.0f,
                1.0f, -1.0f, -1.0f,
                -1.0f, -1.0f, 1.0f,
                1.0f, -1.0f, 1.0f
        };

//        indices = {
//                { 0, 1, 2 },
//                { 1, 3, 2 }
//        };
    }
};
