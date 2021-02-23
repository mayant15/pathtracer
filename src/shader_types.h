#pragma once

#include "color.h"
#include <optix.h>

struct params_t
{
    // Image output buffer
    color_t* image;
    unsigned int image_width;
    unsigned int image_height;

    // Camera
    struct camera
    {
        float3 position;
        float3 u, v, w;
    } camera;

    // Acceleration structure
    OptixTraversableHandle handle;
};

struct miss_data_t
{
    color_t background;
};

struct hitgroup_data_t {};
struct ray_gen_data_t {};
