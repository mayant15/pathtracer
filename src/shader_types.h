#pragma once

#include "color.h"

struct params_t
{
    // Image output buffer
    color_t* image;
    unsigned int image_width;
    unsigned int image_height;

    // Camera
    float3 camera_position;
    float3 camera_u;
    float3 camera_v;
    float3 camera_w;

    // Acceleration structure
    OptixTraversableHandle handle;
};

struct miss_data_t
{
    color_t background;
};

struct hitgroup_data_t {};
struct ray_gen_data_t {};
