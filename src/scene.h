#pragma once

#include "vector_types.h"

#include <vector>
#include <string>

class camera_t
{
public:
    camera_t() = default;
};

struct scene_t
{
    std::vector<float3> vertices;
    camera_t camera;

    explicit scene_t(const std::string& path);
};
