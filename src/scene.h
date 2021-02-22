#pragma once

#include "vector_types.h"
#include <vector>
#include <string>

struct scene_t
{
    std::vector<float3> vertices;

    explicit scene_t(const std::string& path);
};
