#pragma once

#include "color.h"

#include <stb_image_write.h>
#include <string>

void write_image(const std::vector<color_t>& data, const std::string& path, uint2 dims)
{
    stbi_write_png(path.c_str(), dims.x, dims.y, 4, data.data(), dims.x * sizeof (color_t));
}
