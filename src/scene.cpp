#include "scene.h"

scene_t::scene_t(const std::string& path)
{
    // Just a triangle for now
    vertices = {
            { -0.5f, -0.5f, 0.0f },
            { 0.5f,  -0.5f, 0.0f },
            { 0.0f,  0.5f,  0.0f }
    };
}
