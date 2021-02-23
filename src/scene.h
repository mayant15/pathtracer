#pragma once

#include "custom_math.h"
#include "shader_types.h"
#include <tiny_obj_loader.h>
#include <vector>
#include <string>

struct camera_t
{
    float3 position { 1.2f, 0.3f, 2.0f };
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

struct mesh_t
{
    std::vector<float3> vertices;
    std::vector<uint3> indices;

    explicit mesh_t(const std::string& path)
    {
        tinyobj::attrib_t attrib;
        std::vector<tinyobj::shape_t> shapes;
        std::vector<tinyobj::material_t> materials;

        std::string err;
        bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &err, path.c_str());

        if (!ret && !err.empty())
        {
            LOG_ERROR("ERROR: Cannot load OBJ at path %s\n", path.c_str());
        }

        for (size_t i = 0; i < attrib.vertices.size(); i += 3)
        {
            vertices.push_back({
                                       attrib.vertices[i],
                                       attrib.vertices[i + 1],
                                       attrib.vertices[i + 2]
                               });
        }

        for (auto& shape : shapes)
        {
            for (size_t index_offset = 0; index_offset + 2 < shape.mesh.indices.size(); index_offset += 3)
            {
                indices.push_back({
                                          (unsigned int) shape.mesh.indices[index_offset].vertex_index,
                                          (unsigned int) shape.mesh.indices[index_offset + 1].vertex_index,
                                          (unsigned int) shape.mesh.indices[index_offset + 2].vertex_index
                                  });
            }
        }
    }
};

struct scene_t
{
    std::vector<mesh_t> meshes;
    camera_t camera;

    explicit scene_t(const std::string& path)
    {
        meshes.emplace_back("../../assets/suzanne.obj");
    }
};
