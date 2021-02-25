#pragma once

#include "custom_math.h"
#include "shader_types.h"

#include <tiny_obj_loader.h>
#include <tinyexr.h>

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

        // Horizontal half-extent
        float3 u = l * normalize(cross(w, { 0.0f, 1.0f, 0.0f }));

        // Vertical half-extent
        float3 v = (l / aspect_ratio) * normalize(cross(u, w));

        // Setup parameters
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

    unsafe::device_buffer_t d_vertices;
    unsafe::device_buffer_t d_indices;

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

        // Copy to the device
        size_t buffer_size = vertices.size() * sizeof(float3);
        d_vertices.allocate(buffer_size);
        d_vertices.load_data(vertices.data(), buffer_size);

        buffer_size = indices.size() * sizeof(uint3);
        d_indices.allocate(buffer_size);
        d_indices.load_data(indices.data(), buffer_size);
    }

    ~mesh_t()
    {
        d_vertices.free();
        d_indices.free();
    }
};

struct scene_t
{
    mesh_t mesh;
    camera_t camera;

    explicit scene_t(const std::string& path)
            : mesh("../../assets/suzanne.obj")
    {}
};

struct cubemap_t
{
    float* data = nullptr;
    int width = 0;
    int height = 0;

    explicit cubemap_t(const std::string& path)
    {
        const char* error;
        if (LoadEXR(&data, &width, &height, path.c_str(), &error) != TINYEXR_SUCCESS)
        {
            if (error)
            {
                LOG_ERROR("ERROR: Cannot load EXR at path %s\n%s", path.c_str(), error);
                FreeEXRErrorMessage(error);
            }
        }
    }

    ~cubemap_t()
    {
        free(data);
    }
};
