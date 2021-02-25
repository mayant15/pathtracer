#pragma once

#include "custom_math.h"
#include "shader_types.h"

#include <tiny_obj_loader.h>

#include <utility>
#include <vector>
#include <string>

struct camera_t
{
    float3 position { 3.5f, 4.0f, 4.0f };
    float3 look_at { 0.0f, 1.0f, -2.0f };
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

namespace unsafe
{
    struct mesh_t
    {
        std::vector<uint3> indices;

        void load()
        {
            _buffer.allocate(indices.size() * sizeof(uint3));
            _buffer.load_data(indices.data(), _buffer.size);
        }

        void unload()
        {
            _buffer.free();
        }

        [[nodiscard]] CUdeviceptr get_device_ptr() const
        { return reinterpret_cast<CUdeviceptr>(_buffer.data); }

    private:
        unsafe::device_buffer_t _buffer;
    };
}

struct scene_t
{
    std::vector<float3> vertices;
    std::vector<unsafe::mesh_t> meshes;
    camera_t camera;

    explicit scene_t(const std::string& path)
    {
        tinyobj::attrib_t attrib;
        std::vector<tinyobj::shape_t> shapes;
        std::vector<tinyobj::material_t> materials;

        std::string err;
        bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &err, path.c_str());

        if (!ret && !err.empty())
        {
            LOG_ERROR("ERROR: Cannot load scene at path %s\n%s\n", path.c_str(), err.c_str());
        }

        // Load vertices
        for (size_t i = 0; i < attrib.vertices.size(); i += 3)
        {
            vertices.push_back({
                                       attrib.vertices[i],
                                       attrib.vertices[i + 1],
                                       attrib.vertices[i + 2]
                               });
        }

        // Copy to the device
        size_t buffer_size = vertices.size() * sizeof(float3);
        d_vertices.allocate(buffer_size);
        d_vertices.load_data(vertices.data(), buffer_size);

        // Load indices
        meshes.resize(shapes.size());
        for (size_t i = 0; i < shapes.size(); ++i)
        {
            tinyobj::shape_t& shape = shapes[i];
            for (size_t index_offset = 0; index_offset + 2 < shape.mesh.indices.size(); index_offset += 3)
            {
                meshes[i].indices.push_back({
                                                    (unsigned int) shape.mesh.indices[index_offset].vertex_index,
                                                    (unsigned int) shape.mesh.indices[index_offset + 1].vertex_index,
                                                    (unsigned int) shape.mesh.indices[index_offset + 2].vertex_index
                                            });
            }
            meshes[i].load();
        }
    }

    [[nodiscard]] CUdeviceptr get_device_ptr() const
    { return reinterpret_cast<CUdeviceptr>(d_vertices.data); }

    ~scene_t()
    {
        for (auto& mesh : meshes)
        {
            mesh.unload();
        }

        d_vertices.free();
    }

private:
    unsafe::device_buffer_t d_vertices;
};
