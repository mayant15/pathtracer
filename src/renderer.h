#pragma once

#include "types.h"
#include "checkers.h"
#include "buffers.h"
#include "scene.h"

struct render_options_t
{
    unsigned int width;
    unsigned int height;
};

class renderer_t
{
    // Context
    OptixDeviceContext _context = nullptr;
    OptixShaderBindingTable sbt {};

    // Pipeline
    OptixModule _module = nullptr;
    OptixProgramGroup _ray_generation_group = nullptr;
    OptixProgramGroup _miss_group = nullptr;
    OptixPipelineCompileOptions _pipeline_options {};
    OptixPipeline _pipeline = nullptr;

    // Scene
    OptixTraversableHandle _scene_handle {};
    CUdeviceptr _accel_ptr {};

    // Config
    render_options_t _options;

public:
    explicit renderer_t(const render_options_t& opt);
    void render(const device_buffer_t& buffer);
    void load_scene(const scene_t& scene);
    void cleanup();

private:
    char _error_log[2048] {};

    void init_context();
    void init_module();
    void init_programs();
    void init_pipeline();
    void init_sbt();
};
