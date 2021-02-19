#pragma once

#include "types.h"
#include "checkers.h"
#include "buffers.h"

struct render_options_t
{
    unsigned int width;
    unsigned int height;
};

class renderer_t
{
    OptixDeviceContext _context = nullptr;
    OptixShaderBindingTable sbt {};

    OptixModule _module = nullptr;
    OptixProgramGroup _ray_generation_group = nullptr;
    OptixProgramGroup _miss_group = nullptr;
    OptixPipelineCompileOptions _pipeline_options {};
    OptixPipeline _pipeline = nullptr;

    render_options_t _options;

public:
    explicit renderer_t(const render_options_t& opt);
    void render(const device_buffer_t& buffer);
    void cleanup();

private:
    char _error_log[2048] {};

    void init_context();
    void init_module();
    void init_programs();
    void init_pipeline();
    void init_sbt();
};
