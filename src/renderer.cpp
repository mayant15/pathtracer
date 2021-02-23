#include "renderer.h"
#include "module.h"
#include "buffers.h"

// This should be in one translation unit only
#include <optix_function_table_definition.h>

#include <optix_stack_size.h>
#include <optix_stubs.h>

static void logger(unsigned int level, const char* tag, const char* message, void* cbdata)
{
    if (level <= 2)
    {
        LOG_ERROR("[OptiX][%d] %s\n", level, message);
    }
    else
    {
        LOG_INFO("[OptiX][%d] %s\n", level, message);
    }
}

void renderer_t::init_context()
{
    // Initialize CUDA
    CUDA_SAFE_CALL(cudaFree(nullptr));

    // Initialize OptiX
    // '0' is for current context
    CUcontext cuCtx = 0;  // NOLINT(modernize-use-nullptr)
    OPTIX_SAFE_CALL(optixInit());

    // Initialize OptiX device
    OptixDeviceContextOptions options {};
    options.logCallbackFunction = &logger;
    options.logCallbackLevel = 4;
    OPTIX_SAFE_CALL(optixDeviceContextCreate(cuCtx, &options, &_context));
}

void renderer_t::init_module()
{
    OptixModuleCompileOptions module_compile_options {};
    module_compile_options.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;

    _pipeline_options.usesMotionBlur = false;
    _pipeline_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
    _pipeline_options.numPayloadValues = 3;
    _pipeline_options.numAttributeValues = 3;
    _pipeline_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;  // TODO: should be OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
    _pipeline_options.pipelineLaunchParamsVariableName = "params";
    _pipeline_options.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;

    const std::string ptx = load_ptx_string("triangle.ptx");
    size_t sizeof_log = sizeof(_error_log);

    OPTIX_SAFE_CALL(optixModuleCreateFromPTX(_context, &module_compile_options, &_pipeline_options, ptx.c_str(),
                                             ptx.size(), _error_log, &sizeof_log, &_module));
}

void renderer_t::init_programs()
{
    size_t sizeof_log = sizeof(_error_log);
    OptixProgramGroupOptions options {};

    // Raygen program group
    OptixProgramGroupDesc rg_desc {};
    rg_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    rg_desc.raygen.module = _module;
    rg_desc.raygen.entryFunctionName = "__raygen__rg";
    OPTIX_SAFE_CALL(optixProgramGroupCreate(_context, &rg_desc, 1, &options, _error_log, &sizeof_log,
                                            &_raygen_pg));

    // Miss program group
    OptixProgramGroupDesc miss_desc {};
    miss_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    miss_desc.miss.module = _module;
    miss_desc.miss.entryFunctionName = "__miss__ms";
    OPTIX_SAFE_CALL(optixProgramGroupCreate(_context, &miss_desc, 1, &options, _error_log, &sizeof_log,
                                            &_miss_pg));

    // Hitgroup program group
    OptixProgramGroupDesc hitgroup_desc {};
    hitgroup_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hitgroup_desc.hitgroup.moduleCH = _module;
    hitgroup_desc.hitgroup.entryFunctionNameCH = "__closesthit__ch";
    OPTIX_SAFE_CALL(
            optixProgramGroupCreate(_context, &hitgroup_desc, 1, &options, _error_log, &sizeof_log,
                                    &_hitgroup_pg));
}

void renderer_t::init_pipeline()
{
    const uint32_t max_trace_depth = 1;
    std::vector<OptixProgramGroup> groups = { _raygen_pg, _miss_pg, _hitgroup_pg };

    OptixPipelineLinkOptions pipeline_link_options {};
    pipeline_link_options.maxTraceDepth = max_trace_depth;
    pipeline_link_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;

    size_t sizeof_log = sizeof(_error_log);
    OPTIX_SAFE_CALL(
            optixPipelineCreate(_context, &_pipeline_options, &pipeline_link_options, groups.data(), groups.size(),
                                _error_log, &sizeof_log, &_pipeline));

    OptixStackSizes stack_sizes = {};
    for (auto& group : groups)
    {
        OPTIX_SAFE_CALL(optixUtilAccumulateStackSizes(group, &stack_sizes));
    }

    uint32_t direct_callable_stack_size_from_traversal;
    uint32_t direct_callable_stack_size_from_state;
    uint32_t continuation_stack_size;
    OPTIX_SAFE_CALL(
            optixUtilComputeStackSizes(&stack_sizes, max_trace_depth, 0, 0, &direct_callable_stack_size_from_traversal,
                                       &direct_callable_stack_size_from_state, &continuation_stack_size));
    OPTIX_SAFE_CALL(optixPipelineSetStackSize(_pipeline, direct_callable_stack_size_from_traversal,
                                              direct_callable_stack_size_from_state, continuation_stack_size, 2));
}

void renderer_t::init_sbt()
{
    // Raygen record
    ray_gen_sbt_record_t rg_sbt {};
    OPTIX_SAFE_CALL(optixSbtRecordPackHeader(_raygen_pg, &rg_sbt));
    device_buffer_t rg_buffer { sizeof(ray_gen_sbt_record_t), &rg_sbt, true };
    _sbt.raygenRecord = rg_buffer.data();

    // Miss record
    miss_sbt_record_t miss_sbt {};
    miss_sbt.data.background = { 77, 26, 51, 255 };
    OPTIX_SAFE_CALL(optixSbtRecordPackHeader(_miss_pg, &miss_sbt));
    device_buffer_t miss_buffer { sizeof(miss_sbt_record_t), &miss_sbt, true };
    _sbt.missRecordBase = miss_buffer.data();
    _sbt.missRecordStrideInBytes = sizeof(miss_sbt_record_t);
    _sbt.missRecordCount = 1;

    // Hitgroup record
    hitgroup_sbt_record_t hg_sbt {};
    OPTIX_SAFE_CALL(optixSbtRecordPackHeader(_hitgroup_pg, &hg_sbt));
    device_buffer_t hg_buffer { sizeof(hitgroup_sbt_record_t), &hg_sbt, true };
    _sbt.hitgroupRecordBase = hg_buffer.data();
    _sbt.hitgroupRecordStrideInBytes = sizeof(hitgroup_sbt_record_t);
    _sbt.hitgroupRecordCount = 1;
}

renderer_t::renderer_t(const render_options_t& opt)
        : _options(opt)
{
    init_context();

    init_module();
    init_programs();
    init_pipeline();

    init_sbt();
}

void renderer_t::load_scene(const scene_t& scene)
{
    _camera = scene.camera;

    // TODO: See compaction
    OptixAccelBuildOptions accel_options {};
    accel_options.buildFlags = OPTIX_BUILD_FLAG_NONE;
    accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

    // Copy vertices to device
    device_buffer_t d_vertices { sizeof(float3) * scene.vertices.size(), (void*) scene.vertices.data() };
    auto d_vertex_ptr = d_vertices.data();

    // Our build input is a simple list of non-indexed triangle vertices
    const unsigned int triangle_input_flags[1] = { OPTIX_GEOMETRY_FLAG_NONE };
    OptixBuildInput build_input {};
    build_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    build_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    build_input.triangleArray.numVertices = scene.vertices.size();
    build_input.triangleArray.vertexBuffers = &d_vertex_ptr;
    build_input.triangleArray.flags = triangle_input_flags;
    build_input.triangleArray.numSbtRecords = 1;

    // Estimate memory usage
    OptixAccelBufferSizes buffer_sizes;
    OPTIX_SAFE_CALL(optixAccelComputeMemoryUsage(
            _context,
            &accel_options,
            &build_input,
            1, // Number of build inputs
            &buffer_sizes
    ));

    // Allocate estimated memory
    device_buffer_t temp_buffer { buffer_sizes.tempSizeInBytes };

    // The output buffer should persist, we'll save the pointer in the class
    device_buffer_t output_buffer { buffer_sizes.outputSizeInBytes, nullptr, true };

    // Build the structure
    OPTIX_SAFE_CALL(optixAccelBuild(
            _context,
            0,                  // CUDA stream
            &accel_options,
            &build_input,
            1,                  // num build inputs
            temp_buffer.data(),
            buffer_sizes.tempSizeInBytes,
            output_buffer.data(),
            buffer_sizes.outputSizeInBytes,
            &_scene_handle,
            nullptr,            // emitted property list
            0                   // num emitted properties
    ));

    // Save the accel pointer
    _accel_ptr = output_buffer.data();

    // Non-persistent buffers will be cleaned
}

void renderer_t::render(const device_buffer_t& buffer)
{
    CUstream stream;
    CUDA_SAFE_CALL(cudaStreamCreate(&stream));

    // Initialize parameters
    params_t params {};
    params.image = (color_t*) buffer.data();
    params.image_width = _options.width;
    params.image_height = _options.height;
    params.handle = _scene_handle;
    _camera.set_params(params);

    // Copy parameters to device
    device_buffer_t d_params { sizeof(params_t), &params };
    OPTIX_SAFE_CALL(
            optixLaunch(_pipeline, stream, d_params.data(), sizeof(params_t), &_sbt, _options.width, _options.height,
                        1));
    CUDA_SAFE_SYNC();
}

void renderer_t::cleanup()
{
    CUDA_SAFE_CALL(cudaFree(reinterpret_cast<void*>(_sbt.raygenRecord)));
    CUDA_SAFE_CALL(cudaFree(reinterpret_cast<void*>(_sbt.missRecordBase)));
    CUDA_SAFE_CALL(cudaFree(reinterpret_cast<void*>(_sbt.hitgroupRecordBase)));

    CUDA_SAFE_CALL(cudaFree(reinterpret_cast<void*>(_accel_ptr)));

    OPTIX_SAFE_CALL(optixPipelineDestroy(_pipeline));
    OPTIX_SAFE_CALL(optixProgramGroupDestroy(_hitgroup_pg));
    OPTIX_SAFE_CALL(optixProgramGroupDestroy(_miss_pg));
    OPTIX_SAFE_CALL(optixProgramGroupDestroy(_raygen_pg));
    OPTIX_SAFE_CALL(optixModuleDestroy(_module));

    OPTIX_SAFE_CALL(optixDeviceContextDestroy(_context));
}
