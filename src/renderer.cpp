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
    _pipeline_options.numPayloadValues = 2;
    _pipeline_options.numAttributeValues = 2;
    _pipeline_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;  // TODO: should be OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
    _pipeline_options.pipelineLaunchParamsVariableName = "params";

    // TODO: PTX string
    const std::string ptx = load_ptx_string("draw_solid_color.ptx");
    size_t sizeof_log = sizeof(_error_log);

    OPTIX_SAFE_CALL(optixModuleCreateFromPTX(_context, &module_compile_options, &_pipeline_options, ptx.c_str(),
                                             ptx.size(), _error_log, &sizeof_log, &_module));
}

void renderer_t::init_programs()
{
    OptixProgramGroupOptions program_group_options {};

    OptixProgramGroupDesc raygen_prog_group_desc {};
    raygen_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    raygen_prog_group_desc.raygen.module = _module;
    raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__draw";

    size_t sizeof_log = sizeof(_error_log);
    OPTIX_SAFE_CALL(optixProgramGroupCreate(_context, &raygen_prog_group_desc, 1, &program_group_options, _error_log,
                                            &sizeof_log, &_ray_generation_group));

    // Leave miss group's module and entryfunc name null
    OptixProgramGroupDesc miss_prog_group_desc {};
    miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;

    OPTIX_SAFE_CALL(
            optixProgramGroupCreate(_context, &miss_prog_group_desc, 1, &program_group_options, _error_log, &sizeof_log,
                                    &_miss_group));
}

void renderer_t::init_pipeline()
{
    const uint32_t max_trace_depth = 0;
    std::vector<OptixProgramGroup> groups = { _ray_generation_group };

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
    CUdeviceptr raygen_record;
    const size_t raygen_record_size = sizeof(ray_gen_sbt_record_t);
    CUDA_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>( &raygen_record ), raygen_record_size));

    ray_gen_sbt_record_t rg_sbt;
    OPTIX_SAFE_CALL(optixSbtRecordPackHeader(_ray_generation_group, &rg_sbt));
    rg_sbt.data = { 25, 25, 255, 255 };
    CUDA_SAFE_CALL(
            cudaMemcpy(reinterpret_cast<void*>( raygen_record ), &rg_sbt, raygen_record_size, cudaMemcpyHostToDevice));

    CUdeviceptr miss_record;
    size_t miss_record_size = sizeof(miss_sbt_record_t);
    CUDA_SAFE_CALL(cudaMalloc(reinterpret_cast<void**>( &miss_record ), miss_record_size));
    ray_gen_sbt_record_t ms_sbt;
    OPTIX_SAFE_CALL(optixSbtRecordPackHeader(_miss_group, &ms_sbt));
    CUDA_SAFE_CALL(
            cudaMemcpy(reinterpret_cast<void*>( miss_record ), &ms_sbt, miss_record_size, cudaMemcpyHostToDevice));

    sbt.raygenRecord = raygen_record;
    sbt.missRecordBase = miss_record;
    sbt.missRecordStrideInBytes = sizeof(miss_sbt_record_t);
    sbt.missRecordCount = 1;
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
    params.image = (uchar4*) buffer.data();
    params.image_width = _options.width;

    // Copy parameters to device
    device_buffer_t d_params { sizeof(params_t), &params };

    OPTIX_SAFE_CALL(
            optixLaunch(_pipeline, stream, d_params.data(), sizeof(params_t), &sbt, _options.width, _options.height,
                        1));
    CUDA_SAFE_SYNC();
}

void renderer_t::cleanup()
{
    CUDA_SAFE_CALL(cudaFree(reinterpret_cast<void*>(sbt.raygenRecord)));
    CUDA_SAFE_CALL(cudaFree(reinterpret_cast<void*>(sbt.missRecordBase)));

    OPTIX_SAFE_CALL(optixPipelineDestroy(_pipeline));
    OPTIX_SAFE_CALL(optixProgramGroupDestroy(_miss_group));
    OPTIX_SAFE_CALL(optixProgramGroupDestroy(_ray_generation_group));
    OPTIX_SAFE_CALL(optixModuleDestroy(_module));

    OPTIX_SAFE_CALL(optixDeviceContextDestroy(_context));
}
