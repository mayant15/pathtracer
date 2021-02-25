#include "renderer.h"
#include "module.h"

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
    _pipeline_options.numPayloadValues = 4;
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

static void create_raygen_record(OptixProgramGroup& pg, OptixShaderBindingTable& sbt)
{
    ray_gen_sbt_record_t record {};

    OPTIX_SAFE_CALL(optixSbtRecordPackHeader(pg, &record))

    unsafe::device_buffer_t buffer;
    buffer.allocate(sizeof(ray_gen_sbt_record_t));
    buffer.load_data(&record, buffer.size);

    sbt.raygenRecord = reinterpret_cast<CUdeviceptr>(buffer.data);
}

static void create_miss_record(OptixProgramGroup& pg, OptixShaderBindingTable& sbt)
{
    miss_sbt_record_t record {};
    record.data.background = { 135, 206, 235, 255 };

    OPTIX_SAFE_CALL(optixSbtRecordPackHeader(pg, &record))

    unsafe::device_buffer_t buffer;
    buffer.allocate(sizeof(miss_sbt_record_t));
    buffer.load_data(&record, buffer.size);

    sbt.missRecordBase = reinterpret_cast<CUdeviceptr>(buffer.data);
    sbt.missRecordStrideInBytes = sizeof(miss_sbt_record_t);
    sbt.missRecordCount = 1;
}

static void create_hitgroup_record(OptixProgramGroup& pg, OptixShaderBindingTable& sbt, const scene_t* scene)
{
    // Generate hitgroup records for all meshes
    std::vector<hitgroup_sbt_record_t> records(scene->meshes.size());
    for (size_t i = 0; i < records.size(); ++i)
    {
        records[i].data.color = { 255, 119, static_cast<unsigned char>(i * 120), 255 };
        records[i].data.vertices = reinterpret_cast<float3*>(scene->get_device_ptr());
        records[i].data.indices = reinterpret_cast<uint3*>(scene->meshes[i].get_device_ptr());
        OPTIX_SAFE_CALL(optixSbtRecordPackHeader(pg, &records[i]));
    }

    // Copy records to the device
    unsafe::device_buffer_t buffer;
    buffer.allocate(sizeof(hitgroup_sbt_record_t) * records.size());
    buffer.load_data(records.data(), buffer.size);

    // Load the SBT
    sbt.hitgroupRecordBase = reinterpret_cast<CUdeviceptr>(buffer.data);
    sbt.hitgroupRecordStrideInBytes = sizeof(hitgroup_sbt_record_t);
    sbt.hitgroupRecordCount = records.size();
}

void renderer_t::load_sbt()
{
    create_raygen_record(_raygen_pg, _sbt);
    create_miss_record(_miss_pg, _sbt);
    create_hitgroup_record(_hitgroup_pg, _sbt, _scene_ptr);
}

renderer_t::renderer_t(const render_options_t& opt)
        : _options(opt)
{
    init_context();

    init_module();
    init_programs();
    init_pipeline();
}

void renderer_t::build_accel()
{
    // We need a pointer to the CUdeviceptr
    auto d_vertex_tmp_ptr = _scene_ptr->get_device_ptr();
    const unsigned int flags[1] = { OPTIX_GEOMETRY_FLAG_NONE };

    // Prepare build inputs for all meshes
    std::vector<OptixBuildInput> inputs(_scene_ptr->meshes.size());
    for (size_t i = 0; i < inputs.size(); ++i)
    {
        inputs[i].type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

        // vertices
        inputs[i].triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
        inputs[i].triangleArray.vertexStrideInBytes = sizeof(float3);
        inputs[i].triangleArray.numVertices = _scene_ptr->vertices.size();
        inputs[i].triangleArray.vertexBuffers = &d_vertex_tmp_ptr;

        // indices
        inputs[i].triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
        inputs[i].triangleArray.indexStrideInBytes = sizeof(uint3);
        inputs[i].triangleArray.numIndexTriplets = _scene_ptr->meshes[i].indices.size();
        inputs[i].triangleArray.indexBuffer = _scene_ptr->meshes[i].get_device_ptr();

        // SBT offsets
        inputs[i].triangleArray.flags = flags;
        inputs[i].triangleArray.numSbtRecords = 1;
        // TODO: SBT index offsets here?
    }

    OptixAccelBuildOptions accel_options {};
    accel_options.buildFlags = OPTIX_BUILD_FLAG_NONE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

    // Estimate memory usage
    OptixAccelBufferSizes buffer_sizes;
    OPTIX_SAFE_CALL(optixAccelComputeMemoryUsage(
            _context,
            &accel_options,
            inputs.data(),
            inputs.size(),
            &buffer_sizes
    ));

    // Prepare compaction
    device_buffer_t compacted_size_buffer { sizeof(uint64_t) };
    OptixAccelEmitDesc emit;
    emit.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emit.result = compacted_size_buffer.data();

    // Allocate estimated memory
    device_buffer_t tmp { buffer_sizes.tempSizeInBytes };
    device_buffer_t uncompacted { buffer_sizes.outputSizeInBytes };

    // Build the structure
    OPTIX_SAFE_CALL(optixAccelBuild(_context, 0, &accel_options,
                                    inputs.data(), // Build inputs
                                    inputs.size(),
                                    tmp.data(),    // Temporary buffer
                                    tmp.size(),
                                    uncompacted.data(), // Output buffer
                                    uncompacted.size(),
                                    &_traversable_handle, &emit, 1
    ));
    CUDA_SAFE_SYNC();

    // Compaction
    uint64_t compacted_size;
    compacted_size_buffer.fetch(&compacted_size, sizeof(uint64_t));
    _gas.allocate(compacted_size);

    OPTIX_SAFE_CALL(
            optixAccelCompact(_context, 0, _traversable_handle, reinterpret_cast<CUdeviceptr>(_gas.data), _gas.size,
                              &_traversable_handle));
    CUDA_SAFE_SYNC();
}

void renderer_t::load_scene(const scene_t& scene)
{
    _scene_ptr = const_cast<scene_t*>(&scene);
    build_accel();
    load_sbt();
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
    params.handle = _traversable_handle;
    _scene_ptr->camera.set_params(params);

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

    _gas.free();

    OPTIX_SAFE_CALL(optixPipelineDestroy(_pipeline));
    OPTIX_SAFE_CALL(optixProgramGroupDestroy(_hitgroup_pg));
    OPTIX_SAFE_CALL(optixProgramGroupDestroy(_miss_pg));
    OPTIX_SAFE_CALL(optixProgramGroupDestroy(_raygen_pg));
    OPTIX_SAFE_CALL(optixModuleDestroy(_module));

    OPTIX_SAFE_CALL(optixDeviceContextDestroy(_context));
}
