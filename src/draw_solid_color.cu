#include <optix.h>
#include "launch_params.h"

// no name mangling
extern "C" __constant__ params_t params;

// no name mangling
extern "C" __global__ void __raygen__draw() // NOLINT(bugprone-reserved-identifier)
{
    uint3 launch_index = optixGetLaunchIndex();
    ray_gen_data_t* color = (ray_gen_data_t*) optixGetSbtDataPointer();
    params.image[launch_index.y * params.image_width + launch_index.x] = *color;
}
