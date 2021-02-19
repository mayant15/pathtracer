#pragma once

#include "logger.h"

#define OPTIX_SAFE_CALL(func) \
{ \
    OptixResult res = func; \
    if (res != OPTIX_SUCCESS) \
    { \
        LOG_ERROR("Optix call (%s) failed with code %d (line %d)\n", #func, res, __LINE__); \
        exit(2); \
    } \
}
