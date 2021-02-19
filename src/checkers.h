#pragma once

#include "logger.h"

#define OPTIX_SAFE_CALL(FUNC) \
{ \
    OptixResult res = FUNC; \
    if (res != OPTIX_SUCCESS) \
    { \
        LOG_ERROR("OptiX call failed [code: %d][line: %d]: %s\n", res, __LINE__, #FUNC); \
        exit(2); \
    } \
}

#define CUDA_SAFE_CALL(FUNC) \
{ \
    cudaError_t res = FUNC; \
    if (res != cudaSuccess) \
    { \
        std::string message = cudaGetErrorString(res); \
        LOG_ERROR("CUDA call failed [line: %d]: %s\n%s\n", __LINE__, #FUNC, message.c_str()); \
        throw std::runtime_error(message); \
    } \
}

#define CUDA_SAFE_SYNC() \
{ \
    cudaDeviceSynchronize(); \
    CUDA_SAFE_CALL(cudaGetLastError()) \
}
