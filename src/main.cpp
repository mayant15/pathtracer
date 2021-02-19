#include "types.h"
#include "checkers.h"

#include <stdexcept>

void init_optix()
{
    cudaFree(nullptr);
    int numDevices;
    cudaGetDeviceCount(&numDevices);

    if (numDevices == 0)
    {
        throw std::runtime_error("No CUDA capable devices found!");
    }

    LOG_INFO("Found %d CUDA device(s)\n", numDevices);
    OPTIX_SAFE_CALL(optixInit());
}

int main(int ac, char** av)
{
    try
    {
        LOG_INFO("Initializing OptiX\n");

        init_optix();

        LOG_INFO("Success\n");
        LOG_INFO("Clean exit\n");

    } catch (std::runtime_error& e)
    {
        LOG_ERROR("ERROR: %s\n", e.what());
        exit(1);
    }
    return 0;
}
