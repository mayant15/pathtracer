#pragma once

#include <cuda_runtime.h>
#include <cassert>
#include "checkers.h"

class device_buffer_t
{
    void* _pointer;
    size_t _size;
    bool _persists;

public:
    /*!
     * Create a device buffer
     * @param size Size (in bytes) for the buffer
     * @param data (Optional) Data to copy to the device
     */
    explicit device_buffer_t(size_t size, void* data = nullptr, bool persists = false)
            : _size(size),
              _pointer(nullptr),
              _persists(persists)
    {
        cudaMalloc(&_pointer, _size);
        if (data != nullptr)
        {
            CUDA_SAFE_CALL(cudaMemcpy(_pointer, data, _size, cudaMemcpyHostToDevice));
        }
    }

    /*!
     * Copy the contents of the buffer back to the host
     * @param dst  Destination host pointer
     * @param size Size (in bytes) of the data to copy. Cannot be more than the size of the buffer.
     */
    void fetch(void* dst, size_t size)
    {
        assert(_pointer != nullptr);
        assert(size <= _size);
        CUDA_SAFE_CALL(cudaMemcpy(dst, _pointer, size, cudaMemcpyDeviceToHost));
    }

    [[nodiscard]] size_t size() const
    { return _size; }

    [[nodiscard]] CUdeviceptr data() const
    { return (CUdeviceptr) _pointer; }

    ~device_buffer_t()
    {
        if (!_persists)
        {
            try
            {
                CUDA_SAFE_CALL(cudaFree(_pointer));
            }
            catch (const std::exception& e)
            {
                LOG_ERROR("Failed to free resources: %s\n", e.what());
            }
        }
    }
};
