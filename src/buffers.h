#pragma once

#include <cuda_runtime.h>
#include <cassert>

class device_buffer_t
{
    void* _pointer;
    size_t _size;

public:
    /*!
     * Create a device buffer
     * @param size Size (in bytes) for the buffer
     * @param data (Optional) Data to copy to the device
     */
    explicit device_buffer_t(size_t size, void* data = nullptr)
            : _size(size),
              _pointer(nullptr)
    {
        cudaMalloc(&_pointer, _size);
        if (data != nullptr)
        {
            cudaMemcpy(_pointer, data, _size, cudaMemcpyHostToDevice);
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
        cudaMemcpy(dst, _pointer, size, cudaMemcpyDeviceToHost);
    }

    [[nodiscard]] size_t size() const
    { return _size; }

    [[nodiscard]] CUdeviceptr data() const
    { return (CUdeviceptr) _pointer; }

    ~device_buffer_t()
    {
        cudaFree(_pointer);
    }
};
