#pragma once

#include <cuda_runtime.h>
#include <cassert>
#include "checkers.h"

namespace unsafe
{
    struct unmanaged_resource_t
    {
        void* data = nullptr;
        size_t size = 0;

        virtual void allocate(size_t size) = 0;
        virtual void free() = 0;
    };

    /*!
     * Alright, so you want to manage this memory yourself? Here you go.
     */
    class device_buffer_t : public unmanaged_resource_t
    {
    public:
        void allocate(size_t size_in_bytes) override
        {
            CUDA_SAFE_CALL(cudaMalloc(&data, size_in_bytes));
            size = size_in_bytes;
        }

        void load_data(const void* src, size_t size_in_bytes)
        {
            CUDA_SAFE_CALL(cudaMemcpy(data, src, size_in_bytes, cudaMemcpyHostToDevice));
        }

        void fetch_data(void* dst, size_t size_in_bytes)
        {
            CUDA_SAFE_CALL(cudaMemcpy(dst, data, size_in_bytes, cudaMemcpyDeviceToHost));
        }

        void free() override
        {
            CUDA_SAFE_CALL(cudaFree(data));
            size = 0;
        }
    };
}

class device_buffer_t
{
    unsafe::device_buffer_t _data;

public:

    /*!
     * Create a new managed device buffer
     * @param size Size of the buffer in bytes
     * @param data Data to load into the buffer, if any
     */
    explicit device_buffer_t(size_t size, void* data = nullptr)
    {
        // Allocate underlying storage
        assert(size != 0);
        _data.allocate(size);

        // Load data if provided
        if (data != nullptr)
        {
            _data.load_data(data, size);
        }
    }

    /*!
     * Copy data to the device
     * @param src           Source to copy contents from
     * @param size_in_bytes Number of bytes to copy, must be lesser than the length of the buffer
     */
    void load(const void* src, size_t size_in_bytes)
    {
        assert(size_in_bytes <= _data.size);
        _data.load_data(src, size_in_bytes);
    }

    /*!
     * Copy data to the host
     * @param dst           Destination for buffer contents
     * @param size_in_bytes Number of bytes to copy, must be lesser than the length of the buffer
     */
    void fetch(void* dst, size_t size_in_bytes)
    {
        assert(_data.data != nullptr);
        assert(size_in_bytes <= _data.size);
        _data.fetch_data(dst, size_in_bytes);
    }

    /*!
     * Fetch the size of the device buffer
     * @return size_t
     */
    [[nodiscard]] size_t size() const
    { return _data.size; }

    /*!
     * Get a pointer to the allocated device memory
     * @return CUdeviceptr
     */
    [[nodiscard]] CUdeviceptr data() const
    { return reinterpret_cast<CUdeviceptr>(_data.data); }

    ~device_buffer_t()
    {
        try
        {
            _data.free();
        }
        catch (const std::exception& e)
        {
            LOG_ERROR("Failed to free resources: %s\n", e.what());
        }
    }
};
