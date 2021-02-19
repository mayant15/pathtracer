/*
 * Copyright (c) 2020 NVIDIA Corporation.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/// @file
/// @author NVIDIA Corporation
/// @brief  OptiX public API header

#ifndef optix_denoiser_tiling_h
#define optix_denoiser_tiling_h


#include <optix.h>

#include <algorithm>
#include <vector>

#ifdef __cplusplus
extern "C" {
#endif

/** \addtogroup optix_utilities
@{
*/

/// Tile definition
///
/// see #optixUtilDenoiserSplitImage
///
struct OptixUtilDenoiserImageTile
{
    // input tile image
    OptixImage2D input;

    // output tile image
    OptixImage2D output;

    // overlap offsets, parameters for #optixUtilDenoiserInvoke
    unsigned int inputOffsetX;
    unsigned int inputOffsetY;
};

/// Return pixel stride in bytes for the given pixel format
/// if the pixelStrideInBytes member of the image is zero.
/// Otherwise return pixelStrideInBytes from the image.
///
/// \param[in]                  image Image containing the pixel stride
///
inline unsigned int optixUtilGetPixelStride( const OptixImage2D& image )
{
    unsigned int pixelStrideInBytes = image.pixelStrideInBytes;
    if( pixelStrideInBytes == 0 )
    {
        switch( image.format )
        {
            case OPTIX_PIXEL_FORMAT_HALF3:
                pixelStrideInBytes = 3 * sizeof( short );
                break;
            case OPTIX_PIXEL_FORMAT_HALF4:
                pixelStrideInBytes = 4 * sizeof( short );
                break;
            case OPTIX_PIXEL_FORMAT_FLOAT3:
                pixelStrideInBytes = 3 * sizeof( float );
                break;
            case OPTIX_PIXEL_FORMAT_FLOAT4:
                pixelStrideInBytes = 4 * sizeof( float );
                break;
            case OPTIX_PIXEL_FORMAT_UCHAR3:
                pixelStrideInBytes = 3 * sizeof( char );
                break;
            case OPTIX_PIXEL_FORMAT_UCHAR4:
                pixelStrideInBytes = 4 * sizeof( char );
                break;
        }
    }
    return pixelStrideInBytes;
}

/// Split image into 2D tiles given horizontal and vertical tile size
///
/// \param[in]  input            full resolution input image to be split
/// \param[in]  output           full resolution output image
/// \param[in]  overlapWindowSizeInPixels    see #OptixDenoiserSizes, #optixDenoiserComputeMemoryResources
/// \param[in]  tileWidth        maximum width of tiles
/// \param[in]  tileHeight       maximum height of tiles
/// \param[out] tiles            list of tiles covering the input image
///
inline OptixResult optixUtilDenoiserSplitImage(
                                               const OptixImage2D&                     input,
                                               const OptixImage2D&                     output,
                                               unsigned int                            overlapWindowSizeInPixels,
                                               unsigned int                            tileWidth,
                                               unsigned int                            tileHeight,
                                               std::vector<OptixUtilDenoiserImageTile>&    tiles )
{
    if( tileWidth == 0 || tileHeight == 0 )
        return OPTIX_ERROR_INVALID_VALUE;

    unsigned int inPixelStride  = optixUtilGetPixelStride( input );
    unsigned int outPixelStride = optixUtilGetPixelStride( output );

    int inp_w = std::min( tileWidth + 2 * overlapWindowSizeInPixels, input.width );
    int inp_h = std::min( tileHeight + 2 * overlapWindowSizeInPixels, input.height );
    int inp_y = 0, copied_y = 0;

    do
    {
        int inputOffsetY = inp_y == 0 ? 0 : std::max( (int)overlapWindowSizeInPixels, inp_h - ( (int)input.height - inp_y ) );
        int copy_y       = inp_y == 0 ? std::min( input.height, tileHeight + overlapWindowSizeInPixels ) :
                                  std::min( tileHeight, input.height - copied_y );

        int inp_x = 0, copied_x = 0;
        do
        {
            int inputOffsetX = inp_x == 0 ? 0 : std::max( (int)overlapWindowSizeInPixels, inp_w - ( (int)input.width - inp_x ) );
            int copy_x = inp_x == 0 ? std::min( input.width, tileWidth + overlapWindowSizeInPixels ) :
                                      std::min( tileWidth, input.width - copied_x );

            OptixUtilDenoiserImageTile tile;
            tile.input.data               = input.data + ( inp_y - inputOffsetY ) * input.rowStrideInBytes
                                            + ( inp_x - inputOffsetX ) * inPixelStride;
            tile.input.width              = inp_w;
            tile.input.height             = inp_h;
            tile.input.rowStrideInBytes   = input.rowStrideInBytes;
            tile.input.pixelStrideInBytes = input.pixelStrideInBytes;
            tile.input.format             = input.format;

            tile.output.data               = output.data + inp_y * output.rowStrideInBytes + inp_x * outPixelStride;
            tile.output.width              = copy_x;
            tile.output.height             = copy_y;
            tile.output.rowStrideInBytes   = output.rowStrideInBytes;
            tile.output.pixelStrideInBytes = output.pixelStrideInBytes;
            tile.output.format             = output.format;

            tile.inputOffsetX = inputOffsetX;
            tile.inputOffsetY = inputOffsetY;
            tiles.push_back( tile );

            inp_x += inp_x == 0 ? tileWidth + overlapWindowSizeInPixels : tileWidth;
            copied_x += copy_x;
        } while( inp_x < static_cast<int>( input.width ) );

        inp_y += inp_y == 0 ? tileHeight + overlapWindowSizeInPixels : tileHeight;
        copied_y += copy_y;
    } while( inp_y < static_cast<int>( input.height ) );

    return OPTIX_SUCCESS;
}

/// Run denoiser on input layers
/// see #optixDenoiserInvoke
/// additional parameters:

/// Runs the denoiser on the input layers on a single GPU and stream using #optixDenoiserInvoke.
/// If the input layers' dimensions are larger than the specified tile size, the image is divided into
/// tiles using #optixUtilDenoiserSplitImage, and multiple back-to-back invocations are performed in
/// order to reuse the scratch space.  Multiple tiles can be invoked concurrently if
/// #optixUtilDenoiserSplitImage is used directly and multiple scratch allocations for each concurrent
/// invocation are used.

/// The input parameters are the same as #optixDenoiserInvoke except for the addition of the maximum tile size.
///
/// \param[in] denoiser
/// \param[in] stream
/// \param[in] params
/// \param[in] denoiserState
/// \param[in] denoiserStateSizeInBytes
/// \param[in] inputLayers
/// \param[in] numInputLayers
/// \param[in] outputLayer
/// \param[in] scratch
/// \param[in] scratchSizeInBytes
/// \param[in] overlapWindowSizeInPixels
/// \param[in] tileWidth
/// \param[in] tileHeight
inline OptixResult optixUtilDenoiserInvokeTiled(
                                                OptixDenoiser&             denoiser,
                                                CUstream                   stream,
                                                const OptixDenoiserParams* params,
                                                CUdeviceptr                denoiserState,
                                                size_t                     denoiserStateSizeInBytes,
                                                const OptixImage2D*        inputLayers,
                                                unsigned int               numInputLayers,
                                                const OptixImage2D*        outputLayer,
                                                CUdeviceptr                scratch,
                                                size_t                     scratchSizeInBytes,
                                                unsigned int               overlapWindowSizeInPixels,
                                                unsigned int               tileWidth,
                                                unsigned int               tileHeight )
{
    if( !inputLayers || !outputLayer )
        return OPTIX_ERROR_INVALID_VALUE;

    std::vector<std::vector<OptixUtilDenoiserImageTile>> tiles( numInputLayers );
    for( unsigned int l = 0; l < numInputLayers; l++ )
        if( const OptixResult res = optixUtilDenoiserSplitImage( inputLayers[l], *outputLayer, overlapWindowSizeInPixels,
                                                                 tileWidth, tileHeight, tiles[l] ) )
            return res;

    for( size_t t = 0; t < tiles[0].size(); t++ )
    {
        std::vector<OptixImage2D> tlayers;
        for( int l = 0; l < static_cast<int>( numInputLayers ); l++ )
            tlayers.push_back( ( tiles[l] )[t].input );

        if( const OptixResult res =
                optixDenoiserInvoke( denoiser, stream, params, denoiserState, denoiserStateSizeInBytes, &tlayers[0],
                                     numInputLayers, ( tiles[0] )[t].inputOffsetX, ( tiles[0] )[t].inputOffsetY,
                                     &( tiles[0] )[t].output, scratch, scratchSizeInBytes ) )
            return res;
    }
    return OPTIX_SUCCESS;
}

/*@}*/  // end group optix_utilities

#ifdef __cplusplus
}
#endif

#endif  // __optix_optix_stack_size_h__
