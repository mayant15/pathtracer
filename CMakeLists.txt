cmake_minimum_required(VERSION 3.17)
project(pathtracer LANGUAGES C CXX CUDA)
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

find_package(CUDA REQUIRED)

set(OPTIX_INCLUDE_DIRS
    external/optix/include
)
set(PATHTRACER_INCLUDE_DIRS
    external
    ${OPTIX_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
)

set(PATHTRACER_LINK_LIBS
    ${CUDA_LIBRARIES}
)

# Build shaders
set(PTX_PATH ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/ptx)
add_library(shaders OBJECT
    src/triangle.cu
)
set_property(TARGET shaders PROPERTY CUDA_PTX_COMPILATION ON)
target_include_directories(shaders PRIVATE ${OPTIX_INCLUDE_DIRS})

# Build the pathtracer
set(PATHTRACER_SOURCES
    external/stb_image_write.cpp
    external/tiny_obj_loader.cpp

    src/main.cpp
    src/renderer.cpp
)

# Main OptiX renderer
add_executable(pathtracer ${PATHTRACER_SOURCES})
add_dependencies(pathtracer shaders)
target_include_directories(pathtracer PRIVATE ${PATHTRACER_INCLUDE_DIRS})
target_link_libraries(pathtracer PRIVATE ${PATHTRACER_LINK_LIBS})
target_compile_definitions(pathtracer PRIVATE PTX_PATH=\"${PTX_PATH}/\")

make_directory(${PTX_PATH})
add_custom_command(TARGET pathtracer
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_OBJECTS:shaders> ${PTX_PATH}
    COMMENT "Copying shaders..."
)

# Reference implementation of smallpt
add_subdirectory(smallpt)
