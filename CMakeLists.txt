cmake_minimum_required(VERSION 3.17)
project(pathtracer LANGUAGES C CXX CUDA)
set(CMAKE_CXX_STANDARD 17)

find_package(CUDA REQUIRED)

set(OPTIX_INCLUDE_DIRS
    external/optix/include
)

set(PATHTRACER_INCLUDE_DIRS
    ${OPTIX_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
)

set(PATHTRACER_LINK_LIBS
    ${CUDA_LIBRARIES}
)

# Build the pathtracer
set(PATHTRACER_SOURCES
    src/main.cpp
)

add_executable(pathtracer ${PATHTRACER_SOURCES})
target_include_directories(pathtracer PRIVATE ${PATHTRACER_INCLUDE_DIRS})
target_link_libraries(pathtracer PRIVATE ${PATHTRACER_LINK_LIBS})
