cmake_minimum_required(VERSION 3.16)

project(UIRAffineTransform LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(UIR_ENABLE_OPENMP "Enable OpenMP for UIR volume generation" ON)

if(APPLE)
    foreach(_opencv_prefix
        "/opt/homebrew/opt/opencv"
        "/usr/local/opt/opencv"
    )
        if(EXISTS "${_opencv_prefix}/lib/cmake/opencv4")
            list(APPEND CMAKE_PREFIX_PATH "${_opencv_prefix}")
        endif()
    endforeach()
endif()

find_package(OpenCV REQUIRED COMPONENTS core imgcodecs imgproc)

add_executable(uir_affine
    src/main.cpp
    src/synthetic_affine.cpp
)

target_compile_definitions(uir_affine PRIVATE UIR_PROJECT_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
target_include_directories(uir_affine PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(uir_affine PRIVATE ${OpenCV_LIBS})

if(UIR_ENABLE_OPENMP)
    find_package(OpenMP QUIET COMPONENTS CXX)

    if(OpenMP_CXX_FOUND)
        target_link_libraries(uir_affine PRIVATE OpenMP::OpenMP_CXX)
        message(STATUS "OpenMP for uir_affine: ${OpenMP_CXX_VERSION}")
    elseif(APPLE AND CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
        find_path(UIR_LIBOMP_INCLUDE_DIR
            NAMES omp.h
            PATHS
                "/opt/homebrew/opt/libomp/include"
                "/usr/local/opt/libomp/include"
            NO_DEFAULT_PATH
        )
        find_library(UIR_LIBOMP_LIBRARY
            NAMES omp
            PATHS
                "/opt/homebrew/opt/libomp/lib"
                "/usr/local/opt/libomp/lib"
            NO_DEFAULT_PATH
        )

        if(UIR_LIBOMP_INCLUDE_DIR AND UIR_LIBOMP_LIBRARY)
            target_compile_options(uir_affine PRIVATE -Xpreprocessor -fopenmp)
            target_include_directories(uir_affine PRIVATE "${UIR_LIBOMP_INCLUDE_DIR}")
            target_link_libraries(uir_affine PRIVATE "${UIR_LIBOMP_LIBRARY}")
            message(STATUS "OpenMP for uir_affine: AppleClang + ${UIR_LIBOMP_LIBRARY}")
        else()
            message(WARNING "OpenMP requested for uir_affine, but libomp was not found; building sequentially")
        endif()
    else()
        message(WARNING "OpenMP requested for uir_affine, but CMake did not find it; building sequentially")
    endif()
else()
    message(STATUS "OpenMP for uir_affine: disabled")
endif()

message(STATUS "OpenCV version: ${OpenCV_VERSION}")
