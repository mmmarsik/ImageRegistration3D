cmake_minimum_required(VERSION 3.16)

project(UIRAffineTransform LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

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

add_executable(uir_affine src/main.cpp)

target_compile_definitions(uir_affine PRIVATE UIR_PROJECT_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
target_include_directories(uir_affine PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(uir_affine PRIVATE ${OpenCV_LIBS})

message(STATUS "OpenCV version: ${OpenCV_VERSION}")
