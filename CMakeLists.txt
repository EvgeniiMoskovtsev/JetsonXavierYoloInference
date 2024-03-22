cmake_minimum_required(VERSION 3.5)

project(Yolov8CPPInference VERSION 0.1)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

# CUDA
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")
find_package(CUDA 11 REQUIRED)

set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
# !CUDA

# OpenCV
set(CMAKE_PREFIX_PATH "/mnt/d/opencv-4.9.0/build")
set(OpenCV_DIR "/mnt/d/opencv-4.9.0/build")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
# !OpenCV

file(GLOB_RECURSE
        SOURCE_FILES
        LIST_DIRECTORIES true

        "${CMAKE_SOURCE_DIR}/include/yolo/*.h"
        "${CMAKE_SOURCE_DIR}/src/yolo/*.cpp"

        "${CMAKE_SOURCE_DIR}/include/ByteTrack/*.h"
        "${CMAKE_SOURCE_DIR}/src/ByteTrack/*.cpp"
        )


find_package(Eigen3 ${EIGEN_VERSION} QUIET)

add_executable(Yolov8CPPInference main.cpp ${SOURCE_FILES})
target_link_libraries(Yolov8CPPInference ${OpenCV_LIBS}  Eigen3::Eigen)
