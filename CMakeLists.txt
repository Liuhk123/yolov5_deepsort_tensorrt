# CMakeList.txt: 顶层 CMake 项目文件，在此处执行全局配置
# 并包含子项目。
#
cmake_minimum_required (VERSION 2.6)

add_definitions(-std=c++11)

project (TrackCounter)
# 将当前目录所有源文件保存在变量DIR_SRCS中
aux_source_directory(. DIR_SRCS)

# 包含子项目。
add_subdirectory ("yolov5")
add_subdirectory("DeepSORT")

find_package(CUDA REQUIRED)

include_directories(/usr/local/include/eigen3)
# cuda
include_directories(/usr/local/cuda-11.1/include)
link_directories(/usr/local/cuda-11.1/lib64)
# tensorrt
include_directories(/home/source_code/TensorRT-7.2.2.3/include/)
link_directories(/home/source_code/TensorRT-7.2.2.3/lib/)


add_executable(TrackCounter ${DIR_SRCS} main.cpp DeepSORT/track_deepsort.cpp DeepSORT/track_deepsort.h)
# 添加链接库，MyMath是子目录中的链接库名
target_link_libraries(TrackCounter yolov5)
target_link_libraries(TrackCounter deepsort)

find_package(CURL)
if(CURL_FOUND)
 include_directories(${CURL_INCLUDE_DIR})
 target_link_libraries(TrackCounter ${CURL_LIBRARY})
else(CURL_FOUND)
 message(FATAL_ERROR "CURL library not found")
endif(CURL_FOUND)

include_directories(${PROJECT_SOURCE_DIR}/include)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)


add_definitions(-O2 -pthread)


