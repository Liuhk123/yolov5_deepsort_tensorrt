cmake_minimum_required(VERSION 2.6)

add_definitions(-std=c++11)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

find_package(CUDA REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/../include)
# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
# cuda
include_directories(/usr/local/cuda-11.1/include)
link_directories(/usr/local/cuda-11.1/lib64)
# tensorrt
include_directories(/home/source_code/TensorRT-7.2.2.3/include/)
link_directories(/home/source_code/TensorRT-7.2.2.3/lib/)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")

cuda_add_library(myplugins SHARED yololayer.cu)
target_link_libraries(myplugins nvinfer cudart)

set(OpenCV_DIR /home/xuyufeng/projects/sources/opencv_build/share/OpenCV/)
find_package(OpenCV)
include_directories(OpenCV_INCLUDE_DIRS)
#include_directories(/home/xuyufeng/projects/sources/opencv_build/include)
#link_directories(/home/xuyufeng/projects/sources/opencv_build/lib64)
#set(OpenCV_LIBS opencv_calib3d;opencv_core;opencv_dnn;opencv_features2d;opencv_flann;opencv_highgui;opencv_imgcodecs;opencv_imgproc;opencv_ml;opencv_objdetect;opencv_photo;opencv_shape;opencv_stitching;opencv_superres;opencv_video;opencv_videoio;opencv_videostab)

add_library(yolov5 yolov5.cpp yolov5.h Loader.cpp)

target_link_libraries(yolov5 nvinfer)
target_link_libraries(yolov5 cudart)
target_link_libraries(yolov5 myplugins)
target_link_libraries(yolov5 nppig)
target_link_libraries(yolov5 nppicc)
target_link_libraries(yolov5 nppial)
target_link_libraries(yolov5 nppim)
target_link_libraries(yolov5 nppidei)
target_link_libraries(yolov5 ${OpenCV_LIBS})


add_definitions(-O2 -pthread)



