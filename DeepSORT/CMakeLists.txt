cmake_minimum_required(VERSION 2.6)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall")
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

# Try to find OpenCV
set(OpenCV_DIR /home/xuyufeng/projects/sources/opencv_build/share/OpenCV/)
find_package(OpenCV REQUIRED)
if (OpenCV_FOUND)
    # If the package has been found, several variables will
    # be set, you can find the full list with descriptions
    # in the OpenCVConfig.cmake file.
    # Print some message showing some of them
    message(STATUS "OpenCV library status:")
    message(STATUS "    version: ${OpenCV_VERSION}")
    message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
else ()
    message(FATAL_ERROR "Could not locate OpenCV")
endif()

include_directories(${PROJECT_SOURCE_DIR}/KalmanFilter ./DeepAppearanceDescriptor ./MunkresAssignment ./MunkresAssignment/munkres /usr/local/include/eigen3/)

file(GLOB SOURCE_FILES "./KalmanFilter/*.cpp" "./DeepAppearanceDescriptor/*.cpp" "./MunkresAssignment/*.cpp" "./MunkresAssignment/munkres/*.cpp")

message(STATUS "source files ${SOURCE_FILES}")
add_library(deepsort ${SOURCE_FILES})

target_link_libraries(deepsort /usr/lib64/libuuid.so.1)
target_link_libraries (deepsort ${OpenCV_LIBS})


add_definitions(-O2 -pthread)
