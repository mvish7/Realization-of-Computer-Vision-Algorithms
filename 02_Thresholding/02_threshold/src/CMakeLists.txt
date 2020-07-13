cmake_minimum_required(VERSION 2.8.10)

# set project name
project(threshold)

# set compile flags
set(CMAKE_CXX_FLAGS "-std=c++0x")
set(CMAKE_BUILD_TYPE "Release")

# find libraries
find_package(OpenCV REQUIRED)

# set include directories
include_directories(${OpenCV_INCLUDE_DIR})

# make executable
add_executable(${PROJECT_NAME} main.cpp)

# link against libraries
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
