cmake_minimum_required(VERSION 3.15)
project(point_pillars_cuda)

# set compile param
set(CMAKE_EXE_LINKER_FLAGS, "-Wl, --disable-new-dtags")
set(CMAKE_BUILD_TYPE Debug)
# set(CMAKE_BUILD_TYPE "Release")

add_definitions(-std=c++11)

set(THIRD_PARTY_LIBS "")
set(THIRD_PARTY_INCLUDE_DIR "")

find_package(roscpp REQUIRED)
list(APPEND THIRD_PARTY_INCLUDE_DIR ${roscpp_INCLUDE_DIRS})
list(APPEND THIRD_PARTY_LIBS ${roscpp_LIBRARIES})

find_package(yaml-cpp REQUIRED)
list(APPEND THIRD_PARTY_LIBS ${YAML_CPP_LIBRARIES})


# set(PCL_DIR "${CMAKE_SOURCE_DIR}/third_party/pcl-1.9.1/share/pcl-1.9/")    
find_package(PCL REQUIRED)
list(APPEND THIRD_PARTY_INCLUDE_DIR ${PCL_INCLUDE_DIRS})      
list(APPEND THIRD_PARTY_LIBS ${PCL_LIBRARIES})

# add CUDA, TENSORRT, INFERENCE
find_package(CUDA REQUIRED)
# set(TENSORRT_PATH "${CMAKE_SOURCE_DIR}/third_party/TensorRT-8.2.3.0")
set(TENSORRT_PATH "/root/xiaohu/TensorRT-8.4.3.1")
file(GLOB TENSORRT_LIBS "${TENSORRT_PATH}/lib/*")
if(CUDA_FOUND)
    # add includes
    list(APPEND THIRD_PARTY_INCLUDE_DIR ${CUDA_INCLUDE_DIRS})
    list(APPEND THIRD_PARTY_INCLUDE_DIR ${TENSORRT_PATH}/include)
    # add libs
    list(APPEND THIRD_PARTY_LIBS ${CUDA_LIBRARIES})
    list(APPEND THIRD_PARTY_LIBS ${TENSORRT_LIBS})
endif()

include_directories(
  ${THIRD_PARTY_INCLUDE_DIR} 
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/src
)

# config file path
set(ROOT_PATH "${PROJECT_SOURCE_DIR}")

# set config file
set(perception_config_h_in "${CMAKE_SOURCE_DIR}/include/perception_config.h.in")
set(perception_config_h "${CMAKE_SOURCE_DIR}/include/perception_config.h")
configure_file("${perception_config_h_in}" "${perception_config_h}")

set(FILE_DIR "")
list(APPEND FILE_DIR src)
list(APPEND FILE_DIR include)

set(ALL_SRCS "")
foreach(dir ${FILE_DIR})
    file(GLOB_RECURSE TEMP_SRCS "${dir}/*.cpp" "${dir}/*.cc" "${dir}/*.hpp" "${dir}/*.h" "${dir}/*.cu")
    list(APPEND ALL_SRCS ${TEMP_SRCS})
endforeach()

cuda_add_executable(${PROJECT_NAME} ${ALL_SRCS} test_point_pillars_cuda.cpp)
target_link_libraries(${PROJECT_NAME} ${THIRD_PARTY_LIBS})


set(ALL_SRCS2 "")
list(APPEND ALL_SRCS2 ${ALL_SRCS})
cuda_add_executable(${PROJECT_NAME}_ros ${ALL_SRCS2} test_point_pillars_cuda_ros.cpp)
target_link_libraries(${PROJECT_NAME}_ros ${THIRD_PARTY_LIBS})
