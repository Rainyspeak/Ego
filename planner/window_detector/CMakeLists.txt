cmake_minimum_required(VERSION 3.0.2)
project(window_detector)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_compile_options(-Wall)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  sensor_msgs
  nav_msgs
  geometry_msgs
  visualization_msgs
  std_msgs
  std_srvs
)

catkin_package()

include_directories(include ${catkin_INCLUDE_DIRS})

# ROS 节点（薄壳，核心算法在 include/window_detector/frame_detector.h）
add_executable(window_detector_node src/window_detector_node.cpp)
target_link_libraries(window_detector_node ${catkin_LIBRARIES})

# 离线单测（无 ROS 依赖，也可直接 g++ 编译运行）
add_executable(frame_detector_test test/test_frame_detector.cpp)
