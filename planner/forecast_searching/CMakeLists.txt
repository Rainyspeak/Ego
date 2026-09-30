cmake_minimum_required(VERSION 3.0.2)
project(forecast_searching)

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

# ROS 节点（薄壳，核心算法在 include/forecast_searching/frame_detector.h）
# 仿真稀疏云版（forecast_searching_sim.launch 用）
add_executable(forecast_searching_sim src/forecast_searching_sim.cpp)
target_link_libraries(forecast_searching_sim ${catkin_LIBRARIES})

# 实机稠密点云版（real.launch 用，mid360 /cloud_registered：入口预筛 + 稠密参数档，
# 话题/服务/参数名与仿真版完全一致）
add_executable(forecast_searching_real src/forecast_searching_real.cpp)
target_link_libraries(forecast_searching_real ${catkin_LIBRARIES})

# 离线单测（无 ROS 依赖，也可直接 g++ 编译运行）
add_executable(frame_detector_test test/test_frame_detector.cpp)
