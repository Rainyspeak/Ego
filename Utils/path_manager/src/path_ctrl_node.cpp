// path_manager_node —— path_manager 的节点接口：只做入口，主逻辑在
// src/path_manager.cpp（类声明 include/path_manager/path_manager.h）。
// 参数全为私有参数，rosrun 直接传：_lookahead_distance:=4.0 等。

#include "path_manager/path_manager.h"

#include <ros/ros.h>

int main(int argc, char** argv) {
  ros::init(argc, argv, "path_manager_node");
  path_manager::PathManager manager;
  ros::spin();
  return 0;
}
