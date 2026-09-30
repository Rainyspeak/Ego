

#include "path_manager/path_manager.h"

#include <ros/ros.h>

int main(int argc, char** argv) {
  ros::init(argc, argv, "path_manager_node");
  path_manager::PathManager manager;
  ros::spin();
  return 0;
}
