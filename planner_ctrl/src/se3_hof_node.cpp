// 移植自 OpenDrone(Tfly6, Gen3) se3_hopf/src/se3_hopf_node.cpp
#include "planner_ctrl/se3_hof_ctrl.h"

int main(int argc, char **argv){

    ros::init(argc, argv, "se3_hof");
    ros::NodeHandle nh;
    ros::NodeHandle private_nh("~");

    Se3HofCtrl *se3_hof_node = new Se3HofCtrl(nh, private_nh);

    ros::spin();

    return 0;
}
