/**
 * 移植自 OpenDrone(Tfly6, Gen3) se3_hopf/include/se3_hopf/se3_ctrl.h
 * SE3 几何控制器节点类：FSM(连接->OFFBOARD->起飞->任务->降落) + 100Hz 控制
 * Gen3 对齐点：订阅 /planner/output (planner_ctrl/PlannerOutput)、静态参数/动态重配置双模式、
 *             OFFBOARD 预热计数与请求限频、飞行状态广播 /flight_state、降落锁定
 * @author tfly (upstream)
 */

#ifndef PLANNER_CTRL_SE3_HOF_CTRL_H
#define PLANNER_CTRL_SE3_HOF_CTRL_H
#include <ros/ros.h>
#include <mavros_msgs/State.h>
#include <mavros_msgs/SetMode.h>
#include <mavros_msgs/CommandBool.h>
#include <mavros_msgs/AttitudeTarget.h>
#include <dynamic_reconfigure/server.h>
#include <geometry_msgs/PoseStamped.h>
#include <planner_ctrl/PlannerOutput.h>

#include "planner_ctrl/se3_hof.hpp"
#include "planner_ctrl/se3_hof_tuneConfig.h"
#include <std_msgs/Int8.h>
#include <std_srvs/SetBool.h>

using namespace std;

class Se3HofCtrl{
private:
    ros::NodeHandle nh_;
    ros::NodeHandle private_nh_;
    ros::Publisher cmd_pub_, local_pos_pub_, flight_state_pub_;
    ros::Subscriber odom_sub_, imu_sub_, state_sub_, plannerOutput_sub_;
    ros::ServiceClient set_mode_client_;
    ros::ServiceClient arming_client_;
    ros::ServiceServer land_service_;
    ros::Timer exec_timer_;

    mavros_msgs::State currState_;
    mavros_msgs::CommandBool arm_cmd;
    nav_msgs::Odometry desire_odom_;
    Odom_Data_t odom_data_;
    Imu_Data_t imu_data_;
    Desired_State_t desired_state_;
    SE3_HOF_CONTROLLER se3_hof_;

    bool arm_triggered_{false};
    bool offboard_triggered_{false};
    bool auto_takeoff_{false};
    // 起爬到位标志：TAKEOFF 完成(|z-takeoff_height|<0.1)置 true，plannerOutputCallback
    // 在此之前丢弃规划器航点；WAITING_FOR_OFFBOARD→TAKEOFF 复位
    bool takeoff_reached_{false};
    bool use_dynamic_reconfigure_{false};
    bool landing_locked_{false};
    bool enable_auto_offboard_{false};
    bool enable_auto_arm_{false};
    int offboard_warmup_counter_{0};
    int offboard_warmup_count_{80};
    double request_interval_{1.0};
    ros::Time last_mode_request_;
    ros::Time last_arm_request_;
    double msg_expire_time_{0.01};
    ros::Time last_planner_msg_time_;
    bool planner_msg_received_{false};
    double planner_timeout_{0.5};
    double planner_lost_land_time_{0.0};
    ros::Time planner_lost_enter_time_;
    ros::Time last_imu_msg_time_;
    ros::Time last_odom_msg_time_;
    bool imu_msg_received_{false};
    bool odom_msg_received_{false};
    double imu_timeout_{0.5};
    double odom_timeout_{0.5};
    double odom_vel_threshold_{3.0};
    string odom_topic_;  // 里程计来源（参数 odom_topic，默认 mavros；实机 fastlio /Odometry）
    ros::Time last_land_request_;
    double takeoff_height_;
    double takeoff_speed_{0.4};     // TAKEOFF 爬升率(m/s)
    double takeoff_start_z_{0.0};   // 进入 TAKEOFF 时的里程计高度(斜坡起点)
    ros::Time takeoff_start_time_;
    Eigen::Vector3d init_pose_, geo_fence_;;

    Eigen::Vector3d kp_p_, kp_v_, kp_a_, kp_q_, kp_w_, kd_p_, kd_v_, kd_a_, kd_q_, kd_w_;
    double limit_err_p_, limit_err_v_, limit_err_a_, limit_d_err_p_, limit_d_err_v_, limit_d_err_a_;
    double hover_percent_, max_hover_percent_;
    double idle_thrust_{0.15};  // OFFBOARD 预热/解锁接管前发送的油门，必须低于悬停油门
    bool enu_frame_, vel_in_body_;

    dynamic_reconfigure::Server<planner_ctrl::se3_hof_tuneConfig> dynamic_tune_server_;
    dynamic_reconfigure::Server<planner_ctrl::se3_hof_tuneConfig>::CallbackType dynamic_tune_cb_type_;

    enum FlightState { WAITING_FOR_CONNECTED, WAITING_FOR_OFFBOARD, TAKEOFF, MISSION_EXECUTION, LANDING, LANDED, EMERGENCY, PLANNER_LOST } flightState_, prev_flightState_;

    std::string state2string(FlightState state) {
        switch (state) {
        case WAITING_FOR_CONNECTED:
            return "WAITING_FOR_CONNECTED";
        case WAITING_FOR_OFFBOARD:
            return "WAITING_FOR_OFFBOARD";
        case MISSION_EXECUTION:
            return "MISSION_EXECUTION";
        case LANDING:
            return "LANDING";
        case LANDED:
            return "LANDED";
        case TAKEOFF:
            return "TAKEOFF";
        case EMERGENCY:
            return "EMERGENCY";
        case PLANNER_LOST:
            return "PLANNER_LOST";
        default:
            return "UNKNOWN_STATE";
        }
    }

    void execFSMCallback(const ros::TimerEvent &e);

    void send_cmd(const Controller_Output_t &output, bool angle);
    void pubLocalPose(const Eigen::Vector3d &pose);

    bool landCallback(std_srvs::SetBool::Request &request, std_srvs::SetBool::Response &response);
    void OdomCallback(const nav_msgs::Odometry::ConstPtr &msg);
    void IMUCallback(const sensor_msgs::Imu::ConstPtr &msg);
    void StateCallback(const mavros_msgs::State::ConstPtr &msg);
    void plannerOutputCallback(const planner_ctrl::PlannerOutput::ConstPtr &msg);
    bool msgExpired(const ros::Time &stamp, const std::string &what);
    void TrySetOffboard(const ros::Time &now);
    void TryArm(const ros::Time &now);
    void applyTuneConfig(const planner_ctrl::se3_hof_tuneConfig &config);
    void loadStaticTuneConfig();


    void DynamicTuneCallback(planner_ctrl::se3_hof_tuneConfig &config, uint32_t level){
        (void)level;
        ROS_INFO("kp_p: %f %f %f", config.kp_px, config.kp_py, config.kp_pz);
        ROS_INFO("kp_v: %f %f %f", config.kp_vx, config.kp_vy, config.kp_vz);
        ROS_INFO("kp_a: %f %f %f", config.kp_ax, config.kp_ay, config.kp_az);
        ROS_INFO("kp_q: %f %f %f", config.kp_qx, config.kp_qy, config.kp_qz);
        ROS_INFO("kp_w: %f %f %f", config.kp_wx, config.kp_wy, config.kp_wz);

        ROS_INFO("kd_p: %f %f %f", config.kd_px, config.kd_py, config.kd_pz);
        ROS_INFO("kd_v: %f %f %f", config.kd_vx, config.kd_vy, config.kd_vz);
        ROS_INFO("kd_a: %f %f %f", config.kd_ax, config.kd_ay, config.kd_az);
        ROS_INFO("kd_q: %f %f %f", config.kd_qx, config.kd_qy, config.kd_qz);
        ROS_INFO("kd_w: %f %f %f", config.kd_wx, config.kd_wy, config.kd_wz);

        ROS_INFO("limit err   p v a: %f %f %f", config.limit_err_p, config.limit_err_v, config.limit_err_a);
        ROS_INFO("limit d err p v a: %f %f %f", config.limit_d_err_p, config.limit_d_err_v, config.limit_d_err_a);

        applyTuneConfig(config);
        printf("\n");
    }

public:
    Se3HofCtrl(const ros::NodeHandle &nh, const ros::NodeHandle &private_nh);
    ~Se3HofCtrl(){};
};

#endif
