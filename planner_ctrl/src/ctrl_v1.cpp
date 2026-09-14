#include <ros/ros.h>
#include <algorithm>
#include <cmath>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Twist.h>
#include <sensor_msgs/Joy.h>
#include <mavros_msgs/CommandBool.h>
#include <mavros_msgs/SetMode.h>
#include <mavros_msgs/State.h>
#include <mavros_msgs/PositionTarget.h>
#include "quadrotor_msgs/PositionCommand.h"
#include<nav_msgs/Odometry.h>
#include <tf/transform_datatypes.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include "planner_ctrl/Pidparam.hpp"



#define POSITION_CONTROL 0b100111111000   //位置起飞：使用PX/PY/PZ/YAW
#define PLANNER_CONTROL 0b100111000000 //轨迹跟踪：使用位置、速度和YAW（加速度位被忽略）

unsigned short position_mask = POSITION_CONTROL;

float takeoff_height = 1.5f; //全局起飞高度（米）
mavros_msgs::PositionTarget current_goal;
nav_msgs::Odometry position_msg;
//geometry_msgs::PoseStamped target_pos;
mavros_msgs::State current_state;
mavros_msgs::PositionTarget target_pos;


float waypoint_position_tolerance = 0.15f;
float position_x, position_y, position_z, current_yaw, targetpos_x, targetpos_y;
float current_vel_x, current_vel_y, current_vel_z;
// bool target_received = false;
// bool waypoint_hold = false;
float hold_position_x, hold_position_y, hold_position_z, hold_yaw;
bool odom_received = false;
float ego_pos_x, ego_pos_y, ego_pos_z, ego_vel_x, ego_vel_y, ego_vel_z, ego_a_x, ego_a_y, ego_a_z, ego_yaw, ego_yaw_rate; //EGO planner information has position velocity acceleration yaw yaw_dot
bool receive = false;//触发轨迹的条件判断
float pi = 3.14159265;

// Velocity-loop PID parameters are loaded from the node's private namespace.
// The planner velocity is used as feed-forward and the odometry velocity is
// used for feedback.  The resulting command is limited by pid_output_limit.
bool pid_enabled = planner_ctrl::PidParam::kEnabled;
double pid_kp = planner_ctrl::PidParam::kKp;
double pid_ki = planner_ctrl::PidParam::kKi;
double pid_kd = planner_ctrl::PidParam::kKd;
double pid_integral_limit = planner_ctrl::PidParam::kIntegralLimit;
double pid_output_limit = planner_ctrl::PidParam::kOutputLimit;

struct Speed_limit
{
  static constexpr double kControlRate = 50.0;
  static constexpr double kSpeedLimit = 1.5;

  static void limitVelocityNorm(double &vx, double &vy, double &vz, double max_speed)
  {
    //速度限制
    const double speed = std::sqrt(vx * vx + vy * vy + vz * vz);

    if (max_speed > 0.0 && speed > max_speed)
    {
      const double scale = max_speed / speed;
      vx *= scale;
      vy *= scale;
      vz *= scale;
    }
  }
};
class VelocityPid
{
public:
  VelocityPid() = default;

  double integral_x = 0.0;
  double integral_y = 0.0;
  double integral_z = 0.0;
  double previous_error_x = 0.0;
  double previous_error_y = 0.0;
  double previous_error_z = 0.0;
  ros::Time last_update;
  void reset()
  {
    integral_x = integral_y = integral_z = 0.0;
    previous_error_x = previous_error_y = previous_error_z = 0.0;
    last_update = ros::Time(0);
  }
  double update(double desired, double measured,double &integral, double &previous_error,const ros::Time &now);
  void updateTime(const ros::Time &now);

};
double VelocityPid::update(double desired, double measured,double &integral, double &previous_error,const ros::Time &now)
{
    const double error = desired - measured;
    double dt = 1.0 / Speed_limit::kControlRate;
    if (!last_update.isZero())
      dt = (now - last_update).toSec();
    else
      previous_error = error;

    // Ignore a long callback gap so it cannot create a derivative/integral
    // spike after startup or when ROS is paused.
    if (dt <= 0.0 || dt > 0.5)
    {
      dt = 1.0 / Speed_limit::kControlRate;
      previous_error = error;
    }

    integral += error * dt;
    integral = std::max(-pid_integral_limit,
                        std::min(pid_integral_limit, integral));
    const double derivative = (error - previous_error) / dt;
    previous_error = error;
    return desired + pid_kp * error + pid_ki * integral + pid_kd * derivative;
}
void VelocityPid::updateTime(const ros::Time &now)
{
  last_update = now;
}

// struct VelocityPid
// {
//   double integral_x = 0.0;
//   double integral_y = 0.0;
//   double integral_z = 0.0;
//   double previous_error_x = 0.0;
//   double previous_error_y = 0.0;
//   double previous_error_z = 0.0;
//   ros::Time last_update;

//   void reset()
//   {
//     integral_x = integral_y = integral_z = 0.0;
//     previous_error_x = previous_error_y = previous_error_z = 0.0;
//     last_update = ros::Time(0);
//   }

//   double update(double desired, double measured, double &integral,
//                 double &previous_error, const ros::Time &now)
//   {
//     const double error = desired - measured;
//     double dt = 1.0 / Speed_limit::kControlRate;
//     if (!last_update.isZero())
//       dt = (now - last_update).toSec();
//     else
//       previous_error = error;

//     // Ignore a long callback gap so it cannot create a derivative/integral
//     // spike after startup or when ROS is paused.
//     if (dt <= 0.0 || dt > 0.5)
//     {
//       dt = 1.0 / Speed_limit::kControlRate;
//       previous_error = error;
//     }

//     integral += error * dt;
//     integral = std::max(-pid_integral_limit,
//                         std::min(pid_integral_limit, integral));
//     const double derivative = (error - previous_error) / dt;
//     previous_error = error;
//     return desired + pid_kp * error + pid_ki * integral + pid_kd * derivative;
//   }

//   void updateTime(const ros::Time &now) { last_update = now; }
// };

VelocityPid velocity_pid;

void state_cb(const mavros_msgs::State::ConstPtr& msg){
	current_state = *msg;
}

//read vehicle odometry
void position_cb(const nav_msgs::Odometry::ConstPtr& msg)
{
	position_msg=*msg;
	position_x = position_msg.pose.pose.position.x;
	position_y = position_msg.pose.pose.position.y;
	position_z = position_msg.pose.pose.position.z;
	current_vel_x = position_msg.twist.twist.linear.x;
	current_vel_y = position_msg.twist.twist.linear.y;
	current_vel_z = position_msg.twist.twist.linear.z;

	odom_received = true;
	//四元函数的计算
	tf::Quaternion quat;
	tf::quaternionMsgToTF(msg->pose.pose.orientation, quat);
	double roll,pitch,yaw;
  tf::Matrix3x3(quat).getRPY(roll,pitch,yaw);
	current_yaw = yaw;
}
mavros_msgs::PositionTarget pose(const double& x, const double& y, const double& z,const double& yaw)
{
  target_pos.coordinate_frame = mavros_msgs::PositionTarget::FRAME_LOCAL_NED;
  target_pos.header.stamp = ros::Time::now();
  target_pos.type_mask = position_mask;
  target_pos.position.x = x;
  target_pos.position.y = y;
  target_pos.position.z = z;
  target_pos.yaw = yaw;
  return target_pos;
}

//航点读取
// void target_cb(const geometry_msgs::PoseStamped::ConstPtr& msg)
// {
//   target_pos = *msg;
//   targetpos_x = target_pos.pose.position.x;
//   targetpos_y = target_pos.pose.position.y;
//   target_received = true;
//   receive = false;
//   waypoint_hold = true;
//   ROS_INFO("Received RViz waypoint: (%.2f, %.2f)", targetpos_x, targetpos_y);
// }

quadrotor_msgs::PositionCommand ego;
void twist_planner_cb(const quadrotor_msgs::PositionCommand::ConstPtr& msg)//ego的回调函数
{

    receive = true;
	  ego = *msg;
    ego_pos_x = ego.position.x;
    ego_pos_y = ego.position.y;
    ego_pos_z = ego.position.z;
    ego_vel_x = ego.velocity.x;
    ego_vel_y = ego.velocity.y;
    ego_vel_z = ego.velocity.z;
    ego_a_x = ego.acceleration.x;
    ego_a_y = ego.acceleration.y;
    ego_a_z = ego.acceleration.z;
    ego_yaw = ego.yaw;
    ego_yaw_rate = ego.yaw_dot;
}

//判断目标是否到达
// bool Arrival_State()
// {
//   if (!target_received || !receive || !odom_received)
//   {
//     return false;
//   }

//   //xy误差
//   const double error_x = targetpos_x - position_x;
//   const double error_y = targetpos_y - position_y;
//   const double error_z = takeoff_height - position_z;
//   const double position_error = std::sqrt(error_x * error_x + error_y * error_y + error_z * error_z);
//   const bool position_ok = position_error <= waypoint_position_tolerance;

//   if (position_ok)
//   {
//       hold_position_x = targetpos_x;
//       hold_position_y = targetpos_y;
//       hold_position_z = takeoff_height;
//       hold_yaw = current_yaw;
//       waypoint_hold = true;
//       receive = false;
//       target_received = false;
//       ROS_INFO("Reached RViz waypoint, position error = %.3f m, switching to position hold",
//                position_error);
//       return true;
//   }

//   return false;
// }

void Position_Hold()
{
  current_goal.coordinate_frame = mavros_msgs::PositionTarget::FRAME_LOCAL_NED;
  current_goal.header.stamp = ros::Time::now();
  current_goal.type_mask = position_mask;
  current_goal.position.x = hold_position_x;
  current_goal.position.y = hold_position_y;
  current_goal.position.z = hold_position_z;
  current_goal.yaw = hold_yaw;

  current_goal.velocity.x = 0.0;
  current_goal.velocity.y = 0.0;
  current_goal.velocity.z = 0.0;
  current_goal.yaw_rate = 0.0;

  ROS_INFO_THROTTLE(2.0, "hold on：(%.2f, %.2f, %.2f)",
                    hold_position_x, hold_position_y, hold_position_z);
}

void take_off(ros::Publisher &local_pos_pub,ros::ServiceClient &set_mode_client,ros::ServiceClient &arming_client,ros::Rate &rate)
{
  mavros_msgs::SetMode offb_set_mode;
  offb_set_mode.request.custom_mode = "OFFBOARD";

  mavros_msgs::CommandBool arm_cmd;
  arm_cmd.request.value = true;

  // 先发送一段起飞位置 setpoint，再请求 OFFBOARD。
  for (int i = 0; ros::ok() && i < 50; i++)
  {

    local_pos_pub.publish(pose(0,0,takeoff_height,0));
    ros::spinOnce();
    rate.sleep(); 
  }

  while (ros::ok())
  {

    local_pos_pub.publish(pose(0,0,takeoff_height,0));

    if (current_state.mode != "OFFBOARD")
    {
      if (set_mode_client.call(offb_set_mode) && offb_set_mode.response.mode_sent)
      {
        ROS_INFO("Offboard mode enabled");
      }
    }
    else if (!current_state.armed)
    {
      if (arming_client.call(arm_cmd) && arm_cmd.response.success)
      {
        ROS_INFO("arm success, take off");
      }
    }
    else if (position_z <= takeoff_height - 0.2f)
    {
      ROS_INFO_THROTTLE(1.0, "Take off... z=%.2f", position_z);
    }
    else
    {
      hold_position_x = position_x;
      hold_position_y = position_y;
      hold_position_z = takeoff_height;
      hold_yaw = current_yaw;
      //waypoint_hold = true;
      ROS_INFO("Takeoff complete, z=%.2f", position_z);
      return;
    }

    ros::spinOnce();
    rate.sleep();
  }
}

void Planner_Control()
{
  current_goal.coordinate_frame = mavros_msgs::PositionTarget::FRAME_LOCAL_NED;
  current_goal.header.stamp = ros::Time::now();
  current_goal.type_mask = PLANNER_CONTROL;

  current_goal.position.x = ego_pos_x;
  current_goal.position.y = ego_pos_y;
  current_goal.position.z = ego_pos_z;

  double velocity_x = ego_vel_x;
  double velocity_y = ego_vel_y;
  double velocity_z = ego_vel_z;
  if (pid_enabled)
  {
    const ros::Time now = ros::Time::now();
    velocity_x = velocity_pid.update(ego_vel_x, current_vel_x,
                                     velocity_pid.integral_x,
                                     velocity_pid.previous_error_x, now);
    velocity_y = velocity_pid.update(ego_vel_y, current_vel_y,
                                     velocity_pid.integral_y,
                                     velocity_pid.previous_error_y, now);
    velocity_z = velocity_pid.update(ego_vel_z, current_vel_z,
                                     velocity_pid.integral_z,
                                     velocity_pid.previous_error_z, now);
    velocity_pid.updateTime(now);
  }
  Speed_limit::limitVelocityNorm(velocity_x, velocity_y, velocity_z,
                                  pid_output_limit);
  current_goal.velocity.x = velocity_x;
  current_goal.velocity.y = velocity_y;
  current_goal.velocity.z = velocity_z;

  current_goal.yaw = ego_yaw;
  current_goal.yaw_rate = 0.0;

}

int main(int argc, char **argv)
{
	ros::init(argc, argv, "cxr_egoctrl_v1");
	setlocale(LC_ALL,"");
	ros::NodeHandle nh;
	ros::NodeHandle nh_("~");

  nh_.getParam("pid_enabled", pid_enabled);
  nh_.getParam("pid_kp", pid_kp);
  nh_.getParam("pid_ki", pid_ki);
  nh_.getParam("pid_kd", pid_kd);
  nh_.getParam("pid_integral_limit", pid_integral_limit);
  nh_.getParam("pid_output_limit", pid_output_limit);
  pid_integral_limit = std::max(0.0, pid_integral_limit);
  pid_output_limit = std::max(0.1, pid_output_limit);
  ROS_INFO("Velocity PID: enabled=%s kp=%.3f ki=%.3f kd=%.3f integral_limit=%.3f output_limit=%.3f",
           pid_enabled ? "true" : "false", pid_kp, pid_ki, pid_kd,
           pid_integral_limit, pid_output_limit);


	ros::Subscriber state_sub = nh.subscribe<mavros_msgs::State>
	("/mavros/state", 10, state_cb);//读取飞控状态的话题

	ros::Publisher local_pos_pub = nh.advertise<mavros_msgs::PositionTarget>
	("/mavros/setpoint_raw/local", 1); 
	
	ros::service::waitForService("/mavros/cmd/arming");
	ros::service::waitForService("/mavros/set_mode");

	ros::ServiceClient arming_client = nh.serviceClient<mavros_msgs::CommandBool>
	("/mavros/cmd/arming");//解锁飞机的服务端
	ros::ServiceClient set_mode_client = nh.serviceClient<mavros_msgs::SetMode>
	("/mavros/set_mode");//设置飞机飞行模式的服务端
	
	ros::Subscriber twist_sub = nh.subscribe<quadrotor_msgs::PositionCommand>
	("/planner_cmd", 10, twist_planner_cb);
  // ros::Subscriber target_sub = nh.subscribe<geometry_msgs::PoseStamped>
	// ("move_base_simple/goal", 10, target_cb);

	// ros::Subscriber position_sub=nh.subscribe<nav_msgs::Odometry>
  // ("/vins_fusion/odometry",10, position_cb);

  ros::Subscriber position_sub=nh.subscribe<nav_msgs::Odometry>
  ("/mavros/local_position/odom",10, position_cb);

	ros::Rate rate(Speed_limit::kControlRate);
   
	
	take_off(local_pos_pub, set_mode_client, arming_client, rate);

	while(ros::ok())
	{
		// if(receive && odom_received)
		// {
		// 	if(Arrival_State())
		// 	{
		// 		Position_Hold();
		// 	}
		// 	else
		// 	{
		// 		waypoint_hold = false;
		// 		Planner_Control();
		// 	}
		// }
		// else
		// {
		// 	Position_Hold();
		// }
    if (receive && odom_received)
    {
      Planner_Control();
    }

    else
    {
      ROS_WARN_THROTTLE(1.0, "Waiting for ego planner");
      if (odom_received)
      {
        hold_position_x = position_x;
        hold_position_y = position_y;
        hold_position_z = position_z;
        hold_yaw = current_yaw;
      }
      Position_Hold();
      velocity_pid.reset();
    }

		local_pos_pub.publish(current_goal);
		ros::spinOnce();
		rate.sleep();
	}

	return 0;
}
