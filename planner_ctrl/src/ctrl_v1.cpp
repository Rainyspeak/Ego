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


#define VELOCITY2D_CONTROL 0b101111000111 //设置好对应的掩码，从右往左依次对应PX/PY/PZ/VX/VY/VZ/AX/AY/AZ/FORCE/YAW/YAW-RATE
#define POSITION_CONTROL 0b100111111000   //位置起飞：使用PX/PY/PZ/YAW
#define PLANNER_CONTROL 0b100111000000 //轨迹跟踪：使用位置、速度和YAW（加速度位被忽略）

unsigned short velocity_mask = VELOCITY2D_CONTROL;    
unsigned short position_mask = POSITION_CONTROL;

float takeoff_height = 1.5f; //全局起飞高度（米）
mavros_msgs::PositionTarget current_goal;
nav_msgs::Odometry position_msg;
geometry_msgs::PoseStamped target_pos;
mavros_msgs::State current_state;



int now_yaw = 0;
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

void state_cb(const mavros_msgs::State::ConstPtr& msg){
	current_state = *msg;
}

//read vehicle odometry
void position_cb(const nav_msgs::Odometry::ConstPtr&msg)
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
    current_goal.coordinate_frame = mavros_msgs::PositionTarget::FRAME_LOCAL_NED;
    current_goal.header.stamp = ros::Time::now();
    current_goal.type_mask = position_mask;
    current_goal.position.x = 0.0;
    current_goal.position.y = 0.0;
    current_goal.position.z = takeoff_height;
    current_goal.yaw = now_yaw;

    local_pos_pub.publish(current_goal);
    ros::spinOnce();
    rate.sleep(); 
  }

  while (ros::ok())
  {
    current_goal.coordinate_frame = mavros_msgs::PositionTarget::FRAME_LOCAL_NED;
    current_goal.header.stamp = ros::Time::now();
    current_goal.type_mask = position_mask;
    current_goal.position.x = 0.0;
    current_goal.position.y = 0.0;
    current_goal.position.z = takeoff_height;
    current_goal.yaw = now_yaw;

    local_pos_pub.publish(current_goal);

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
  Speed_limit::limitVelocityNorm(velocity_x, velocity_y, velocity_z, Speed_limit::kSpeedLimit);
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
    }

		local_pos_pub.publish(current_goal);
		ros::spinOnce();
		rate.sleep();
	}

	return 0;
}
