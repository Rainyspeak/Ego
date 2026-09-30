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
#include <nav_msgs/Odometry.h>
#include <tf/transform_datatypes.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include "planner_ctrl/Pidparam.hpp"

#include <Eigen/Dense>


Eigen::Vector3d ego_pos, ego_vel, ego_acc;  // yaw 为标量，见下方 ego_yaw/ego_yaw_rate

#define POSITION_CONTROL 0b100111111000   //位置起飞：使用PX/PY/PZ/YAW
#define PLANNER_CONTROL 0b100111000000 //轨迹跟踪：使用位置、速度和YAW（加速度位被忽略）

unsigned short position_mask = POSITION_CONTROL;

float takeoff_height = 1.1f; //全局起飞高度（米）
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
bool odom_received = false;
float ego_yaw, ego_yaw_rate; //EGO planner yaw/yaw_dot（标量）；位置速度加速度在 Eigen 向量 ego_pos/ego_vel/ego_acc
bool receive = false;//触发轨迹的条件判断
bool planner_timed_out = false;//规划器超时后原地悬停，等它重新规划
constexpr double kPlannerTimeout = 0.05; // 50 ms 内未收到新指令即过期
ros::SteadyTime last_planner_receive_time;
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
double position_kp = planner_ctrl::PidParam::kPositionKp;
double position_error_limit = planner_ctrl::PidParam::kPositionErrorLimit;
// 加速度前馈超前时间(s)：v_cmd = v + a·τ 等价沿轨迹前视 τ 秒，抵消指令链路
// 总滞后（规划延迟+控制离散+PX4 内环响应+机体动力学）；0 = 关闭。
// 内环（角速率/姿态/速度）已标定良好，0.1 起步，滞后未消可加到 0.15~0.25
double acc_ff_time = 0.1;

struct Speed_limit
{
  static constexpr double kControlRate = 50.0; // 每 20 ms 检查
  // 绝对速度兜底（异常保护），须高于规划器 max_vel：实机 0.8 / 仿真 2.0，
  // 取 3.0 正常不触发；正常工况的速度约束由规划器 max_vel 负责
  static constexpr double kSpeedLimit = 3.0;

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
  VelocityPid(){};
  ~VelocityPid(){};

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

    //返回修正的pid速度 
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

quadrotor_msgs::PositionCommand ego_twist;
void twist_planner_cb(const quadrotor_msgs::PositionCommand::ConstPtr& msg)
{
    last_planner_receive_time = ros::SteadyTime::now();
    ego_twist = *msg;
    receive = true;
    planner_timed_out = false;
    ego_pos(0) = ego_twist.position.x;
    ego_pos(1) = ego_twist.position.y;
    ego_pos(2) = ego_twist.position.z;
    ego_vel(0) = ego_twist.velocity.x;
    ego_vel(1) = ego_twist.velocity.y;
    ego_vel(2) = ego_twist.velocity.z;
    ego_acc(0) = ego_twist.acceleration.x;
    ego_acc(1) = ego_twist.acceleration.y;
    ego_acc(2) = ego_twist.acceleration.z;
    ego_yaw = ego_twist.yaw;
    ego_yaw_rate = ego_twist.yaw_dot;

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

void take_off(ros::Publisher &local_pos_pub,ros::ServiceClient &set_mode_client,ros::ServiceClient &arming_client,ros::Rate &rate)
{
  mavros_msgs::SetMode offb_set_mode;
  offb_set_mode.request.custom_mode = "OFFBOARD";

  mavros_msgs::CommandBool arm_cmd;
  arm_cmd.request.value = true;

  // 单循环依次完成：切入 OFFBOARD → 解锁 → 爬升到起飞高度。
  // setpoint 从第一帧就在发送，OFFBOARD 请求被拒会自动重试，无需单独预发送阶段。
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
    else if (position_z > takeoff_height - 0.2f)
    {
      ROS_INFO("Takeoff complete, hover at (0, 0, %.2f), waiting for planner", takeoff_height);
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

  current_goal.position.x = ego_pos(0);
  current_goal.position.y = ego_pos(1);
  current_goal.position.z = ego_pos(2);

  double velocity_x = ego_vel(0);
  double velocity_y = ego_vel(1);
  double velocity_z = ego_vel(2);

  // 加速度前馈：v(t+τ) ≈ v(t) + a·τ，指令提前 τ 秒到达，补相位滞后；
  velocity_x += acc_ff_time * ego_acc(0);
  velocity_y += acc_ff_time * ego_acc(1);
  velocity_z += acc_ff_time * ego_acc(2);

  // Position feedback: correct the planner feed-forward velocity using the
  // live odometry position before entering the velocity PID loop.
  // 误差基准是 EGO 期望位置 ego_pos（Eigen 向量，回调里赋值）
  const double position_error_x = std::max(-position_error_limit,
                                           std::min(position_error_limit,
                                                    ego_pos(0) - position_x));
  const double position_error_y = std::max(-position_error_limit,
                                           std::min(position_error_limit,
                                                    ego_pos(1) - position_y));
  const double position_error_z = std::max(-position_error_limit,
                                           std::min(position_error_limit,
                                                    ego_pos(2) - position_z));
  velocity_x += position_kp * position_error_x;
  velocity_y += position_kp * position_error_y;
  velocity_z += position_kp * position_error_z;
  if (pid_enabled)
  {
    const ros::Time now = ros::Time::now();
    // pid_output_limit 只截断 PID 反馈修正量，不截断规划前馈——旧实现直接
    // 限幅总输出，规划 max_vel 超过该值时前馈被削，产生系统性轨迹滞后
    auto clamp = [](double v, double lim) {
      return std::max(-lim, std::min(lim, v));
    };
    velocity_x += clamp(velocity_pid.update(velocity_x, current_vel_x,
                                            velocity_pid.integral_x,
                                            velocity_pid.previous_error_x, now) - velocity_x,
                        pid_output_limit);
    velocity_y += clamp(velocity_pid.update(velocity_y, current_vel_y,
                                            velocity_pid.integral_y,
                                            velocity_pid.previous_error_y, now) - velocity_y,
                        pid_output_limit);
    velocity_z += clamp(velocity_pid.update(velocity_z, current_vel_z,
                                            velocity_pid.integral_z,
                                            velocity_pid.previous_error_z, now) - velocity_z,
                        pid_output_limit);
    velocity_pid.updateTime(now);
  }
  // 绝对速度兜底（异常保护）：须高于规划器 max_vel，正常工况不触发
  Speed_limit::limitVelocityNorm(velocity_x, velocity_y, velocity_z,
                                  Speed_limit::kSpeedLimit);
  current_goal.velocity.x = velocity_x;
  current_goal.velocity.y = velocity_y;
  current_goal.velocity.z = velocity_z;

  current_goal.yaw = ego_yaw;
  current_goal.yaw_rate = 0.0;

}

int main(int argc, char **argv)
{
	ros::init(argc, argv, "ctrl_v1");
	setlocale(LC_ALL,"");
	ros::NodeHandle nh;
	ros::NodeHandle nh_("~");

  nh_.getParam("pid_enabled", pid_enabled);
  nh_.getParam("pid_kp", pid_kp);
  nh_.getParam("pid_ki", pid_ki);
  nh_.getParam("pid_kd", pid_kd);
  nh_.getParam("pid_integral_limit", pid_integral_limit);
  nh_.getParam("pid_output_limit", pid_output_limit);
  nh_.getParam("position_kp", position_kp);
  nh_.getParam("position_error_limit", position_error_limit);
  nh_.getParam("acc_ff_time", acc_ff_time);
  nh_.getParam("takeoff_height", takeoff_height);
  pid_integral_limit = std::max(0.0, pid_integral_limit);
  pid_output_limit = std::max(0.1, pid_output_limit);
  position_kp = std::max(0.0, position_kp);
  position_error_limit = std::max(0.0, position_error_limit);
  acc_ff_time = std::max(0.0, acc_ff_time);
  // ROS_INFO("Velocity PID: enabled=%s kp=%.3f ki=%.3f kd=%.3f integral_limit=%.3f output_limit=%.3f",
  //          pid_enabled ? "true" : "false", pid_kp, pid_ki, pid_kd,
  //          pid_integral_limit, pid_output_limit);


	ros::Subscriber state_sub = nh.subscribe<mavros_msgs::State>
	("/mavros/state", 10, state_cb);//读取飞控状态的话题

	ros::Publisher local_pos_pub = nh.advertise<mavros_msgs::PositionTarget>
	("/mavros/setpoint_raw/local", 1); 
	
	ros::service::waitForService("/mavros/cmd/arming");
	ros::service::waitForService("/mavros/set_mode");

	ros::ServiceClient arming_client = nh.serviceClient<mavros_msgs::CommandBool>
	("/mavros/cmd/arming");
	ros::ServiceClient set_mode_client = nh.serviceClient<mavros_msgs::SetMode>
	("/mavros/set_mode");
	
	ros::Subscriber twist_sub = nh.subscribe<quadrotor_msgs::PositionCommand>
	("/planner_cmd", 1, twist_planner_cb, ros::TransportHints().tcpNoDelay());
  // ros::Subscriber target_sub = nh.subscribe<geometry_msgs::PoseStamped>
	// ("move_base_simple/goal", 10, target_cb);

	// ros::Subscriber position_sub=nh.subscribe<nav_msgs::Odometry>
  // ("/vins_fusion/odometry",10, position_cb);

  ros::Subscriber position_sub=nh.subscribe<nav_msgs::Odometry>
  ("/Odometry",10, position_cb);

	ros::Rate takeoff_rate(Speed_limit::kControlRate);
   
	
	take_off(local_pos_pub, set_mode_client, arming_client, takeoff_rate);
  // 使用实际时间调度
  ros::WallRate rate(Speed_limit::kControlRate);

	while(ros::ok())
	{
    // 先接收最新指令，再判断有效期，避免多使用上一周期的速度。
    ros::spinOnce();
    if (receive && (ros::SteadyTime::now() - last_planner_receive_time).toSec() >= kPlannerTimeout)
    {
      receive = false;
      // 超时瞬间锁定当前位置，原地悬停等规划器重新规划
      if (odom_received)
      {
        planner_timed_out = true;
        current_goal = pose(position_x, position_y, position_z, current_yaw);
      }
      ROS_WARN("Planner command timeout (%.0f ms), holding position, waiting for re-plan",
               kPlannerTimeout * 1000.0);
    }
    if (receive && odom_received)
    {
      Planner_Control();
    }

    else
    {
      if (!planner_timed_out)
        current_goal = pose(0, 0, takeoff_height, 0);
        velocity_pid.reset();
    }

    current_goal.header.stamp = ros::Time::now();
		local_pos_pub.publish(current_goal);
		rate.sleep();
	}

	return 0;
}
