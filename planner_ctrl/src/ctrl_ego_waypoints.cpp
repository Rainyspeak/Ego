#include <ros/ros.h>

#include <algorithm>
#include <cmath>

#include <mavros_msgs/CommandBool.h>
#include <mavros_msgs/PositionTarget.h>
#include <mavros_msgs/SetMode.h>
#include <mavros_msgs/State.h>
#include <nav_msgs/Odometry.h>
#include <quadrotor_msgs/PositionCommand.h>
#include <std_msgs/Empty.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2/utils.h>

namespace
{
constexpr unsigned short kPositionControl = 0b100111111000;   // position + yaw
constexpr unsigned short kTrajectoryControl = 0b100111000000; // position + velocity + yaw
constexpr double kControlRate = 50.0;

mavros_msgs::State current_state;
mavros_msgs::PositionTarget current_goal;
quadrotor_msgs::PositionCommand ego_command;

bool odom_received = false;
bool trajectory_received = false;
bool trajectory_completed = false;
bool timeout_hold_active = false;
int previous_trajectory_id = -1;

double position_x = 0.0;
double position_y = 0.0;
double position_z = 0.0;
double current_yaw = 0.0;
double current_velocity_x = 0.0;
double current_velocity_y = 0.0;
double current_velocity_z = 0.0;
double terminal_x = 0.0;
double terminal_y = 0.0;
double terminal_z = 0.0;
double terminal_yaw = 0.0;
double timeout_x = 0.0;
double timeout_y = 0.0;
double timeout_z = 0.0;
double timeout_yaw = 0.0;

double takeoff_height = 1.0;
double takeoff_hover_time = 2.0;
double max_speed = 1.0;
double trajectory_timeout = 0.3;
ros::Time last_trajectory_time;

void stateCallback(const mavros_msgs::State::ConstPtr &msg)
{
  current_state = *msg;
}

void odomCallback(const nav_msgs::Odometry::ConstPtr &msg)
{
  position_x = msg->pose.pose.position.x;
  position_y = msg->pose.pose.position.y;
  position_z = msg->pose.pose.position.z;
  current_yaw = tf2::getYaw(msg->pose.pose.orientation);
  current_velocity_x = msg->twist.twist.linear.x;
  current_velocity_y = msg->twist.twist.linear.y;
  current_velocity_z = msg->twist.twist.linear.z;
  odom_received = true;
}

void trajectoryCallback(const quadrotor_msgs::PositionCommand::ConstPtr &msg)
{
  trajectory_received = true;
  last_trajectory_time = ros::Time::now();
  timeout_hold_active = false;
  ego_command = *msg;

  const int trajectory_id = static_cast<int>(msg->trajectory_id);
  if (trajectory_id != previous_trajectory_id)
  {
    previous_trajectory_id = trajectory_id;
    trajectory_completed = false;
    ROS_INFO("开始跟踪EGO局部B样条，trajectory_id=%d", trajectory_id);
  }

  if (msg->trajectory_flag == quadrotor_msgs::PositionCommand::TRAJECTORY_STATUS_COMPLETED)
  {
    terminal_x = msg->position.x;
    terminal_y = msg->position.y;
    terminal_z = msg->position.z;
    terminal_yaw = msg->yaw;
    if (!trajectory_completed)
    {
      trajectory_completed = true;
      ROS_INFO("EGO最终B样条执行完成，切换终点悬停：(%.2f, %.2f, %.2f)",
               terminal_x, terminal_y, terminal_z);
    }
  }
}

bool trajectoryFresh()
{
  return trajectory_received && !last_trajectory_time.isZero() &&
         (ros::Time::now() - last_trajectory_time).toSec() <= trajectory_timeout;
}

void limitVelocityNorm(double &vx, double &vy, double &vz)
{
  const double speed = std::sqrt(vx * vx + vy * vy + vz * vz);
  if (max_speed > 0.0 && speed > max_speed)
  {
    const double scale = max_speed / speed;
    vx *= scale;
    vy *= scale;
    vz *= scale;
  }
}

void setPositionHold(double x, double y, double z, double yaw)
{
  current_goal.coordinate_frame = mavros_msgs::PositionTarget::FRAME_LOCAL_NED;
  current_goal.header.stamp = ros::Time::now();
  current_goal.type_mask = kPositionControl;
  current_goal.position.x = x;
  current_goal.position.y = y;
  current_goal.position.z = z;
  current_goal.velocity.x = 0.0;
  current_goal.velocity.y = 0.0;
  current_goal.velocity.z = 0.0;
  current_goal.acceleration_or_force.x = 0.0;
  current_goal.acceleration_or_force.y = 0.0;
  current_goal.acceleration_or_force.z = 0.0;
  current_goal.yaw = yaw;
  current_goal.yaw_rate = 0.0;
}

void setTrajectoryCommand()
{
  double vx = ego_command.velocity.x;
  double vy = ego_command.velocity.y;
  double vz = ego_command.velocity.z;
  limitVelocityNorm(vx, vy, vz);

  current_goal.coordinate_frame = mavros_msgs::PositionTarget::FRAME_LOCAL_NED;
  current_goal.header.stamp = ros::Time::now();
  current_goal.type_mask = kTrajectoryControl;
  current_goal.position.x = ego_command.position.x;
  current_goal.position.y = ego_command.position.y;
  current_goal.position.z = ego_command.position.z;
  current_goal.velocity.x = vx;
  current_goal.velocity.y = vy;
  current_goal.velocity.z = vz;
  current_goal.acceleration_or_force.x = 0.0;
  current_goal.acceleration_or_force.y = 0.0;
  current_goal.acceleration_or_force.z = 0.0;
  current_goal.yaw = ego_command.yaw;
  current_goal.yaw_rate = 0.0;

  ROS_INFO_THROTTLE(0.5, "EGO预设航点轨迹速度：vel_xyz=%.2f",
                    std::sqrt(vx * vx + vy * vy + vz * vz));
}

void setTimeoutHold()
{
  if (!timeout_hold_active)
  {
    timeout_x = position_x;
    timeout_y = position_y;
    timeout_z = position_z;
    timeout_yaw = current_yaw;
    timeout_hold_active = true;
  }
  setPositionHold(timeout_x, timeout_y, timeout_z, timeout_yaw);
  ROS_ERROR_THROTTLE(1.0, "EGO轨迹指令超时，冻结当前位置保持");
}
} // namespace

int main(int argc, char **argv)
{
  ros::init(argc, argv, "ctrl_ego_waypoints");
  ros::NodeHandle nh;
  ros::NodeHandle pnh("~");

  pnh.param("takeoff_height", takeoff_height, 1.0);
  pnh.param("takeoff_hover_time", takeoff_hover_time, 2.0);
  pnh.param("max_speed", max_speed, 1.0);
  pnh.param("trajectory_timeout", trajectory_timeout, 0.3);
  takeoff_hover_time = std::max(0.0, takeoff_hover_time);
  max_speed = std::max(0.1, max_speed);
  trajectory_timeout = std::max(0.1, trajectory_timeout);

  ros::Subscriber state_sub = nh.subscribe("/mavros/state", 10, stateCallback);
  ros::Subscriber trajectory_sub =
      nh.subscribe("/planner_cmd", 10, trajectoryCallback);
  ros::Subscriber odom_sub =
      nh.subscribe("/mavros/local_position/odom", 10, odomCallback);
  ros::Publisher setpoint_pub =
      nh.advertise<mavros_msgs::PositionTarget>("/mavros/setpoint_raw/local", 1);
  ros::Publisher preset_start_pub =
      nh.advertise<std_msgs::Empty>("/ego_preset_start", 1, true);
  ros::ServiceClient arming_client =
      nh.serviceClient<mavros_msgs::CommandBool>("/mavros/cmd/arming");
  ros::ServiceClient set_mode_client =
      nh.serviceClient<mavros_msgs::SetMode>("/mavros/set_mode");

  ros::Rate rate(kControlRate);
  while (ros::ok() && !current_state.connected)
  {
    ros::spinOnce();
    rate.sleep();
  }

  // Keep the original controller behavior: stream takeoff setpoints before
  // requesting OFFBOARD, then request OFFBOARD and arming automatically.
  for (int i = 100; ros::ok() && i > 0; --i)
  {
    setPositionHold(0.0, 0.0, takeoff_height, 0.0);
    setpoint_pub.publish(current_goal);
    ros::spinOnce();
    rate.sleep();
  }

  mavros_msgs::SetMode offboard_mode;
  offboard_mode.request.custom_mode = "OFFBOARD";
  mavros_msgs::CommandBool arm_command;
  arm_command.request.value = true;
  int mode_arm_stage = 0;
  bool preset_start_sent = false;
  ros::Time takeoff_stable_start;
  ros::Time last_request = ros::Time::now();

  while (ros::ok())
  {
    if (current_state.mode != "OFFBOARD" && mode_arm_stage == 0 &&
        ros::Time::now() - last_request > ros::Duration(5.0))
    {
      if (set_mode_client.call(offboard_mode) && offboard_mode.response.mode_sent)
      {
        ROS_INFO("Offboard mode enabled");
        mode_arm_stage = 1;
      }
      last_request = ros::Time::now();
    }

    else if (!current_state.armed && mode_arm_stage == 1 &&
             ros::Time::now() - last_request > ros::Duration(2.0))
    {
      if (arming_client.call(arm_command) && arm_command.response.success)
      {
        ROS_INFO("arm success, take off");
        mode_arm_stage = 2;
      }
      last_request = ros::Time::now();
    }

    // Generate the EGO preset path only after a real stable takeoff hover.
    // Otherwise the B-spline clock would run during OFFBOARD/arming/takeoff.
    if (!preset_start_sent && odom_received && current_state.armed &&
        current_state.mode == "OFFBOARD")
    {
      const double altitude_error = std::abs(position_z - takeoff_height);
      const double speed = std::sqrt(current_velocity_x * current_velocity_x +
                                     current_velocity_y * current_velocity_y +
                                     current_velocity_z * current_velocity_z);
      if (altitude_error <= 0.15 && speed <= 0.10)
      {
        if (takeoff_stable_start.isZero())
          takeoff_stable_start = ros::Time::now();
        else if ((ros::Time::now() - takeoff_stable_start).toSec() >=
                 takeoff_hover_time)
        {
          preset_start_pub.publish(std_msgs::Empty());
          preset_start_sent = true;
          ROS_INFO("Stable takeoff hover for %.2f s; starting EGO preset mission",
                   takeoff_hover_time);
        }
      }
      else
      {
        takeoff_stable_start = ros::Time(0);
      }
    }

    if (!trajectory_received || !odom_received)
    {
      setPositionHold(0.0, 0.0, takeoff_height, 0.0);
      if (!odom_received)
        ROS_WARN_THROTTLE(1.0, "等待MAVROS里程计，不执行EGO预设航点轨迹");
    }
    else if (!trajectoryFresh())
    {
      setTimeoutHold();
    }
    else if (trajectory_completed)
    {
      // Preset-waypoint controller keeps the original final-B-spline hold.
      setPositionHold(terminal_x, terminal_y, terminal_z, terminal_yaw);
      ROS_INFO_THROTTLE(2.0, "最终航点悬停：(%.2f, %.2f, %.2f)",
                        terminal_x, terminal_y, terminal_z);
    }
    else
    {
      setTrajectoryCommand();
    }

    setpoint_pub.publish(current_goal);
    ros::spinOnce();
    rate.sleep();
  }

  return 0;
}
