#pragma once

// Default values for the velocity-loop PID controller.
namespace planner_ctrl
{
struct PidParam
{
  static constexpr bool kEnabled = true;
  static constexpr double kKp = 1.0;
  static constexpr double kKi = 0.0;
  static constexpr double kKd = 0.0;
  static constexpr double kIntegralLimit = 1.0;
  static constexpr double kOutputLimit = 1.5;
  // Position feedback gains convert position error (m) to velocity (m/s).
  static constexpr double kPositionKp = 0.8;
  static constexpr double kPositionErrorLimit = 1.0;
};
}  // namespace planner_ctrl
