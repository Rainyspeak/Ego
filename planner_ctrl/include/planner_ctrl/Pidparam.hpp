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
};
}  // namespace planner_ctrl
