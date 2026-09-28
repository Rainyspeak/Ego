#ifndef PATH_MANAGER_PATH_MANAGER_H
#define PATH_MANAGER_PATH_MANAGER_H

#include <cmath>
#include <string>
#include <vector>

#include <eigen3/Eigen/Dense>

#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <ros/ros.h>
#include <std_msgs/Int32.h>

namespace path_manager {

// path_manager —— Super mission_manager 的 EGO 移植 + waypoint_generator 功能并入
// （寄存航点 + 单点 goal 输入 + 20 Hz 跟随式目标 + 框心热存储/硬化 + 回溯）。
//
// 任务输入四路（回溯模式 recall_mode_ 下除显式 path_topic 新任务外全部屏蔽）：
//   1. path_topic（nav_msgs/Path）：整条多航点任务路径，替换任务（空 Path 清空）；
//   2. goal_topic（PoseStamped，默认 /move_base_simple/goal）：单点 goal 直接成为
//      新的单航点任务（z ≤ -0.1 丢弃）；
//   3. enqueue_topic（PoseStamped，/path_manager/enqueue）：单点入队（距尾 <
//      min_enqueue_spacing 更新队尾，否则 O(1) 追加）；
//   4. enqueue_batch_topic（nav_msgs/Path，/path_manager/enqueue_path）：注视
//      三航点组批量入队——批次与队尾逐点（≤4 点）邻近 ≤ batch_match_dist 则
//      原位更新（EMA 精化刷新本组）；队尾不匹配先做已入列校验
//      （revalidateEnqueued）：批次与容器内已有航点命中过半即同框重识别，
//      原位刷新/缺员补插，不重复追加；其余是新框：**乐观入列**（第一帧立即
//      追加，入列零延迟）+ 后置二次校验（未确认组被远离新组替换时按幻影
//      撤销）。队列持久增长成完整穿框路径，进度只进不退。
//
// 引导：20 Hz（timer_dt）把 odom 投影求 progress，发布 progress+lookahead 的
// 单点 Path 到 goal_path_topic。框心热存储（center_topic）登记的未越过框心
// 会把有效 lookahead 沿"距框心剩余弧长"线性收窄到下限（穿心轴硬化，连续
// 斜坡；下限 = center_lookahead + lookahead_speed_gain·|v|）。队列断供时沿
// 末段外推 tail_extrapolate_max（继续飞行等待）。完成判定对有效终点（含
// 外推），距尾 ≤ reach_fallback_dist 兜底，dwell 后停发。
//
// 回溯（recall_topic，PoseStamped=home）：队列整体反转 + 队尾接 home，carrot
// 倒数框原路返回起飞原点；重复触发幂等。
//
// 接线（FSM 吃 Path，故输出也是 Path）：
//   下游: goal_path_topic 默认 /path_manager/waypoints，直接接 EGO FSM 航点话题。
//   另发 frame_count_topic（std_msgs/Int32，latch）：centers_ 中已越过（passed）
//   的框心数 = 穿越计数，供 mission_state 做任务进度（与检测器 done 取最大）。

// ROS 消息 ↔ Eigen 边界转换（内部几何一律 Eigen::Vector3d）
inline Eigen::Vector3d toVec3(const geometry_msgs::Point& p) {
  return Eigen::Vector3d(p.x, p.y, p.z);
}
inline geometry_msgs::Point toPoint(const Eigen::Vector3d& v) {
  geometry_msgs::Point p;
  p.x = v.x();
  p.y = v.y();
  p.z = v.z();
  return p;
}

inline geometry_msgs::Quaternion yawToQuaternion(double yaw) {
  geometry_msgs::Quaternion q;
  q.w = std::cos(yaw * 0.5);
  q.z = std::sin(yaw * 0.5);
  return q;
}

struct SampledPoint {
  Eigen::Vector3d position = Eigen::Vector3d::Zero();
  double progress = 0.0;
  size_t segment_index = 0;
};

class PathManager {
 public:
  PathManager();

 private:
  void odomCallback(const nav_msgs::Odometry::ConstPtr& msg);
  void pathCallback(const nav_msgs::Path::ConstPtr& msg);
  void goalCallback(const geometry_msgs::PoseStamped::ConstPtr& msg);
  void enqueueCallback(const geometry_msgs::PoseStamped::ConstPtr& msg);
  void enqueueBatchCallback(const nav_msgs::Path::ConstPtr& msg);
  void centerCallback(const geometry_msgs::PoseStamped::ConstPtr& msg);
  void recallCallback(const geometry_msgs::PoseStamped::ConstPtr& msg);
  void setMission(const nav_msgs::Path& mission);
  void appendToMission(const geometry_msgs::PoseStamped& wp);
  void enqueueBatch(const nav_msgs::Path& batch);
  bool revalidateEnqueued(const nav_msgs::Path& batch);
  size_t nearestEnqueuedIndex(const Eigen::Vector3d& p,
                              const std::vector<size_t>& skip) const;
  void rebuildLengths();
  void publishFrameCount();
  void timerCallback(const ros::TimerEvent&);

  double totalLength() const;
  SampledPoint sampleAtProgress(double progress) const;
  SampledPoint closestPointOnPath(const Eigen::Vector3d& p) const;
  double tangentYaw(size_t segment_index) const;

  // nearestEnqueuedIndex 的"无命中"哨兵
  static constexpr size_t kNoSlot = static_cast<size_t>(-1);

  ros::NodeHandle nh_;
  ros::NodeHandle pnh_;
  ros::Subscriber odom_sub_;
  ros::Subscriber path_sub_;
  ros::Subscriber goal_sub_;
  ros::Subscriber enqueue_sub_;
  ros::Subscriber enqueue_batch_sub_;
  ros::Subscriber center_sub_;
  ros::Subscriber recall_sub_;
  ros::Publisher goal_pub_;
  ros::Publisher frame_count_pub_;
  ros::Timer timer_;

  nav_msgs::Odometry odom_;
  nav_msgs::Path path_;
  std::vector<double> cumulative_lengths_;

  std::string path_topic_;
  std::string goal_topic_;
  std::string enqueue_topic_;
  std::string enqueue_batch_topic_;
  std::string center_topic_;
  std::string recall_topic_;
  std::string goal_path_topic_;
  std::string frame_count_topic_;
  std::string odom_topic_;
  std::string frame_;
  double timer_dt_ = 0.05;
  double lookahead_ = 5.0;
  double reached_dist_ = 0.20;
  double dwell_ = 0.5;
  double min_enqueue_spacing_ = 0.5;
  double batch_match_dist_ = 1.2;   // 批次与队尾的逐点匹配半径（原位更新判定）
  double enqueued_match_dist_ = 1.2; // 已入列校验匹配半径：批次与容器内已有
                                     // 航点最近邻命中（重复帧认定/缺员判定）
  double reach_fallback_dist_ = 0.5; // 队尾静止兜底：距尾 ≤ 此值且 carrot 不动即完成
  int max_queue_size_ = 0;          // 0 = 不限
  double tail_extrapolate_max_ = 0.0; // 队尾外推上限，0 = 关闭

  // ---- 框心热存储（累积航点任务）：框心 = 穿越关键点，登记后由自适应
  // lookahead 硬化长程引导路径对穿心段的贴附（carrot 沿 P1→框心→P3 轴推进，
  // 不被大 lookahead 跳过），检测丢失也不丢约束 ----
  struct CenterEntry {
    Eigen::Vector3d point = Eigen::Vector3d::Zero();
    bool passed = false;
  };
  std::vector<CenterEntry> centers_;  // FIFO，上限 64
  double center_lookahead_ = 0.0;     // 框心处的前视下限。0=关（默认，纯
                                      // lookahead 长轨迹跟随——2026-09-28 用户
                                      // 回归此行为；>0 开启框心锚+P1 锚约束）
  double center_gate_window_ = 0.0;   // 前视收窄斜坡跨度：距框心该弧长内
                                      // lookahead 线性收窄，0 = lookahead+1
  double p1_lookahead_ = 2.5;         // 对准点 P1 处的前视下限：距 P1 同跨度
                                      // 也收窄（carrot 贴入 P1 的路径段，转弯
                                      // 提前完成；只锚框心会让 carrot 越过 P1
                                      // 直奔框心，急转弯来不及）
  double lookahead_speed_gain_ = 0.0; // 框心处前视下限的速度增益：下限 =
                                      // center_lookahead + gain·|v|，纯跟踪
                                      // 经典 L=k·v+L0；0 = 固定下限（旧语义）
  double center_match_dist_ = 1.0;    // 框心登记去重半径（同框观测更新）

  bool recall_mode_ = false;  // 回溯模式：反转队列回 home，屏蔽动态任务输入
                              //（仅显式 path_topic 新任务可解除）

  // ---- 新组乐观入列 + 后置二次校验（发散撤销）：新框第一帧立即入列（入列
  // 延迟=0，流畅优先）；入列后队尾精化批次累计确认，未确认（hits <
  // enqueue_verify_count）期间若新组与本组远离 ≥ enqueue_retract_dist 且在
  // enqueue_retract_time 内 → 本组按幻影撤销（队列尾部整组移除，进度投影
  // 只进不退可越过短暂回撤）----
  nav_msgs::Path last_group_;
  size_t last_group_hits_ = 0;      // 队尾精化累计的确认批次数
  ros::Time last_group_stamp_;
  bool enqueue_verify_ = true;      // false = 完全旧行为（无撤销）
  int enqueue_verify_count_ = 2;    // 确认所需精化批次（10 Hz 下 ~0.2 s）
  double enqueue_retract_dist_ = 2.0;  // 新组与本组逐点全远离阈值（判幻影）
  double enqueue_retract_time_ = 3.0;  // 撤销窗口：入列后该时长内未确认才可撤

  bool has_odom_ = false;
  bool has_path_ = false;
  bool complete_ = false;
  ros::Time inside_since_;
};

}  // namespace path_manager

#endif  // PATH_MANAGER_PATH_MANAGER_H
