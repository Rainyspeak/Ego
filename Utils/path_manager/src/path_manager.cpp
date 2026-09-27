// path_manager 主逻辑（节点接口见 path_ctrl_node.cpp，类声明与行为/接线
// 说明在 include/path_manager/path_manager.h）。内部几何一律 Eigen::Vector3d，
// ROS 消息仅在边界转换。

#include "path_manager/path_manager.h"

#include <algorithm>
#include <limits>

namespace path_manager {

// C++14：constexpr 静态成员被按引用使用（vector 初始化）需类外定义
constexpr size_t PathManager::kNoSlot;

PathManager::PathManager() : nh_(), pnh_("~") {
  pnh_.param<std::string>("path_topic", path_topic_, "/waypoint_generator/waypoints");
  pnh_.param<std::string>("goal_topic", goal_topic_, "/move_base_simple/goal");
  pnh_.param<std::string>("enqueue_topic", enqueue_topic_, "/path_manager/enqueue");
  pnh_.param<std::string>("enqueue_batch_topic", enqueue_batch_topic_, "/path_manager/enqueue_path");
  pnh_.param<std::string>("center_topic", center_topic_, "/path_manager/frame_centers");
  pnh_.param<std::string>("recall_topic", recall_topic_, "/path_manager/recall");
  pnh_.param<std::string>("goal_path_topic", goal_path_topic_, "/path_manager/waypoints");
  pnh_.param<std::string>("frame_count_topic", frame_count_topic_, "/path_manager/frame_count");
  pnh_.param<std::string>("odom_topic", odom_topic_, "/odom_world");
  pnh_.param<std::string>("frame", frame_, "world");
  pnh_.param<double>("timer_dt", timer_dt_, 0.05);            // 20 Hz
  pnh_.param<double>("lookahead_distance", lookahead_, 5.0);  // 建议 ≥ fsm/planning_horizon
  pnh_.param<double>("goal_reached_distance", reached_dist_, 0.20);
  pnh_.param<double>("goal_dwell_time", dwell_, 0.5);
  pnh_.param<double>("min_enqueue_spacing", min_enqueue_spacing_, 0.5);
  pnh_.param<double>("batch_match_dist", batch_match_dist_, 1.2);
  pnh_.param<double>("enqueued_match_dist", enqueued_match_dist_, 1.2);
  pnh_.param("validate_before_enqueue", validate_before_enqueue_, true);
  pnh_.param("validate_count", validate_count_, 2);
  pnh_.param("validate_match_dist", validate_match_dist_, 0.4);
  pnh_.param<double>("reach_fallback_dist", reach_fallback_dist_, 0.5);
  pnh_.param<double>("center_lookahead", center_lookahead_, 2.0);
  pnh_.param<double>("center_gate_window", center_gate_window_, 0.0);
  pnh_.param<double>("center_match_dist", center_match_dist_, 1.0);
  pnh_.param<double>("lookahead_speed_gain", lookahead_speed_gain_, 0.0);
  pnh_.param<int>("max_queue_size", max_queue_size_, 0);
  pnh_.param<double>("tail_extrapolate_max", tail_extrapolate_max_, 0.0);
  timer_dt_ = std::max(timer_dt_, 0.01);
  lookahead_ = std::max(lookahead_, 0.0);
  reached_dist_ = std::max(reached_dist_, 0.0);
  dwell_ = std::max(dwell_, 0.0);
  min_enqueue_spacing_ = std::max(min_enqueue_spacing_, 0.0);
  batch_match_dist_ = std::max(batch_match_dist_, 0.0);
  enqueued_match_dist_ = std::max(enqueued_match_dist_, 0.0);
  validate_count_ = std::max(validate_count_, 1);
  validate_match_dist_ = std::max(validate_match_dist_, 0.05);
  reach_fallback_dist_ = std::max(reach_fallback_dist_, 0.0);
  center_lookahead_ = std::max(center_lookahead_, 0.0);
  center_match_dist_ = std::max(center_match_dist_, 0.0);
  lookahead_speed_gain_ = std::max(lookahead_speed_gain_, 0.0);
  max_queue_size_ = std::max(max_queue_size_, 0);
  tail_extrapolate_max_ = std::max(tail_extrapolate_max_, 0.0);

  odom_sub_ = nh_.subscribe(odom_topic_, 10, &PathManager::odomCallback, this);
  path_sub_ = nh_.subscribe(path_topic_, 1, &PathManager::pathCallback, this);
  goal_sub_ = nh_.subscribe(goal_topic_, 10, &PathManager::goalCallback, this);
  enqueue_sub_ = nh_.subscribe(enqueue_topic_, 10, &PathManager::enqueueCallback, this);
  enqueue_batch_sub_ =
      nh_.subscribe(enqueue_batch_topic_, 1, &PathManager::enqueueBatchCallback, this);
  center_sub_ = nh_.subscribe(center_topic_, 10, &PathManager::centerCallback, this);
  recall_sub_ = nh_.subscribe(recall_topic_, 1, &PathManager::recallCallback, this);
  goal_pub_ = nh_.advertise<nav_msgs::Path>(goal_path_topic_, 5);
  frame_count_pub_ = nh_.advertise<std_msgs::Int32>(frame_count_topic_, 1, /*latch=*/true);
  timer_ = nh_.createTimer(ros::Duration(timer_dt_), &PathManager::timerCallback, this);
  publishFrameCount();  // latch 初值 0：晚启动的订阅者（mission_state 重启）立即拿当前计数

  ROS_INFO("[path_manager] 替换输入 Path<-%s goal<-%s；入队<-%s；框心<-%s；回溯<-%s；"
           "carrot Path -> %s @ %.0f Hz (lookahead %.1f m, 外推 %.1f m, odom %s)",
           path_topic_.c_str(), goal_topic_.c_str(), enqueue_topic_.c_str(),
           center_topic_.c_str(), recall_topic_.c_str(), goal_path_topic_.c_str(),
           1.0 / timer_dt_, lookahead_, tail_extrapolate_max_, odom_topic_.c_str());
}

void PathManager::odomCallback(const nav_msgs::Odometry::ConstPtr& msg) {
  odom_ = *msg;
  has_odom_ = true;
}

// 回溯任务（recall）：任务侧穿框结束后触发——把飞行过程中存储的注视航点
// 队列整体反转（倒数框原路返回），队尾（原起点侧）接触发消息携带的 home
// （起飞原点），carrot 沿反路引导回 home；回溯期间屏蔽动态任务输入
//（goal/入队/框心——反向时检测器再锁框不能往回路上挂新点），仅显式
// path_topic 新任务可解除。重复触发幂等忽略。
void PathManager::recallCallback(const geometry_msgs::PoseStamped::ConstPtr& msg) {
  if (recall_mode_) {
    return;  // 幂等：任务侧周期重发安全
  }
  recall_mode_ = true;
  if (!path_.poses.empty()) {
    std::reverse(path_.poses.begin(), path_.poses.end());
  }
  geometry_msgs::PoseStamped home = *msg;
  home.pose.orientation.w = 1.0;
  path_.poses.push_back(home);
  path_.header = msg->header;
  rebuildLengths();
  complete_ = false;
  inside_since_ = ros::Time();
  for (CenterEntry& ce : centers_) {
    ce.passed = true;  // 回程不再做框心硬化（历史框心全部放行）
  }
  publishFrameCount();  // 回溯态计数=登记总数（进度语义到此为止）
  has_path_ = true;
  const Eigen::Vector3d h = toVec3(home.pose.position);
  ROS_INFO("[path_manager] RECALL 回溯任务：队列反转 %zu 点 + home(%.2f, %.2f, %.2f)，"
           "回程总长 %.2f m",
           path_.poses.size() - 1, h.x(), h.y(), h.z(), totalLength());
}

void PathManager::pathCallback(const nav_msgs::Path::ConstPtr& msg) {
  setMission(*msg);
}

// waypoint_generator(manual-lonely-waypoint) 的功能并入：单点 goal 直接成为
// 新的单航点任务——与旧链 goal→waypoint_generator→单点 Path→寄存 完全等效
void PathManager::goalCallback(const geometry_msgs::PoseStamped::ConstPtr& msg) {
  if (recall_mode_) {
    ROS_WARN_STREAM_THROTTLE(5.0, "[path_manager] 回溯模式：忽略 goal 输入");
    return;
  }
  if (msg->pose.position.z <= -0.1) {
    ROS_WARN("[path_manager] 无效 goal (z=%.2f ≤ -0.1)，丢弃", msg->pose.position.z);
    return;
  }
  nav_msgs::Path mission;
  mission.header = msg->header;
  mission.poses.push_back(*msg);
  setMission(mission);
}

// 几何特征提取的入队入口：看到就发、不等到达——参考路径在线增长，
// carrot 保持在前，实现"未达先引、不刹停等待"
void PathManager::enqueueCallback(const geometry_msgs::PoseStamped::ConstPtr& msg) {
  if (recall_mode_) {
    ROS_WARN_STREAM_THROTTLE(5.0, "[path_manager] 回溯模式：忽略入队输入");
    return;
  }
  if (msg->pose.position.z <= -0.1) {
    ROS_WARN("[path_manager] 无效 enqueue (z=%.2f ≤ -0.1)，丢弃", msg->pose.position.z);
    return;
  }
  appendToMission(*msg);
}

void PathManager::enqueueBatchCallback(const nav_msgs::Path::ConstPtr& msg) {
  enqueueBatch(*msg);
}

// 批量入队（注视三航点组）：批次与队尾逐点邻近（≤ batch_match_dist）→ 原位
// 更新尾部（本框三航点的 EMA 精化实时刷新，不追加重复点）；队尾不匹配先做
// 已入列校验（revalidateEnqueued）——容器内命中过半 = 同框重识别，原位刷新/
// 缺员补插，不重复追加；命中不足才是新框整批追加。队列持久增长成完整穿框
// 路径：飞过的组留在队里，进度投影只进不退，carrot 沿队引导。估计漂移超过
// 匹配半径（校验也命中不了）时退化为整批追加（旧组留在队中成为形状点，
// 投影可越过），与幻影锁定问题的退化行为同级。
void PathManager::enqueueBatch(const nav_msgs::Path& batch) {
  if (recall_mode_) {
    ROS_WARN_STREAM_THROTTLE(5.0, "[path_manager] 回溯模式：忽略批量入队（反向锁框不挂新点）");
    return;
  }
  if (batch.poses.empty()) {
    return;
  }
  complete_ = false;
  inside_since_ = ros::Time();

  const size_t n = batch.poses.size();
  if (!path_.poses.empty()) {
    const size_t k = std::min(std::min(n, path_.poses.size()), static_cast<size_t>(4));
    bool match = true;
    for (size_t i = 0; i < k; ++i) {
      if ((toVec3(batch.poses[n - 1 - i].pose.position) -
           toVec3(path_.poses[path_.poses.size() - 1 - i].pose.position))
              .norm() > batch_match_dist_) {
        match = false;
        break;
      }
    }
    if (match) {  // 原位更新尾部 k 个点
      for (size_t i = 0; i < k; ++i) {
        path_.poses[path_.poses.size() - 1 - i] = batch.poses[n - 1 - i];
      }
      rebuildLengths();
      has_path_ = true;
      ROS_INFO_STREAM_THROTTLE(2.0, "[path_manager] 批量入队：队尾 " << k
                                << " 点原位更新");
      return;
    }
    // 队尾不匹配 ≠ 新框：已经入列的框被再次识别（解锁重锁、就近粘滞回切、
    // 估计漂移超出队尾匹配窗）同样过不了队尾匹配——已入列校验确认同框则
    // 原位刷新/缺员补插，绝不整批重复追加（追加会让 carrot 折返已飞过的框）
    if (revalidateEnqueued(batch)) {
      return;
    }
  }
  // 新框二次校验闸门（含首框）：连续一致观测确认后才入队，未确认挂起等待
  if (validate_before_enqueue_ && !validateNewFrame(batch)) {
    return;
  }
  const size_t before = path_.poses.size();
  for (const geometry_msgs::PoseStamped& p : batch.poses) {
    appendToMission(p);
  }
  ROS_INFO_STREAM_THROTTLE(2.0, "[path_manager] 批量入队：" << n << " 点追加，队列 "
                            << before << " -> " << path_.poses.size() << " 点");
}

// 已入列校验（重复帧防护）：已经入列的框再次被识别（解锁后重锁、就近粘滞
// 回切到旧框、估计漂移超出队尾匹配窗）时，队尾匹配失败但并非新框。本方法
// 把批次注视三航点 [P1 对准, P2 框心, P3 穿出] 与 3D 容器内全部已入列航点做
// 互斥最近邻匹配（半径 enqueued_match_dist_，一个队列槽位至多配一个批次点）：
//   命中过半（三航点组 = 2/3）→ 认定本框已入列。命中点原位刷新（重识别的
//     估计通常更准，EMA 精化跟随），缺员（FIFO 裁剪拆组/单点漂移出窗）按批内
//     顺序补插回本组槽位（保持 P1→P2→P3 链形），不整批重复追加。
//   命中不足半数 → 新框，返回 false 由调用方整批追加。
// 残余退化：缺员补插不清理漂移出窗的同组旧点（旧点成为形状点，投影可越过），
// 与整批追加退化同级但规模小得多。
bool PathManager::revalidateEnqueued(const nav_msgs::Path& batch) {
  const size_t n = batch.poses.size();
  std::vector<size_t> slot(n, kNoSlot);
  std::vector<size_t> claimed;
  claimed.reserve(n);
  size_t hits = 0;
  for (size_t i = 0; i < n; ++i) {
    const size_t m = nearestEnqueuedIndex(toVec3(batch.poses[i].pose.position), claimed);
    if (m == kNoSlot) continue;
    slot[i] = m;
    claimed.push_back(m);
    ++hits;
  }
  if (hits < (n + 1) / 2) {
    return false;  // 证据不足：不能排除新框，维持整批追加语义
  }

  for (size_t i = 0; i < n; ++i) {  // 先刷新（覆写不动下标），后补插
    if (slot[i] != kNoSlot) path_.poses[slot[i]] = batch.poses[i];
  }
  size_t inserted = 0;
  for (size_t i = n; i-- > 0;) {  // 逆序：插入只右移更后方已处理完的槽位
    if (slot[i] != kNoSlot) continue;
    size_t pos = path_.poses.size();
    for (size_t j = i + 1; j < n; ++j) {
      if (slot[j] != kNoSlot) {
        pos = slot[j];  // 插到组内下一个命中槽位之前
        break;
      }
    }
    if (pos == path_.poses.size()) {
      for (size_t j = i; j-- > 0;) {
        if (slot[j] != kNoSlot) {
          pos = slot[j] + 1;  // 组内无后方命中：插到上一个命中槽位之后
          break;
        }
      }
    }
    path_.poses.insert(path_.poses.begin() + pos, batch.poses[i]);
    ++inserted;
  }
  rebuildLengths();
  has_path_ = true;
  ROS_INFO_STREAM_THROTTLE(2.0, "[path_manager] 已入列校验：本框已在队列（命中 "
                              << hits << "/" << n << " 点原位刷新"
                              << (inserted > 0 ? "，缺员补插 " + std::to_string(inserted) + " 点" : "")
                              << "），不重复追加，队列 " << path_.poses.size() << " 点");
  return true;
}

// 新框二次校验闸门（validateNewFrame）：过队尾匹配与已入列校验后，批次是
// "未见过的框"——高速下平面拟合的中心滑动/单帧野值（幻影平面）正是以新框
// 面目出现的，直接入队会污染引导路径（carrot 被拽向幻影）。闸门要求连续
// validate_count 次观测逐点一致（≤ validate_match_dist，锁定 EMA 的正常帧间
// 抖动远小于该值）才确认；确认即直接入队，只多等 ~1 个检测周期（10 Hz 下
// 0.1 s × (count-1)）。候选跳变（超阈值）= 估计不稳 → 替换候选重新计数；
// 挂起候选 2 s 无新观测过期作废。已入列组的 EMA 精化走队尾原位更新路径，
// 不经过本闸门——live-following 语义保留。
bool PathManager::validateNewFrame(const nav_msgs::Path& batch) {
  const ros::Time now = ros::Time::now();
  if (pending_hits_ > 0 && !pending_stamp_.isZero() &&
      (now - pending_stamp_).toSec() > 2.0) {
    pending_hits_ = 0;  // 检测中断后的旧候选不能当新框的确认依据
    pending_batch_.poses.clear();
  }

  bool match = pending_hits_ > 0 && pending_batch_.poses.size() == batch.poses.size();
  if (match) {
    for (size_t i = 0; i < batch.poses.size(); ++i) {
      if ((toVec3(batch.poses[i].pose.position) -
           toVec3(pending_batch_.poses[i].pose.position)).norm() > validate_match_dist_) {
        match = false;
        break;
      }
    }
  }
  if (match) {
    ++pending_hits_;
    pending_batch_.poses = batch.poses;  // 跟随最新观测（慢漂移下确认值是新的）
  } else {
    pending_batch_ = batch;  // 首个观测或跳变替换：重新计数
    pending_hits_ = 1;
    ROS_INFO_STREAM_THROTTLE(2.0, "[path_manager] 二次校验：新框候选登记 ("
                              << batch.poses.size() << " 点)，等待确认");
  }
  pending_stamp_ = now;

  if (pending_hits_ >= static_cast<size_t>(validate_count_)) {
    pending_hits_ = 0;
    pending_batch_.poses.clear();
    ROS_INFO("[path_manager] 二次校验通过：新框 %zu 点直接入队", batch.poses.size());
    return true;  // 校验完成 → 调用方直接入队
  }
  return false;
}

// 3D 容器内距 p 最近且距离 ≤ enqueued_match_dist_ 的航点下标（跳过 skip 中
// 已被批次其他点占用的槽位，保证组员↔槽位一一对应）；无命中返回 kNoSlot。
// 线性扫描：队列 ~百点级 × 批 3 点 × 10 Hz，开销可忽略
size_t PathManager::nearestEnqueuedIndex(const Eigen::Vector3d& p,
                                         const std::vector<size_t>& skip) const {
  const double r_sq = enqueued_match_dist_ * enqueued_match_dist_;
  size_t best = kNoSlot;
  double best_sq = r_sq;
  for (size_t i = 0; i < path_.poses.size(); ++i) {
    if (std::find(skip.begin(), skip.end(), i) != skip.end()) continue;
    const double d_sq = (toVec3(path_.poses[i].pose.position) - p).squaredNorm();
    if (d_sq <= best_sq) {
      best_sq = d_sq;
      best = i;
    }
  }
  return best;
}

// 框心热存储入口（累积航点任务）：检测到框心即登记/更新（去重半径内同框观测
// 原位刷新，检测丢失不清除——约束随登记常驻，直到被穿越通过或漏过放行）
void PathManager::centerCallback(const geometry_msgs::PoseStamped::ConstPtr& msg) {
  if (recall_mode_) {
    return;  // 回程不登记新框心（不硬化回程路径）
  }
  const Eigen::Vector3d p = toVec3(msg->pose.position);
  for (CenterEntry& ce : centers_) {
    if ((ce.point - p).norm() < center_match_dist_) {
      ce.point = p;  // 同框观测更新（EMA 精化跟随）
      return;
    }
  }
  centers_.push_back(CenterEntry{p, false});
  if (centers_.size() > 64) {
    centers_.erase(centers_.begin());  // FIFO 上限：最老的先出
  }
  ROS_INFO("[path_manager] 框心登记：第 %zu 个 (%.2f, %.2f, %.2f)",
           centers_.size(), p.x(), p.y(), p.z());
}

void PathManager::appendToMission(const geometry_msgs::PoseStamped& wp) {
  complete_ = false;
  inside_since_ = ros::Time();

  if (path_.poses.empty()) {
    path_ = nav_msgs::Path();
    path_.header = wp.header;
    path_.poses.push_back(wp);
    cumulative_lengths_.assign(1, 0.0);
    has_path_ = true;
    const Eigen::Vector3d p = toVec3(wp.pose.position);
    ROS_INFO("[path_manager] 队列起始：首航点 (%.2f, %.2f, %.2f)", p.x(), p.y(), p.z());
    return;
  }

  const Eigen::Vector3d tail = toVec3(path_.poses.back().pose.position);
  const Eigen::Vector3d p = toVec3(wp.pose.position);
  const double d = (tail - p).norm();
  if (d < min_enqueue_spacing_) {
    // 同一特征的观测更新：替换队尾（只重算最后一段弧长）
    path_.poses.back() = wp;
    if (path_.poses.size() >= 2) {
      cumulative_lengths_.back() =
          cumulative_lengths_[cumulative_lengths_.size() - 2] +
          (toVec3(path_.poses[path_.poses.size() - 2].pose.position) - p).norm();
    }
    ROS_INFO_STREAM_THROTTLE(2.0, "[path_manager] 队尾观测更新 -> (" << p.x() << ", "
                              << p.y() << ", " << p.z() << ")");
    return;
  }

  if (max_queue_size_ > 0 && path_.poses.size() >= static_cast<size_t>(max_queue_size_)) {
    path_.poses.erase(path_.poses.begin());
    rebuildLengths();
  }

  path_.poses.push_back(wp);
  cumulative_lengths_.push_back(cumulative_lengths_.back() + d);
  has_path_ = true;
  ROS_INFO_STREAM_THROTTLE(2.0, "[path_manager] 航点入队：第 " << path_.poses.size()
                           << " 个，距队尾 " << d << " m，总长 "
                           << cumulative_lengths_.back() << " m");
}

void PathManager::rebuildLengths() {
  cumulative_lengths_.assign(path_.poses.size(), 0.0);
  for (size_t i = 1; i < path_.poses.size(); ++i) {
    cumulative_lengths_[i] =
        cumulative_lengths_[i - 1] +
        (toVec3(path_.poses[i - 1].pose.position) - toVec3(path_.poses[i].pose.position)).norm();
  }
}

// 框心穿越计数输出：centers_ 中已越过（passed）的框心数。穿越判定 = 路径
// 进度越过已登记框心 0.5 m，与检测器锁定状态解耦——锁丢失/幻影重锁时
// window_detector 的穿越事件（status done=N）会漏计（实测 6-7/9），此计数
// 以存储框心为准补齐；回溯进入时全部置 passed（计数=登记总数）。latch。
void PathManager::publishFrameCount() {
  std_msgs::Int32 msg;
  msg.data = static_cast<int>(std::count_if(
      centers_.begin(), centers_.end(), [](const CenterEntry& ce) { return ce.passed; }));
  frame_count_pub_.publish(msg);
}

// 替换语义输入共用：寄存任务路径并复位完成状态（空路径 = 清空任务）。
// 显式新任务同时解除回溯模式
void PathManager::setMission(const nav_msgs::Path& mission) {
  recall_mode_ = false;
  path_ = mission;
  cumulative_lengths_.clear();
  complete_ = false;
  inside_since_ = ros::Time();

  if (path_.poses.empty()) {
    has_path_ = false;
    ROS_WARN("[path_manager] 收到空路径，任务清空");
    return;
  }

  rebuildLengths();
  has_path_ = true;
  ROS_INFO("[path_manager] 寄存任务路径 %zu 个航点，总长 %.2f m",
           path_.poses.size(), cumulative_lengths_.back());
}

double PathManager::totalLength() const {
  return cumulative_lengths_.empty() ? 0.0 : cumulative_lengths_.back();
}

// 按弧长在路径上取点（线性插值）
SampledPoint PathManager::sampleAtProgress(double progress) const {
  SampledPoint sampled;
  if (path_.poses.empty()) {
    return sampled;
  }
  if (path_.poses.size() == 1) {
    sampled.position = toVec3(path_.poses.front().pose.position);
    return sampled;
  }

  const double s = std::max(0.0, std::min(progress, totalLength()));
  sampled.progress = s;
  if (s >= totalLength()) {
    sampled.position = toVec3(path_.poses.back().pose.position);
    sampled.segment_index = path_.poses.size() - 2;
    return sampled;
  }

  auto upper = std::upper_bound(cumulative_lengths_.begin(), cumulative_lengths_.end(), s);
  size_t idx = 0;
  if (upper != cumulative_lengths_.begin()) {
    idx = static_cast<size_t>(std::distance(cumulative_lengths_.begin(), upper) - 1);
    idx = std::min(idx, path_.poses.size() - 2);
  }
  const double seg_len =
      std::max(cumulative_lengths_[idx + 1] - cumulative_lengths_[idx], 1e-6);
  const double ratio = (s - cumulative_lengths_[idx]) / seg_len;
  const Eigen::Vector3d a = toVec3(path_.poses[idx].pose.position);
  const Eigen::Vector3d b = toVec3(path_.poses[idx + 1].pose.position);
  sampled.position = a + (b - a) * ratio;
  sampled.segment_index = idx;
  return sampled;
}

// 里程计位置到路径的最近点（逐段投影），progress = 投影点的累计弧长。
// 距离并列时取 progress 更大的——自回近路径不会退回已走过的段
SampledPoint PathManager::closestPointOnPath(const Eigen::Vector3d& p) const {
  SampledPoint best;
  double best_dist_sq = std::numeric_limits<double>::infinity();
  if (path_.poses.empty()) {
    return best;
  }
  if (path_.poses.size() == 1) {
    best.position = toVec3(path_.poses.front().pose.position);
    return best;
  }

  for (size_t i = 0; i + 1 < path_.poses.size(); ++i) {
    const Eigen::Vector3d a = toVec3(path_.poses[i].pose.position);
    const Eigen::Vector3d b = toVec3(path_.poses[i + 1].pose.position);
    const Eigen::Vector3d ab = b - a;
    const double ab_sq = ab.squaredNorm();

    double ratio = 0.0;
    if (ab_sq > 1e-9) {
      ratio = std::max(0.0, std::min(1.0, (p - a).dot(ab) / ab_sq));
    }

    const Eigen::Vector3d proj = a + ab * ratio;
    const double dist_sq = (p - proj).squaredNorm();
    const double progress = cumulative_lengths_[i] + std::sqrt(ab_sq) * ratio;

    if (dist_sq < best_dist_sq - 1e-9 ||
        (std::abs(dist_sq - best_dist_sq) <= 1e-9 && progress > best.progress)) {
      best_dist_sq = dist_sq;
      best.position = proj;
      best.progress = progress;
      best.segment_index = i;
    }
  }
  return best;
}

// 目标航向取所在段的切向；退化段向后/向前找最近的非退化段
double PathManager::tangentYaw(size_t segment_index) const {
  if (path_.poses.size() < 2) {
    return 0.0;
  }
  const size_t last = path_.poses.size() - 2;
  for (size_t k = segment_index; k <= last; ++k) {
    const Eigen::Vector3d d =
        toVec3(path_.poses[k + 1].pose.position) - toVec3(path_.poses[k].pose.position);
    if (std::hypot(d.x(), d.y()) > 1e-6) {
      return std::atan2(d.y(), d.x());
    }
  }
  for (size_t k = std::min(segment_index, last);; --k) {
    const Eigen::Vector3d d =
        toVec3(path_.poses[k + 1].pose.position) - toVec3(path_.poses[k].pose.position);
    if (std::hypot(d.x(), d.y()) > 1e-6) {
      return std::atan2(d.y(), d.x());
    }
    if (k == 0) {
      break;
    }
  }
  return 0.0;
}

void PathManager::timerCallback(const ros::TimerEvent&) {
  if (!has_odom_ || !has_path_ || complete_) {
    return;
  }

  const Eigen::Vector3d current = toVec3(odom_.pose.pose.position);
  const SampledPoint closest = closestPointOnPath(current);

  // 有效终点：开启队尾外推时 = 队尾 + 末段方向 × 外推上限。检测丢失/下一框
  // 未入列（不完美识别）期间 carrot 沿末段持续前移——无人机"继续飞行等待
  // 识别的后续航点入列"，而不是到队尾就收尾悬停；只有走到外推上限才停发
  //（此时仍无新批次，交给上层搜索行为）。外推点在未测绘空间，杂乱场景慎用
  //（框链近似共线时安全）。
  Eigen::Vector3d effective_final = toVec3(path_.poses.back().pose.position);
  Eigen::Vector3d tail_dir = Eigen::Vector3d::Zero();
  if (tail_extrapolate_max_ > 0.0 && path_.poses.size() >= 2) {
    const Eigen::Vector3d prev_pt = toVec3(path_.poses[path_.poses.size() - 2].pose.position);
    const Eigen::Vector3d tail_pt = toVec3(path_.poses.back().pose.position);
    const Eigen::Vector3d seg = tail_pt - prev_pt;
    if (seg.norm() > 1e-6) {
      tail_dir = seg.normalized();
      effective_final = tail_pt + tail_dir * tail_extrapolate_max_;
    }
  }

  // 到达判定：距有效终点进圈 + progress 走到末端，持续 dwell 才算完成。
  // 兜底分支：距尾 ≤ reach_fallback_dist 也算到达——无人机轻微越过头点时
  // （0.2 m 完成半径外），队尾落在机后会经 carrot 把飞机往回拽，"会退"；
  // 放宽判定让 carrot 停发，EGO 收尾悬停，新批次入队即恢复
  const bool at_final =
      ((current - effective_final).norm() <= reached_dist_ ||
       (reach_fallback_dist_ > 0.0 &&
        (current - effective_final).norm() <= reach_fallback_dist_)) &&
      closest.progress >= totalLength() - std::max(lookahead_, 0.5);
  if (at_final) {
    const ros::Time now = ros::Time::now();
    if (inside_since_.isZero()) {
      inside_since_ = now;
    }
    if (dwell_ <= 0.0 || (now - inside_since_).toSec() >= dwell_) {
      complete_ = true;
      ROS_INFO("[path_manager] 任务完成：终点 (%.2f, %.2f, %.2f) 停留 %.1f s，停发目标",
               effective_final.x(), effective_final.y(), effective_final.z(), dwell_);
      return;
    }
  } else {
    inside_since_ = ros::Time();
  }

  // ---- 框心硬化（连续自适应前视，MPCC lag/contour 思想的工程落地）----
  // 未越过的框心把有效 lookahead 从全量沿"距框心剩余弧长"线性收窄到下限：
  // carrot 沿 P1→框心→P3 穿心轴逐段推进，引导线贴轴，不被大 lookahead 一步
  // 跳过穿心段而离轴切角。连续斜坡替代旧版"进窗口即硬切 center_lookahead"
  // 的阶梯逻辑（进窗瞬间 5.0→2.0 单帧跳变，carrot 目标突退数米，EGO 被迫
  // 重规划）。下限随速度抬升：刹车距离 v²/2a 随速度增长，固定下限在高速时
  // 会把 carrot 压进减速区（纯跟踪经典 L=k·v+L0，k=lookahead_speed_gain，
  // 0=固定下限）。越过框心 0.5 m 即恢复全量加速（不为平滑牺牲出框加速）。
  // 非"必须经过"的到点语义：carrot 始终在前（下限 > 刹车距离），不停、不
  // 访问。框心落到机后仍未越过（漏过）：放行告警，绝不回拽
  double lookahead_eff = lookahead_;
  if (center_lookahead_ > 0.0 && !centers_.empty()) {
    const double span = std::max(center_gate_window_ > 0.0 ? center_gate_window_ : lookahead_ + 1.0, 0.5);
    // twist.linear 是 Vector3（toVec3 只吃 Point）；模长对体/世界系旋转不变
    const geometry_msgs::Vector3& w = odom_.twist.twist.linear;
    const double v = std::sqrt(w.x * w.x + w.y * w.y + w.z * w.z);
    const double floor_lo = std::min(center_lookahead_ + lookahead_speed_gain_ * v, lookahead_);
    double d_min = std::numeric_limits<double>::infinity();
    for (CenterEntry& ce : centers_) {
      if (ce.passed) {
        continue;
      }
      const SampledPoint proj = closestPointOnPath(ce.point);
      if (proj.progress < closest.progress - 0.5) {
        ce.passed = true;  // 已在后方：漏过放行，不回拽
        ROS_WARN("[path_manager] 框心漏过放行 (%.2f, %.2f, %.2f)",
                 ce.point.x(), ce.point.y(), ce.point.z());
        publishFrameCount();  // 穿越计数 +1（越过已登记框心）
        continue;
      }
      d_min = std::min(d_min, std::fabs(proj.progress - closest.progress));
    }
    if (d_min < span) {
      const double t = std::max(0.0, std::min(1.0, d_min / span));
      lookahead_eff = floor_lo + (lookahead_ - floor_lo) * t;
    }
  }

  SampledPoint target =
      sampleAtProgress(std::min(closest.progress + lookahead_eff, totalLength()));

  // 队尾外推（可选，默认关）：队列断供（检测丢失/下一框未入列）时沿末段方向
  // 虚拟延伸 carrot，保持目标在减速区之外——继续飞行等待后续航点入列；
  // 外推量封顶 tail_extrapolate_max，与上方完成判定的有效终点一致
  if (tail_extrapolate_max_ > 0.0 && tail_dir.norm() > 0.5 &&
      closest.progress + lookahead_eff > totalLength()) {
    const double extra =
        std::min(closest.progress + lookahead_eff - totalLength(), tail_extrapolate_max_);
    target.position += tail_dir * extra;
  }

  nav_msgs::Path goal;
  goal.header.stamp = ros::Time::now();
  goal.header.frame_id =
      path_.header.frame_id.empty() ? frame_ : path_.header.frame_id;
  geometry_msgs::PoseStamped pose;
  pose.header = goal.header;
  pose.pose.position = toPoint(target.position);
  pose.pose.orientation = yawToQuaternion(tangentYaw(target.segment_index));
  goal.poses.push_back(pose);
  goal_pub_.publish(goal);
}

}  // namespace path_manager
