// forecast_searching_sim：方案A 方形框检测与穿框航点生成 —— ROS 薄壳节点。
// 核心算法在 include/forecast_searching/frame_detector.h（零依赖）
#include <ros/ros.h>

#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <std_msgs/String.h>
#include <std_srvs/Trigger.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include <cmath>
#include <string>
#include <vector>

#include "forecast_searching/frame_detector.h"

namespace wd = forecast_searching;
// Vec3 是 std::array 别名，ADL 找不到 forecast_searching 的自由运算符，需显式引入
using forecast_searching::operator+;
using forecast_searching::operator-;
using forecast_searching::operator*;

namespace {

wd::Vec3 toVec3(const geometry_msgs::Point& p) { return wd::vec3(p.x, p.y, p.z); }
geometry_msgs::Point toPoint(const wd::Vec3& v) {
  geometry_msgs::Point p;
  p.x = v[0];
  p.y = v[1];
  p.z = v[2];
  return p;
}

// +X 轴对齐到 n 的四元数（stable_frame 的姿态约定：+X = 飞行方向）
geometry_msgs::Quaternion quatFromXAxis(const wd::Vec3& n) {
  geometry_msgs::Quaternion q;
  const wd::Vec3 x = wd::vec3(1, 0, 0);
  const double d = wd::dot(x, n);
  if (d > 1.0 - 1e-6) {  // 近似 +X：噪声叉积归一化会放大成怪参数，直接给 identity
    q.w = 1.0;
    return q;
  }
  wd::Vec3 c = wd::cross(x, n);
  const double cn = wd::norm(c);
  if (cn < 1e-9) {  // 平行或反平行
    if (d > 0) {
      q.w = 1.0;
      return q;
    }
    q.x = 0.0;
    q.y = 0.0;
    q.z = 1.0;
    q.w = 0.0;
    return q;
  }
  c = c * (1.0 / cn);
  const double s = std::sqrt(0.5 * (1.0 + d));
  const double t = 0.5 / s;
  q.x = c[0] * t;
  q.y = c[1] * t;
  q.z = c[2] * t;
  q.w = s;
  return q;
}

}  // namespace

class WindowDetectorNode {
 public:
  WindowDetectorNode() : nh_(), pnh_("~") {
    readParams();
    detector_ = wd::FrameDetector(dp_);
    stab_ = wd::FrameStabilizer(sp_);

    cloud_sub_ = pnh_.subscribe("cloud", 1, &WindowDetectorNode::cloudCallback, this);
    odom_sub_ = pnh_.subscribe("odom", 2, &WindowDetectorNode::odomCallback, this);

    waypoints_pub_ = pnh_.advertise<nav_msgs::Path>("waypoints", 1, /*latch=*/true);
    frame_pub_ = pnh_.advertise<geometry_msgs::PoseStamped>("stable_frame", 1, /*latch=*/true);
    markers_pub_ = pnh_.advertise<visualization_msgs::MarkerArray>("markers", 1);
    status_pub_ = pnh_.advertise<std_msgs::String>("status", 1);

    reset_srv_ = pnh_.advertiseService("reset", &WindowDetectorNode::resetSrv, this);
    commit_srv_ = pnh_.advertiseService("commit_crossing", &WindowDetectorNode::commitSrv, this);
    cancel_srv_ = pnh_.advertiseService("cancel_crossing", &WindowDetectorNode::cancelSrv, this);
    pause_srv_ = pnh_.advertiseService("pause", &WindowDetectorNode::pauseSrv, this);

    if (publish_goal_) {
      goal_pub_ = nh_.advertise<geometry_msgs::PoseStamped>(goal_topic_, 1, /*latch=*/true);
    }
    if (publish_path_) {
      path_pub_ = nh_.advertise<nav_msgs::Path>(path_out_topic_, 1);
      center_pub_ = nh_.advertise<geometry_msgs::PoseStamped>(center_out_topic_, 5);
    }

    timer_ = nh_.createTimer(ros::Duration(1.0 / std::max(sp_.expected_rate, 0.1)),
                             &WindowDetectorNode::timerCallback, this);

    ROS_INFO_STREAM("forecast_searching_sim ready. state=SEARCHING lock_count="
                    << sp_.lock_count << " unlock_miss=" << sp_.unlock_miss_count
                    << " approach=" << sp_.approach_dist << " exit=" << sp_.exit_dist
                    << (publish_goal_ ? " goal=" + goal_topic_ : "")
                    << (publish_goal_ ? " goal_exit_dist=" + std::to_string(goal_exit_dist_) : "")
                    << (publish_path_ ? " path=" + path_out_topic_ + " (三航点入队)" : ""));
  }

 private:
  void readParams() {
    // ---- 检测参数（文档 5 节 DetectorParams）----
    pnh_.param("min_range", dp_.min_range, dp_.min_range);
    pnh_.param("max_range", dp_.max_range, dp_.max_range);
    pnh_.param("voxel_leaf", dp_.voxel_leaf, dp_.voxel_leaf);
    pnh_.param("cluster_tol", dp_.cluster_tol, dp_.cluster_tol);
    pnh_.param("min_cluster_points", dp_.min_cluster_points, dp_.min_cluster_points);
    pnh_.param("plane_inlier_dist", dp_.plane_inlier_dist, dp_.plane_inlier_dist);
    pnh_.param("plane_refine_iters", dp_.plane_refine_iters, dp_.plane_refine_iters);
    pnh_.param("min_plane_inliers", dp_.min_plane_inliers, dp_.min_plane_inliers);
    pnh_.param("max_plane_splits", dp_.max_plane_splits, dp_.max_plane_splits);
    pnh_.param("min_frame_size", dp_.min_frame_size, dp_.min_frame_size);
    pnh_.param("max_frame_size", dp_.max_frame_size, dp_.max_frame_size);
    pnh_.param("max_aspect_ratio", dp_.max_aspect_ratio, dp_.max_aspect_ratio);
    pnh_.param("hole_check_ratio", dp_.hole_check_ratio, dp_.hole_check_ratio);
    pnh_.param("require_hole", dp_.require_hole, dp_.require_hole);
    pnh_.param("max_frame_thickness", dp_.max_frame_thickness, dp_.max_frame_thickness);
    pnh_.param("ring_match_min", dp_.ring_match_min, dp_.ring_match_min);
    pnh_.param("hole_fill_ratio_min", dp_.hole_fill_ratio_min, dp_.hole_fill_ratio_min);
    // 圆环开弧分支（近距开弧 RANSAC 圆拟合，闭合空洞路径失败时启用）
    pnh_.param("enable_circle_arc", dp_.enable_circle_arc, dp_.enable_circle_arc);
    pnh_.param("circle_min_arc_coverage", dp_.circle_min_arc_coverage, dp_.circle_min_arc_coverage);
    pnh_.param("circle_inlier_tol", dp_.circle_inlier_tol, dp_.circle_inlier_tol);
    pnh_.param("circle_min_inlier_ratio", dp_.circle_min_inlier_ratio, dp_.circle_min_inlier_ratio);
    // 已知环径约束（比赛环标准尺寸）：圆环精化固定半径、只解圆心与法向——
    // 斜视角/短弧下圆欠约束（自由拟合中心可偏 0.45 m），固定已知径后大幅改善
    pnh_.param("circle_known_diameter", dp_.circle_known_diameter, dp_.circle_known_diameter);
    // 骨架环检测（方向 A）：无环时端点桥接的缺口上限（格）；enable_skeleton 可整体关闭
    pnh_.param("skeleton_gap_cells", dp_.skeleton_gap_cells, dp_.skeleton_gap_cells);
    pnh_.param("enable_skeleton", dp_.enable_skeleton, dp_.enable_skeleton);
    pnh_.param("max_view_angle_deg", dp_.max_view_angle_deg, dp_.max_view_angle_deg);
    pnh_.param("extent_clip_quantile", dp_.extent_clip_quantile, dp_.extent_clip_quantile);
    pnh_.param("max_input_points", dp_.max_input_points, dp_.max_input_points);
    // 无先验多框歧义时的策略：true=就近优先（持续可锁），false=报歧义不锁定
    pnh_.param("prefer_nearest_on_ambiguous", dp_.prefer_nearest, dp_.prefer_nearest);
    // 就近粘滞的切换余量(m)：其余框须比当前选中框近这么多才切换，防逐帧跳变
    pnh_.param("nearest_switch_margin", dp_.nearest_switch_margin, dp_.nearest_switch_margin);

    double dx = 0, dy = 0, dz = 0;
    pnh_.param("target_search_direction_x", dx, 0.0);
    pnh_.param("target_search_direction_y", dy, 0.0);
    pnh_.param("target_search_direction_z", dz, 0.0);
    if (dx != 0 || dy != 0 || dz != 0) {
      dp_.has_target_direction = true;
      dp_.target_direction = wd::normalized(wd::vec3(dx, dy, dz));
    }
    pnh_.param("target_size", dp_.target_size, 0.0);
    dp_.has_target_size = dp_.target_size > 0.05;

    // ---- 稳定器参数（文档 5 节 StabilizerParams）----
    pnh_.param("lock_count", sp_.lock_count, sp_.lock_count);
    pnh_.param("unlock_miss_count", sp_.unlock_miss_count, sp_.unlock_miss_count);
    pnh_.param("history_len", sp_.history_len, sp_.history_len);
    pnh_.param("center_consistency", sp_.center_consistency, sp_.center_consistency);
    pnh_.param("angle_consistency_deg", sp_.angle_consistency_deg, sp_.angle_consistency_deg);
    pnh_.param("center_ema_alpha", sp_.center_ema_alpha, sp_.center_ema_alpha);
    pnh_.param("normal_ema_alpha", sp_.normal_ema_alpha, sp_.normal_ema_alpha);
    pnh_.param("size_ema_alpha", sp_.size_ema_alpha, sp_.size_ema_alpha);
    // 预测模型（Holt 阻尼趋势）：beta 0 = 纯 EMA 旧行为；>0 时锁定后的精化
    // 带趋势外推（收敛段提前到位），status 带 |pred= 一步预测残差
    pnh_.param("trend_beta", sp_.trend_beta, sp_.trend_beta);
    pnh_.param("trend_damping", sp_.trend_damping, sp_.trend_damping);
    pnh_.param("approach_dist", sp_.approach_dist, sp_.approach_dist);
    pnh_.param("exit_dist", sp_.exit_dist, sp_.exit_dist);
    pnh_.param("include_center_wp", sp_.include_center_wp, sp_.include_center_wp);
    pnh_.param("expected_rate", sp_.expected_rate, sp_.expected_rate);
    pnh_.param("freeze_on_commit", sp_.freeze_on_commit, sp_.freeze_on_commit);

    // ---- 简版 EGO 接入：锁定后把目标点发到 goal 话题（默认 /move_base_simple/goal，
    //      经 waypoint_generator 单航点下发 EGO，与 RViz 2D Nav Goal 同链）----
    pnh_.param("publish_goal", publish_goal_, false);
    pnh_.param<std::string>("goal_topic", goal_topic_, "/move_base_simple/goal");
    // goal 落在框心后方（沿穿框法向）多远：0=恰为框心；>0 越过框平面，保证"穿过"
    pnh_.param("goal_exit_dist", goal_exit_dist_, 0.5);
    // 自定义航点高度：默认 1.0 固定高度；传 <0（如 -1）用检测框心高度（跟随模式）
    pnh_.param("goal_z", goal_z_, 1.0);

    // ---- 三步穿框（GPA-Teleoperation 注视航点思想 / 技术说明 §6.1 / 方案A §4.4）：
    // ① 提取框心与法向 → ② goal 先给框心正前方对准点 P1=c−n·goal_approach_dist，
    // 机身速度摆到法向后 → ③ 切到穿出点 P_out=c+n·goal_exit_dist 直穿框心。
    // 单 goal 直发 P_out 时 EGO 可能斜切入框（撞框主因之一），分段强制先对准再穿。
    pnh_.param("staged_goal", staged_goal_, true);
    pnh_.param("goal_approach_dist", goal_approach_dist_, 1.5);
    // 到 P1 距离 < 该值(m) 或已越过 P1 深度平面 → 切入穿越段
    pnh_.param("goal_switch_tol", goal_switch_tol_, 0.5);
    // 首个对准航点与锁定时机位（起飞后≈起飞悬停点）距离小于该值(m)即取消：
    // 冗余航点会让 EGO 生成退化短轨迹干扰起飞后的首次规划
    pnh_.param("cancel_first_wp_dist", cancel_first_wp_dist_, 0.5);

    // ---- 注视三航点快速入队（接 path_manager enqueue_batch）：锁定期间把本框
    // [P1 对准点, P2 框心, P3 穿出点] 作为一组批量入队——path_manager 队尾
    // 原位更新（EMA 精化实时刷新）、新框整批追加，队列持久增长成完整穿框
    // 路径，carrot 沿队引导，进度只进不退（P1 滞后落机后时投影直接越过，
    // 不会拽回；不再需要 ALIGN→CROSS 的 goal 相位切换与首航点冗余判定——
    // 冻结提交时本组即冻结值）。与 publish_goal 互斥（同开时 goal 被禁用）。----
    pnh_.param("publish_path", publish_path_, false);
    pnh_.param<std::string>("path_out_topic", path_out_topic_, "/path_manager/enqueue_path");
    // 框心热存储输出：锁定期间把框心（EMA/冻结值）持续发到 center_out_topic，
    // path_manager 登记后硬化长程引导路径对穿心段的贴附（自适应 lookahead，
    // carrot 沿 P1→框心→P3 轴推进；检测丢失不丢约束）
    pnh_.param<std::string>("center_out_topic", center_out_topic_, "/path_manager/frame_centers");
    if (publish_path_ && publish_goal_) {
      ROS_WARN("publish_goal 与 publish_path 同时开启：goal 已禁用（避免双写打架）");
      publish_goal_ = false;
    }

    // 时间窗点云累积：稀疏 LiDAR 单帧只打中框的一两条边时，把窗口期内的世界系
    // 点累积起来补齐轮廓再检测（无人机移动不影响，点已在世界系）
    pnh_.param("accumulate", accumulate_, true);
    pnh_.param("accumulate_time", accumulate_time_, 2.0);
    pnh_.param("accumulate_voxel", accumulate_voxel_, 0.05);
    // 速度自适应累积窗：窗口内机体弧长封顶 accumulate_max_dist——高速时收缩
    // 窗口，世界系累积的配准误差/框壁壳厚预算不随速度增长（提速后框壁多层
    // 壳 → 平面拟合/中心估计跳变 → 锁定不稳的主因）；0 = 固定窗口（旧行为）
    pnh_.param("accumulate_max_dist", accumulate_max_dist_, accumulate_max_dist_);
    pnh_.param("accumulate_min_time", accumulate_min_time_, 0.3);
    accum_ = wd::PointAccumulator(accumulate_time_, accumulate_voxel_);

    // ---- 已穿越框过滤 ----
    pnh_.param("enable_traversed_filter", enable_traversed_filter_, true);
    // 检测中心与已穿越框中心距离 < 该值(m) 视为同一框
    pnh_.param("traversed_match_dist", traversed_match_dist_, 1.0);
    // 沿锁定法向越过框平面该距离(m) 判定"已穿越"
    pnh_.param("traversed_mark_clear", traversed_mark_clear_, 0.15);
    // 轴向横向偏差 < 框半径+该余量(m) 才算穿过（绕框侧飞不算）
    pnh_.param("traversed_axis_margin", traversed_axis_margin_, 0.5);

  }

  // ---- 回调 ----

  void odomCallback(const nav_msgs::OdometryConstPtr& msg) {
    const wd::Vec3 p = toVec3(msg->pose.pose.position);
    // 速度估计：odom 位置差分（fastlio 不填 twist，不能依赖）；EMA 平滑
    const ros::WallTime now_w = ros::WallTime::now();
    const double dt = (now_w - last_odom_wall_).toSec();
    if (last_odom_wall_ != ros::WallTime(0) && dt > 1e-3 && dt < 0.5) {
      const double v = wd::norm(p - last_odom_pos_) / dt;
      speed_est_ = 0.3 * v + 0.7 * speed_est_;
    }
    last_odom_pos_ = p;
    last_odom_wall_ = now_w;
    odom_pos_ = p;
    has_odom_ = true;
    stab_.onOdomFrame();
  }

  void cloudCallback(const sensor_msgs::PointCloud2ConstPtr& msg) {
    if (paused_) {  // 任务结束后暂停检测：只保 status 心跳
      publishStatus("PAUSED");
      return;
    }
    stab_.onCloudFrame();
    if (!has_odom_) {
      publishStatus("NO_ODOM");
      return;
    }
    std::vector<double> pts = parseCloud(*msg);

    // 稀疏单帧 → 时间窗累积后再检测（速度自适应：弧长封顶，高速收缩窗口）
    if (accumulate_) {
      if (accumulate_max_dist_ > 0.0) {
        const double w = std::max(accumulate_min_time_,
                                  std::min(accumulate_time_,
                                           accumulate_max_dist_ / std::max(speed_est_, 0.1)));
        accum_.setWindowTime(w);
        ROS_INFO_STREAM_THROTTLE(5.0, "forecast_searching: 累积窗 " << w
                                 << " s (v=" << speed_est_ << " m/s, pts=" << pts.size() / 3 << ")");
      }
      accum_.insert(pts);
      pts = accum_.points();
    }
    last_points_ = static_cast<int>(pts.size() / 3);

    wd::FrameDetection det;
    const wd::DetectStatus st = detector_.detect(pts, odom_pos_, det,
                                                 have_sticky_ ? &sticky_center_ : nullptr);
    last_detect_status_ = wd::detectStatusToString(st);
    if (st == wd::DetectStatus::OK) {
      sticky_center_ = det.center;  // 就近粘滞：记住本帧选中的框
      have_sticky_ = true;
    }

    // 穿越记账：越过锁定框平面即拉黑该框并解锁（在喂检测结果前做，
    // 本帧起就不再往稳定器里送这个框）
    checkTraversed();

    const bool was_locked = stab_.locked();
    if (st == wd::DetectStatus::OK) {
      stab_.onDetection(det);
    } else {
      stab_.onMiss();  // 含 AMBIGUOUS_TARGET：按无有效检测处理
    }
    // 新锁定（含穿越解锁后的重锁）：记录锁定时机位（用于对准航点冗余/高度中性判定）。
    // 首个框整体跳过 ALIGN：第一个 goal 直接是穿框目标（框心方向），
    // 避免起飞后"对准→穿越"两段切换的高度抖动；后续框正常走三流程。
    if (!was_locked && stab_.locked()) {
      goal_cross_phase_ = !first_frame_seen_;  // 首帧 true=直接 CROSS（跳过 ALIGN）
      first_frame_seen_ = true;
      lock_origin_ = odom_pos_;
    }
    handleTransition(was_locked);
    publishStatus(last_detect_status_);

    if (stab_.locked()) {
      // 目标不固定：锁定期间每帧跟随最新 EMA 锁定值（冻结期间为冻结值）
      if (publish_goal_) publishGoal();
      if (publish_path_) publishPath();
      if (!stab_.frozen()) {
        publishLockedOutputs(msg->header);
      } else {
        publishMarkers(msg->header);  // 冻结期间只刷新可视化，Path 不再重发
      }
    }
  }

  void timerCallback(const ros::TimerEvent&) {
    const bool was_locked = stab_.locked();
    stab_.onTimerTick();
    handleTransition(was_locked, "TIMEOUT");
    // 输入静默期也保持 status 可观测（每周期 10 Hz，小消息）
    publishStatus(last_detect_status_);
  }

  // ---- 服务 ----

  bool resetSrv(std_srvs::Trigger::Request&, std_srvs::Trigger::Response& res) {
    paused_ = false;  // pause 的逆操作：恢复检测
    stab_.reset();
    goal_cross_phase_ = false;
    clearOutputs();
    traversed_.clear();  // reset 连已穿越框记录一起清：重新允许锁定全部框
    traversed_events_ = 0;
    detector_.setExclusions(std::vector<wd::FrameDetector::Exclusion>());
    have_sticky_ = false;  // 就近粘滞也复位
    res.success = true;
    res.message = "reset";
    publishStatus("RESET");
    ROS_INFO("forecast_searching: reset");
    return true;
  }

  bool commitSrv(std_srvs::Trigger::Request&, std_srvs::Trigger::Response& res) {
    if (stab_.commit()) {
      std_msgs::Header h;
      h.stamp = ros::Time::now();
      h.frame_id = frame_id_;
      publishLockedOutputs(h);
      res.success = true;
      res.message = "crossing committed, waypoints frozen";
      publishStatus(last_detect_status_);
      ROS_INFO("forecast_searching: crossing committed, waypoints frozen");
    } else {
      res.success = false;
      res.message = "not locked (or freeze_on_commit disabled), nothing to commit";
    }
    return true;
  }

  bool cancelSrv(std_srvs::Trigger::Request&, std_srvs::Trigger::Response& res) {
    stab_.cancelCommit();
    res.success = true;
    res.message = "crossing cancelled";
    publishStatus(last_detect_status_);
    ROS_INFO("forecast_searching: crossing cancelled");
    return true;
  }

  // 任务达标收尾用：暂停检测（清输出/解锁/停粘滞，cloudCallback 早退）——
  // 不再锁定、不再往 path_manager 入队新框，已穿越计数冻结保留。
  // reset 服务恢复检测。地面悬停检查时也可手动调用
  bool pauseSrv(std_srvs::Trigger::Request&, std_srvs::Trigger::Response& res) {
    paused_ = true;
    stab_.reset();
    clearOutputs();
    have_sticky_ = false;
    res.success = true;
    res.message = "paused (call reset to resume)";
    publishStatus("PAUSED");
    ROS_INFO("forecast_searching: paused (done=%d kept)", traversed_events_);
    return true;
  }

  // ---- 工具 ----

  // PointCloud2 → 扁平 double 数组；遍历 fields 取 x/y/z（float32），
  // 跳过 NaN，超过 max_input_points 按步长抽稀。
  std::vector<double> parseCloud(const sensor_msgs::PointCloud2& msg) {
    frame_id_ = msg.header.frame_id;
    const size_t n = static_cast<size_t>(msg.width) * msg.height;
    size_t stride = 1;
    if (n > static_cast<size_t>(dp_.max_input_points)) {
      stride = n / static_cast<size_t>(dp_.max_input_points) + 1;
    }
    std::vector<double> pts;
    pts.reserve(3 * (n / stride + 1));
    sensor_msgs::PointCloud2ConstIterator<float> ix(msg, "x"), iy(msg, "y"), iz(msg, "z");
    for (size_t i = 0; i < n; ++i, ++ix, ++iy, ++iz) {
      if (i % stride != 0) continue;
      const double x = *ix, y = *iy, z = *iz;
      if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) continue;
      pts.push_back(x);
      pts.push_back(y);
      pts.push_back(z);
    }
    return pts;
  }

  // 已穿越框记账。锁定期间监测机体相对框平面的有向距离 s = (机位置-框心)·法向：
  // 法向来自锁定时的检测序列，固定指向远离接近侧——穿越前 s<0，穿越后 s>0。
  // s 越过阈值且横向偏差在框半径内 → 记入排除表、立即解锁停发 goal
  // （EGO 侧 latch 的最后一个 goal 继续引导飞出穿出段），后续检测直接跳过该框。
  void checkTraversed() {
    if (!stab_.locked()) {
      saw_approach_side_ = false;  // 每次锁定从框前接近重新观察
      return;
    }
    const wd::FrameDetection& f = stab_.vizFrame();
    const wd::Vec3 rel = odom_pos_ - f.center;
    const double s = wd::dot(rel, f.normal);
    if (s < -0.1) {
      saw_approach_side_ = true;
      return;
    }
    if (!enable_traversed_filter_ || !saw_approach_side_ || s <= traversed_mark_clear_) return;
    const double lat = wd::norm(rel - f.normal * s);
    const double radius = 0.5 * std::max(f.width, f.height);
    if (lat > radius + traversed_axis_margin_) return;  // 绕到框侧面外：不算穿越
    // ===== 穿越点（物理真值）=====
    // 标记位置用机体实际穿越点（里程计），不用锁定中心——实测方框锁定中心可沿
    // 法向偏 ±1.2-1.5 m（幻影平面），打在偏移中心上会漏掉真实框区域 → 二次锁定
    // → 同框重复穿。穿越点必在真实框平面附近，排除区必然罩住真实框。
    const wd::Vec3 cp = odom_pos_ - f.normal * (s - traversed_mark_clear_);
    // ===== 穿越拉黑（球形兜底 + 平面胶囊）=====
    //      球形 = 穿越点 ± max(match_dist, radius)，旧语义；
    //      平面胶囊 = 板厚 1.0 m × 面内半径 3.2 m ——半框锁定的中心沿框内水平轴
    //      偏 ±1.5 m，第二锁定中心距第一穿越点 ~3 m，球形盖不住同框另一侧；
    //      法向估计始终准确，胶囊罩住整个已穿框平面区域，斜交邻框靠板厚豁免
    ++traversed_events_;  // 每次穿越事件计 1（traversed_ 含球形+胶囊两条排除项，不能当计数）
    traversed_.push_back(wd::FrameDetector::Exclusion{cp, std::max(traversed_match_dist_, radius)});
    {
      wd::FrameDetector::Exclusion plane_ex;
      plane_ex.center = cp;
      plane_ex.radius = 3.2;          // 面内：1.5(中心偏移) + 1.18(框半宽) + 余量
      plane_ex.normal = f.normal;
      plane_ex.slab = 1.0;
      traversed_.push_back(plane_ex);
    }
    detector_.setExclusions(traversed_);
    saw_approach_side_ = false;
    stab_.reset();
    clearOutputs();
    goal_cross_phase_ = false;
    const int n_done = traversed_events_;
    ROS_WARN("forecast_searching: frame TRAVERSED at (%.2f, %.2f, %.2f) excluded (total %d), unlocked",
             cp[0], cp[1], cp[2], n_done);
  }

  void handleTransition(bool was_locked, const std::string& reason = "") {
    if (was_locked && !stab_.locked()) {
      clearOutputs();
      publishStatus(reason.empty() ? ("UNLOCKED after " + last_detect_status_) : ("UNLOCKED_" + reason));
      ROS_WARN("forecast_searching: UNLOCKED (%s)",
               reason.empty() ? last_detect_status_.c_str() : reason.c_str());
    }
  }

  // 当前目标点（含 ALIGN→CROSS 分段状态迁移），publishGoal/publishPath 共用。
  // 三步穿框：先对准 P1=c−n·approach（ALIGN 段），到位后切 P_out=c+n·exit
  // （CROSS 段）；分段状态随锁定/穿越复位，新框从 ALIGN 重新开始。
  wd::Vec3 currentGoalTarget() {
    const wd::FrameDetection& f = stab_.vizFrame();
    wd::Vec3 target;
    if (staged_goal_ && !goal_cross_phase_) {
      const wd::Vec3 p1 = f.center - f.normal * goal_approach_dist_;
      // s<0 在框前；到 P1 足够近，或已越过 P1 深度（避免在对准点卡死）
      const double s = wd::dot(odom_pos_ - f.center, f.normal);
      // 首个对准航点冗余判定：P1 距锁定时机位（起飞后首个锁定≈起飞悬停点）不足
      // cancel_first_wp_dist 时，该航点与当前位置几乎重合，EGO 会为它生成退化
      // 短轨迹干扰规划——取消 ALIGN，本框直接进入 CROSS 穿越段
      if (wd::norm(p1 - lock_origin_) < cancel_first_wp_dist_ ||
          wd::norm(odom_pos_ - p1) < goal_switch_tol_ || s > -goal_approach_dist_ * 0.3) {
        goal_cross_phase_ = true;
        target = f.center + f.normal * goal_exit_dist_;
      } else {
        // 对准段高度中性（跟随模式）：只做水平对准，不改变当前高度——
        // 起飞高度不被第一个航点拉走；爬到穿框高度交给 CROSS 段
        target = p1;
        if (goal_z_ < 0.0) target[2] = lock_origin_[2];
      }
    } else {
      target = f.center + f.normal * goal_exit_dist_;
    }
    if (goal_z_ >= 0.0) target[2] = goal_z_;  // 自定义航点高度
    return target;
  }

  // 简版 EGO 接入：锁定期间把最新目标点持续发到 goal 话题（EMA 跟随识别结果，
  // 冻结期间为冻结值）。
  void publishGoal() {
    const wd::FrameDetection& f = stab_.vizFrame();
    const wd::Vec3 target = currentGoalTarget();
    geometry_msgs::PoseStamped goal;
    goal.header.stamp = ros::Time::now();
    goal.header.frame_id = frame_id_;
    goal.pose.position = toPoint(target);
    goal.pose.orientation = quatFromXAxis(f.normal);
    goal_pub_.publish(goal);
    ROS_INFO_STREAM_THROTTLE(5.0, "forecast_searching: goal[" << (goal_cross_phase_ ? "CROSS" : "ALIGN")
                             << "] -> " << goal_topic_ << " (" << target[0] << ", "
                             << target[1] << ", " << target[2] << ")");
  }

  // 注视三航点批量入队输出：本框 [P1 对准点, P2 框心, P3 穿出点] 一组发到
  // path_out_topic（enqueue_batch）。path_manager 侧队尾逐点邻近则原位更新
  // （EMA 精化刷新本组），新框整批追加——队列即完整穿框路径。
  void publishPath() {
    const wd::FrameDetection& f = stab_.vizFrame();
    nav_msgs::Path batch;
    batch.header.stamp = ros::Time::now();
    batch.header.frame_id = frame_id_;
    auto addPose = [&](wd::Vec3 t) {
      geometry_msgs::PoseStamped pose;
      pose.header = batch.header;
      pose.pose.position = toPoint(t);
      pose.pose.orientation.w = 1.0;
      batch.poses.push_back(pose);
    };
    wd::Vec3 p1 = f.center - f.normal * goal_approach_dist_;
    if (goal_z_ < 0.0)
      p1[2] = lock_origin_[2];  // 对准点高度中性（跟随模式）：不被首航点拉高度
    else
      p1[2] = goal_z_;
    addPose(p1);
    wd::Vec3 p2 = f.center;
    if (goal_z_ >= 0.0) p2[2] = goal_z_;
    addPose(p2);
    // 框心同步登记到热存储（穿心轴约束点；path_manager 侧去重更新）
    geometry_msgs::PoseStamped cp;
    cp.header = batch.header;
    cp.pose.position = toPoint(p2);
    cp.pose.orientation.w = 1.0;
    center_pub_.publish(cp);
    wd::Vec3 p3 = f.center + f.normal * goal_exit_dist_;
    if (goal_z_ >= 0.0) p3[2] = goal_z_;
    addPose(p3);
    path_pub_.publish(batch);
    ROS_INFO_STREAM_THROTTLE(5.0, "forecast_searching: batch[P1,P2,P3] -> " << path_out_topic_
                             << " (" << p1[0] << "," << p1[1] << "," << p1[2] << ")-("
                             << p3[0] << "," << p3[1] << "," << p3[2] << ")");
  }

  void publishStatus(const std::string& detect_status) {
    std_msgs::String s;
    const int n_done = traversed_events_;
    char pred_buf[24] = {0};
    if (stab_.locked() && stab_.lastPredResidual() >= 0.0) {
      std::snprintf(pred_buf, sizeof(pred_buf), "|pred=%.2f", stab_.lastPredResidual());
    }
    s.data = (stab_.locked() ? std::string("LOCKED") : std::string("SEARCHING")) +
             "|detect=" + detect_status +
             "|progress=" + std::to_string(stab_.lockProgress()) + "/" + std::to_string(sp_.lock_count) +
             "|miss=" + std::to_string(stab_.missCount()) + "/" + std::to_string(sp_.unlock_miss_count) +
             "|frozen=" + (stab_.frozen() ? "1" : "0") +
             "|pts=" + std::to_string(last_points_) +
             pred_buf +
             "|done=" + std::to_string(n_done);
    status_pub_.publish(s);
  }

  void publishLockedOutputs(const std_msgs::Header& hdr) {
    const wd::FrameDetection& f = stab_.vizFrame();
    const auto wps = stab_.waypoints();

    nav_msgs::Path path;
    path.header = hdr;
    if (path.header.frame_id.empty()) path.header.frame_id = frame_id_;
    for (const wd::Vec3& p : wps) {
      geometry_msgs::PoseStamped pose;
      pose.header = path.header;
      pose.pose.position = toPoint(p);
      pose.pose.orientation.w = 1.0;
      path.poses.push_back(pose);
    }
    waypoints_pub_.publish(path);

    geometry_msgs::PoseStamped frame_msg;
    frame_msg.header = path.header;
    frame_msg.pose.position = toPoint(f.center);
    frame_msg.pose.orientation = quatFromXAxis(f.normal);
    frame_pub_.publish(frame_msg);

    publishMarkers(hdr);
  }

  void publishMarkers(const std_msgs::Header& hdr) {
    const wd::FrameDetection& f = stab_.vizFrame();
    const auto wps = stab_.waypoints();
    const std::string frame = hdr.frame_id.empty() ? frame_id_ : hdr.frame_id;

    visualization_msgs::MarkerArray array;
    ros::Time now = ros::Time::now();

    // 框线框（粉）：中心 + 面内轴重建四角
    wd::Vec3 u = stab_.vizAxis();
    if (wd::norm(u) < 0.5 || std::fabs(wd::dot(u, f.normal)) > 0.1) u = wd::anyPerpendicular(f.normal);
    else u = wd::normalized(u - f.normal * wd::dot(u, f.normal));
    const wd::Vec3 v = wd::normalized(wd::cross(f.normal, u));
    const double hw = 0.5 * f.width, hh = 0.5 * f.height;
    const wd::Vec3 c00 = f.center - u * hw - v * hh;
    const wd::Vec3 c10 = f.center + u * hw - v * hh;
    const wd::Vec3 c11 = f.center + u * hw + v * hh;
    const wd::Vec3 c01 = f.center - u * hw + v * hh;
    visualization_msgs::Marker wire;
    wire.header.frame_id = frame;
    wire.header.stamp = now;
    wire.ns = "frame_outline";
    wire.id = 0;
    wire.type = visualization_msgs::Marker::LINE_LIST;
    wire.action = visualization_msgs::Marker::ADD;
    wire.scale.x = 0.03;
    wire.color.r = 1.0f;
    wire.color.g = 0.2f;
    wire.color.b = 0.8f;
    wire.color.a = 1.0f;
    for (const wd::Vec3* p : {&c00, &c10, &c10, &c11, &c11, &c01, &c01, &c00}) {
      wire.points.push_back(toPoint(*p));
    }
    array.markers.push_back(wire);

    // 法向箭头（绿，飞行方向）
    visualization_msgs::Marker arrow;
    arrow.header = wire.header;
    arrow.ns = "normal_arrow";
    arrow.id = 1;
    arrow.type = visualization_msgs::Marker::ARROW;
    arrow.action = visualization_msgs::Marker::ADD;
    arrow.scale.x = 0.05;
    arrow.scale.y = 0.12;
    arrow.scale.z = 0.12;
    arrow.color.r = 0.1f;
    arrow.color.g = 0.9f;
    arrow.color.b = 0.2f;
    arrow.color.a = 1.0f;
    arrow.points.push_back(toPoint(f.center));
    arrow.points.push_back(toPoint(f.center + f.normal * 0.8));
    array.markers.push_back(arrow);

    // 航点球 + 连线
    if (!wps.empty()) {
      visualization_msgs::Marker spheres;
      spheres.header = wire.header;
      spheres.ns = "waypoints";
      spheres.id = 2;
      spheres.type = visualization_msgs::Marker::SPHERE_LIST;
      spheres.action = visualization_msgs::Marker::ADD;
      spheres.scale.x = spheres.scale.y = spheres.scale.z = 0.1;
      spheres.color.r = 0.2f;
      spheres.color.g = 0.6f;
      spheres.color.b = 1.0f;
      spheres.color.a = 1.0f;
      visualization_msgs::Marker line;
      line.header = wire.header;
      line.ns = "waypoint_line";
      line.id = 3;
      line.type = visualization_msgs::Marker::LINE_STRIP;
      line.action = visualization_msgs::Marker::ADD;
      line.scale.x = 0.02;
      line.color.r = 1.0f;
      line.color.g = 1.0f;
      line.color.b = 1.0f;
      line.color.a = 0.9f;
      for (const wd::Vec3& p : wps) {
        spheres.points.push_back(toPoint(p));
        line.points.push_back(toPoint(p));
      }
      array.markers.push_back(spheres);
      array.markers.push_back(line);
    }
    markers_pub_.publish(array);
  }

  // 解锁/复位：清空已发布航点（latch 覆盖）并删除所有 marker
  void clearOutputs() {
    nav_msgs::Path path;
    path.header.stamp = ros::Time::now();
    path.header.frame_id = frame_id_;
    waypoints_pub_.publish(path);
    frame_pub_.publish(geometry_msgs::PoseStamped());
    visualization_msgs::MarkerArray array;
    visualization_msgs::Marker del;
    del.header.frame_id = frame_id_;
    del.header.stamp = ros::Time::now();
    del.action = visualization_msgs::Marker::DELETEALL;
    array.markers.push_back(del);
    markers_pub_.publish(array);
  }

  ros::NodeHandle nh_, pnh_;
  ros::Subscriber cloud_sub_, odom_sub_;
  ros::Publisher waypoints_pub_, frame_pub_, markers_pub_, status_pub_;
  ros::Publisher goal_pub_;
  ros::ServiceServer reset_srv_, commit_srv_, cancel_srv_, pause_srv_;
  ros::Timer timer_;

  wd::DetectorParams dp_;
  wd::StabilizerParams sp_;
  wd::FrameDetector detector_;
  wd::FrameStabilizer stab_;

  bool has_odom_ = false;
  wd::Vec3 odom_pos_ = {{0, 0, 0}};
  int last_points_ = 0;
  bool accumulate_ = true;
  double accumulate_time_ = 2.0;
  double accumulate_voxel_ = 0.05;
  double accumulate_max_dist_ = 2.0;  // 速度自适应累积窗弧长上限，0 = 固定窗口；
                                       // 2.0 = 常规竞速速度（≤2 m/s）下不缩窗
  double accumulate_min_time_ = 0.3;  // 自适应窗口下限（防点数不足）
  wd::PointAccumulator accum_;
  double speed_est_ = 0.0;                       // odom 差分速度（EMA）
  wd::Vec3 last_odom_pos_ = {{0, 0, 0}};
  ros::WallTime last_odom_wall_;
  std::string last_detect_status_ = "INIT";
  std::string frame_id_ = "world";
  bool publish_goal_ = false;
  std::string goal_topic_ = "/move_base_simple/goal";
  double goal_exit_dist_ = 0.5;
  double goal_z_ = 1.0;

  // 注视三航点批量入队（publish_path 模式）
  bool publish_path_ = false;
  std::string path_out_topic_ = "/path_manager/enqueue_path";
  std::string center_out_topic_ = "/path_manager/frame_centers";
  ros::Publisher path_pub_;
  ros::Publisher center_pub_;

  // 三步穿框分段状态：false=ALIGN（对准 P1），true=CROSS（直穿 P_out）
  bool staged_goal_ = true;
  double goal_approach_dist_ = 1.5;
  double goal_switch_tol_ = 0.5;
  double cancel_first_wp_dist_ = 0.5;
  bool goal_cross_phase_ = false;
  bool first_frame_seen_ = false;        // 首框跳过 ALIGN（直去框心，防起步高度抖动）
  wd::Vec3 lock_origin_ = {{0, 0, 0}};  // 本次锁定时机的机体位置（首个航点冗余判定基准）

  // 已穿越框过滤状态
  bool enable_traversed_filter_ = true;
  double traversed_match_dist_ = 1.0;
  double traversed_mark_clear_ = 0.15;
  double traversed_axis_margin_ = 0.5;
  bool saw_approach_side_ = false;
  int traversed_events_ = 0;  // 穿越事件计数（done 语义：每次锁定穿越计 1）
  std::vector<wd::FrameDetector::Exclusion> traversed_;


  // 就近粘滞：上一帧就近策略选中的候选中心
  bool have_sticky_ = false;
  wd::Vec3 sticky_center_ = {{0, 0, 0}};

  // 任务暂停（pause 服务）：true 时不再检测/入队，reset 恢复
  bool paused_ = false;
};

int main(int argc, char** argv) {
  ros::init(argc, argv, "forecast_searching_sim");
  WindowDetectorNode node;
  ros::spin();
  return 0;
}
