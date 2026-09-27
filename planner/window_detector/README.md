# window_detector —— 方形框检测与穿框航点生成（方案 A）

按《[方案A_方形框检测与穿框航点生成.md](../../方案A_方形框检测与穿框航点生成.md)》实现。
输入世界系点云（FAST-LIO 配准）+ 里程计，输出**稳定**的穿框航点 P1/P2/P3。
本阶段不接入 EGO-Planner（接入方案见设计文档第 7 节）。

## 结构

```
include/window_detector/frame_detector.h   核心算法，零依赖纯 C++14（无 ROS/PCL/Eigen）
src/window_detector_node.cpp               ROS 薄壳节点
test/test_frame_detector.cpp               T1–T17 离线单测（g++ 直跑）
launch/window_detector_sim.launch          仿真（读 sim 链路点云/里程计）
launch/window_detector.launch              实机（FAST-LIO 输入）
```

三层职责分离（稳定性来自"单帧检测 + 多帧状态机"两层结构）：
单帧检测（无状态，宁严勿滥）→ 时间稳定器（SEARCHING/LOCKED 状态机，去抖/防误检/防丢锁）
→ 航点生成（锁定后解析计算）。

## 话题 / 服务

| 方向 | 名称 | 类型 | 说明 |
|---|---|---|---|
| 订阅 | `~cloud`（sim 重映射到 `/livox/lidar_world`；实机 `/cloud_registered`） | sensor_msgs/PointCloud2 | 世界系点云，queue=1 |
| 订阅 | `~odom`（实机 `/Odometry`） | nav_msgs/Odometry | 仅取位置 |
| 发布 | `~waypoints` | nav_msgs/Path（latch） | P1 对准 / P2 框心 / P3 穿出；解锁即清空 |
| 发布 | `~stable_frame` | geometry_msgs/PoseStamped（latch） | 位置=框中心，+X=飞行方向 |
| 发布 | `~markers` | visualization_msgs/MarkerArray | 粉框线框、绿法向箭头、航点球+连线 |
| 发布 | `~status` | std_msgs/String | 每帧：`LOCKED\|detect=OK\|progress=5/5\|miss=0/8\|frozen=0` |
| 服务 | `~reset` | std_srvs/Trigger | 清锁重来 |
| 服务 | `~commit_crossing` | std_srvs/Trigger | 冻结当前航点（穿越启动时调用一次） |
| 服务 | `~cancel_crossing` | std_srvs/Trigger | 取消冻结 |
| 发布 | `/move_base_simple/goal`（可选） | geometry_msgs/PoseStamped（latch） | `publish_goal:=true` 时，锁定期间每帧发布最新目标 = 框心 + 法向 × `goal_exit_dist`（EMA 跟随识别结果，冻结期间为冻结值），与 RViz 2D Nav Goal 同链：waypoint_generator → EGO 规划执行 |

## 运行

```bash
# 1) 离线单测（无需 ROS）
g++ -std=c++14 -O2 -I include test/test_frame_detector.cpp -o /tmp/frame_detector_test && /tmp/frame_detector_test

# 2) 构建本包（工作区根目录）
catkin build window_detector

# 3) 仿真：与 ego_planner sim.launch 同跑，直接读仿真链路的世界系点云与里程计
#    （/livox/lidar_world + /mavros/local_position/odom），检测锁定 → goal → EGO 规划执行
roslaunch window_detector window_detector_sim.launch

# 冻结验证：锁定后
rosservice call /window_detector/commit_crossing    # 航点冻结，扰动检测不再改变 Path
rosservice call /window_detector/cancel_crossing

# 4) 实机（需 FAST-LIO 已在跑；S1–S5 验收步骤见设计文档 6.3 节）
roslaunch window_detector window_detector.launch
```

## 简版 EGO 接入（goal 接口）

`window_detector.launch` 默认 `publish_goal:=true`：锁定期间把最新目标持续发到
`/move_base_simple/goal`（约 10 Hz，不固定锁死初值；commit 冻结期间发冻结值），走与
RViz 2D Nav Goal 完全相同的链路（waypoint_generator `manual-lonely-waypoint` →
`/waypoint_generator/waypoints` → EGO FSM，需 `flight_type=1`），EGO 持续朝当前目标规划
执行。解锁后停止发布，目标 latch 在最后值。

- **`goal_exit_dist`（新增参数，默认 0.5 m）**：goal 落在检测框心后方（沿穿框法向）的
  距离。要穿过框，局部目标必须越过框平面——0 = 恰为框心（轨迹到框平面减速悬停），
  建议 0.3–1.0 m。
- **`goal_z`（默认 −1）**：自定义航点高度。−1 = 跟随检测框心高度；在 launch 里改成
  ≥0 的值（如 1.2）即强制 goal 用该高度。
- **重要**：goal 只改目标，不改地图。`mid360.xml` 的 `grid_map/obstacles_inflation=0.30`
  会把 <1.2 m 框的洞完全堵死，EGO 将绕框而非穿框——实飞穿越前必须按设计文档 7.2 调到
  0.10–0.15（`optimization/dist0` 同步到 ~0.10）。
- 单目标接口不做轨迹过框校验；对精度要求高时再按文档 7.3 扩展 FSM 多航点接口。

## 参数

全部走私有命名空间 `~`，默认值与调参指南见设计文档第 5 节（检测参数 + 稳定器参数两张表）。
实机必调：`min_frame_size`（实际框边长 − 0.15）、`max_frame_size`（实际框边长 + 0.5）。
多目标消歧（可选）：`target_search_direction_x/y/z` 与 `target_size`，二者均未配置且场景中出现
两个合格框时返回 `AMBIGUOUS_TARGET`、保持 SEARCHING。

## 实现说明与偏差

- **检测输入**：实机直接订阅 FAST-LIO 原始配准云 `/cloud_registered`（未过 cloud_preprocess 的
  0.15 m 体素——该体素会抹掉细框点）。检测器自带 ROI ≥ 0.5 m 球裁剪（等效挡掉机体自打点）与
  0.05 m 体素质心降采样；`/cloud_ego_registered` → grid_map 链路完全不动。
- **尺寸度量偏差**（对文档 4.2(b) 的必要补充）：正方形框的面内协方差各向同性，PCA 面内主轴
  方向任意，直接量宽高会得到旋转包围盒（边长虚大至 √2 倍、长矩形误判方形）。实现中先做面内
  **最小面积矩形方向搜索**（两阶段角扫描）再取分位数包围盒，否则无法满足 T1/T6。
- **鲁棒性（平面拆分 + 封闭空洞优先）**：框黏在墙/支架上会聚成一簇，直接拟平面会失败或被大结构
  主导。`max_plane_splits`（默认 2）控制两层拆分：主平面校验失败后剥离其内点对残余结构再提
  平面；主平面提不出足够内点时（框贴近平行墙）按平面两侧拆开各试。几何度量采用**封闭空洞优先**：
  存在被材料包围的封闭空洞时以空洞几何为准（框心=空洞中心、尺寸=空洞包围盒+管厚），并施加三道
  形状无关的几何约束——空洞**矩形度** ≥ `hole_fill_ratio_min`（0.60，三角形/L 形/狭缝出局，
  圆≈0.73 通过）、**周界特征匹配度** ≥ `ring_match_min`（0.60：洞边界上 60% 的方向是
  ≤ `max_frame_thickness` 0.30 m 的薄框边即通过——多柱支撑的框柱子只占少数方向不会被
  误杀，厚墙开窗全周超厚仍被拒；管厚只从薄框边方向推算）、空洞尺寸窗；封闭空洞存在但约束不过则直接拒绝（斜三角形的斜向
  包围盒钻不了尺寸/宽高比的空子）。无封闭空洞（C 形缺边、稀疏部分可见）退回包围盒+中心净空路径。
  栅格分析前做 1 格形态学膨胀，封住旋转重采样/稀疏采样在连续边上留下的单格缺口。框某条边整体缺失（洞与外界连通、洪泛分不出洞）时做**四方向虚拟封口**：用开口两翼材料的延伸线补一行虚拟占据再找封闭洞，旋转无关——顶边打不到点（传感器垂直 FOV/稀疏常见）或任一边缺失都能检出（T20–T26）。
  全部失败时 status 报"走得最远"的候选原因，便于定位。
- **时间窗累积**：`accumulate_time`（默认 2 s）内的世界系点按体素去重累积后再检测——稀疏
  LiDAR 单帧只打中框的一两条边时逐帧补齐轮廓（T19）。
- **超时规则**：节点定时器周期 1/`expected_rate`；点云或里程计静默超 2/`expected_rate` 时，
  LOCKED 每周期计一次 miss（8 个周期解锁），SEARCHING 清空候选窗口。
- **冻结语义**：`commit_crossing` 后 c、n、P1/P2/P3 与 Path 冻结（markers 继续刷新）；
  `cancel_crossing` 恢复 EMA；任何路径解锁时冻结与航点一并清除（不输出陈旧航点）。
- **零依赖**：PointCloud2 用 `sensor_msgs` 迭代器解析；核心算法（Jacobi 特征分解、体素哈希、
  并查集聚类、空洞连通域）自带，无 PCL/Eigen。

## 已知局限（升级路径见设计文档第 8 节）

框嵌在墙/网上时聚类切不开（调小 `cluster_tol`，仍不行→方案 B）；远距单帧点数不足时降
`min_cluster_points`/`voxel_leaf`；与 EGO 集成前需按第 7 节调 `obstacles_inflation` 并扩展
FSM 多航点接口（当前 FSM 回调只读 `Path.poses[0]`）。
