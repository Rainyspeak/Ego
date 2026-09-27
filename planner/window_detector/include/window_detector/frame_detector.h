#ifndef WINDOW_DETECTOR_FRAME_DETECTOR_H
#define WINDOW_DETECTOR_FRAME_DETECTOR_H

//方形框检测与穿框航点生成

#include <algorithm>
#include <cstdio>
#include <array>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdint>
#include <deque>
#include <functional>
#include <numeric>
#include <random>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace window_detector {

static const double kPi = 3.14159265358979323846;

// ---------------- 基础向量运算 ----------------

using Vec3 = std::array<double, 3>;

inline Vec3 vec3(double x, double y, double z) { return Vec3{{x, y, z}}; }
inline Vec3 operator+(const Vec3& a, const Vec3& b) { return Vec3{{a[0] + b[0], a[1] + b[1], a[2] + b[2]}}; }
inline Vec3 operator-(const Vec3& a, const Vec3& b) { return Vec3{{a[0] - b[0], a[1] - b[1], a[2] - b[2]}}; }
inline Vec3 operator-(const Vec3& a) { return Vec3{{-a[0], -a[1], -a[2]}}; }
inline Vec3 operator*(const Vec3& a, double s) { return Vec3{{a[0] * s, a[1] * s, a[2] * s}}; }
inline Vec3 operator*(double s, const Vec3& a) { return a * s; }
inline double dot(const Vec3& a, const Vec3& b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
inline Vec3 cross(const Vec3& a, const Vec3& b) {
  return Vec3{{a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]}};
}
inline double norm(const Vec3& a) { return std::sqrt(dot(a, a)); }
inline Vec3 normalized(const Vec3& a) {
  const double n = norm(a);
  return n > 1e-12 ? a * (1.0 / n) : Vec3{{0, 0, 0}};
}
inline double angleBetween(const Vec3& a, const Vec3& b) {
  const double d = dot(a, b) / std::max(norm(a) * norm(b), 1e-12);
  return std::acos(std::max(-1.0, std::min(1.0, d)));
}
// 任意一个与 n 垂直的单位向量
inline Vec3 anyPerpendicular(const Vec3& n) {
  const Vec3 helper = std::fabs(n[2]) < 0.9 ? vec3(0, 0, 1) : vec3(1, 0, 0);
  return normalized(cross(n, helper));
}

// 对称 3×3 矩阵 Jacobi 特征分解；eval 升序，evec[i] 为对应单位特征向量
inline void symEigen3(const double a_in[3][3], double eval[3], Vec3 evec[3]) {
  double a[3][3] = {{a_in[0][0], a_in[0][1], a_in[0][2]},
                    {a_in[1][0], a_in[1][1], a_in[1][2]},
                    {a_in[2][0], a_in[2][1], a_in[2][2]}};
  double v[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
  for (int sweep = 0; sweep < 64; ++sweep) {
    const double off = std::fabs(a[0][1]) + std::fabs(a[0][2]) + std::fabs(a[1][2]);
    const double diag = std::fabs(a[0][0]) + std::fabs(a[1][1]) + std::fabs(a[2][2]);
    if (off <= 1e-14 * std::max(diag, 1e-300)) break;
    for (int p = 0; p < 3; ++p) {
      for (int q = p + 1; q < 3; ++q) {
        if (std::fabs(a[p][q]) < 1e-300) continue;
        const double theta = 0.5 * std::atan2(2.0 * a[p][q], a[q][q] - a[p][p]);
        const double c = std::cos(theta), s = std::sin(theta);
        for (int k = 0; k < 3; ++k) {  // A <- J^T A J（列变换 + 行变换）
          const double akp = a[k][p], akq = a[k][q];
          a[k][p] = c * akp - s * akq;
          a[k][q] = s * akp + c * akq;
        }
        for (int k = 0; k < 3; ++k) {
          const double apk = a[p][k], aqk = a[q][k];
          a[p][k] = c * apk - s * aqk;
          a[q][k] = s * apk + c * aqk;
        }
        for (int k = 0; k < 3; ++k) {  // V <- V J
          const double vkp = v[k][p], vkq = v[k][q];
          v[k][p] = c * vkp - s * vkq;
          v[k][q] = s * vkp + c * vkq;
        }
      }
    }
  }
  int order[3] = {0, 1, 2};
  std::sort(order, order + 3, [&](int i, int j) { return a[i][i] < a[j][j]; });
  for (int i = 0; i < 3; ++i) {
    eval[i] = a[order[i]][order[i]];
    evec[i] = normalized(vec3(v[0][order[i]], v[1][order[i]], v[2][order[i]]));
  }
}

// ---------------- 参数（默认值 = 文档第 5 节） ----------------

struct DetectorParams {
  double min_range = 0.5;              // ROI 近距（兼挡机体/桨叶自打点）
  double max_range = 8.0;              // ROI 远距
  double voxel_leaf = 0.05;            // 体素降采样
  double cluster_tol = 0.25;           // 欧式聚类连通阈值
  int min_cluster_points = 40;         // 最小簇点数（体素点）
  double plane_inlier_dist = 0.05;     // 平面内点半径
  int plane_refine_iters = 2;          // 平面精化轮数
  int min_plane_inliers = 25;          // 每轮最少平面内点
  // 平面拆分轮数（鲁棒性核心）：框黏在墙/支架上成一簇时，先取主平面校验，
  // 失败后剥离该平面内点、对残余结构再提平面（互不共面的框面会被拆出来）；
  // 主平面提不出时按平面两侧拆（处理框贴近平行墙的场景）。0=关闭（回到旧行为）
  int max_plane_splits = 2;
  double min_frame_size = 0.55;        // 框边长下限
  double max_frame_size = 2.5;         // 框边长上限
  // 宽高比上限：支持正方形/长方形/圆形框（圆的外接包围盒宽≈高）。
  // 原方案 A 为 1.35（仅正方形），实测场景含长方形框后放宽至 2.0。
  double max_aspect_ratio = 2.0;
  double hole_check_ratio = 0.25;      // 中心净空半径比
  bool require_hole = true;            // 空洞校验开关
  // 空洞路径几何约束（把"框"钉死为闭合、薄壁、矩形/圆形孔洞）：
  // 空洞路径几何约束（特征匹配度模型）："框"= 闭合、以薄壁为主、矩形/圆形的孔洞。
  // 匹配度语义：洞边界上 ≥ring_match_min 的方向是薄框边即通过——
  // 多柱支撑的框（柱子只在部分方向撑出）不会被误杀；厚墙开窗全周超厚仍被拒。
  double max_frame_thickness = 0.30;   // 单个方向"薄框边"的厚度上限(m)
  double ring_match_min = 0.60;        // 周界特征匹配度下限（薄框边方向占比）
  double hole_fill_ratio_min = 0.60;   // 空洞矩形度下限：三角/L 形/狭缝出局（圆≈0.73）

  // 骨架环检测（方向 A）：无环时骨架最近端点对距离 ≤ 该格数 → 桥接虚拟材料找洞。
  // 值=5 格（res=0.05 时约 0.25 m）：覆盖边中段漏采缺口，又不至于把 C 形开口强闭
  int skeleton_gap_cells = 5;
  // 骨架路径开关（A/B 定位用：仿真实测方框中心偏移 1.1-2.1 m 致同框重穿，
  // 关闭后回退旧空洞路径，用于分离骨架路径的回归）
  bool enable_skeleton = true;

  // 圆环开弧分支（2026-09-26）：近距时圆环被垂直视场(59°)/min_range 盲区裁成开弧，
  // 无闭合空洞（洞经缺口与外界连通）、包围盒中心偏移 0.3-0.5 m，标准路径必拒。
  // 对平面内点做 RANSAC 圆拟合直接给出圆心/半径；仅当闭合空洞路径未产出有效框时
  // 启用（完整圆环仍走空洞路径，矩形框因角/边不同心必被圆约束拒掉，行为不变）。
  bool enable_circle_arc = true;
  // 已知环径约束（比赛环标准尺寸）：>0.05 时圆环精化固定半径、只解圆心与法向
  //（|q−c|=r 线性化），短弧/斜视角下圆欠约束、中心漂移大（实测 51°+55% 偏置弧
  // 自由拟合中心偏 0.45 m），固定已知半径后条件数大幅改善。0=关闭（自由拟合）
  double circle_known_diameter = 0.0;
  double circle_min_arc_coverage = 0.60;  // 弧段角覆盖下限（36 扇区占比，直线/小弧出局）
  double circle_inlier_tol = 0.08;        // 圆上内点径向容差(m)：管厚0.05+体素0.05+噪声
  double circle_tol_ratio = 0.05;         // 容差同时 ≤ 该比例×直径：防方框三边被内切圆
                                         // 误拟合（方框需要的最小偏差≈0.1×边长必超此限）
  double circle_min_inlier_ratio = 0.80;  // 圆内点占平面内点比例下限：真环≥95%，
                                          // 方框边落入内切圆带的占比约60%（T3实测59%），
                                          // 0.80 可同时拒两者之误检
  double max_view_angle_deg = 65.0;    // 视角上限
  double extent_clip_quantile = 0.02;  // 尺寸分位裁剪
  int max_input_points = 200000;       // 单帧点数上限

  // 多目标消歧先验（均默认未配置）
  bool has_target_direction = false;
  Vec3 target_direction = {{1, 0, 0}};
  bool has_target_size = false;
  double target_size = 1.0;
  // 无先验且 ≥2 候选时的策略：false=报 AMBIGUOUS_TARGET（旧行为，会一直锁不上）；
  // true=取离无人机最近的候选。多框场景（fly.world 散布 4~5 个环）两框同时进
  // ROI 是常态，歧义即死锁——就近优先可保证航点持续发布，先验配置时仍以先验为准。
  // 配合切换余量使用：与上一帧选中框的匹配半径 0.6 m，其余候选须比它近
  // nearest_switch_margin 以上才会切换——两框距离相近时选择不会逐帧来回跳
  //（跳一次清一次锁定窗口，5 帧一致性永远凑不齐）。
  bool prefer_nearest = true;
  double nearest_switch_margin = 0.5;
};

struct StabilizerParams {
  int lock_count = 5;                    // 锁定所需一致帧数
  int unlock_miss_count = 8;             // 解锁所需连续 miss
  int history_len = 20;                  // 候选滑窗容量
  double center_consistency = 0.35;      // 一致性中心阈值 (m)
  double angle_consistency_deg = 30.0;   // 一致性角度阈值
  double center_ema_alpha = 0.35;
  double normal_ema_alpha = 0.35;
  double size_ema_alpha = 0.25;
  double approach_dist = 1.5;            // P1 距框
  double exit_dist = 2.5;                // P3 距框
  bool include_center_wp = true;         // 是否含 P2（框中心）
  double expected_rate = 10.0;           // 预期点云/里程计频率 (Hz)
  bool freeze_on_commit = true;          // commit 后冻结航点
};

// ---------------- 单帧检测 ----------------

enum class DetectStatus {
  OK = 0,
  NOT_ENOUGH_POINTS,
  NO_CLUSTER,
  NOT_PLANAR,
  TOO_SMALL_OR_LARGE,
  NOT_SQUARE,
  HOLE_FILLED,
  BAD_VIEW_ANGLE,
  AMBIGUOUS_TARGET,
};

// 校验流水线深度（用于失败原因上报：报"走得最远"的候选，而不是最大簇的原因）
inline int stageDepth(DetectStatus s) {
  switch (s) {
    case DetectStatus::TOO_SMALL_OR_LARGE: return 1;
    case DetectStatus::NOT_SQUARE: return 2;
    case DetectStatus::HOLE_FILLED: return 3;
    case DetectStatus::BAD_VIEW_ANGLE: return 4;
    case DetectStatus::AMBIGUOUS_TARGET: return 4;
    case DetectStatus::OK: return 5;
    default: return 0;  // NOT_ENOUGH_POINTS / NO_CLUSTER / NOT_PLANAR
  }
}

inline const char* detectStatusToString(DetectStatus s) {
  switch (s) {
    case DetectStatus::OK: return "OK";
    case DetectStatus::NOT_ENOUGH_POINTS: return "NOT_ENOUGH_POINTS";
    case DetectStatus::NO_CLUSTER: return "NO_CLUSTER";
    case DetectStatus::NOT_PLANAR: return "NOT_PLANAR";
    case DetectStatus::TOO_SMALL_OR_LARGE: return "TOO_SMALL_OR_LARGE";
    case DetectStatus::NOT_SQUARE: return "NOT_SQUARE";
    case DetectStatus::HOLE_FILLED: return "HOLE_FILLED";
    case DetectStatus::BAD_VIEW_ANGLE: return "BAD_VIEW_ANGLE";
    case DetectStatus::AMBIGUOUS_TARGET: return "AMBIGUOUS_TARGET";
  }
  return "UNKNOWN";
}

struct FrameDetection {
  Vec3 center = {{0, 0, 0}};  // 框中心（投影包围盒中心）
  Vec3 normal = {{0, 0, 0}};  // 单位法向，指向远离无人机一侧（飞行方向）
  Vec3 axis_u = {{1, 0, 0}};  // 面内宽度方向单位轴（可视化用）
  double width = 0.0;         // 面内包围盒宽 (m)
  double height = 0.0;        // 面内包围盒高 (m)
  int inlier_count = 0;
};

class FrameDetector {
 public:
  using Points = std::vector<double>;  // 扁平 N*3，世界系

  explicit FrameDetector(const DetectorParams& params = DetectorParams()) : p_(params) {}

  // 已穿越框排除表：候选中心与表内中心距离 < radius 即视为同一框，跳过。
  // Mid360 全向感知，穿越后身后的框仍会被检出，且法向自动翻转成"远离无人机"
  // 方向后 goal 会落到框的另一侧，把无人机往回拉——来回穿框死循环的根源。
  struct Exclusion {
    Vec3 center = {{0, 0, 0}};
    double radius = 1.0;
    // 平面模式（normal 非零）：排除"已穿框的整块平面区域"——
    // |dot(c-center, normal)| < slab（板厚）且 面内距离 < radius。
    // 背景：方框局部视角锁定中心可沿框内水平轴偏 ±1.5 m（半框锁定），
    // 球形排除盖不住同一框的另一侧（第二锁定中心距第一穿越点 ~3 m），
    // 加大球半径又会误伤 3.6 m 外邻框；而法向估计始终准确（偏移都在面内），
    // 平面胶囊能用小板厚(1.0 m)豁免斜交/平行的邻框，同时罩住整个已穿框。
    Vec3 normal = {{0, 0, 0}};
    double slab = 1.0;
  };
  void setExclusions(const std::vector<Exclusion>& ex) { exclusions_ = ex; }
  const std::vector<Exclusion>& exclusions() const { return exclusions_; }

  // 单帧检测。输入一帧世界系点云与无人机位置，输出 0 或 1 个框检测，或失败原因。
  // 状态有二：排除表（已穿越框，命中的候选不进入锁定与歧义判定）与就近粘滞
  // （sticky_center 指向上一帧选中的框时，同框加切换余量防逐帧跳变）。
  // candidates_out（可选）：回传全部通过校验的非排除候选框——前视下一框用
  //（"未达先引"：锁定当前框期间从候选里读下一框，不等穿越解锁）。
  DetectStatus detect(const Points& points, const Vec3& drone_pos, FrameDetection& out,
                      const Vec3* sticky_center = nullptr,
                      std::vector<FrameDetection>* candidates_out = nullptr) {
    // ① ROI 球裁剪
    Points roi;
    roi.reserve(points.size());
    const size_t n_raw = points.size() / 3;
    for (size_t i = 0; i < n_raw; ++i) {
      const Vec3 q = {{points[3 * i], points[3 * i + 1], points[3 * i + 2]}};
      const double r = norm(q - drone_pos);
      if (r >= p_.min_range && r <= p_.max_range) {
        roi.push_back(q[0]);
        roi.push_back(q[1]);
        roi.push_back(q[2]);
      }
    }
    if (roi.size() / 3 < static_cast<size_t>(p_.min_cluster_points)) {
      return DetectStatus::NOT_ENOUGH_POINTS;
    }

    // ② 体素降采样（体素质心）
    const Points vox = voxelDownsample(roi, p_.voxel_leaf);

    // ③ 欧式聚类（并查集 + 空间哈希），按点数降序
    std::vector<std::vector<int>> clusters = euclideanClusters(vox, p_.cluster_tol);
    std::sort(clusters.begin(), clusters.end(),
              [](const std::vector<int>& a, const std::vector<int>& b) { return a.size() > b.size(); });

    // ④ 候选簇逐个评估（包围盒对角线预筛 + 平面拆分五步校验），保留全部通过簇；
    //    全部失败时报"走得最远"的候选原因（而非最大簇的原因，便于定位真框卡在哪步）
    std::vector<FrameDetection> candidates;
    DetectStatus best_fail = DetectStatus::NO_CLUSTER;
    int best_depth = -1;
    int evaluated = 0;
    for (const std::vector<int>& cluster : clusters) {
      if (static_cast<int>(cluster.size()) < p_.min_cluster_points) break;  // 已降序
      if (!bboxDiagPass(vox, cluster)) continue;
      if (++evaluated > kMaxEvaluatedClusters) break;
      FrameDetection det;
      const DetectStatus st = evaluateCluster(vox, cluster, drone_pos, det, &best_fail, &best_depth);
      // 已穿越框：不进入候选（不会被锁定，也不占多目标歧义的计数名额）
      if (st == DetectStatus::OK && !isExcluded(det.center)) candidates.push_back(det);
      if (static_cast<int>(candidates.size()) >= kMaxCandidates) break;
    }
    if (candidates_out) *candidates_out = candidates;
    if (candidates.empty()) return best_depth >= 0 ? best_fail : DetectStatus::NO_CLUSTER;
    if (candidates.size() == 1) {
      out = candidates[0];
      return DetectStatus::OK;
    }

    // 多目标处理：有先验 → 评分取最高；无先验 → 按策略歧义或就近优先
    const bool no_prior = !p_.has_target_direction && !p_.has_target_size;
    if (no_prior && !p_.prefer_nearest) return DetectStatus::AMBIGUOUS_TARGET;
    int best = -1;
    double best_score = -1e300;
    int tie_count = 0;
    for (size_t i = 0; i < candidates.size(); ++i) {
      double s = no_prior ? 0.0 : candidateScore(candidates[i], drone_pos);
      if (no_prior) {
        s -= norm(candidates[i].center - drone_pos);  // 就近优先
        // 粘滞余量：与上一帧选中框同体 → 加切换余量，只有显著更近的候选才能夺走
        if (sticky_center && norm(candidates[i].center - *sticky_center) < 0.6)
          s += p_.nearest_switch_margin;
      }
      if (s > best_score + 1e-9) {
        best_score = s;
        best = static_cast<int>(i);
        tie_count = 1;
      } else if (std::fabs(s - best_score) <= 1e-9) {
        ++tie_count;
      }
    }
    if (best < 0 || tie_count > 1) return DetectStatus::AMBIGUOUS_TARGET;
    out = candidates[best];
    return DetectStatus::OK;
  }

 private:
  static const int kMaxEvaluatedClusters = 32;
  static const int kMaxCandidates = 16;

  bool isExcluded(const Vec3& c) const {
    for (const Exclusion& e : exclusions_) {
      if (norm(e.normal) > 0.5) {
        // 平面模式：板厚 + 面内半径双条件
        const Vec3 d = c - e.center;
        const double h = dot(d, e.normal);           // 面外（法向）分量
        if (std::fabs(h) >= e.slab) continue;
        const double lat = norm(d - e.normal * h);   // 面内分量
        if (lat < e.radius) return true;
      } else if (norm(e.center - c) < e.radius) {
        return true;  // 球模式（旧语义）
      }
    }
    return false;
  }

  std::vector<Exclusion> exclusions_;

  // 空间格键：|idx| < 2^20 安全
  static int64_t voxelIndexKey(int ix, int iy, int iz) {
    return ((static_cast<int64_t>(ix) + (1 << 20)) << 42) |
           ((static_cast<int64_t>(iy) + (1 << 20)) << 21) |
           (static_cast<int64_t>(iz) + (1 << 20));
  }

  static std::vector<double> voxelDownsample(const Points& pts, double leaf) {
    struct Acc {
      double x = 0, y = 0, z = 0;
      int n = 0;
    };
    std::unordered_map<int64_t, Acc> grid;
    grid.reserve(pts.size() / 3);
    const size_t n = pts.size() / 3;
    for (size_t i = 0; i < n; ++i) {
      const int ix = static_cast<int>(std::floor(pts[3 * i] / leaf));
      const int iy = static_cast<int>(std::floor(pts[3 * i + 1] / leaf));
      const int iz = static_cast<int>(std::floor(pts[3 * i + 2] / leaf));
      Acc& a = grid[voxelIndexKey(ix, iy, iz)];
      a.x += pts[3 * i];
      a.y += pts[3 * i + 1];
      a.z += pts[3 * i + 2];
      ++a.n;
    }
    std::vector<double> out;
    out.reserve(grid.size() * 3);
    for (const auto& kv : grid) {
      out.push_back(kv.second.x / kv.second.n);
      out.push_back(kv.second.y / kv.second.n);
      out.push_back(kv.second.z / kv.second.n);
    }
    return out;
  }

  static std::vector<std::vector<int>> euclideanClusters(const Points& pts, double tol) {
    const int n = static_cast<int>(pts.size() / 3);
    std::vector<int> parent(n);
    std::iota(parent.begin(), parent.end(), 0);
    std::function<int(int)> find = [&](int x) {
      while (parent[x] != x) {
        parent[x] = parent[parent[x]];
        x = parent[x];
      }
      return x;
    };
    auto unite = [&](int a, int b) {
      const int ra = find(a), rb = find(b);
      if (ra != rb) parent[ra] = rb;
    };

    // 以 tol 为格长的空间哈希，27 邻域查并
    std::unordered_map<int64_t, std::vector<int>> cells;
    std::vector<std::array<int, 3>> cell_of(n);
    for (int i = 0; i < n; ++i) {
      const int ix = static_cast<int>(std::floor(pts[3 * i] / tol));
      const int iy = static_cast<int>(std::floor(pts[3 * i + 1] / tol));
      const int iz = static_cast<int>(std::floor(pts[3 * i + 2] / tol));
      cell_of[i] = {{ix, iy, iz}};
      cells[voxelIndexKey(ix, iy, iz)].push_back(i);
    }
    const double tol2 = tol * tol;
    for (int i = 0; i < n; ++i) {
      const int cx = cell_of[i][0], cy = cell_of[i][1], cz = cell_of[i][2];
      for (int dx = -1; dx <= 1; ++dx)
        for (int dy = -1; dy <= 1; ++dy)
          for (int dz = -1; dz <= 1; ++dz) {
            const auto it = cells.find(voxelIndexKey(cx + dx, cy + dy, cz + dz));
            if (it == cells.end()) continue;
            for (const int j : it->second) {
              if (j <= i) continue;
              const double ex = pts[3 * i] - pts[3 * j];
              const double ey = pts[3 * i + 1] - pts[3 * j + 1];
              const double ez = pts[3 * i + 2] - pts[3 * j + 2];
              if (ex * ex + ey * ey + ez * ez <= tol2) unite(i, j);
            }
          }
    }
    std::unordered_map<int, std::vector<int>> groups;
    for (int i = 0; i < n; ++i) groups[find(i)].push_back(i);
    std::vector<std::vector<int>> out;
    out.reserve(groups.size());
    for (auto& kv : groups) out.push_back(std::move(kv.second));
    return out;
  }

  // 包围盒对角线预筛：diag ∈ [0.5·min_size, 2.5·max_size]
  bool bboxDiagPass(const Points& pts, const std::vector<int>& idx) const {
    Vec3 lo = {{1e9, 1e9, 1e9}}, hi = {{-1e9, -1e9, -1e9}};
    for (const int i : idx) {
      for (int k = 0; k < 3; ++k) {
        lo[k] = std::min(lo[k], pts[3 * i + k]);
        hi[k] = std::max(hi[k], pts[3 * i + k]);
      }
    }
    const double diag = norm(hi - lo);
    return diag >= 0.5 * p_.min_frame_size && diag <= 2.5 * p_.max_frame_size;
  }

  struct PlaneFit {
    Vec3 mean = {{0, 0, 0}}, normal = {{0, 0, 0}}, b1 = {{0, 0, 0}}, b2 = {{0, 0, 0}};
  };

  static PlaneFit pcaPlane(const Points& pts, const std::vector<int>& idx) {
    PlaneFit f;
    const size_t n = idx.size();
    Vec3 m = {{0, 0, 0}};
    for (const int i : idx) m = m + vec3(pts[3 * i], pts[3 * i + 1], pts[3 * i + 2]);
    m = m * (1.0 / static_cast<double>(n));
    double cov[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
    for (const int i : idx) {
      const Vec3 d = {{pts[3 * i] - m[0], pts[3 * i + 1] - m[1], pts[3 * i + 2] - m[2]}};
      for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) cov[r][c] += d[r] * d[c];
    }
    const double inv = 1.0 / static_cast<double>(n);
    for (int r = 0; r < 3; ++r)
      for (int c = 0; c < 3; ++c) cov[r][c] *= inv;
    double eval[3];
    Vec3 evec[3];
    symEigen3(cov, eval, evec);
    f.mean = m;
    f.normal = evec[0];
    f.b1 = evec[1];
    f.b2 = evec[2];
    return f;
  }

  struct Extents {
    double u_lo, u_hi, v_lo, v_hi;
    double width() const { return u_hi - u_lo; }
    double height() const { return v_hi - v_lo; }
  };

  // 原地取上下两个分位数（双 nth_element，O(n)，替代整列排序）
  static std::pair<double, double> quantileBounds(std::vector<double>* s, double q) {
    if (s->empty()) return {0.0, 0.0};
    const int n = static_cast<int>(s->size());
    int i0 = static_cast<int>(std::floor(q * (n - 1)));
    int i1 = static_cast<int>(std::floor((1.0 - q) * (n - 1)));
    i0 = std::max(0, std::min(i0, n - 1));
    i1 = std::max(0, std::min(i1, n - 1));
    if (i0 > i1) std::swap(i0, i1);
    std::nth_element(s->begin(), s->begin() + i1, s->end());
    const double hi = (*s)[i1];
    if (i0 == i1) return {hi, hi};
    std::nth_element(s->begin(), s->begin() + i0, s->begin() + i1);
    return {(*s)[i0], hi};
  }

  // 分位数裁剪的面内包围盒（剔面内杂点）
  static Extents clippedExtents(std::vector<double> u, std::vector<double> v, double q) {
    Extents e;
    std::tie(e.u_lo, e.u_hi) = quantileBounds(&u, q);
    std::tie(e.v_lo, e.v_hi) = quantileBounds(&v, q);
    return e;
  }

  // 最小面积矩形方向搜索（两阶段角扫描），返回面内旋转角
  static double minAreaAngle(const std::vector<double>& u, const std::vector<double>& v, double q) {
    auto areaAt = [&](double th, double* w, double* h) {
      const double c = std::cos(th), s = std::sin(th);
      std::vector<double> a(u.size()), b(u.size());
      for (size_t i = 0; i < u.size(); ++i) {
        a[i] = u[i] * c + v[i] * s;
        b[i] = -u[i] * s + v[i] * c;
      }
      const Extents e = clippedExtents(a, b, q);
      *w = e.width();
      *h = e.height();
      return (*w) * (*h);
    };
    double best = 0.0, best_area = 1e300;
    for (int deg = 0; deg < 90; deg += 2) {  // 粗扫 2°
      double w = 0, h = 0;
      const double ar = areaAt(deg * kPi / 180.0, &w, &h);
      if (ar < best_area) {
        best_area = ar;
        best = deg * kPi / 180.0;
      }
    }
    for (double d = -2.0; d <= 2.0; d += 0.1) {  // 细扫 ±2° 步 0.1°
      double w = 0, h = 0;
      const double th = best + d * kPi / 180.0;
      const double ar = areaAt(th, &w, &h);
      if (ar < best_area) {
        best_area = ar;
        best = th;
      }
    }
    return best;
  }

  // 候选簇评估入口：平面拆分 + 五步校验。
  // 鲁棒性核心：框黏在墙/支架上会与它们聚成一簇，直接拟平面会失败或被墙主导。
  // 策略：①先取主平面做 b–e 校验；②失败（实心/尺寸超限）时剥离该平面内点，
  // 对残余结构再提平面——互不共面的框面会被拆出来；③主平面提不出足够内点时
  // （框贴近平行墙，PCA 平面落在两层中间），按平面两侧把簇拆开各试一次。
  DetectStatus evaluateCluster(const Points& pts, const std::vector<int>& cluster,
                               const Vec3& drone_pos, FrameDetection& out,
                               DetectStatus* best_fail, int* best_depth) const {
    return evalPlaneSet(pts, cluster, drone_pos, out, best_fail, best_depth, 0);
  }

  DetectStatus evalPlaneSet(const Points& pts, const std::vector<int>& set,
                            const Vec3& drone_pos, FrameDetection& out,
                            DetectStatus* best_fail, int* best_depth, int depth) const {
    auto consider = [&](DetectStatus st) {
      if (stageDepth(st) > *best_depth) {
        *best_depth = stageDepth(st);
        *best_fail = st;
      }
    };
    if (static_cast<int>(set.size()) < p_.min_plane_inliers) {
      consider(DetectStatus::NOT_PLANAR);
      return DetectStatus::NOT_PLANAR;
    }

    // a) 迭代平面拟合（PCA → 选内点 → PCA）
    std::vector<int> work = set;
    PlaneFit fit = pcaPlane(pts, work);
    bool first_sel_ok = true;
    for (int it = 0; it < p_.plane_refine_iters; ++it) {
      std::vector<int> sel;
      sel.reserve(work.size());
      for (const int i : work) {
        const Vec3 q = {{pts[3 * i], pts[3 * i + 1], pts[3 * i + 2]}};
        if (std::fabs(dot(q - fit.mean, fit.normal)) <= p_.plane_inlier_dist) sel.push_back(i);
      }
      if (static_cast<int>(sel.size()) < p_.min_plane_inliers) {
        if (it == 0) first_sel_ok = false;
        break;  // 无法继续精化，用当前 fit/work
      }
      work = std::move(sel);
      fit = pcaPlane(pts, work);
    }

    // 主平面不成立（首内点不足）：拆平面两侧（平行双平面场景）
    if (!first_sel_ok) {
      if (depth < p_.max_plane_splits) {
        std::vector<int> side[2];
        for (const int i : set) {
          const Vec3 q = {{pts[3 * i], pts[3 * i + 1], pts[3 * i + 2]}};
          side[dot(q - fit.mean, fit.normal) >= 0 ? 0 : 1].push_back(i);
        }
        for (int k = 0; k < 2; ++k) {
          if (static_cast<int>(side[k].size()) < p_.min_plane_inliers) continue;
          if (evalPlaneSet(pts, side[k], drone_pos, out, best_fail, best_depth, depth + 1) ==
              DetectStatus::OK) {
            return DetectStatus::OK;
          }
        }
      }
      consider(DetectStatus::NOT_PLANAR);
      return DetectStatus::NOT_PLANAR;
    }

    // b–e) 对平面内点做几何校验
    const DetectStatus st = evaluatePlaneInliers(pts, work, fit, drone_pos, out);
    if (st == DetectStatus::OK) return st;
    consider(st);

    // 拆残差：剥掉本平面内点，残余结构里可能藏着框（框黏墙/支架场景）
    if ((st == DetectStatus::HOLE_FILLED || st == DetectStatus::TOO_SMALL_OR_LARGE ||
         st == DetectStatus::NOT_SQUARE) &&
        depth < p_.max_plane_splits) {
      std::vector<int> work_sorted(work);
      std::sort(work_sorted.begin(), work_sorted.end());
      std::vector<int> remainder;
      remainder.reserve(set.size());
      for (const int i : set) {
        if (!std::binary_search(work_sorted.begin(), work_sorted.end(), i)) remainder.push_back(i);
      }
      if (static_cast<int>(remainder.size()) >= p_.min_plane_inliers) {
        if (evalPlaneSet(pts, remainder, drone_pos, out, best_fail, best_depth, depth + 1) ==
            DetectStatus::OK) {
          return DetectStatus::OK;
        }
      }
    }
    return st;
  }

  // 面内栅格空洞连通域提取（方案 B 核心，作为包围盒度量的兜底）：
  // 在框平面内找"被点云包围的最大有界空洞"（不与栅格外界连通的空格连通块），
  // 框心 = 空洞中心，尺寸 = 空洞包围盒 + 两侧边界厚度。可救回：
  // 框面内黏着支柱（撑爆包围盒）、框嵌在共面墙上（窗洞）等场景。
  struct HoleRing {
    bool ok = false;      // 通过全部约束，可作为框
    bool closed = false;  // 存在被材料包围的封闭空洞（即使未过约束）
    double uc = 0, vc = 0, w = 0, h = 0;
  };

  // ===================== 骨架环检测（方向 A，2026-09-26） =====================
  // 拓扑定义："框 = 细材料闭合成环"，与形状无关（方/圆/六边形通吃）。
  // 关键思路：
  //   ① Zhang-Suen 细化把材料逐层剥成 1 格宽骨架——细化保持同伦类型：
  //      环剥完还是环（含一个洞），实心板剥成树/线段（无洞），C 形剥成开路径；
  //   ② 环秩 β = E − V + 1（骨架 8 连通分量内，V=格点数 E=邻接边数）：
  //      闭合细环 β=1、路径/树 β=0——"是不是闭合细环"从一堆度量启发式
  //      （尺寸/宽高比/圆拟合）坍缩成一个纯拓扑数，形状免疫；
  //   ③ 无环时看骨架端点（度=1 的格点）：最近端点对距离 ≤ skeleton_gap_cells
  //      说明只是缺口（边中段漏采/部分可见），两点间补虚拟材料线再找洞——
  //      把旧路径"矩形四方向直边封口"推广到任意形状与任意朝向的边中段缺口；
  //   ④ 闭合（真环或桥接）后洞的属性检查沿用同一套度量约束：矩形度
  //      （三角≈0.5 出局）、周界厚度匹配度（厚墙开窗出局）、尺寸窗——
  //      拓扑定"是不是框"，度量只定"框的参数"，两层职责分离。
  // 输出语义与 extractHoleRing 完全兼容；无环且端点不闭合时返回 closed=false，
  // 调用方回退旧路径（其矩形虚拟封口等行为保留）。
  HoleRing skeletonRing(const std::vector<double>& u2, const std::vector<double>& v2) const {
    HoleRing out;
    const int n = static_cast<int>(u2.size());
    if (n < p_.min_plane_inliers) return out;
    const double res = std::max(0.03, p_.voxel_leaf);

    // ---- 栅格化（与旧路径一致：绝对格坐标 + 膨胀 1 格，保持渗漏语义）----
    auto cellKey = [](int iu, int iv) -> int64_t {
      return ((int64_t)iu << 32) ^ (int64_t)(uint32_t)iv;
    };
    std::unordered_set<int64_t> occ;
    int iu_lo = INT_MAX, iu_hi = INT_MIN, iv_lo = INT_MAX, iv_hi = INT_MIN;
    for (int i = 0; i < n; ++i) {
      const int iu = static_cast<int>(std::floor(u2[i] / res));
      const int iv = static_cast<int>(std::floor(v2[i] / res));
      occ.insert(cellKey(iu, iv));
    }
    std::unordered_set<int64_t> occ_d;
    for (const int64_t k : occ) {
      const int iu = (int)(k >> 32), iv = (int)((int32_t)(k & 0xffffffffLL));
      for (int dx = -1; dx <= 1; ++dx)
        for (int dy = -1; dy <= 1; ++dy) {
          occ_d.insert(cellKey(iu + dx, iv + dy));
          iu_lo = std::min(iu_lo, iu + dx);
          iu_hi = std::max(iu_hi, iu + dx);
          iv_lo = std::min(iv_lo, iv + dy);
          iv_hi = std::max(iv_hi, iv + dy);
        }
    }
    const int cols = iu_hi - iu_lo + 1, rows = iv_hi - iv_lo + 1;
    if (cols < 6 || rows < 6 || cols > 400 || rows > 400) return out;
    auto idOf = [&](int cu, int cv) { return cv * cols + cu; };
    auto inGrid = [&](int cu, int cv) { return cu >= 0 && cv >= 0 && cu < cols && cv < rows; };
    auto occAt = [&](int cu, int cv) {
      return inGrid(cu, cv) && occ_d.count(cellKey(cu + iu_lo, cv + iv_lo)) > 0;
    };

    // ---- ① Zhang-Suen 细化：迭代剥除满足条件的边界格，直到 1 格宽骨架 ----
    // A(p)=邻域序列 0→1 跳变数（=1 保证不拆断连通/不蚀穿环），B(p)=邻居数 2..6
    // 保留（端点 B=1 与孤立点 B=0 不剥，骨架端点由此留存，供缺口判定用）
    std::vector<char> skel(static_cast<size_t>(cols) * rows, 0);
    for (int cv = 0; cv < rows; ++cv)
      for (int cu = 0; cu < cols; ++cu)
        if (occAt(cu, cv)) skel[idOf(cu, cv)] = 1;
    auto neighborsOn = [&](int cu, int cv, int* p2p9) {
      // p2..p9 从正上起逆时针：上、右上、右、右下、下、左下、左、左上
      static const int dx[8] = {0, 1, 1, 1, 0, -1, -1, -1};
      static const int dy[8] = {1, 1, 0, -1, -1, -1, 0, 1};
      for (int k = 0; k < 8; ++k) {
        const int nx = cu + dx[k], ny = cv + dy[k];
        p2p9[k] = inGrid(nx, ny) ? skel[idOf(nx, ny)] : 0;
      }
    };
    bool changed = true;
    while (changed) {
      changed = false;
      for (int step = 0; step < 2; ++step) {
        std::vector<int> del;
        for (int cv = 1; cv < rows - 1; ++cv) {
          for (int cu = 1; cu < cols - 1; ++cu) {
            const int id = idOf(cu, cv);
            if (!skel[id]) continue;
            int q[8];
            neighborsOn(cu, cv, q);
            int B = 0;
            for (int k = 0; k < 8; ++k) B += q[k];
            if (B < 2 || B > 6) continue;
            int A = 0;
            for (int k = 0; k < 8; ++k)
              if (q[k] == 0 && q[(k + 1) % 8] == 1) ++A;
            if (A != 1) continue;
            // 两个子迭代交替删东南/西北角，保证骨架居中对称
            if (step == 0) {
              if (q[0] * q[2] * q[4] == 0 && q[2] * q[4] * q[6] == 0) del.push_back(id);
            } else {
              if (q[0] * q[2] * q[6] == 0 && q[0] * q[4] * q[6] == 0) del.push_back(id);
            }
          }
        }
        if (!del.empty()) {
          for (const int id : del) skel[id] = 0;
          changed = true;
        }
      }
    }

    // ---- ② 毛刺修剪：反复删度=1 的格（噪声须/支柱接缝毛刺），骨架主干不受影响 ----
    for (int round = 0; round < 3; ++round) {
      std::vector<int> del;
      for (int cv = 0; cv < rows; ++cv)
        for (int cu = 0; cu < cols; ++cu) {
          const int id = idOf(cu, cv);
          if (!skel[id]) continue;
          int q[8];
          neighborsOn(cu, cv, q);
          int deg = 0;
          for (int k = 0; k < 8; ++k) deg += q[k];
          if (deg == 1) del.push_back(id);
        }
      if (del.empty()) break;
      for (const int id : del) skel[id] = 0;
    }

    // ---- ③ 环秩检测：骨架 8 连通分量内 β = E − V + 1 ----
    //    同时收集端点（度=1）供缺口桥接
    std::vector<char> vis(static_cast<size_t>(cols) * rows, 0);
    bool has_cycle = false;
    int ep1 = -1, ep2 = -1;  // 最近端点对（格子 id）
    double ep_best = 1e9;
    for (int cv0 = 0; cv0 < rows; ++cv0) {
      for (int cu0 = 0; cu0 < cols; ++cu0) {
        const int id0 = idOf(cu0, cv0);
        if (!skel[id0] || vis[id0]) continue;
        // BFS 该分量，累计 V 与 E（每条无向边只计一次：只向 +u/+v 方向邻居扩展计数）
        std::vector<int> stack{id0};
        vis[id0] = 1;
        long long V = 0, E = 0;
        std::vector<int> cells;
        for (size_t qi = 0; qi < stack.size(); ++qi) {
          const int id = stack[qi];
          const int cu = id % cols, cv = id / cols;
          ++V;
          cells.push_back(id);
          int q[8];
          neighborsOn(cu, cv, q);
          int deg = 0;
          static const int dxdy[8][2] = {{0, 1}, {1, 1}, {1, 0}, {1, -1},
                                         {0, -1}, {-1, -1}, {-1, 0}, {-1, 1}};
          for (int k = 0; k < 8; ++k) {
            if (!q[k]) continue;
            ++deg;
            const int nid = idOf(cu + dxdy[k][0], cv + dxdy[k][1]);
            if (!vis[nid]) {
              vis[nid] = 1;
              stack.push_back(nid);
            }
            // 无向边计数：仅当邻居 id > 当前 id（避免重复）
            if (nid > id) ++E;
          }
          if (deg == 1) cells.push_back(-1);  // 占位（下面统一处理端点，见 cells2）
        }
        (void)cells;
        (void)ep1;  // 端点配对在下面统一扫描
        const long long beta = E - V + 1;
        if (beta >= 1) has_cycle = true;
      }
    }
    // 端点配对：全局最近端点对（缺口桥接候选）
    std::vector<int> endpoints;
    for (int cv = 0; cv < rows; ++cv)
      for (int cu = 0; cu < cols; ++cu) {
        const int id = idOf(cu, cv);
        if (!skel[id]) continue;
        int q[8];
        neighborsOn(cu, cv, q);
        int deg = 0;
        for (int k = 0; k < 8; ++k) deg += q[k];
        if (deg == 1) endpoints.push_back(id);
      }
    for (size_t i = 0; i < endpoints.size(); ++i)
      for (size_t j = i + 1; j < endpoints.size(); ++j) {
        const int a = endpoints[i], b = endpoints[j];
        const double du = (a % cols) - (b % cols), dv = (a / cols) - (b / cols);
        const double d = std::sqrt(du * du + dv * dv);
        if (d < ep_best) {
          ep_best = d;
          ep1 = a;
          ep2 = b;
        }
      }

#ifdef WINDOW_DETECTOR_DEBUG
    std::fprintf(stderr, "[skelring] cycle=%d endpoints=%zu ep_gap=%.1f\n",
                 (int)has_cycle, endpoints.size(), ep_best);
#endif

    // ---- ④ 闭合判定与洞提取 ----
    // 有环 → 直接找洞；无环 → 端点近距才桥接（否则交给旧路径处理开形状）
    if (!has_cycle) {
      if (ep1 < 0 || ep_best > p_.skeleton_gap_cells) return out;  // closed=false → 回退旧路径
      // 桥接：端点对之间步进补虚拟材料（形状无关的"虚拟封口"推广）
      const double au = ep1 % cols, av = ep1 / cols;
      const double bu = ep2 % cols, bv = ep2 / cols;
      const int steps = std::max(1, (int)std::ceil(ep_best));
      for (int s = 0; s <= steps; ++s) {
        const double t = double(s) / steps;
        const int cu = (int)std::lround(au + (bu - au) * t);
        const int cv = (int)std::lround(av + (bv - av) * t);
        if (inGrid(cu, cv)) occ_d.insert(cellKey(cu + iu_lo, cv + iv_lo));
      }
#ifdef WINDOW_DETECTOR_DEBUG
      std::fprintf(stderr, "[skelring] bridged gap of %.1f cells\n", ep_best);
#endif
    }

    // 外界洪泛 → 有界空域 = 洞（与旧路径同法；在可能含桥接虚拟格的掩码上）
    std::vector<char> empty(static_cast<size_t>(cols) * rows, 1);
    for (int cv = 0; cv < rows; ++cv)
      for (int cu = 0; cu < cols; ++cu)
        if (occAt(cu, cv)) empty[idOf(cu, cv)] = 0;
    std::vector<char> exterior(static_cast<size_t>(cols) * rows, 0);
    std::vector<int> flood;
    auto pushExt = [&](int cu, int cv) {
      if (!inGrid(cu, cv)) return;
      const int id = idOf(cu, cv);
      if (exterior[id] || !empty[id]) return;
      exterior[id] = 1;
      flood.push_back(id);
    };
    for (int cu = 0; cu < cols; ++cu) {
      pushExt(cu, 0);
      pushExt(cu, rows - 1);
    }
    for (int cv = 0; cv < rows; ++cv) {
      pushExt(0, cv);
      pushExt(cols - 1, cv);
    }
    for (size_t qi = 0; qi < flood.size(); ++qi) {
      const int cu = flood[qi] % cols, cv = flood[qi] / cols;
      pushExt(cu + 1, cv);
      pushExt(cu - 1, cv);
      pushExt(cu, cv + 1);
      pushExt(cu, cv - 1);
    }
    // 最大有界空域 = 洞
    int best_cnt = 0, blo = 0, bhi = 0, bwlo = 0, bwhi = 0;
    std::unordered_set<int64_t> hole_cells;
    {
      std::vector<char> vis2(static_cast<size_t>(cols) * rows, 0);
      std::vector<int> comp;
      for (int cv0 = 0; cv0 < rows; ++cv0)
        for (int cu0 = 0; cu0 < cols; ++cu0) {
          const int id0 = idOf(cu0, cv0);
          if (vis2[id0] || exterior[id0] || !empty[id0]) continue;
          comp.clear();
          comp.push_back(id0);
          vis2[id0] = 1;
          int lo = INT_MAX, hi = INT_MIN, wlo = INT_MAX, whi = INT_MIN;
          for (size_t qi = 0; qi < comp.size(); ++qi) {
            const int cu = comp[qi] % cols, cv = comp[qi] / cols;
            wlo = std::min(wlo, cu);
            whi = std::max(whi, cu);
            lo = std::min(lo, cv);
            hi = std::max(hi, cv);
            const int nb[4][2] = {{cu + 1, cv}, {cu - 1, cv}, {cu, cv + 1}, {cu, cv - 1}};
            for (int k = 0; k < 4; ++k) {
              if (!inGrid(nb[k][0], nb[k][1])) continue;
              const int nid = idOf(nb[k][0], nb[k][1]);
              if (vis2[nid] || exterior[nid] || !empty[nid]) continue;
              vis2[nid] = 1;
              comp.push_back(nid);
            }
          }
          if ((int)comp.size() > best_cnt) {
            best_cnt = (int)comp.size();
            bwlo = wlo; bwhi = whi; blo = lo; bhi = hi;
            hole_cells.clear();
            for (const int c : comp) hole_cells.insert(c);
          }
        }
    }
    // ---- 洞的属性约束（与旧路径同套度量：矩形度/周界厚度/尺寸补偿）----
    auto finishRing = [&](HoleRing& r) {
      if (best_cnt < 6) return;
      r.closed = true;  // 闭合存在（真环或桥接）——即使约束不过也不再回退包围盒路径
      const int u_span = bwhi - bwlo + 1, v_span = bhi - blo + 1;
      if (u_span < 4 || v_span < 4) return;
      const double fill = static_cast<double>(best_cnt) /
                          static_cast<double>(u_span * v_span);
      if (fill < p_.hole_fill_ratio_min) return;  // 三角/狭缝出局（拓扑收下，度量筛掉）
      // 物理坐标（±1 格抵消膨胀）
      const double hu_lo = (bwlo - 1 + iu_lo) * res, hu_hi = (bwhi + 2 + iu_lo) * res;
      const double hv_lo = (blo - 1 + iv_lo) * res, hv_hi = (bhi + 2 + iv_lo) * res;
      const double hc_u = 0.5 * (hu_lo + hu_hi), hc_v = 0.5 * (hv_lo + hv_hi);
      // 洞边界材料格 + 径向厚度 marching（薄框边方向占比 ≥ ring_match_min）
      std::vector<std::pair<int, int>> bcells;
      for (int cv = blo - 1; cv <= bhi + 1; ++cv)
        for (int cu = bwlo - 1; cu <= bwhi + 1; ++cu) {
          if (!occAt(cu, cv)) continue;
          bool near_hole = false;
          for (int dy = -1; dy <= 1 && !near_hole; ++dy)
            for (int dx = -1; dx <= 1; ++dx) {
              if (!inGrid(cu + dx, cv + dy)) continue;
              if (hole_cells.count(idOf(cu + dx, cv + dy))) { near_hole = true; break; }
            }
          if (near_hole) bcells.push_back({cu, cv});
        }
      if (bcells.size() < 8) return;
      const int max_run = std::max(2, (int)std::ceil(p_.max_frame_thickness / res));
      std::vector<double> runs;
      for (const auto& bc : bcells) {
        const double pu = (bc.first + iu_lo + 0.5) * res - hc_u;
        const double pv = (bc.second + iv_lo + 0.5) * res - hc_v;
        const double len = std::sqrt(pu * pu + pv * pv);
        if (len < 1e-6) continue;
        const double du = pu / len, dv = pv / len;
        int run = 0;
        double cu = bc.first, cv = bc.second;
        while (run < max_run && occAt((int)std::lround(cu), (int)std::lround(cv))) {
          ++run;
          cu += du;
          cv += dv;
        }
        runs.push_back(run >= max_run ? max_run + 1 : run);
      }
      int thin_dirs = 0;
      std::vector<double> thin_runs;
      for (const double r2 : runs)
        if (r2 * res <= p_.max_frame_thickness) {
          ++thin_dirs;
          thin_runs.push_back(r2);
        }
      if (thin_dirs == 0 ||
          static_cast<double>(thin_dirs) / runs.size() < p_.ring_match_min)
        return;  // 周界多数方向超厚：厚墙开窗/箱体围合，不是框
      std::sort(thin_runs.begin(), thin_runs.end());
      const double tube = std::max(0.0, std::min(thin_runs[thin_runs.size() / 2] - 3.0, 3.0)) * res;
      r.ok = true;
      r.uc = hc_u;
      r.vc = hc_v;
      r.w = (hu_hi - hu_lo) + 2.0 * tube;
      r.h = (hv_hi - hv_lo) + 2.0 * tube;
    };
    HoleRing r;
    finishRing(r);
#ifdef WINDOW_DETECTOR_DEBUG
    std::fprintf(stderr, "[skelring] hole=%d closed=%d ok=%d\n", best_cnt, (int)r.closed, (int)r.ok);
#endif
    return r;
  }
  HoleRing extractHoleRing(const std::vector<double>& u2, const std::vector<double>& v2) const {
    HoleRing ring;
    auto med = [](std::vector<double> s) {
      if (s.empty()) return 0.0;
      std::sort(s.begin(), s.end());
      return s[s.size() / 2];
    };
    const double res = std::max(0.03, p_.voxel_leaf);
    if (static_cast<int>(u2.size()) < p_.min_plane_inliers) return ring;
    // 绝对格坐标占据
    std::unordered_set<int64_t> occ;
    occ.reserve(u2.size() * 2);
    auto cellKey = [](int iu, int iv) -> int64_t {
      return ((int64_t)iu << 32) ^ (int64_t)(uint32_t)iv;
    };
    int iu_lo = INT_MAX, iu_hi = INT_MIN, iv_lo = INT_MAX, iv_hi = INT_MIN;
    for (size_t i = 0; i < u2.size(); ++i) {
      const int iu = static_cast<int>(std::floor(u2[i] / res));
      const int iv = static_cast<int>(std::floor(v2[i] / res));
      occ.insert(cellKey(iu, iv));
    }
    // 膨胀 1 格（形态学闭合）：旋转重采样/稀疏采样会在连续边上留单格缺口，
    // 4-连通洪水会从缺口渗漏。后续连通域/边界/厚度分析全用膨胀后的占据。
    std::unordered_set<int64_t> occ_d;
    occ_d.reserve(occ.size() * 9);
    for (const int64_t k : occ) {
      const int iu = static_cast<int>(k >> 32);
      const int iv = static_cast<int>((int32_t)(k & 0xffffffffLL));
      for (int dx = -1; dx <= 1; ++dx)
        for (int dy = -1; dy <= 1; ++dy) {
          occ_d.insert(cellKey(iu + dx, iv + dy));
          iu_lo = std::min(iu_lo, iu + dx);
          iu_hi = std::max(iu_hi, iu + dx);
          iv_lo = std::min(iv_lo, iv + dy);
          iv_hi = std::max(iv_hi, iv + dy);
        }
    }
    auto occHas = [&](int iu, int iv) { return occ_d.count(cellKey(iu, iv)) > 0; };
    const int cols = iu_hi - iu_lo + 1, rows = iv_hi - iv_lo + 1;
    if (cols < 6 || rows < 6) return ring;
    auto idOf = [&](int cu, int cv) { return cv * cols + cu; };
    auto inGrid = [&](int cu, int cv) { return cu >= 0 && cv >= 0 && cu < cols && cv < rows; };
    std::vector<char> empty(static_cast<size_t>(cols) * rows, 1);
    for (int cv = 0; cv < rows; ++cv)
      for (int cu = 0; cu < cols; ++cu)
        if (occHas(cu + iu_lo, cv + iv_lo)) empty[cv * cols + cu] = 0;
    // 外界洪泛：seed_top=true 从四边下种（完全封闭判定）；
    //           seed_top=false 不从顶边下种（容忍"只向上开放"的洞：框顶点云缺失场景）
    auto floodExterior = [&](bool seed_top) {
      std::vector<char> exterior(static_cast<size_t>(cols) * rows, 0);
      std::vector<int> flood;
      flood.reserve(static_cast<size_t>(cols) * rows);
      auto pushExt = [&](int cu, int cv) {
        if (!inGrid(cu, cv)) return;
        const int id = idOf(cu, cv);
        if (exterior[id] || !empty[id]) return;
        exterior[id] = 1;
        flood.push_back(id);
      };
      for (int cu = 0; cu < cols; ++cu) {
        pushExt(cu, 0);
        if (seed_top) pushExt(cu, rows - 1);
      }
      for (int cv = 0; cv < rows; ++cv) {
        pushExt(0, cv);
        pushExt(cols - 1, cv);
      }
      for (size_t qi = 0; qi < flood.size(); ++qi) {
        const int cu = flood[qi] % cols, cv = flood[qi] / cols;
        pushExt(cu + 1, cv);
        pushExt(cu - 1, cv);
        pushExt(cu, cv + 1);
        pushExt(cu, cv - 1);
      }
      return exterior;
    };
    // 在给定外界掩码下找最大有界空域
    struct Comp {
      int cnt = 0;
      int box[4] = {0, 0, 0, 0};       // cu_lo, cu_hi, cv_lo, cv_hi（相对格）
      std::unordered_set<int64_t> cells;
    };
    auto findBestComp = [&](const std::vector<char>& exterior) {
      Comp best;
      std::vector<char> vis(static_cast<size_t>(cols) * rows, 0);
      std::vector<int> comp;
      for (int cv0 = 0; cv0 < rows; ++cv0) {
        for (int cu0 = 0; cu0 < cols; ++cu0) {
          const int id0 = idOf(cu0, cv0);
          if (vis[id0] || exterior[id0] || !empty[id0]) continue;
          comp.clear();
          comp.push_back(id0);
          vis[id0] = 1;
          int blo = INT_MAX, bhi = INT_MIN, bwlo = INT_MAX, bwhi = INT_MIN;
          for (size_t qi = 0; qi < comp.size(); ++qi) {
            const int cu = comp[qi] % cols, cv = comp[qi] / cols;
            bwlo = std::min(bwlo, cu);
            bwhi = std::max(bwhi, cu);
            blo = std::min(blo, cv);
            bhi = std::max(bhi, cv);
            const int nb[4][2] = {{cu + 1, cv}, {cu - 1, cv}, {cu, cv + 1}, {cu, cv - 1}};
            for (int k = 0; k < 4; ++k) {
              if (!inGrid(nb[k][0], nb[k][1])) continue;
              const int nid = idOf(nb[k][0], nb[k][1]);
              if (vis[nid] || exterior[nid] || !empty[nid]) continue;
              vis[nid] = 1;
              comp.push_back(nid);
            }
          }
          if ((int)comp.size() > best.cnt) {
            best.cnt = (int)comp.size();
            best.box[0] = bwlo;
            best.box[1] = bwhi;
            best.box[2] = blo;
            best.box[3] = bhi;
            best.cells.clear();
            for (const int c : comp) best.cells.insert(c);
          }
        }
      }
      return best;
    };
    auto occAt = [&](int cu, int cv) -> bool {
      return inGrid(cu, cv) && occHas(cu + iu_lo, cv + iv_lo);
    };
    // 约束校验 + 几何重建（top_open 时 v 上界由侧边材料上缘推定，忽略向上方向）
    auto buildRing = [&](const Comp& comp, bool top_open) {
      HoleRing r;  // ok=false 默认
      if (comp.cnt < 6) return r;
      r.closed = true;
      // v 上界：顶开放时取洞列范围内材料的最上缘（侧边/顶边残余），完全封闭时用洞自身 bbox
      int cv_top = comp.box[3];
      if (top_open) {
        for (int cu = comp.box[0] - 2; cu <= comp.box[1] + 2; ++cu) {
          for (int cv = rows - 1; cv > comp.box[3]; --cv) {
            if (occAt(cu, cv)) {
              cv_top = std::max(cv_top, cv);
              break;
            }
          }
        }
      }
      const int u_span = comp.box[1] - comp.box[0] + 1;
      const int v_span = cv_top - comp.box[2] + 1;
      if (u_span < 4 || v_span < 4) return r;
      // 矩形度：洞格数（顶开放时只计推定 bbox 内的） / bbox 格数
      int cnt_in = 0;
      if (top_open) {
        for (const int64_t c : comp.cells) {
          const int cu = (int)(c % cols), cv = (int)(c / cols);
          if (cu >= comp.box[0] && cu <= comp.box[1] && cv >= comp.box[2] && cv <= cv_top) ++cnt_in;
        }
      } else {
        cnt_in = comp.cnt;
      }
      const double fill = static_cast<double>(cnt_in) /
                          static_cast<double>(u_span * v_span);
#ifdef WINDOW_DETECTOR_DEBUG
      std::fprintf(stderr,
                   "[holering] top_open=%d cnt=%d box=(%d,%d,%d,%d) cv_top=%d span=%dx%d fill=%.2f\n",
                   (int)top_open, comp.cnt, comp.box[0], comp.box[1], comp.box[2], comp.box[3],
                   cv_top, u_span, v_span, fill);
#endif
      if (fill < p_.hole_fill_ratio_min) return r;
      // 物理坐标（相对格 → 外扩 1 格抵消膨胀；顶开放时上界用 cv_top+膨胀补偿）
      const double hole_u_lo = (comp.box[0] - 1 + iu_lo) * res;
      const double hole_u_hi = (comp.box[1] + 2 + iu_lo) * res;
      const double hole_v_lo = (comp.box[2] - 1 + iv_lo) * res;
      const double hole_v_hi = (cv_top + 2 + iv_lo) * res;
      const double hole_c_u = 0.5 * (hole_u_lo + hole_u_hi);
      const double hole_c_v = 0.5 * (hole_v_lo + hole_v_hi);
      // 洞边界材料格（8 邻接洞格；顶开放时排除 cv > cv_top 的上缘——那里本来就没材料）
      std::vector<std::pair<int, int>> bcells;
      for (int cv = comp.box[2] - 1; cv <= cv_top + 1; ++cv) {
        for (int cu = comp.box[0] - 1; cu <= comp.box[1] + 1; ++cu) {
          if (!occAt(cu, cv)) continue;
          if (top_open && cv > cv_top) continue;
          bool near_hole = false;
          for (int dy = -1; dy <= 1 && !near_hole; ++dy)
            for (int dx = -1; dx <= 1; ++dx) {
              if (!inGrid(cu + dx, cv + dy)) continue;
              if (comp.cells.count(idOf(cu + dx, cv + dy))) {
                near_hole = true;
                break;
              }
            }
          if (near_hole) bcells.push_back({cu, cv});
        }
      }
      if (bcells.size() < 8) return r;
      // 周界特征匹配度 + 管厚（顶开放时 marching 方向自然只来自三边，向上无材料）
      const int max_run = std::max(2, (int)std::ceil(p_.max_frame_thickness / res));
      std::vector<double> runs;
      runs.reserve(bcells.size());
      for (const auto& bc : bcells) {
        const double pu = (bc.first + iu_lo + 0.5) * res - hole_c_u;
        const double pv = (bc.second + iv_lo + 0.5) * res - hole_c_v;
        const double len = std::sqrt(pu * pu + pv * pv);
        if (len < 1e-6) continue;
        const double du = pu / len, dv = pv / len;
        int run = 0;
        double cu = bc.first, cv = bc.second;
        while (run < max_run && occAt((int)std::lround(cu), (int)std::lround(cv))) {
          ++run;
          cu += du;
          cv += dv;
        }
        runs.push_back(run >= max_run ? max_run + 1 : run);  // 触顶=至少这么厚
      }
      if (runs.empty()) return r;
      int thin_dirs = 0;
      std::vector<double> thin_runs;
      thin_runs.reserve(runs.size());
      for (const double r2 : runs) {
        if (r2 * res <= p_.max_frame_thickness) {
          ++thin_dirs;
          thin_runs.push_back(r2);
        }
      }
      if (thin_dirs == 0 || static_cast<double>(thin_dirs) / runs.size() < p_.ring_match_min) {
        return r;  // 大部分方向超厚：厚墙开窗/箱体围合，不是框
      }
      const double tube = std::max(0.0, std::min(med(thin_runs) - 3.0, 3.0)) * res;
      r.ok = true;
      r.uc = hole_c_u;
      r.vc = hole_c_v;
      r.w = (hole_u_hi - hole_u_lo) + 2.0 * tube;
      r.h = (hole_v_hi - hole_v_lo) + 2.0 * tube;
      return r;
    };
    // 优先完全封闭洞
    HoleRing r1 = buildRing(findBestComp(floodExterior(true)), false);
    if (r1.ok) return r1;
#ifdef WINDOW_DETECTOR_DEBUG
    std::fprintf(stderr, "[holering] r1.ok=%d r1.closed=%d\n", (int)r1.ok, (int)r1.closed);
#endif
    if (r1.closed) return r1;  // 完全封闭但约束不过：保持拒绝（不给包围盒路径留后门）
    // 开口兜底（框某条边点云缺失：传感器垂直 FOV/稀疏常见）：洞与外界连通，
    // 洪泛分不出"洞"。四方向尝试虚拟封口——用开口两翼材料的延伸线补一行虚拟占据，
    // 再找封闭洞。旋转无关（u2/v2 是最小面积旋转后的坐标，开口可能朝任意方向）。
    // 真实边细、虚拟边细 → 匹配度通过；两厚壁夹缝场景厚壁超限仍被拒。
    auto refreshEmpty = [&]() {
      for (int cv = 0; cv < rows; ++cv)
        for (int cu = 0; cu < cols; ++cu)
          empty[cv * cols + cu] = occHas(cu + iu_lo, cv + iv_lo) ? 0 : 1;
    };
    auto rowExtreme = [&](int cv, bool max_side) {  // 行内最右/最左材料列
      int best = -1;
      for (int cu = 0; cu < cols; ++cu) {
        if (!occAt(cu, cv)) continue;
        if (best < 0) best = cu;
        if (max_side) best = cu;
      }
      return best;
    };
    auto colExtreme = [&](int cu, bool max_side) {  // 列内最上/最下材料行
      int best = -1;
      for (int cv = 0; cv < rows; ++cv) {
        if (!occAt(cu, cv)) continue;
        if (best < 0) best = cv;
        if (max_side) best = cv;
      }
      return best;
    };
    const std::unordered_set<int64_t> occ_snapshot = occ_d;
    HoleRing best_virtual;  // closed 标记：封口后确有洞但约束不过 → 保持拒绝
    for (int side = 0; side < 4 && !best_virtual.ok; ++side) {
      occ_d = occ_snapshot;
      int line = -1;
      bool insert_column = (side < 2);
      if (side == 0) {  // +u 侧开口：上下两行各自最右材料的较小值处竖直封
        const int f1 = rowExtreme(rows - 1, true), f2 = rowExtreme(0, true);
        if (f1 >= 0 && f2 >= 0 && std::abs(f1 - f2) <= 3) line = std::min(f1, f2);
      } else if (side == 1) {  // −u 侧
        const int f1 = rowExtreme(rows - 1, false), f2 = rowExtreme(0, false);
        if (f1 >= 0 && f2 >= 0 && std::abs(f1 - f2) <= 3) line = std::max(f1, f2);
      } else if (side == 2) {  // +v 侧开口：左右两列各自最上材料的较小值处水平封
        const int f1 = colExtreme(cols - 1, true), f2 = colExtreme(0, true);
        if (f1 >= 0 && f2 >= 0 && std::abs(f1 - f2) <= 3) line = std::min(f1, f2);
      } else {  // −v 侧
        const int f1 = colExtreme(cols - 1, false), f2 = colExtreme(0, false);
        if (f1 >= 0 && f2 >= 0 && std::abs(f1 - f2) <= 3) line = std::max(f1, f2);
      }
      if (line < 0) continue;
      for (int k = 0; k < (insert_column ? rows : cols); ++k) {
        if (insert_column) occ_d.insert(cellKey(line + iu_lo, k + iv_lo));
        else occ_d.insert(cellKey(k + iu_lo, line + iv_lo));
      }
      refreshEmpty();
      HoleRing rv = buildRing(findBestComp(floodExterior(true)), false);
      if (rv.ok) {
        best_virtual = rv;
        break;
      }
      if (rv.closed) best_virtual.closed = true;  // 记录"封口后有洞但非框形"
      occ_d = occ_snapshot;
      refreshEmpty();
    }
#ifdef WINDOW_DETECTOR_DEBUG
    std::fprintf(stderr, "[holering] virtual best.ok=%d closed=%d\n", (int)best_virtual.ok,
                 (int)best_virtual.closed);
#endif
    if (best_virtual.closed) {
      // 封口后确有洞：约束过 → 用之；不过 → 保持拒绝（不给包围盒路径留后门）
      return best_virtual;
    }
    return HoleRing();  // 无封闭洞：交回包围盒+中心净空路径
  }

  // 圆环开弧分支：对平面内点(u2,v2)做 RANSAC 圆拟合 + Kåsa 代数最小二乘精化。
  // 开弧（视场/盲区裁切）无闭合空洞，本分支直接给圆心/半径；四重约束防误检：
  // ① 内点占比 ≥ circle_min_inlier_ratio（矩形框角/边不同心必出局）
  // ② 直径 ∈ [min_frame_size, max_frame_size]
  // ③ 弧段角覆盖 ≥ circle_min_arc_coverage（直线段能被拟成大圆，但只占小角段）
  // ④ 内点容差同时受 circle_tol_ratio×直径 封顶（方框三边距内切圆仅 0.1×边长，
  //    固定容差会把它误拟合；实心板的环带占比也随容差收紧跌破下限）
  bool fitCircleArc(const std::vector<double>& u, const std::vector<double>& v,
                    const PlaneFit& fit, const Vec3& b1, const Vec3& b2,
                    const Vec3& drone_pos, FrameDetection& out) const {
    const int n = static_cast<int>(u.size());
    if (!p_.enable_circle_arc || n < std::max(3, p_.min_plane_inliers)) return false;

    const double r_lo = 0.5 * p_.min_frame_size, r_hi = 1.25 * p_.max_frame_size;
    std::mt19937 rng(20260926u);  // 固定种子：同一点云结果可复现
    std::uniform_int_distribution<int> uni(0, n - 1);
    auto tolOf = [&](double r) {
      return std::min(p_.circle_inlier_tol, p_.circle_tol_ratio * 2.0 * r);
    };

    auto circumcircle = [&](int i, int j, int k, double* cu, double* cv, double* r) {
      const double x1 = u[i], y1 = v[i], x2 = u[j], y2 = v[j], x3 = u[k], y3 = v[k];
      const double d = 2.0 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2));
      if (std::fabs(d) < 1e-9) return false;
      const double s1 = x1 * x1 + y1 * y1, s2 = x2 * x2 + y2 * y2, s3 = x3 * x3 + y3 * y3;
      *cu = (s1 * (y2 - y3) + s2 * (y3 - y1) + s3 * (y1 - y2)) / d;
      *cv = (s1 * (x3 - x2) + s2 * (x1 - x3) + s3 * (x2 - x1)) / d;
      *r = std::sqrt((x1 - *cu) * (x1 - *cu) + (y1 - *cv) * (y1 - *cv));
      return *r > r_lo && *r < r_hi;
    };
    auto countInliers = [&](double cu, double cv, double r, std::vector<char>* mask) {
      int cnt = 0;
      if (mask) mask->assign(n, 0);
      const double tol = tolOf(r);
      for (int i = 0; i < n; ++i) {
        const double du = u[i] - cu, dv = v[i] - cv;
        const double dr = std::sqrt(du * du + dv * dv) - r;
        if (std::fabs(dr) <= tol) {
          ++cnt;
          if (mask) (*mask)[i] = 1;
        }
      }
      return cnt;
    };

    double best_cu = 0, best_cv = 0, best_r = 0;
    int best_cnt = 0;
    for (int it = 0; it < 48; ++it) {
      const int i = uni(rng), j = uni(rng), k = uni(rng);
      if (i == j || j == k || i == k) continue;
      double cu, cv, r;
      if (!circumcircle(i, j, k, &cu, &cv, &r)) continue;
      const int cnt = countInliers(cu, cv, r, nullptr);
      if (cnt > best_cnt) {
        best_cnt = cnt;
        best_cu = cu;
        best_cv = cv;
        best_r = r;
      }
    }
    if (best_cnt < p_.min_plane_inliers) return false;

    // Kåsa 精化两轮：u²+v² = a·u + b·v + c，(a/2, b/2) 为圆心（3×3 正规方程克拉默求解）
    for (int round = 0; round < 2; ++round) {
      std::vector<char> mask;
      countInliers(best_cu, best_cv, best_r, &mask);
      double A[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}}, B[3] = {0, 0, 0};
      for (int i = 0; i < n; ++i) {
        if (!mask[i]) continue;
        const double x = u[i], y = v[i], s = x * x + y * y;
        A[0][0] += x * x; A[0][1] += x * y; A[0][2] += x;
        A[1][1] += y * y; A[1][2] += y;
        A[2][2] += 1.0;
        B[0] += s * x; B[1] += s * y; B[2] += s;
      }
      A[1][0] = A[0][1]; A[2][0] = A[0][2]; A[2][1] = A[1][2];
      const double det = A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1]) -
                         A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0]) +
                         A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]);
      if (std::fabs(det) < 1e-9) break;
      const double a = (B[0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1]) -
                        A[0][1] * (B[1] * A[2][2] - A[1][2] * B[2]) +
                        A[0][2] * (B[1] * A[2][1] - A[1][1] * B[2])) / det;
      const double b = (A[0][0] * (B[1] * A[2][2] - A[1][2] * B[2]) -
                        B[0] * (A[1][0] * A[2][2] - A[1][2] * A[2][0]) +
                        A[0][2] * (A[1][0] * B[2] - B[0] * A[2][0])) / det;
      const double cc = (A[0][0] * (A[1][1] * B[2] - B[1] * A[2][1]) -
                         A[0][1] * (A[1][0] * B[2] - B[0] * A[2][0]) +
                         B[0] * (A[1][0] * A[2][1] - A[1][1] * A[2][0])) / det;
      const double cu = 0.5 * a, cv = 0.5 * b;
      const double r2 = cc + cu * cu + cv * cv;
      if (!(r2 > 0.01)) break;
      const double r = std::sqrt(r2);
      if (r < r_lo || r > r_hi) break;
      const int cnt = countInliers(cu, cv, r, nullptr);
      if (cnt < best_cnt) break;
      best_cnt = cnt; best_cu = cu; best_cv = cv; best_r = r;
    }

    if (best_cnt < p_.min_plane_inliers) {
#ifdef WINDOW_DETECTOR_DEBUG
      std::fprintf(stderr, "[circlearc] too few ransac inliers %d\n", best_cnt);
#endif
      return false;
    }
    if (static_cast<double>(best_cnt) / n < p_.circle_min_inlier_ratio) {
#ifdef WINDOW_DETECTOR_DEBUG
      std::fprintf(stderr, "[circlearc] ratio %.2f < %.2f (cnt=%d n=%d)\n",
                   static_cast<double>(best_cnt) / n, p_.circle_min_inlier_ratio, best_cnt, n);
#endif
      return false;
    }
    const double diam = 2.0 * best_r;
    if (diam < p_.min_frame_size || diam > p_.max_frame_size) {
#ifdef WINDOW_DETECTOR_DEBUG
      std::fprintf(stderr, "[circlearc] diam %.2f outside [%.2f, %.2f]\n",
                   diam, p_.min_frame_size, p_.max_frame_size);
#endif
      return false;
    }

    // 弧段角覆盖（36 扇区）
    std::vector<char> mask;
    countInliers(best_cu, best_cv, best_r, &mask);
    bool sector[36] = {false};
    int occ = 0;
    for (int i = 0; i < n; ++i) {
      if (!mask[i]) continue;
      const double th = std::atan2(v[i] - best_cv, u[i] - best_cu) + kPi;
      const int s = std::max(0, std::min(35, static_cast<int>(th / (2.0 * kPi) * 36.0)));
      if (!sector[s]) {
        sector[s] = true;
        ++occ;
      }
    }
    if (static_cast<double>(occ) / 36.0 < p_.circle_min_arc_coverage) {
#ifdef WINDOW_DETECTOR_DEBUG
      std::fprintf(stderr, "[circlearc] coverage %.2f < %.2f (occ=%d cnt=%d n=%d)\n",
                   static_cast<double>(occ) / 36.0, p_.circle_min_arc_coverage, occ, best_cnt, n);
#endif
      return false;
    }

    // 视角校验与主路径一致：法向定向为飞行方向后须大致正对
    const Vec3 center = fit.mean + b1 * best_cu + b2 * best_cv;
    Vec3 nrm = fit.normal;
    if (dot(fit.mean - drone_pos, nrm) < 0) nrm = -nrm;
    const Vec3 los = normalized(center - drone_pos);
    if (dot(nrm, los) < std::cos(p_.max_view_angle_deg * kPi / 180.0)) return false;

    out.center = center;
    out.normal = nrm;
    out.axis_u = b1;
    out.width = diam;
    out.height = diam;
    out.inlier_count = best_cnt;
    return true;
  }

  // ===================== 圆环 3D 交替精化（斜视角几何约束，2026-09-27）=====================
  // 动机：PCA 平面法向对弧段覆盖不对称有一阶偏置（点云质心≠圆心），Kåsa 代数
  // 拟合对开弧中心也有拉偏——斜视角圆环（视线偏离环法向 30°+）的中心/法向
  // 漂移明显，P1/P2/P3 航点随之歪斜。把"圆心↔法向"绑定到圆约束上交替精化：
  //   ① 以当前圆为准重选内点（板厚带 × 径向带）；
  //   ② 以圆心为中心的协方差最小特征向量 = 新法向——按圆心中心化消掉质心偏置；
  //   ③ 过质心的新平面上 Kåsa 重拟合圆 → 新圆心/半径；
  // 迭代 3 轮；任一步退化（内点不足/行列式奇异/半径越窗）即中止，保留原检测值。
  void refineCircle3D(const Points& pts, const std::vector<int>& work,
                      FrameDetection& out) const {
    Vec3 c = out.center;
    Vec3 n = out.normal;
    if (norm(n) < 0.5) return;
    // 初值半径：已知环径 > 平面内点半中位径向距离 > (w+h)/2。骨架洞包围盒在
    // 短弧/偏置密度下严重畸变（实测 55% 偏置弧 w=1.30/h=2.15），(w+h)/2 初值
    // 会离真值过远、首轮内点选择直接清空
    double r;
    if (p_.circle_known_diameter > 0.05) {
      r = 0.5 * p_.circle_known_diameter;
    } else {
      std::vector<double> rads;
      rads.reserve(work.size());
      for (const int i : work) {
        const Vec3 d = vec3(pts[3 * i], pts[3 * i + 1], pts[3 * i + 2]) - c;
        const double dn = dot(d, n);
        if (std::fabs(dn) > p_.plane_inlier_dist) continue;
        rads.push_back(norm(d - n * dn));
      }
      if (rads.empty()) return;
      std::nth_element(rads.begin(), rads.begin() + rads.size() / 2, rads.end());
      r = rads[rads.size() / 2];
    }
    bool ok = false;
    for (int iter = 0; iter < 3 && r > 1e-3; ++iter) {
      // ① 内点重选：|dot(p−c, n)| ≤ 板厚 且 |‖p−c‖−r| ≤ 径向带
      //（首轮 3×宽带：骨架路径给的初值半径可能有偏，宽带先收敛再收紧）
      const double tol_r = (iter == 0 ? 3.0 : 1.5) * p_.circle_inlier_tol;
      std::vector<int> sel;
      sel.reserve(work.size());
      for (const int i : work) {
        const Vec3 d = vec3(pts[3 * i], pts[3 * i + 1], pts[3 * i + 2]) - c;
        const double dn = dot(d, n);
        if (std::fabs(dn) > p_.plane_inlier_dist) continue;
        if (std::fabs(norm(d - n * dn) - r) > tol_r) continue;
        sel.push_back(i);
      }
      if (static_cast<int>(sel.size()) < p_.min_plane_inliers) break;
      // ② 圆心处协方差 → 新法向（最小特征向量；弧段覆盖不对称的偏置被中心化消除）
      Vec3 m = {{0, 0, 0}};
      for (const int i : sel) m = m + vec3(pts[3 * i], pts[3 * i + 1], pts[3 * i + 2]);
      m = m * (1.0 / static_cast<double>(sel.size()));
      double cov[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
      for (const int i : sel) {
        const Vec3 d = vec3(pts[3 * i], pts[3 * i + 1], pts[3 * i + 2]) - m;
        for (int a = 0; a < 3; ++a)
          for (int b = a; b < 3; ++b) cov[a][b] += d[a] * d[b];
      }
      const double inv = 1.0 / static_cast<double>(sel.size());
      for (int a = 0; a < 3; ++a)
        for (int b = a; b < 3; ++b) {
          cov[a][b] *= inv;
          cov[b][a] = cov[a][b];
        }
      double eval[3];
      Vec3 evec[3];
      symEigen3(cov, eval, evec);
      Vec3 nn = evec[0];
      if (dot(nn, n) < 0) nn = -nn;  // 保持原定向（远离无人机一侧）
      // ③ 新平面（过质心 m）上圆拟合：
      //    已知环径 → 固定半径解圆心（|q−c|=r 线性化：2x·cu + 2y·cv − w = x²+y²−r²，
      //    w=cu²+cv² 作松弛变量，最小二乘解 cu/cv）——短弧/斜视角的强几何约束；
      //    否则 Kåsa 自由拟合：u²+v² = a·u + b·v + c
      const Vec3 b1 = anyPerpendicular(nn);
      const Vec3 b2 = normalized(cross(nn, b1));
      double cu, cv, r_new;
      if (p_.circle_known_diameter > 0.05) {
        const double r_fix = 0.5 * p_.circle_known_diameter;
        double A[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}}, B[3] = {0, 0, 0};
        for (const int i : sel) {
          const Vec3 d = vec3(pts[3 * i], pts[3 * i + 1], pts[3 * i + 2]) - m;
          const double x = dot(d, b1), y = dot(d, b2);
          const double row[3] = {2.0 * x, 2.0 * y, -1.0};
          const double rhs = x * x + y * y - r_fix * r_fix;
          for (int a2 = 0; a2 < 3; ++a2) {
            B[a2] += row[a2] * rhs;
            for (int b3 = a2; b3 < 3; ++b3) A[a2][b3] += row[a2] * row[b3];
          }
        }
        A[1][0] = A[0][1]; A[2][0] = A[0][2]; A[2][1] = A[1][2];
        const double det = A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1]) -
                           A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0]) +
                           A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]);
        if (std::fabs(det) < 1e-9) return;
        cu = (B[0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1]) -
              A[0][1] * (B[1] * A[2][2] - A[1][2] * B[2]) +
              A[0][2] * (B[1] * A[2][1] - A[1][1] * B[2])) / det;
        cv = (A[0][0] * (B[1] * A[2][2] - A[1][2] * B[2]) -
              B[0] * (A[1][0] * A[2][2] - A[1][2] * A[2][0]) +
              A[0][2] * (A[1][0] * B[2] - B[0] * A[2][0])) / det;
        r_new = r_fix;
      } else {
        double A[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}}, B[3] = {0, 0, 0};
        for (const int i : sel) {
          const Vec3 d = vec3(pts[3 * i], pts[3 * i + 1], pts[3 * i + 2]) - m;
          const double x = dot(d, b1), y = dot(d, b2), s = x * x + y * y;
          A[0][0] += x * x; A[0][1] += x * y; A[0][2] += x;
          A[1][1] += y * y; A[1][2] += y;
          A[2][2] += 1.0;
          B[0] += s * x; B[1] += s * y; B[2] += s;
        }
        A[1][0] = A[0][1]; A[2][0] = A[0][2]; A[2][1] = A[1][2];
        const double det = A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1]) -
                           A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0]) +
                           A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]);
        if (std::fabs(det) < 1e-9) return;
        const double a = (B[0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1]) -
                          A[0][1] * (B[1] * A[2][2] - A[1][2] * B[2]) +
                          A[0][2] * (B[1] * A[2][1] - A[1][1] * B[2])) / det;
        const double b = (A[0][0] * (B[1] * A[2][2] - A[1][2] * B[2]) -
                          B[0] * (A[1][0] * A[2][2] - A[1][2] * A[2][0]) +
                          A[0][2] * (A[1][0] * B[2] - B[0] * A[2][0])) / det;
        const double cc = (A[0][0] * (A[1][1] * B[2] - B[1] * A[2][1]) -
                           A[0][1] * (A[1][0] * B[2] - B[0] * A[2][0]) +
                           B[0] * (A[1][0] * A[2][1] - A[1][1] * A[2][0])) / det;
        cu = 0.5 * a;
        cv = 0.5 * b;
        const double r2 = cc + cu * cu + cv * cv;
        if (!(r2 > 1e-4)) break;
        r_new = std::sqrt(r2);
      }
      if (r_new < 0.5 * p_.min_frame_size || r_new > 1.25 * p_.max_frame_size) break;
      c = m + b1 * cu + b2 * cv;  // 新圆心落在新平面上
      n = nn;
      r = r_new;
      ok = true;
    }
    if (!ok) return;  // 收敛失败：保留原检测值
    // 圆一致性终判（分扇区径向平坦度）：环的径向距离沿方位平坦（噪声级起伏）；
    // 方框有 4 角，扇区中位径向从中点↔对角起伏 ~35%（径向带占比区分不了——
    // 收敛后方框点也 ~90% 落带内）。≥6 个有效扇区且相对起伏 <0.2 才判环，
    // 否则整体回退（方框/非圆不被圆约束拉偏尺寸）
    const Vec3 g1 = anyPerpendicular(n);
    const Vec3 g2 = normalized(cross(n, g1));
    double sec_sum[12] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    int sec_cnt[12] = {0};
    for (const int i : work) {
      const Vec3 d = vec3(pts[3 * i], pts[3 * i + 1], pts[3 * i + 2]) - c;
      const double dn = dot(d, n);
      if (std::fabs(dn) > p_.plane_inlier_dist) continue;
      const Vec3 e = d - n * dn;
      const double rr = norm(e);
      if (std::fabs(rr - r) > 2.5 * p_.circle_inlier_tol) continue;
      const int k = static_cast<int>(
          (std::atan2(dot(e, g2), dot(e, g1)) + kPi) / (2.0 * kPi) * 12.0);
      const int kk = std::max(0, std::min(11, k));
      sec_sum[kk] += rr;
      ++sec_cnt[kk];
    }
    int populated = 0;
    double rmin = 1e9, rmax = 0.0;
    for (int k = 0; k < 12; ++k) {
      if (sec_cnt[k] < 3) continue;
      ++populated;
      const double rm = sec_sum[k] / sec_cnt[k];
      rmin = std::min(rmin, rm);
      rmax = std::max(rmax, rm);
    }
    if (populated < 6 || (rmax - rmin) / std::max(r, 1e-6) >= 0.20) return;
    out.center = c;
    out.normal = n;
    out.width = 2.0 * r;
    out.height = 2.0 * r;
  }

  // b–e：面内投影 + 最小面积方向对齐 + 分位裁剪尺寸度量 + 尺寸/方形 + 空洞 + 视角
  DetectStatus evaluatePlaneInliers(const Points& pts, const std::vector<int>& work,
                                    const PlaneFit& fit, const Vec3& drone_pos,
                                    FrameDetection& out) const {
    std::vector<double> u(work.size()), v(work.size());
    for (size_t i = 0; i < work.size(); ++i) {
      const Vec3 q = {{pts[3 * work[i]], pts[3 * work[i] + 1], pts[3 * work[i] + 2]}};
      const Vec3 d = q - fit.mean;
      u[i] = dot(d, fit.b1);
      v[i] = dot(d, fit.b2);
    }
    const double th = minAreaAngle(u, v, p_.extent_clip_quantile);
    const double c = std::cos(th), s = std::sin(th);
    const Vec3 b1 = fit.b1 * c + fit.b2 * s;
    const Vec3 b2 = fit.b1 * (-s) + fit.b2 * c;
    std::vector<double> u2(work.size()), v2(work.size());
    for (size_t i = 0; i < work.size(); ++i) {
      u2[i] = u[i] * c + v[i] * s;
      v2[i] = -u[i] * s + v[i] * c;
    }
    const Extents e = clippedExtents(u2, v2, p_.extent_clip_quantile);
    double w = e.width(), h = e.height();
    Vec3 center = fit.mean + b1 * ((e.u_lo + e.u_hi) / 2.0) + b2 * ((e.v_lo + e.v_hi) / 2.0);

    auto sizeStatus = [&](double W, double H) -> DetectStatus {
      if (W < p_.min_frame_size || W > p_.max_frame_size || H < p_.min_frame_size ||
          H > p_.max_frame_size) {
        return DetectStatus::TOO_SMALL_OR_LARGE;
      }
      if (std::max(W, H) / std::max(std::min(W, H), 1e-9) > p_.max_aspect_ratio) {
        return DetectStatus::NOT_SQUARE;
      }
      return DetectStatus::OK;
    };

    // c) 尺寸/方形 + d) 空洞校验（封闭空洞优先）
    //    存在封闭空洞时以空洞几何为准（框心/尺寸由空洞边界重建，黏支柱/嵌墙均正确）；
    //    封闭空洞存在但约束不过（三角形/厚壁孔）→ 直接拒绝，不再退回包围盒路径
    //    （否则斜三角形的斜向包围盒会钻尺寸/宽高比的空子）。
    //    无封闭空洞（C 形缺边、稀疏部分可见）→ 老路径：包围盒 + 中心净空圆。
    // 圆环开弧分支（优先）：圆拟合对完整环/开弧/厚壁(重影)环都给出准确圆心；
    // 矩形框（角/边不同心，需 ~0.1×边长 容差）与实心板（环带占比不足）必被四重
    // 约束拒掉，随后落入原空洞/包围盒路径，矩形框行为不变。
    if (p_.enable_circle_arc &&
        fitCircleArc(u2, v2, fit, b1, b2, drone_pos, out)) {
      return DetectStatus::OK;
    }

    // ===== 框提取（方向 A：骨架环优先，旧空洞路径兜底）=====
    // 骨架环检测给出形状无关的闭合判定（真环 / 端点桥接闭合）：
    //   ok=true            → 直接采用（方框/六边形/带缺口框，中心来自洞几何）
    //   closed && !ok      → 闭合但非框（三角/厚墙）：维持拒绝语义，不回退
    //   !closed（无环且端点不闭合）→ 交旧路径（矩形四方向虚拟封口等原行为）
    HoleRing ring;
    if (p_.require_hole && p_.enable_skeleton) {
      ring = skeletonRing(u2, v2);
      if (!ring.ok && !ring.closed) ring = extractHoleRing(u2, v2);
    } else if (p_.require_hole) {
      ring = extractHoleRing(u2, v2);
    }
    DetectStatus st = sizeStatus(w, h);
    if (ring.closed) {
      if (!ring.ok) return DetectStatus::HOLE_FILLED;
      st = sizeStatus(ring.w, ring.h);
      if (st != DetectStatus::OK) return st;
      w = ring.w;
      h = ring.h;
      center = fit.mean + b1 * ring.uc + b2 * ring.vc;
    } else if (st == DetectStatus::OK && p_.require_hole) {
      const double uc = (e.u_lo + e.u_hi) / 2.0, vc = (e.v_lo + e.v_hi) / 2.0;
      const double r_hole = p_.hole_check_ratio * std::min(w, h);
      for (size_t i = 0; i < u2.size() && st == DetectStatus::OK; ++i) {
        const double du = u2[i] - uc, dv = v2[i] - vc;
        if (du * du + dv * dv < r_hole * r_hole) st = DetectStatus::HOLE_FILLED;
      }
    }
    if (st != DetectStatus::OK) return st;

    // e) 视角校验：法向定向为飞行方向（远离无人机）后须大致正对
    Vec3 n = fit.normal;
    if (dot(fit.mean - drone_pos, n) < 0) n = -n;
    const Vec3 los = normalized(center - drone_pos);
    if (dot(n, los) < std::cos(p_.max_view_angle_deg * kPi / 180.0)) {
      return DetectStatus::BAD_VIEW_ANGLE;
    }

    out.center = center;
    out.normal = n;
    out.axis_u = b1;
    out.width = w;
    out.height = h;
    out.inlier_count = static_cast<int>(work.size());
    // 圆约束统一精化：拓扑（圆弧/骨架环/空洞）定"是框"之后，凡径向一致的
    // （真环，含 <60% 覆盖被判骨架路径的短弧/斜视角环）都用"圆心↔法向↔半径"
    // 绑定精化参数；方框被自门控豁免。挂在统一出口而非圆分支后——短弧环
    // 走的是骨架路径，圆分支挂载够不着
    refineCircle3D(pts, work, out);
    return DetectStatus::OK;
  }

  // 多候选先验评分：方向对齐度 + 尺寸匹配度，各 [0,1]
  double candidateScore(const FrameDetection& d, const Vec3& drone_pos) const {
    double score = 0.0;
    if (p_.has_target_direction) {
      const Vec3 los = normalized(d.center - drone_pos);
      score += std::max(0.0, dot(los, normalized(p_.target_direction)));
    }
    if (p_.has_target_size) {
      const double mean_size = 0.5 * (d.width + d.height);
      score += std::max(0.0, 1.0 - std::fabs(mean_size - p_.target_size) / p_.target_size);
    }
    return score;
  }

  DetectorParams p_;
};



// 稀疏 LiDAR（仿真 Mid-360 / 实机远距细框）单帧往往只打中框的一两条边，
// 单帧检测凑不齐几何。把窗口期内（默认 2 s）的世界系点按体素去重累积，
// 逐帧补齐框的完整轮廓后再检测。点为世界系，无人机移动不影响配准。
class PointAccumulator {
 public:
  using ClockFn = std::function<double()>;

  PointAccumulator(double window_sec = 2.0, double voxel = 0.05, ClockFn clock = nullptr)
      : window_sec_(window_sec), voxel_(voxel), clock_(clock) {}

  void insert(const std::vector<double>& pts) {
    const double t = now();
    const size_t n = pts.size() / 3;
    for (size_t i = 0; i < n; ++i) {
      const int ix = static_cast<int>(std::floor(pts[3 * i] / voxel_));
      const int iy = static_cast<int>(std::floor(pts[3 * i + 1] / voxel_));
      const int iz = static_cast<int>(std::floor(pts[3 * i + 2] / voxel_));
      // 同体素取最新点（刷新时间戳，保证活跃体素不因窗口过期被清掉）
      voxels_[key(ix, iy, iz)] = {vec3(pts[3 * i], pts[3 * i + 1], pts[3 * i + 2]), t};
    }
  }

  // 窗口内的累计点（扁平 N*3）
  std::vector<double> points() const {
    const double t = now();
    std::vector<double> out;
    out.reserve(voxels_.size() * 3);
    for (const auto& kv : voxels_) {
      if (t - kv.second.second <= window_sec_) {
        out.push_back(kv.second.first[0]);
        out.push_back(kv.second.first[1]);
        out.push_back(kv.second.first[2]);
      }
    }
    return out;
  }

  size_t size() const { return voxels_.size(); }
  void clear() { voxels_.clear(); }
  // 窗口可在线调整（速度自适应收缩：高速时压窗保证累积弧长/壳厚预算恒定；
  // 缩窗后过期体素由 points() 的时间过滤自然淘汰）
  void setWindowTime(double sec) { window_sec_ = std::max(sec, 0.05); }

 private:
  static int64_t key(int ix, int iy, int iz) {
    return ((static_cast<int64_t>(ix) + (1 << 20)) << 42) |
           ((static_cast<int64_t>(iy) + (1 << 20)) << 21) |
           (static_cast<int64_t>(iz) + (1 << 20));
  }
  double now() const {
    return clock_ ? clock_()
                  : std::chrono::duration_cast<std::chrono::duration<double>>(
                        std::chrono::steady_clock::now().time_since_epoch()).count();
  }

  double window_sec_;
  double voxel_;
  ClockFn clock_;
  std::unordered_map<int64_t, std::pair<Vec3, double>> voxels_;
};

// ---------------- 时间稳定器（状态机） ----------------

enum class StabState { SEARCHING = 0, LOCKED = 1 };

class FrameStabilizer {
 public:
  // 可注入时钟（秒），单测用；默认 steady_clock
  using ClockFn = std::function<double()>;

  explicit FrameStabilizer(const StabilizerParams& params = StabilizerParams(),
                           ClockFn clock = nullptr)
      : p_(params), clock_(clock) {
    const double t = now();
    last_cloud_t_ = t;
    last_odom_t_ = t;
    last_det_t_ = t;
  }

  void setParams(const StabilizerParams& p) { p_ = p; }
  const StabilizerParams& params() const { return p_; }

  // ---- 每帧驱动 ----

  void onCloudFrame() { stampCloud(); }
  void onOdomFrame() {
    last_odom_t_ = now();
    got_odom_ = true;
  }

  // 单帧检测成功（隐含点云存活）
  void onDetection(const FrameDetection& d) {
    stampCloud();
    last_det_t_ = now();
    got_det_ = true;
    if (state_ == StabState::SEARCHING) {
      if (!window_.empty() && !consistentWith(medianOfWindow(), d)) window_.clear();
      window_.push_back(d);
      while (static_cast<int>(window_.size()) > p_.history_len) window_.pop_front();
      viz_axis_ = d.axis_u;
      if (static_cast<int>(window_.size()) >= p_.lock_count) {
        frame_ = medianOfWindow();  // 锁定初值取窗口中位数（抗离群）
        state_ = StabState::LOCKED;
        miss_count_ = 0;
        window_.clear();
      }
      return;
    }
    // LOCKED
    if (consistentWith(frame_, d)) {
      miss_count_ = 0;
      viz_axis_ = d.axis_u;
      if (!frozen_) emaUpdate(d);
    } else {
      // 冻结（已提交穿越）期间挂起 miss：近距盲区的坏检测不应打断直穿，
      // 解锁只应来自穿越记账(checkTraversed→reset)或取消服务
      if (frozen_) return;
      ++miss_count_;
      if (miss_count_ >= p_.unlock_miss_count) unlock();
    }
  }

  // 单帧无有效检测（点云帧到了但检测失败/歧义）
  void onMiss() {
    stampCloud();
    if (state_ == StabState::LOCKED) {
      if (frozen_) return;  // 同上：冻结期间不因检测丢失解锁
      ++miss_count_;
      if (miss_count_ >= p_.unlock_miss_count) unlock();
    }
  }

  // 定时器周期回调（周期 = 1/expected_rate）。消息超时路径：
  // 点云或里程计静默超 2/expected_rate → LOCKED 每周期计一次 miss；
  // SEARCHING 的有效检测间隔超时 → 清空候选窗口。
  void onTimerTick() {
    const double t = now();
    const double timeout = 2.0 / std::max(p_.expected_rate, 0.1);
    const bool cloud_stale = got_cloud_ && (t - last_cloud_t_ > timeout);
    const bool odom_stale = got_odom_ && (t - last_odom_t_ > timeout);
    const bool det_gap = got_det_ && (t - last_det_t_ > timeout);
    if (state_ == StabState::SEARCHING) {
      if (det_gap || cloud_stale || odom_stale) window_.clear();
      return;
    }
    if (cloud_stale || odom_stale) {
      if (frozen_) return;  // 冻结穿越期间输入静默不解锁（穿越通常 3-5 s 内完成）
      ++miss_count_;
      if (miss_count_ >= p_.unlock_miss_count) unlock();
    }
  }

  // ---- 状态查询 ----

  bool locked() const { return state_ == StabState::LOCKED; }
  StabState state() const { return state_; }
  const FrameDetection& frame() const { return frame_; }
  // 可视化/对外输出用锁定帧（冻结时返回冻结快照）
  const FrameDetection& vizFrame() const { return frozen_ ? frozen_frame_ : frame_; }
  Vec3 vizAxis() const { return viz_axis_; }
  bool frozen() const { return frozen_; }
  int missCount() const { return miss_count_; }
  int lockProgress() const {
    return state_ == StabState::LOCKED
               ? p_.lock_count
               : std::min<int>(static_cast<int>(window_.size()), p_.lock_count);
  }

  // ---- 穿越提交冻结 ----

  // 锁定且 freeze_on_commit 开启时冻结当前 c、n 与航点；返回是否成功
  bool commit() {
    if (state_ != StabState::LOCKED || !p_.freeze_on_commit || frozen_) return false;
    frozen_ = true;
    frozen_frame_ = frame_;
    frozen_wps_ = computeWaypoints(frame_);
    return true;
  }
  void cancelCommit() { frozen_ = false; }

  // ---- 航点 ----

  // 未锁定返回空；冻结时返回冻结航点
  std::vector<Vec3> waypoints() const {
    if (state_ != StabState::LOCKED) return {};
    if (frozen_) return frozen_wps_;
    return computeWaypoints(frame_);
  }

  void reset() {
    state_ = StabState::SEARCHING;
    window_.clear();
    miss_count_ = 0;
    frozen_ = false;
    frozen_wps_.clear();
  }

 private:
  static double steadySeconds() {
    return std::chrono::duration_cast<std::chrono::duration<double>>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
  }
  double now() const { return clock_ ? clock_() : steadySeconds(); }
  void stampCloud() {
    last_cloud_t_ = now();
    got_cloud_ = true;
  }

  void unlock() {
    state_ = StabState::SEARCHING;
    window_.clear();
    miss_count_ = 0;
    frozen_ = false;
    frozen_wps_.clear();  // 不输出陈旧航点
  }

  bool consistentWith(const FrameDetection& ref, const FrameDetection& d) const {
    if (norm(ref.center - d.center) > p_.center_consistency) return false;
    if (angleBetween(ref.normal, d.normal) * 180.0 / kPi > p_.angle_consistency_deg) return false;
    return true;
  }

  FrameDetection medianOfWindow() const {
    FrameDetection ref;
    const size_t n = window_.size();
    if (n == 0) return ref;
    std::vector<double> cx, cy, cz, ws, hs;
    Vec3 nsum = {{0, 0, 0}};
    cx.reserve(n);
    cy.reserve(n);
    cz.reserve(n);
    for (const FrameDetection& d : window_) {
      cx.push_back(d.center[0]);
      cy.push_back(d.center[1]);
      cz.push_back(d.center[2]);
      nsum = nsum + d.normal;
      ws.push_back(d.width);
      hs.push_back(d.height);
    }
    ref.center = vec3(medianOf(cx), medianOf(cy), medianOf(cz));
    ref.normal = normalized(nsum);
    ref.width = medianOf(ws);
    ref.height = medianOf(hs);
    return ref;
  }

  static double medianOf(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return n % 2 == 1 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
  }

  void emaUpdate(const FrameDetection& d) {
    frame_.center = frame_.center + (d.center - frame_.center) * p_.center_ema_alpha;
    Vec3 nd = dot(d.normal, frame_.normal) < 0 ? -d.normal : d.normal;  // 同号对齐
    frame_.normal = normalized(frame_.normal + (nd - frame_.normal) * p_.normal_ema_alpha);
    frame_.width += (d.width - frame_.width) * p_.size_ema_alpha;
    frame_.height += (d.height - frame_.height) * p_.size_ema_alpha;
  }

  std::vector<Vec3> computeWaypoints(const FrameDetection& f) const {
    std::vector<Vec3> wps;
    wps.push_back(f.center - f.normal * p_.approach_dist);  // P1 对准点（己方）
    if (p_.include_center_wp) wps.push_back(f.center);      // P2 框中心
    wps.push_back(f.center + f.normal * p_.exit_dist);      // P3 穿出点（对面）
    return wps;
  }

  StabilizerParams p_;
  ClockFn clock_;
  StabState state_ = StabState::SEARCHING;
  std::deque<FrameDetection> window_;
  FrameDetection frame_;
  Vec3 viz_axis_ = {{1, 0, 0}};
  int miss_count_ = 0;
  bool frozen_ = false;
  FrameDetection frozen_frame_;
  std::vector<Vec3> frozen_wps_;
  double last_cloud_t_ = 0.0, last_odom_t_ = 0.0, last_det_t_ = 0.0;
  bool got_cloud_ = false, got_odom_ = false, got_det_ = false;
};

}  // namespace window_detector

#endif  // WINDOW_DETECTOR_FRAME_DETECTOR_H
