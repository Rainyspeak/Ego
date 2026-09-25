// Square-frame ("window") detection core: single-frame geometric detection,
// temporal stabilization state machine and crossing-waypoint generation.
//
// Pure C++14, no ROS / PCL / Eigen dependency on purpose, so the whole
// pipeline can be unit-tested offline (see test/test_frame_detector.cpp).
// The ROS node in src/window_detector_node.cpp is a thin wrapper around this.
//
// Single-frame detection (method A: clustering + PCA):
//   crop sphere around drone -> voxel downsample -> euclidean clustering
//   -> per candidate cluster: PCA plane fit with inlier refinement
//   -> in-plane extents (quantile-clipped) -> size / squareness / hole /
//   view-angle validation -> frame center + travel-direction normal.
//
// Temporal stabilizer:
//   SEARCHING --lock_count consistent detections--> LOCKED
//   LOCKED --unlock_miss_count missed/inconsistent detections--> SEARCHING
//   Locked state is initialized from the median of the candidate window and
//   then updated by EMA, so the output is jitter-free and outlier-rejecting.

#ifndef WINDOW_DETECTOR_FRAME_DETECTOR_H
#define WINDOW_DETECTOR_FRAME_DETECTOR_H

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <limits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace window_detector {

// ============================== math helpers ==============================

struct Vec3 {
  float x = 0.f, y = 0.f, z = 0.f;

  Vec3() = default;
  Vec3(float fx, float fy, float fz) : x(fx), y(fy), z(fz) {}

  Vec3 operator+(const Vec3& o) const { return Vec3(x + o.x, y + o.y, z + o.z); }
  Vec3 operator-(const Vec3& o) const { return Vec3(x - o.x, y - o.y, z - o.z); }
  Vec3 operator-() const { return Vec3(-x, -y, -z); }
  Vec3 operator*(float s) const { return Vec3(x * s, y * s, z * s); }
  Vec3 operator/(float s) const { return Vec3(x / s, y / s, z / s); }
  Vec3& operator+=(const Vec3& o) { x += o.x; y += o.y; z += o.z; return *this; }

  float dot(const Vec3& o) const { return x * o.x + y * o.y + z * o.z; }
  Vec3 cross(const Vec3& o) const {
    return Vec3(y * o.z - z * o.y, z * o.x - x * o.z, x * o.y - y * o.x);
  }
  float norm() const { return std::sqrt(dot(*this)); }
  float distanceTo(const Vec3& o) const { return (*this - o).norm(); }
  Vec3 normalized() const {
    const float n = norm();
    return n > 1e-9f ? *this / n : Vec3(0.f, 0.f, 0.f);
  }
};

inline Vec3 operator*(float s, const Vec3& v) { return v * s; }

inline float degToRad(float deg) { return deg * 0.017453292519943295776f; }

// Angle in [0, pi/2] ignoring the sign of the vectors (for plane normals).
inline float unsignedAngle(const Vec3& a, const Vec3& b) {
  const float d = std::fabs(a.dot(b)) / std::max(a.norm() * b.norm(), 1e-12f);
  return std::acos(std::min(std::max(d, -1.f), 1.f));
}

// Orthonormal in-plane basis (b1, b2) for a unit-ish normal n. Used for
// drawing the frame outline and for the synthetic-cloud generator in tests.
inline void frameBasisFromNormal(const Vec3& n_in, Vec3* b1, Vec3* b2) {
  const Vec3 n = n_in.normalized();
  Vec3 up(0.f, 0.f, 1.f);
  if (std::fabs(n.dot(up)) > 0.9f) up = Vec3(0.f, 1.f, 0.f);
  *b1 = n.cross(up).normalized();
  *b2 = n.cross(*b1).normalized();
}

// Jacobi eigen decomposition of a symmetric 3x3 matrix.
// Input a[i][j] row-major symmetric; output eigenvalues ascending and
// eigenvectors as COLUMNS of vec (vec[row][col]).
inline void symEigen3x3(const float a_in[3][3], float eval[3], float vec[3][3]) {
  float a[3][3];
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) a[i][j] = a_in[i][j];
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) vec[i][j] = (i == j) ? 1.f : 0.f;

  for (int sweep = 0; sweep < 64; ++sweep) {
    const float off = std::fabs(a[0][1]) + std::fabs(a[0][2]) + std::fabs(a[1][2]);
    if (off < 1e-12f) break;
    for (int p = 0; p < 2; ++p) {
      for (int q = p + 1; q < 3; ++q) {
        const float apq = a[p][q];
        if (std::fabs(apq) < 1e-15f) continue;
        const float theta = (a[q][q] - a[p][p]) / (2.f * apq);
        const float t = (theta >= 0.f ? 1.f : -1.f) /
                        (std::fabs(theta) + std::sqrt(theta * theta + 1.f));
        const float c = 1.f / std::sqrt(t * t + 1.f);
        const float s = t * c;
        for (int k = 0; k < 3; ++k) {
          const float akp = a[k][p], akq = a[k][q];
          a[k][p] = c * akp - s * akq;
          a[k][q] = s * akp + c * akq;
        }
        for (int k = 0; k < 3; ++k) {
          const float apk = a[p][k], aqk = a[q][k];
          a[p][k] = c * apk - s * aqk;
          a[q][k] = s * apk + c * aqk;
        }
        for (int k = 0; k < 3; ++k) {
          const float vkp = vec[k][p], vkq = vec[k][q];
          vec[k][p] = c * vkp - s * vkq;
          vec[k][q] = s * vkp + c * vkq;
        }
      }
    }
  }

  struct Eig {
    float value;
    float vec[3];
  };
  Eig e[3];
  for (int i = 0; i < 3; ++i) {
    e[i].value = a[i][i];
    for (int r = 0; r < 3; ++r) e[i].vec[r] = vec[r][i];
  }
  std::sort(e, e + 3, [](const Eig& l, const Eig& r) { return l.value < r.value; });
  for (int i = 0; i < 3; ++i) {
    eval[i] = e[i].value;
    float len = 0.f;
    for (int r = 0; r < 3; ++r) len += e[i].vec[r] * e[i].vec[r];
    len = std::sqrt(std::max(len, 1e-30f));
    for (int r = 0; r < 3; ++r) vec[r][i] = e[i].vec[r] / len;
  }
}

// PCA of a point set: centroid + eigenvectors (columns, ascending eigenvalue).
// Column 0 is the plane normal (smallest variance), columns 1/2 span the plane.
inline void computePca(const std::vector<Vec3>& pts, Vec3* mean, float eval[3], float vec[3][3]) {
  const size_t n = pts.size();
  Vec3 m(0.f, 0.f, 0.f);
  for (const auto& p : pts) m += p;
  m = m / static_cast<float>(n);
  float cxx = 0.f, cyy = 0.f, czz = 0.f, cxy = 0.f, cxz = 0.f, cyz = 0.f;
  for (const auto& p : pts) {
    const Vec3 d = p - m;
    cxx += d.x * d.x; cyy += d.y * d.y; czz += d.z * d.z;
    cxy += d.x * d.y; cxz += d.x * d.z; cyz += d.y * d.z;
  }
  const float a[3][3] = {{cxx, cxy, cxz}, {cxy, cyy, cyz}, {cxz, cyz, czz}};
  symEigen3x3(a, eval, vec);
  if (mean) *mean = m;
}

// Quantile (0..1) of a sample; copies the input because nth_element mutates.
inline float quantile(std::vector<float> v, float q) {
  if (v.empty()) return 0.f;
  if (v.size() == 1) return v[0];
  q = std::min(std::max(q, 0.f), 1.f);
  const size_t k = static_cast<size_t>(std::floor(q * static_cast<float>(v.size() - 1)));
  std::nth_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(k), v.end());
  return v[k];
}

inline int64_t cellKey(int64_t ix, int64_t iy, int64_t iz) {
  return ((ix & 0x1FFFFF) << 42) | ((iy & 0x1FFFFF) << 21) | (iz & 0x1FFFFF);
}

inline std::vector<Vec3> voxelDownsample(const std::vector<Vec3>& pts, float leaf) {
  if (leaf <= 0.f) return pts;
  struct Cell {
    float sx = 0.f, sy = 0.f, sz = 0.f;
    int n = 0;
  };
  std::unordered_map<int64_t, Cell> grid;
  grid.reserve(pts.size() / 2 + 1);
  const float inv = 1.f / leaf;
  for (const auto& p : pts) {
    const int64_t ix = static_cast<int64_t>(std::floor(p.x * inv));
    const int64_t iy = static_cast<int64_t>(std::floor(p.y * inv));
    const int64_t iz = static_cast<int64_t>(std::floor(p.z * inv));
    Cell& c = grid[cellKey(ix, iy, iz)];
    c.sx += p.x; c.sy += p.y; c.sz += p.z; ++c.n;
  }
  std::vector<Vec3> out;
  out.reserve(grid.size());
  for (const auto& kv : grid) {
    const Cell& c = kv.second;
    out.push_back(Vec3(c.sx, c.sy, c.sz) / static_cast<float>(c.n));
  }
  return out;
}

// Union-find euclidean clustering over a spatial hash grid (cell size = tol).
inline std::vector<std::vector<int>> euclideanClusters(const std::vector<Vec3>& pts, float tol) {
  std::vector<std::vector<int>> clusters;
  const size_t n = pts.size();
  if (n == 0) return clusters;

  std::vector<int> parent(n);
  for (size_t i = 0; i < n; ++i) parent[i] = static_cast<int>(i);
  // find with path halving
  auto find = [&parent](int x) {
    while (parent[x] != x) {
      parent[x] = parent[parent[x]];
      x = parent[x];
    }
    return x;
  };

  const float cell = std::max(tol, 1e-3f);
  const float inv = 1.f / cell;
  std::unordered_map<int64_t, std::vector<int>> grid;
  grid.reserve(n / 2 + 1);
  std::vector<int64_t> ix(n), iy(n), iz(n);
  for (size_t i = 0; i < n; ++i) {
    ix[i] = static_cast<int64_t>(std::floor(pts[i].x * inv));
    iy[i] = static_cast<int64_t>(std::floor(pts[i].y * inv));
    iz[i] = static_cast<int64_t>(std::floor(pts[i].z * inv));
    grid[cellKey(ix[i], iy[i], iz[i])].push_back(static_cast<int>(i));
  }

  const float tol2 = tol * tol;
  for (size_t i = 0; i < n; ++i) {
    for (int dx = -1; dx <= 1; ++dx)
      for (int dy = -1; dy <= 1; ++dy)
        for (int dz = -1; dz <= 1; ++dz) {
          const auto it = grid.find(cellKey(ix[i] + dx, iy[i] + dy, iz[i] + dz));
          if (it == grid.end()) continue;
          for (const int j : it->second) {
            if (j <= static_cast<int>(i)) continue;
            if (pts[i].distanceTo(pts[static_cast<size_t>(j)]) > tol) continue;
            const int ri = find(static_cast<int>(i)), rj = find(j);
            if (ri != rj) parent[ri] = rj;
          }
        }
  }

  std::unordered_map<int, std::vector<int>> groups;
  groups.reserve(n / 4 + 1);
  for (size_t i = 0; i < n; ++i) groups[find(static_cast<int>(i))].push_back(static_cast<int>(i));
  clusters.reserve(groups.size());
  for (auto& kv : groups) clusters.push_back(std::move(kv.second));
  return clusters;
}

// ============================ single-frame detector ============================

enum class DetectStatus {
  Ok = 0,
  EmptyCloud,         // nothing left after range crop / downsample
  NoCandidateCluster, // no cluster large enough to be a frame
  TooFewInliers,
  NotPlanar,
  NotSquare,
  TooSmallOrLarge,
  HoleFilled,   // looks like a plate / wall, not a frame with a hole
  BadViewAngle, // frame seen too obliquely for a trustworthy center estimate
};

inline const char* detectStatusName(DetectStatus s) {
  switch (s) {
    case DetectStatus::Ok: return "OK";
    case DetectStatus::EmptyCloud: return "EMPTY_CLOUD";
    case DetectStatus::NoCandidateCluster: return "NO_CANDIDATE_CLUSTER";
    case DetectStatus::TooFewInliers: return "TOO_FEW_INLIERS";
    case DetectStatus::NotPlanar: return "NOT_PLANAR";
    case DetectStatus::NotSquare: return "NOT_SQUARE";
    case DetectStatus::TooSmallOrLarge: return "TOO_SMALL_OR_LARGE";
    case DetectStatus::HoleFilled: return "HOLE_FILLED";
    case DetectStatus::BadViewAngle: return "BAD_VIEW_ANGLE";
  }
  return "UNKNOWN";
}

struct DetectorParams {
  // region of interest: sphere around the drone
  float min_range = 0.5f;  // body + lidar blind zone
  float max_range = 8.0f;

  // downsample + clustering
  float voxel_leaf = 0.05f;
  float cluster_tol = 0.25f;     // max gap inside one connected frame
  int min_cluster_points = 40;   // in voxel points; lower (e.g. 20) for far frames

  // plane fit
  float plane_inlier_dist = 0.05f;
  int plane_refine_iters = 2;
  int min_plane_inliers = 25;
  float min_inlier_ratio = 0.6f;

  // frame validation
  float min_frame_size = 0.55f;     // min side length [m], tune to actual frame
  float max_frame_size = 2.5f;      // max side length [m]
  float squareness_ratio = 1.35f;   // max(width/height) ratio
  float hole_check_ratio = 0.25f;   // required clear circle radius = ratio * min side
  bool require_hole = true;
  float max_view_angle_deg = 65.f;  // max angle between normal and drone->frame
  float extent_clip_quantile = 0.02f;  // trims in-plane outliers in extent estimate

  int max_input_points = 200000;
};

struct FrameDetection {
  Vec3 center;   // center of the opening (world frame)
  Vec3 normal;   // unit plane normal, oriented along travel direction
  float width = 0.f;   // frame side along in-plane axis 1
  float height = 0.f;  // frame side along in-plane axis 2
  int inlier_count = 0;
  int cluster_points = 0;
};

class FrameDetector {
 public:
  FrameDetector() : FrameDetector(DetectorParams()) {}
  explicit FrameDetector(const DetectorParams& p) : p_(p) {}

  const DetectorParams& params() const { return p_; }

  DetectStatus detect(const std::vector<Vec3>& cloud, const Vec3& drone_pos,
                      FrameDetection* out = nullptr) {
    std::vector<Vec3> roi;
    roi.reserve(cloud.size());
    for (const auto& q : cloud) {
      const float r = q.distanceTo(drone_pos);
      if (r >= p_.min_range && r <= p_.max_range) roi.push_back(q);
    }
    if (static_cast<int>(roi.size()) < p_.min_cluster_points)
      return DetectStatus::EmptyCloud;

    const std::vector<Vec3> pts = voxelDownsample(roi, p_.voxel_leaf);
    if (static_cast<int>(pts.size()) < p_.min_cluster_points)
      return DetectStatus::EmptyCloud;

    std::vector<std::vector<int>> clusters = euclideanClusters(pts, p_.cluster_tol);
    std::sort(clusters.begin(), clusters.end(),
              [](const std::vector<int>& a, const std::vector<int>& b) {
                return a.size() > b.size();
              });

    DetectStatus best_fail = DetectStatus::NoCandidateCluster;
    bool have_fail = false;
    for (const auto& cl : clusters) {
      if (static_cast<int>(cl.size()) < p_.min_cluster_points) break;  // sorted desc

      Vec3 lo(std::numeric_limits<float>::max(), std::numeric_limits<float>::max(),
              std::numeric_limits<float>::max());
      Vec3 hi(std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest(),
              std::numeric_limits<float>::lowest());
      for (const int idx : cl) {
        const Vec3& q = pts[static_cast<size_t>(idx)];
        lo.x = std::min(lo.x, q.x); lo.y = std::min(lo.y, q.y); lo.z = std::min(lo.z, q.z);
        hi.x = std::max(hi.x, q.x); hi.y = std::max(hi.y, q.y); hi.z = std::max(hi.z, q.z);
      }
      const float diag = (hi - lo).norm();
      if (diag < 0.5f * p_.min_frame_size) continue;
      if (diag > 2.5f * p_.max_frame_size) continue;

      std::vector<Vec3> cpts;
      cpts.reserve(cl.size());
      for (const int idx : cl) cpts.push_back(pts[static_cast<size_t>(idx)]);

      FrameDetection det;
      const DetectStatus st = evaluateCluster(cpts, drone_pos, &det);
      if (st == DetectStatus::Ok) {
        if (out) *out = det;
        return st;
      }
      if (!have_fail) {
        best_fail = st;
        have_fail = true;
      }
    }
    return have_fail ? best_fail : DetectStatus::NoCandidateCluster;
  }

 private:
  DetectStatus evaluateCluster(const std::vector<Vec3>& pts, const Vec3& drone_pos,
                               FrameDetection* out) {
    // iterative plane fit: PCA -> inlier selection -> PCA (refine_iters times)
    std::vector<Vec3> work = pts;
    Vec3 mean;
    float eval[3];
    float vec[3][3];
    for (int it = 0; it <= p_.plane_refine_iters; ++it) {
      if (work.size() < 8) return DetectStatus::TooFewInliers;
      computePca(work, &mean, eval, vec);
      if (it == p_.plane_refine_iters) break;
      Vec3 n0(vec[0][0], vec[1][0], vec[2][0]);
      std::vector<Vec3> inl;
      inl.reserve(work.size());
      for (const auto& q : work)
        if (std::fabs((q - mean).dot(n0)) <= p_.plane_inlier_dist) inl.push_back(q);
      if (static_cast<int>(inl.size()) < p_.min_plane_inliers)
        return DetectStatus::TooFewInliers;
      work.swap(inl);
    }
    if (static_cast<float>(work.size()) <
        p_.min_inlier_ratio * static_cast<float>(pts.size()))
      return DetectStatus::NotPlanar;

    Vec3 n(vec[0][0], vec[1][0], vec[2][0]);
    Vec3 b1(vec[0][1], vec[1][1], vec[2][1]);
    Vec3 b2(vec[0][2], vec[1][2], vec[2][2]);
    n = n.normalized();
    b1 = b1.normalized();
    b2 = b2.normalized();

    // orient the normal along the drone -> frame direction
    if ((mean - drone_pos).dot(n) < 0.f) n = -n;

    // in-plane extents with outlier clipping
    std::vector<float> u, v;
    u.reserve(work.size());
    v.reserve(work.size());
    for (const auto& q : work) {
      const Vec3 d = q - mean;
      u.push_back(d.dot(b1));
      v.push_back(d.dot(b2));
    }
    const float qclip = std::min(std::max(p_.extent_clip_quantile, 0.f), 0.25f);
    const float ulo = quantile(u, qclip), uhi = quantile(u, 1.f - qclip);
    const float vlo = quantile(v, qclip), vhi = quantile(v, 1.f - qclip);
    const float width = uhi - ulo;
    const float height = vhi - vlo;
    const float uc = 0.5f * (ulo + uhi);
    const float vc = 0.5f * (vlo + vhi);
    const Vec3 center = mean + b1 * uc + b2 * vc;

    if (width < p_.min_frame_size || height < p_.min_frame_size ||
        width > p_.max_frame_size || height > p_.max_frame_size)
      return DetectStatus::TooSmallOrLarge;
    if (std::max(width, height) / std::min(width, height) > p_.squareness_ratio)
      return DetectStatus::NotSquare;

    if (p_.require_hole) {
      const float hole_r = p_.hole_check_ratio * std::min(width, height);
      for (size_t i = 0; i < work.size(); ++i) {
        const float du = u[i] - uc, dv = v[i] - vc;
        if (std::sqrt(du * du + dv * dv) < hole_r) return DetectStatus::HoleFilled;
      }
    }

    const Vec3 dir = (center - drone_pos).normalized();
    if (dir.dot(n) < std::cos(degToRad(p_.max_view_angle_deg)))
      return DetectStatus::BadViewAngle;

    if (out) {
      out->center = center;
      out->normal = n;
      out->width = width;
      out->height = height;
      out->inlier_count = static_cast<int>(work.size());
      out->cluster_points = static_cast<int>(pts.size());
    }
    return DetectStatus::Ok;
  }

  DetectorParams p_;
};

// ============================ temporal stabilizer ============================

struct StabilizerParams {
  int lock_count = 5;          // consistent detections required to lock
  int unlock_miss_count = 8;   // consecutive misses to unlock (hysteresis)
  int history_len = 20;        // candidate window length while searching

  float center_consistency = 0.35f;    // [m]
  float angle_consistency_deg = 30.f;

  float center_ema_alpha = 0.35f;
  float normal_ema_alpha = 0.35f;
  float size_ema_alpha = 0.25f;

  float approach_dist = 1.5f;  // waypoint behind the frame (own side)
  float exit_dist = 2.5f;      // waypoint beyond the frame
  bool include_center_wp = true;
};

struct StableFrame {
  Vec3 center;
  Vec3 travel_dir;        // unit, from own side through the opening
  float width = 0.f;
  float height = 0.f;
};

class FrameStabilizer {
 public:
  enum class State { SEARCHING = 0, LOCKED = 1 };

  FrameStabilizer() : FrameStabilizer(StabilizerParams()) {}
  explicit FrameStabilizer(const StabilizerParams& p) : p_(p) {}

  void reset() {
    state_ = State::SEARCHING;
    hist_.clear();
    miss_count_ = 0;
    stable_ = StableFrame();
  }

  // Feed one detection. Returns true when the stable frame was (re)published,
  // i.e. the lock just happened or a consistent detection updated the EMA.
  bool update(const FrameDetection& det) {
    if (state_ == State::SEARCHING) {
      if (!hist_.empty()) {
        const Vec3 ref_c = medianCenter(hist_);
        const Vec3 ref_n = meanNormal(hist_);
        if (!consistentWith(ref_c, ref_n, det.center, det.normal)) hist_.clear();
      }
      hist_.push_back(det);
      const int cap = std::max(p_.history_len, p_.lock_count);
      while (static_cast<int>(hist_.size()) > cap) hist_.pop_front();
      if (static_cast<int>(hist_.size()) >= p_.lock_count) {
        state_ = State::LOCKED;
        stable_.center = medianCenter(hist_);
        stable_.travel_dir = meanNormal(hist_);
        stable_.width = medianSize(hist_, true);
        stable_.height = medianSize(hist_, false);
        miss_count_ = 0;
        return true;
      }
      return false;
    }

    if (consistentWith(stable_.center, stable_.travel_dir, det.center, det.normal)) {
      Vec3 n = det.normal;
      if (n.dot(stable_.travel_dir) < 0.f) n = -n;
      stable_.center = stable_.center + (det.center - stable_.center) * p_.center_ema_alpha;
      stable_.travel_dir =
          (stable_.travel_dir + (n - stable_.travel_dir) * p_.normal_ema_alpha).normalized();
      stable_.width += (det.width - stable_.width) * p_.size_ema_alpha;
      stable_.height += (det.height - stable_.height) * p_.size_ema_alpha;
      miss_count_ = 0;
      return true;
    }
    miss();
    return false;
  }

  // Feed a miss (cloud arrived but no valid detection).
  void miss() {
    if (state_ != State::LOCKED) return;
    if (++miss_count_ >= p_.unlock_miss_count) {
      state_ = State::SEARCHING;
      hist_.clear();
      miss_count_ = 0;
    }
  }

  bool locked() const { return state_ == State::LOCKED; }
  State state() const { return state_; }
  const StableFrame& stable() const { return stable_; }
  int lockProgress() const { return static_cast<int>(hist_.size()); }
  int missCount() const { return miss_count_; }
  const StabilizerParams& params() const { return p_; }

 private:
  static Vec3 medianCenter(const std::deque<FrameDetection>& h) {
    std::vector<float> xs, ys, zs;
    xs.reserve(h.size()); ys.reserve(h.size()); zs.reserve(h.size());
    for (const auto& d : h) {
      xs.push_back(d.center.x);
      ys.push_back(d.center.y);
      zs.push_back(d.center.z);
    }
    return Vec3(quantile(xs, 0.5f), quantile(ys, 0.5f), quantile(zs, 0.5f));
  }

  static Vec3 meanNormal(const std::deque<FrameDetection>& h) {
    Vec3 sum(0.f, 0.f, 0.f);
    const Vec3& ref = h.front().normal;
    for (const auto& d : h) {
      Vec3 n = d.normal;
      if (n.dot(ref) < 0.f) n = -n;
      sum += n;
    }
    return sum.normalized();
  }

  static float medianSize(const std::deque<FrameDetection>& h, bool width) {
    std::vector<float> vals;
    vals.reserve(h.size());
    for (const auto& d : h) vals.push_back(width ? d.width : d.height);
    return quantile(vals, 0.5f);
  }

  bool consistentWith(const Vec3& ref_c, const Vec3& ref_n, const Vec3& c, const Vec3& n) const {
    return c.distanceTo(ref_c) <= p_.center_consistency &&
           unsignedAngle(n, ref_n) <= degToRad(p_.angle_consistency_deg);
  }

  StabilizerParams p_;
  State state_ = State::SEARCHING;
  std::deque<FrameDetection> hist_;
  StableFrame stable_;
  int miss_count_ = 0;
};

// P1 (approach) -> [P2 (center)] -> P3 (exit), empty when travel_dir degenerate.
inline std::vector<Vec3> crossingWaypoints(const StableFrame& f, const StabilizerParams& p) {
  std::vector<Vec3> wps;
  if (f.travel_dir.norm() < 0.5f) return wps;
  wps.push_back(f.center - f.travel_dir * p.approach_dist);
  if (p.include_center_wp) wps.push_back(f.center);
  wps.push_back(f.center + f.travel_dir * p.exit_dist);
  return wps;
}

}  // namespace window_detector

#endif  // WINDOW_DETECTOR_FRAME_DETECTOR_H
