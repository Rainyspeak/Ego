// 方案A 离线单元测试 T1–T17（设计文档 6.1 节）
// 运行：g++ -std=c++14 -O2 -Wall -I include test/test_frame_detector.cpp -o frame_detector_test && ./frame_detector_test
// 无 ROS / PCL / Eigen 依赖，任意平台可编译。

#include "forecast_searching/frame_detector.h"

#include <cstdarg>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>

using namespace forecast_searching;

static int g_pass = 0, g_fail = 0;

static void report(bool ok, const std::string& name, const std::string& detail = "") {
  if (ok) {
    ++g_pass;
    std::printf("[PASS] %s\n", name.c_str());
  } else {
    ++g_fail;
    std::printf("[FAIL] %s   %s\n", name.c_str(), detail.c_str());
  }
}

static std::string fmt(const char* f, ...) {
  char buf[512];
  va_list ap;
  va_start(ap, f);
  vsnprintf(buf, sizeof(buf), f, ap);
  va_end(ap);
  return std::string(buf);
}

// ---------------- 合成点云生成器 ----------------

// 方形框：四条边均匀采样 + 管厚抖动 + 高斯噪声。drop_right_edge 丢弃完整一条边（T4）。
static std::vector<double> makeFrameCloud(const Vec3& c, const Vec3& n_raw, double width, double height,
                                          int pts_per_edge, double tube, double sigma,
                                          std::mt19937& rng, bool drop_right_edge = false) {
  const Vec3 n = normalized(n_raw);
  const Vec3 u = anyPerpendicular(n);
  const Vec3 v = normalized(cross(n, u));
  std::vector<double> out;
  std::uniform_real_distribution<double> uni(-0.5, 0.5);
  std::normal_distribution<double> gauss(0.0, sigma);
  auto add = [&](const Vec3& p) {
    out.push_back(p[0]);
    out.push_back(p[1]);
    out.push_back(p[2]);
  };
  for (int side = 0; side < 2; ++side) {  // 两条水平边（沿 u，v = ±h/2）
    const double vsign = side == 0 ? 1.0 : -1.0;
    for (int i = 0; i < pts_per_edge; ++i) {
      const Vec3 p = c + u * (uni(rng) * width) +
                     v * (vsign * height / 2.0 + uni(rng) * tube) + n * gauss(rng);
      add(p);
    }
  }
  for (int side = 0; side < 2; ++side) {  // 两条垂直边（沿 v，u = ±w/2）
    const double usign = side == 0 ? 1.0 : -1.0;
    if (drop_right_edge && usign > 0) continue;
    for (int i = 0; i < pts_per_edge; ++i) {
      const Vec3 p = c + u * (usign * width / 2.0 + uni(rng) * tube) +
                     v * (uni(rng) * height) + n * gauss(rng);
      add(p);
    }
  }
  return out;
}

// 实心板：面内均匀填充 + 高斯噪声（T5/T8）
static std::vector<double> makeBoardCloud(const Vec3& c, const Vec3& n_raw, double width, double height,
                                          int total_pts, double sigma, std::mt19937& rng) {
  const Vec3 n = normalized(n_raw);
  const Vec3 u = anyPerpendicular(n);
  const Vec3 v = normalized(cross(n, u));
  std::vector<double> out;
  std::uniform_real_distribution<double> uni(-0.5, 0.5);
  std::normal_distribution<double> gauss(0.0, sigma);
  for (int i = 0; i < total_pts; ++i) {
    const Vec3 p = c + u * (uni(rng) * width) + v * (uni(rng) * height) + n * gauss(rng);
    out.push_back(p[0]);
    out.push_back(p[1]);
    out.push_back(p[2]);
  }
  return out;
}

// 圆环：均匀角度采样 + 管厚抖动 + 高斯噪声（T18）。
// arc_span < 2π 时只生成以 θ=0 为中心、跨度 arc_span 的开弧（T27/T28 近距视场裁切）
static std::vector<double> makeRingCloud(const Vec3& c, const Vec3& n_raw, double diameter,
                                         int total_pts, double tube, double sigma,
                                         std::mt19937& rng, double arc_span = 2.0 * kPi) {
  const Vec3 n = normalized(n_raw);
  const Vec3 u = anyPerpendicular(n);
  const Vec3 v = normalized(cross(n, u));
  std::vector<double> out;
  std::uniform_real_distribution<double> uni(-0.5, 0.5);
  std::normal_distribution<double> gauss(0.0, sigma);
  for (int i = 0; i < total_pts; ++i) {
    const double th = (arc_span >= 2.0 * kPi - 1e-9) ? uni(rng) * 2.0 * kPi
                                                     : uni(rng) * arc_span;
    const double rr = diameter / 2.0 + uni(rng) * tube;
    const Vec3 p = c + u * (std::cos(th) * rr) + v * (std::sin(th) * rr) + n * gauss(rng);
    out.push_back(p[0]);
    out.push_back(p[1]);
    out.push_back(p[2]);
  }
  return out;
}

// 单条边：edge 0/1=上/下（沿 u），2/3=左/右（沿 v）——T19 稀疏逐帧用
static std::vector<double> makeEdgeCloud(const Vec3& c, const Vec3& n_raw, double width, double height,
                                         int edge, int pts, double tube, double sigma,
                                         std::mt19937& rng) {
  const Vec3 n = normalized(n_raw);
  const Vec3 u = anyPerpendicular(n);
  const Vec3 v = normalized(cross(n, u));
  std::vector<double> out;
  std::uniform_real_distribution<double> uni(-0.5, 0.5);
  std::normal_distribution<double> gauss(0.0, sigma);
  for (int i = 0; i < pts; ++i) {
    Vec3 p;
    if (edge < 2) {
      const double vsign = edge == 0 ? 1.0 : -1.0;
      p = c + u * (uni(rng) * width) + v * (vsign * height / 2.0 + uni(rng) * tube) + n * gauss(rng);
    } else {
      const double usign = edge == 2 ? 1.0 : -1.0;
      p = c + u * (usign * width / 2.0 + uni(rng) * tube) + v * (uni(rng) * height) + n * gauss(rng);
    }
    out.push_back(p[0]);
    out.push_back(p[1]);
    out.push_back(p[2]);
  }
  return out;
}

// 箱内随机离群点（T2）
static void addOutliers(std::vector<double>& pts, const Vec3& c, double box_half, int count,
                        std::mt19937& rng) {
  std::uniform_real_distribution<double> uni(-box_half, box_half);
  for (int i = 0; i < count; ++i) {
    const Vec3 p = c + vec3(uni(rng), uni(rng), uni(rng));
    pts.push_back(p[0]);
    pts.push_back(p[1]);
    pts.push_back(p[2]);
  }
}

// 检测应有的法向：远离无人机一侧（飞行方向）
static Vec3 expectedFlightNormal(const Vec3& n, const Vec3& c, const Vec3& drone) {
  return dot(n, c - drone) > 0 ? n : -n;
}

static FrameDetection makeDet(const Vec3& c, const Vec3& n, double w, double h) {
  FrameDetection d;
  d.center = c;
  d.normal = normalized(n);
  d.axis_u = anyPerpendicular(d.normal);
  d.width = w;
  d.height = h;
  d.inlier_count = 100;
  return d;
}

// ---------------- 单帧检测 T1–T9 ----------------

static void t1_ideal() {
  std::mt19937 rng(42);
  const Vec3 drone = vec3(0, 0, 1.2), c = vec3(5, 0, 1.2), n = vec3(1, 0, 0);
  auto pts = makeFrameCloud(c, n, 1.0, 1.0, 375, 0.02, 0.005, rng);  // 1500 点
  FrameDetector detector;
  FrameDetection out;
  const DetectStatus st = detector.detect(pts, drone, out);
  const double cerr = norm(out.center - c);
  const double aerr = angleBetween(out.normal, expectedFlightNormal(n, c, drone)) * 180.0 / kPi;
  const double size = 0.5 * (out.width + out.height);
  report(st == DetectStatus::OK && cerr < 0.03 && aerr < 3.0 && std::fabs(size - 1.0) < 0.06,
         "T1 ideal frame",
         fmt("st=%s c=(%.3f,%.3f,%.3f) cerr=%.3f aerr=%.2f size=%.3f", detectStatusToString(st),
             out.center[0], out.center[1], out.center[2], cerr, aerr, size));
}

static void t2_noise_outliers() {
  std::mt19937 rng(7);
  const Vec3 drone = vec3(0, 0, 1.2), c = vec3(5, 0, 1.2), n = vec3(1, 0, 0);
  auto pts = makeFrameCloud(c, n, 1.0, 1.0, 375, 0.02, 0.01, rng);
  addOutliers(pts, c, 1.5, 150, rng);
  FrameDetector detector;
  FrameDetection out;
  const DetectStatus st = detector.detect(pts, drone, out);
  const double cerr = norm(out.center - c);
  const double aerr = angleBetween(out.normal, expectedFlightNormal(n, c, drone)) * 180.0 / kPi;
  report(st == DetectStatus::OK && cerr < 0.04 && aerr < 3.0, "T2 noise + outliers",
         fmt("st=%s cerr=%.3f aerr=%.2f", detectStatusToString(st), cerr, aerr));
}

static void t3_tilted() {
  std::mt19937 rng(11);
  const Vec3 drone = vec3(0, 0, 1.2), c = vec3(4, 0.5, 1.3), n = vec3(1, 0.35, -0.15);
  auto pts = makeFrameCloud(c, n, 1.0, 1.0, 375, 0.02, 0.008, rng);
  FrameDetector detector;
  FrameDetection out;
  const DetectStatus st = detector.detect(pts, drone, out);
  const double cerr = norm(out.center - c);
  const double aerr = angleBetween(out.normal, expectedFlightNormal(n, c, drone)) * 180.0 / kPi;
  const double size = 0.5 * (out.width + out.height);
  report(st == DetectStatus::OK && cerr < 0.05 && aerr < 3.0 && std::fabs(size - 1.0) < 0.10,
         "T3 tilted + rotated frame",
         fmt("st=%s cerr=%.3f aerr=%.2f size=%.3f", detectStatusToString(st), cerr, aerr, size));
}

static void t4_missing_edge() {
  std::mt19937 rng(23);
  const Vec3 drone = vec3(0, 0, 1.2), c = vec3(4, 0, 1.2), n = vec3(1, 0, 0);
  auto pts = makeFrameCloud(c, n, 1.0, 1.0, 375, 0.02, 0.008, rng, /*drop_right_edge=*/true);
  FrameDetector detector;
  FrameDetection out;
  const DetectStatus st = detector.detect(pts, drone, out);
  const double cerr = norm(out.center - c);
  const double size = 0.5 * (out.width + out.height);
  report(st == DetectStatus::OK && cerr < 0.05 && std::fabs(size - 1.0) < 0.10, "T4 one edge missing",
         fmt("st=%s cerr=%.3f size=%.3f", detectStatusToString(st), cerr, size));
}

static void t5_board() {
  std::mt19937 rng(5);
  const Vec3 drone = vec3(0, 0, 1.2), c = vec3(4, 0, 1.2), n = vec3(1, 0, 0);
  auto pts = makeBoardCloud(c, n, 1.0, 1.0, 2000, 0.008, rng);
  FrameDetector detector;
  FrameDetection out;
  const DetectStatus st = detector.detect(pts, drone, out);
  report(st == DetectStatus::HOLE_FILLED, "T5 solid board rejected",
         fmt("st=%s", detectStatusToString(st)));
}

static void t6_nonsquare() {
  std::mt19937 rng(13);
  const Vec3 drone = vec3(0, 0, 1.2), c = vec3(4, 0, 1.2), n = vec3(1, 0, 0);
  // 宽高比 2.35 > 2.0 上限（两边均 ≥0.55，先过尺寸再挂比例）
  auto pts = makeFrameCloud(c, n, 1.6, 0.65, 500, 0.02, 0.008, rng);
  FrameDetector detector;
  FrameDetection out;
  const DetectStatus st = detector.detect(pts, drone, out);
  report(st == DetectStatus::NOT_SQUARE, "T6 overlong rectangle rejected",
         fmt("st=%s", detectStatusToString(st)));
}

static void t6b_rectangle_accepted() {
  std::mt19937 rng(13);
  const Vec3 drone = vec3(0, 0, 1.2), c = vec3(4, 0, 1.2), n = vec3(1, 0, 0);
  auto pts = makeFrameCloud(c, n, 1.0, 0.62, 375, 0.02, 0.008, rng);
  FrameDetector detector;
  FrameDetection out;
  const DetectStatus st = detector.detect(pts, drone, out);
  const double wmax = std::max(out.width, out.height), wmin = std::min(out.width, out.height);
  report(st == DetectStatus::OK && std::fabs(wmax - 1.0) < 0.12 && std::fabs(wmin - 0.62) < 0.10,
         "T6b rectangle frame accepted", fmt("st=%s %.3fx%.3f", detectStatusToString(st), out.width,
                                             out.height));
}

static void t7_small() {
  std::mt19937 rng(17);
  const Vec3 drone = vec3(0, 0, 1.2), c = vec3(4, 0, 1.2), n = vec3(1, 0, 0);
  auto pts = makeFrameCloud(c, n, 0.4, 0.4, 400, 0.08, 0.008, rng);
  FrameDetector detector;
  FrameDetection out;
  const DetectStatus st = detector.detect(pts, drone, out);
  report(st == DetectStatus::TOO_SMALL_OR_LARGE, "T7 small frame rejected",
         fmt("st=%s", detectStatusToString(st)));
}

static void t8_wall() {
  std::mt19937 rng(19);
  const Vec3 drone = vec3(0, 0, 1.2), c = vec3(4, 0, 1.2), n = vec3(1, 0, 0);
  auto pts = makeBoardCloud(c, n, 2.0, 2.0, 4000, 0.008, rng);
  FrameDetector detector;
  FrameDetection out;
  const DetectStatus st = detector.detect(pts, drone, out);
  report(st != DetectStatus::OK, "T8 big wall rejected", fmt("st=%s", detectStatusToString(st)));
}

static void t9_oblique() {
  std::mt19937 rng(29);
  const Vec3 c = vec3(4, 0, 1.2), n = vec3(1, 0, 0);
  // 视线与法向夹角 68°（> 65° 上限），距离 5 m
  const Vec3 drone = c - vec3(5.0 * std::cos(68.0 * kPi / 180.0), 5.0 * std::sin(68.0 * kPi / 180.0), 0.0);
  auto pts = makeFrameCloud(c, n, 1.0, 1.0, 375, 0.02, 0.008, rng);
  FrameDetector detector;
  FrameDetection out;
  const DetectStatus st = detector.detect(pts, drone, out);
  report(st == DetectStatus::BAD_VIEW_ANGLE, "T9 oblique view rejected",
         fmt("st=%s", detectStatusToString(st)));
}

// ---------------- 稳定器 T10–T14, T17 ----------------

static void t10_lock() {
  FrameStabilizer stab;
  const Vec3 c = vec3(2, 0, 1.2);
  const Vec3 n = vec3(1, 0, 0);
  for (int i = 0; i < 4; ++i) stab.onDetection(makeDet(c, n, 1.0, 1.0));
  const bool not_locked_at_4 = !stab.locked() && stab.lockProgress() == 4;
  stab.onDetection(makeDet(c, n, 1.0, 1.0));
  const bool locked_at_5 = stab.locked();
  const double cerr = norm(stab.frame().center - c);
  report(not_locked_at_4 && locked_at_5 && cerr < 1e-9, "T10 lock at 5th frame",
         fmt("at4=%d at5=%d cerr=%.6f", (int)not_locked_at_4, (int)locked_at_5, cerr));
}

static void t11_waypoints() {
  FrameStabilizer stab;
  const Vec3 c = vec3(2, 0, 1.2);
  const Vec3 n = vec3(0, 1, 0);
  for (int i = 0; i < 5; ++i) stab.onDetection(makeDet(c, n, 1.0, 1.0));
  const auto wps = stab.waypoints();
  bool ok = wps.size() == 3;
  if (ok) {
    const Vec3 p1 = wps[0], p2 = wps[1], p3 = wps[2];
    ok = ok && std::fabs(norm(p1 - c) - 1.5) < 1e-9 && dot(normalized(p1 - c), -n) > 0.999;
    ok = ok && norm(p2 - c) < 1e-9;
    ok = ok && std::fabs(norm(p3 - c) - 2.5) < 1e-9 && dot(normalized(p3 - c), n) > 0.999;
  }
  report(ok, "T11 waypoint geometry P1/P2/P3", fmt("n=%zu", wps.size()));

  // include_center_wp=false → 只输出 P1/P3
  StabilizerParams sp;
  sp.include_center_wp = false;
  FrameStabilizer stab2(sp);
  for (int i = 0; i < 5; ++i) stab2.onDetection(makeDet(c, n, 1.0, 1.0));
  report(stab2.waypoints().size() == 2, "T11b include_center_wp=false", "");
}

static void t12_unlock() {
  FrameStabilizer stab;
  const Vec3 c = vec3(2, 0, 1.2);
  const Vec3 n = vec3(1, 0, 0);
  for (int i = 0; i < 5; ++i) stab.onDetection(makeDet(c, n, 1.0, 1.0));
  for (int i = 0; i < 7; ++i) stab.onMiss();
  const bool still_locked = stab.locked();
  stab.onMiss();
  const bool unlocked_at_8 = !stab.locked() && stab.waypoints().empty();
  report(still_locked && unlocked_at_8, "T12 unlock at 8th miss", "");
}

static void t13_false_detection() {
  FrameStabilizer stab;
  const Vec3 c = vec3(2, 0, 1.2);
  const Vec3 n = vec3(1, 0, 0);
  for (int i = 0; i < 5; ++i) stab.onDetection(makeDet(c, n, 1.0, 1.0));
  const auto wps_before = stab.waypoints();
  stab.onDetection(makeDet(c + vec3(1.5, 0, 0), n, 1.2, 1.2));  // 1.5 m 外的"另一个矩形"
  const bool unchanged = stab.waypoints() == wps_before && norm(stab.frame().center - c) < 1e-9;
  const bool not_unlocked = stab.locked() && stab.missCount() == 1;
  report(unchanged && not_unlocked, "T13 false detection rejected", "");
}

static void t14_jitter() {
  FrameStabilizer stab;
  const Vec3 c = vec3(3, 0.5, 1.2);
  const Vec3 n = vec3(1, 0, 0);
  std::mt19937 rng(101);
  std::normal_distribution<double> gauss(0.0, 0.05);
  for (int i = 0; i < 5; ++i) stab.onDetection(makeDet(c, n, 1.0, 1.0));
  for (int i = 0; i < 15; ++i) {
    const Vec3 jitter = vec3(gauss(rng), gauss(rng), gauss(rng));
    stab.onDetection(makeDet(c + jitter, n, 1.0, 1.0));
  }
  const double err = norm(stab.frame().center - c);
  report(stab.locked() && err < 0.10, "T14 jitter suppressed", fmt("err=%.3f", err));
}

// ---------------- 多目标 T15 ----------------

static void t15_ambiguous() {
  std::mt19937 rng(31);
  const Vec3 drone = vec3(0, 0, 1.2);
  const Vec3 ca = vec3(4, 0, 1.2), cb = vec3(4, 2.5, 1.2), n = vec3(1, 0, 0);
  auto pts = makeFrameCloud(ca, n, 1.0, 1.0, 375, 0.02, 0.008, rng);
  auto pts_b = makeFrameCloud(cb, n, 1.4, 1.4, 500, 0.02, 0.008, rng);
  pts.insert(pts.end(), pts_b.begin(), pts_b.end());

  // 无先验：默认就近优先（不再歧义死锁），选中更近的 A
  FrameDetector detector;
  FrameDetection out;
  const DetectStatus st = detector.detect(pts, drone, out);
  report(st == DetectStatus::OK && norm(out.center - ca) < 0.05,
         "T15a two frames prefer nearest by default",
         fmt("st=%s cerrA=%.3f", detectStatusToString(st), norm(out.center - ca)));

  // 显式关闭就近优先：回到歧义
  DetectorParams dp_reject;
  dp_reject.prefer_nearest = false;
  FrameDetector det_reject(dp_reject);
  FrameDetection out_r;
  const DetectStatus st_r = det_reject.detect(pts, drone, out_r);
  report(st_r == DetectStatus::AMBIGUOUS_TARGET, "T15a2 two frames ambiguous when disabled",
         fmt("st=%s", detectStatusToString(st_r)));

  // 歧义结果喂稳定器：不锁定、无航点
  FrameStabilizer stab;
  stab.onMiss();  // AMBIGUOUS_TARGET 按无有效检测处理
  report(!stab.locked() && stab.waypoints().empty(), "T15b ambiguous keeps SEARCHING", "");

  // 尺寸先验 1.0 m → 选中 A
  DetectorParams dp;
  dp.has_target_size = true;
  dp.target_size = 1.0;
  FrameDetector det_size(dp);
  const DetectStatus st2 = det_size.detect(pts, drone, out);
  report(st2 == DetectStatus::OK && norm(out.center - ca) < 0.05 && norm(out.center - cb) > 1.0,
         "T15c size prior resolves A",
         fmt("st=%s cerrA=%.3f", detectStatusToString(st2), norm(out.center - ca)));

  // 方向先验 +X → 选中 A
  DetectorParams dp2;
  dp2.has_target_direction = true;
  dp2.target_direction = vec3(1, 0, 0);
  FrameDetector det_dir(dp2);
  const DetectStatus st3 = det_dir.detect(pts, drone, out);
  report(st3 == DetectStatus::OK && norm(out.center - ca) < 0.05, "T15d direction prior resolves A",
         fmt("st=%s", detectStatusToString(st3)));

  // 就近粘滞：选中 B 后，A 只比 B 近 0.15 m（< 余量 0.5）→ 仍选 B；
  // 把无人机挪近使 A 比 B 近 1.2 m（> 余量）→ 切换到 A
  const Vec3 drone2 = vec3(0, 1.0, 1.2);  // dA≈4.12, dB≈4.27：A 近 0.15 m
  FrameDetection out_s;
  const Vec3* sticky = &cb;
  const DetectStatus st_s = detector.detect(pts, drone2, out_s, sticky);
  const bool stay = st_s == DetectStatus::OK && norm(out_s.center - cb) < 0.05;
  const Vec3 drone3 = vec3(2.5, 0.2, 1.2);  // dA≈1.51, dB≈2.75：A 近 1.23 m
  const DetectStatus st_s2 = detector.detect(pts, drone3, out_s, sticky);
  const bool swit = st_s2 == DetectStatus::OK && norm(out_s.center - ca) < 0.05;
  report(stay && swit, "T15e nearest sticky margin (stay within, switch beyond)",
         fmt("stay=%d st=%s swit=%d st2=%s", (int)stay, detectStatusToString(st_s),
             (int)swit, detectStatusToString(st_s2)));
}

// ---------------- 消息超时 T16 ----------------

static void t16_timeout() {
  // 可注入假时钟：每次调用前进 0.5 s（远超 2/10Hz = 0.2 s 超时阈值）
  double fake_t = 100.0;
  FrameStabilizer::ClockFn clock = [&fake_t]() {
    fake_t += 0.5;
    return fake_t;
  };

  // 锁定后消息静默：按周期计 miss，第 8 次解锁
  FrameStabilizer stab(StabilizerParams(), clock);
  const Vec3 c = vec3(2, 0, 1.2);
  const Vec3 n = vec3(1, 0, 0);
  for (int i = 0; i < 5; ++i) stab.onDetection(makeDet(c, n, 1.0, 1.0));
  bool ok = stab.locked();
  for (int i = 0; i < 7; ++i) stab.onTimerTick();
  ok = ok && stab.locked();
  for (int i = 0; i < 1; ++i) stab.onTimerTick();
  ok = ok && !stab.locked() && stab.waypoints().empty();
  report(ok, "T16a cloud silence unlocks after 8 ticks", "");

  // SEARCHING：有效检测间隔超时 → 清空候选窗口
  FrameStabilizer stab2(StabilizerParams(), clock);
  for (int i = 0; i < 3; ++i) stab2.onDetection(makeDet(c, n, 1.0, 1.0));
  const bool had_progress = stab2.lockProgress() == 3;
  stab2.onTimerTick();  // now - last_det > 0.2 s → 清窗
  const bool cleared = stab2.lockProgress() == 0;
  stab2.onDetection(makeDet(c, n, 1.0, 1.0));
  stab2.onDetection(makeDet(c, n, 1.0, 1.0));
  const bool restarts = stab2.lockProgress() == 2 && !stab2.locked();
  report(had_progress && cleared && restarts, "T16b detection gap clears SEARCHING window", "");
}

// ---------------- 航点冻结 T17 ----------------

static void t17_freeze() {
  FrameStabilizer stab;
  const Vec3 c = vec3(3, 0, 1.2);
  const Vec3 n = vec3(1, 0, 0);

  // 未锁定时 commit 必须失败
  const bool commit_rejected = !stab.commit();

  for (int i = 0; i < 5; ++i) stab.onDetection(makeDet(c, n, 1.0, 1.0));
  const bool committed = stab.commit() && stab.frozen();
  const auto wps_frozen = stab.waypoints();
  const Vec3 c_frozen = stab.frame().center;

  // 冻结后继续喂有抖动的检测（一致性范围内）：c、n、航点不变
  std::mt19937 rng(55);
  std::normal_distribution<double> gauss(0.0, 0.05);
  for (int i = 0; i < 5; ++i) {
    stab.onDetection(makeDet(c + vec3(gauss(rng), gauss(rng), gauss(rng)), n, 1.0, 1.0));
  }
  const bool frozen_stable = stab.waypoints() == wps_frozen && stab.frame().center == c_frozen &&
                             stab.frozen();

  // 取消后恢复 EMA 跟踪
  stab.cancelCommit();
  stab.onDetection(makeDet(c + vec3(0.2, 0, 0), n, 1.0, 1.0));
  const bool resumed = !stab.frozen() && norm(stab.frame().center - c) > 0.05;

  // 解锁时冻结与航点一并清除
  for (int i = 0; i < 8; ++i) stab.onMiss();
  const bool cleared = !stab.locked() && !stab.frozen() && stab.waypoints().empty();

  report(commit_rejected && committed && frozen_stable && resumed && cleared, "T17 waypoint freeze",
         fmt("cr=%d cm=%d fs=%d rs=%d cl=%d", (int)commit_rejected, (int)committed,
             (int)frozen_stable, (int)resumed, (int)cleared));
}

// ---------------- 圆环 T18 / 稀疏累积 T19 ----------------

static void t18_circle() {
  std::mt19937 rng(37);
  const Vec3 drone = vec3(0, 0, 1.2), c = vec3(4, 0, 1.2), n = vec3(1, 0, 0);
  auto pts = makeRingCloud(c, n, 1.0, 1500, 0.02, 0.008, rng);
  FrameDetector detector;
  FrameDetection out;
  const DetectStatus st = detector.detect(pts, drone, out);
  const double cerr = norm(out.center - c);
  const double aerr = angleBetween(out.normal, expectedFlightNormal(n, c, drone)) * 180.0 / kPi;
  const double size = 0.5 * (out.width + out.height);
  report(st == DetectStatus::OK && cerr < 0.03 && aerr < 3.0 && std::fabs(size - 1.0) < 0.12,
         "T18 circular ring accepted",
         fmt("st=%s cerr=%.3f aerr=%.2f size=%.3f", detectStatusToString(st), cerr, aerr, size));
}

static void t19_accumulate() {
  std::mt19937 rng(41);
  const Vec3 drone = vec3(0, 0, 1.2), c = vec3(4, 0, 1.2), n = vec3(1, 0, 0);
  FrameDetector detector;
  FrameDetection out;

  // 单帧只打中一条边：检测必须失败（高 0.02 m，远小于下限）
  auto single = makeEdgeCloud(c, n, 1.0, 1.0, 0, 400, 0.02, 0.008, rng);
  const DetectStatus st_single = detector.detect(single, drone, out);

  // 时间窗累积：四帧各亮一条边 → 拼出完整框
  double fake_t = 100.0;
  PointAccumulator::ClockFn clock = [&fake_t]() {
    fake_t += 0.1;
    return fake_t;
  };
  PointAccumulator acc(2.0, 0.05, clock);
  for (int edge = 0; edge < 4; ++edge) {
    acc.insert(makeEdgeCloud(c, n, 1.0, 1.0, edge, 400, 0.02, 0.008, rng));
  }
  const DetectStatus st_acc = detector.detect(acc.points(), drone, out);
  const double cerr = norm(out.center - c);
  report(st_single != DetectStatus::OK && st_acc == DetectStatus::OK && cerr < 0.05,
         "T19 sparse edges assembled by accumulation",
         fmt("single=%s acc=%s cerr=%.3f", detectStatusToString(st_single),
             detectStatusToString(st_acc), cerr));
}

// ---------------- 鲁棒性：平面拆分 T20–T22 ----------------

// 框后 0.15 m 贴一面更大的实心平行墙（会并成一簇，PCA 平面落在两层中间）
static void t20_parallel_wall() {
  std::mt19937 rng(53);
  const Vec3 drone = vec3(0, 0, 1.2), c = vec3(4, 0, 1.2), n = vec3(1, 0, 0);
  auto pts = makeFrameCloud(c, n, 1.0, 1.0, 400, 0.02, 0.008, rng);
  auto wall = makeBoardCloud(vec3(4.15, 0, 1.25), n, 3.0, 2.5, 3000, 0.008, rng);
  pts.insert(pts.end(), wall.begin(), wall.end());
  FrameDetector detector;
  FrameDetection out;
  const DetectStatus st = detector.detect(pts, drone, out);
  const double cerr = norm(out.center - c);
  report(st == DetectStatus::OK && cerr < 0.08, "T20 frame glued to parallel wall",
         fmt("st=%s cerr=%.3f", detectStatusToString(st), cerr));
}

// 框底共面立柱伸到地面（面内黏着结构撑爆包围盒 → 空洞连通域兜底救回）
static void t21_glued_post() {
  std::mt19937 rng(59);
  const Vec3 drone = vec3(0, 0, 1.2), c = vec3(4, 0, 1.2), n = vec3(1, 0, 0);
  auto pts = makeFrameCloud(c, n, 1.0, 1.0, 300, 0.02, 0.008, rng);
  // 立柱：与框共面（x≈4），y ∈ [-0.06,0.06]，z ∈ [0, 0.75]（上端接框底边）
  std::uniform_real_distribution<double> uni(-0.5, 0.5);
  std::normal_distribution<double> gauss(0.0, 0.008);
  for (int i = 0; i < 2000; ++i) {
    const Vec3 p = c + vec3(gauss(rng), uni(rng) * 0.12, -1.2 + uni(rng) * 1.5);
    pts.push_back(p[0]);
    pts.push_back(p[1]);
    pts.push_back(p[2]);
  }
  FrameDetector detector;
  FrameDetection out;
  const DetectStatus st = detector.detect(pts, drone, out);
  const double cerr = norm(out.center - c);
  const double size = 0.5 * (out.width + out.height);
  report(st == DetectStatus::OK && cerr < 0.08 && std::fabs(size - 1.0) < 0.15,
         "T21 frame with in-plane post", fmt("st=%s cerr=%.3f size=%.3f",
                                             detectStatusToString(st), cerr, size));
}

// 共面墙开窗：0.6 m 厚边带的窗洞——"厚壁孔"不是框，应被边框厚度约束拒绝
static void t23_wall_window() {
  std::mt19937 rng(67);
  const Vec3 drone = vec3(0, 0, 1.2), c = vec3(4, 0, 1.2), n = vec3(1, 0, 0);
  const Vec3 u = anyPerpendicular(normalized(n));
  const Vec3 v = normalized(cross(normalized(n), u));
  std::uniform_real_distribution<double> uni(-0.5, 0.5);
  std::normal_distribution<double> gauss(0.0, 0.008);
  std::vector<double> pts;
  for (int i = 0; i < 4000; ++i) {
    const double su = uni(rng) * 2.0, sv = uni(rng) * 2.0;
    if (std::fabs(su) < 0.4 && std::fabs(sv) < 0.4) continue;  // 0.8×0.8 窗洞，边带 0.6
    const Vec3 p = c + u * su + v * sv + normalized(n) * gauss(rng);
    pts.push_back(p[0]);
    pts.push_back(p[1]);
    pts.push_back(p[2]);
  }
  FrameDetector detector;
  FrameDetection out;
  const DetectStatus st = detector.detect(pts, drone, out);
  report(st != DetectStatus::OK, "T23 thick-wall window rejected",
         fmt("st=%s", detectStatusToString(st)));
}

// 三角形围合空洞：有界但非矩形/圆形（矩形度≈0.5），不是框
static void t24_triangle_enclosure() {
  std::mt19937 rng(71);
  const Vec3 drone = vec3(0, 0, 1.2), c = vec3(4, 0, 1.2), n = vec3(1, 0, 0);
  const Vec3 nrm = normalized(n);
  const Vec3 u = anyPerpendicular(nrm);
  const Vec3 v = normalized(cross(nrm, u));
  std::uniform_real_distribution<double> uni(0.0, 1.0);
  std::normal_distribution<double> gauss(0.0, 0.008);
  std::vector<double> pts;
  // 三条边围出直角三角形（两直角边 + 斜边），边长 1 m
  for (int i = 0; i < 3; ++i) {
    for (int k = 0; k < 1200; ++k) {
      const double t = uni(rng);
      double su, sv;
      if (i == 0) { su = -0.5; sv = -0.5 + t; }         // 左竖边
      else if (i == 1) { su = -0.5 + t; sv = -0.5; }    // 下横边
      else { su = -0.5 + t; sv = 0.5 - t; }             // 斜边
      const Vec3 p = c + u * su + v * sv + nrm * gauss(rng);
      pts.push_back(p[0]);
      pts.push_back(p[1]);
      pts.push_back(p[2]);
    }
  }
  FrameDetector detector;
  FrameDetection out;
  const DetectStatus st = detector.detect(pts, drone, out);
  report(st != DetectStatus::OK, "T24 triangle enclosure rejected",
         fmt("st=%s", detectStatusToString(st)));
}

// 失败原因上报"走得最远"的候选：最大簇（长条）挂尺寸，较小簇（实心板）挂空洞 → 应报 HOLE_FILLED
static void t22_deepest_failure() {
  std::mt19937 rng(61);
  const Vec3 drone = vec3(0, 0, 1.2), n = vec3(1, 0, 0);
  // 长条（更大簇）：4×0.3 竖直条 → TOO_SMALL_OR_LARGE
  auto beam = makeBoardCloud(vec3(4, -2.0, 1.2), n, 0.3, 4.0, 2500, 0.008, rng);
  // 实心小板（较小簇）：0.6×0.6 → HOLE_FILLED（走得更远）
  auto board = makeBoardCloud(vec3(4, 1.5, 1.2), n, 0.6, 0.6, 800, 0.008, rng);
  beam.insert(beam.end(), board.begin(), board.end());
  FrameDetector detector;
  FrameDetection out;
  const DetectStatus st = detector.detect(beam, drone, out);
  report(st == DetectStatus::HOLE_FILLED, "T22 deepest failure reported",
         fmt("st=%s", detectStatusToString(st)));
}

// 多柱支撑的框（用户真实结构）：框由多根柱子撑起，柱子只占周界少数方向 → 匹配度通过
static void t25_pillar_supported() {
  std::mt19937 rng(73);
  const Vec3 drone = vec3(0, 0, 1.2), c = vec3(4, 0, 1.2), n = vec3(1, 0, 0);
  auto pts = makeFrameCloud(c, n, 1.0, 1.0, 300, 0.02, 0.008, rng);
  // 5 根支撑柱：底边下方，从地面 z≈0 伸到框底 0.7，另 2 根贴侧边下段
  std::uniform_real_distribution<double> uni(-0.5, 0.5);
  std::normal_distribution<double> gauss(0.0, 0.008);
  const Vec3 nrm = normalized(n);
  const Vec3 u = anyPerpendicular(nrm);
  const Vec3 v = normalized(cross(nrm, u));
  for (int p = 0; p < 5; ++p) {
    const double py = -0.45 + 0.225 * p;
    for (int k = 0; k < 700; ++k) {
      const Vec3 q = c + u * (py + uni(rng) * 0.1) + v * (-1.2 + uni(rng) * 1.5) +
                     nrm * uni(rng) * 0.1 + vec3(gauss(rng), 0, 0);
      pts.push_back(q[0]); pts.push_back(q[1]); pts.push_back(q[2]);
    }
  }
  for (int p = 0; p < 2; ++p) {
    const double py = p == 0 ? -0.5 : 0.5;
    for (int k = 0; k < 400; ++k) {
      const Vec3 q = c + u * (py + uni(rng) * 0.1) + v * (-1.2 + uni(rng) * 0.8) +
                     nrm * uni(rng) * 0.1 + vec3(gauss(rng), 0, 0);
      pts.push_back(q[0]); pts.push_back(q[1]); pts.push_back(q[2]);
    }
  }
  FrameDetector detector;
  FrameDetection out;
  const DetectStatus st = detector.detect(pts, drone, out);
  const double cerr = norm(out.center - c);
  const double size = 0.5 * (out.width + out.height);
  report(st == DetectStatus::OK && cerr < 0.08 && std::fabs(size - 1.0) < 0.15,
         "T25 pillar-supported frame accepted", fmt("st=%s cerr=%.3f size=%.3f",
                                                    detectStatusToString(st), cerr, size));
}

// 框顶边点云整体缺失（传感器垂直 FOV/稀疏常见）：三边 + 底部双柱，洞只向上开放
static void t26_missing_top() {
  std::mt19937 rng(79);
  const Vec3 drone = vec3(0, 0, 1.2), c = vec3(4, 0, 1.2), n = vec3(1, 0, 0);
  // 只生成下/左/右三条边（drop 上边：v=-1 侧为上? 以 makeFrameCloud 的定义，vs=+1 为上）
  auto pts = makeFrameCloud(c, n, 1.0, 1.0, 300, 0.02, 0.008, rng);
  // 手工剔除上边（v≈+0.5 的水平边）：仅保留 |Δ| 判定后重新生成三边
  pts.clear();
  const Vec3 nrm = normalized(n);
  const Vec3 u = anyPerpendicular(nrm);
  const Vec3 v = normalized(cross(nrm, u));
  std::uniform_real_distribution<double> uni(-0.5, 0.5);
  std::normal_distribution<double> gauss(0.0, 0.008);
  for (int i = 0; i < 900; ++i) {  // 下边
    const Vec3 p = c + u * (uni(rng) * 1.0) + v * (-0.5 + uni(rng) * 0.02) + nrm * gauss(rng);
    pts.push_back(p[0]); pts.push_back(p[1]); pts.push_back(p[2]);
  }
  for (int side = 0; side < 2; ++side) {  // 左右边
    const double us = side == 0 ? -0.5 : 0.5;
    for (int i = 0; i < 500; ++i) {
      const Vec3 p = c + u * (us + uni(rng) * 0.02) + v * (uni(rng) * 1.0) + nrm * gauss(rng);
      pts.push_back(p[0]); pts.push_back(p[1]); pts.push_back(p[2]);
    }
  }
  for (int p = 0; p < 2; ++p) {  // 底部双柱
    const double py = p == 0 ? -0.4 : 0.4;
    for (int k = 0; k < 600; ++k) {
      const Vec3 q = c + u * (py + uni(rng) * 0.1) + v * (-1.2 + uni(rng) * 1.4) +
                     nrm * uni(rng) * 0.1 + vec3(gauss(rng), 0, 0);
      pts.push_back(q[0]); pts.push_back(q[1]); pts.push_back(q[2]);
    }
  }
  FrameDetector detector;
  FrameDetection out;
  const DetectStatus st = detector.detect(pts, drone, out);
  const double cerr = norm(out.center - c);
  const double size = 0.5 * (out.width + out.height);
  report(st == DetectStatus::OK && cerr < 0.10 && std::fabs(size - 1.0) < 0.20,
         "T26 missing top edge (top-open hole)", fmt("st=%s cerr=%.3f size=%.3f",
                                                     detectStatusToString(st), cerr, size));
}

// ---------------- 圆环开弧分支 T27–T29（2026-09-26：近距视场裁切成开弧）----------------

// T27：70% 开弧（Ø2.15，模拟近距裁切）→ RANSAC 圆拟合检出，中心准确
static void t27_open_arc_detected() {
  std::mt19937 rng(101);
  const Vec3 drone = vec3(0, 0, 1.2), c = vec3(3.5, 0, 1.2), n = vec3(1, 0, 0);
  auto pts = makeRingCloud(c, n, 2.15, 1400, 0.05, 0.008, rng, 0.7 * 2.0 * kPi);
  FrameDetector detector;
  FrameDetection out;
  const DetectStatus st = detector.detect(pts, drone, out);
  const double cerr = norm(out.center - c);
  const double size = 0.5 * (out.width + out.height);
  report(st == DetectStatus::OK && cerr < 0.10 && std::fabs(size - 2.15) < 0.25,
         "T27 70% open arc detected by circle fit",
         fmt("st=%s cerr=%.3f size=%.3f", detectStatusToString(st), cerr, size));
}

// T28：40% 短弧 → 角覆盖不足，必须拒绝（短弧可拟出任意大圆，中心不可信）
static void t28_short_arc_rejected() {
  std::mt19937 rng(102);
  const Vec3 drone = vec3(0, 0, 1.2), c = vec3(3.5, 0, 1.2), n = vec3(1, 0, 0);
  auto pts = makeRingCloud(c, n, 2.15, 900, 0.05, 0.008, rng, 0.4 * 2.0 * kPi);
  FrameDetector detector;
  FrameDetection out;
  const DetectStatus st = detector.detect(pts, drone, out);
  report(st != DetectStatus::OK, "T28 40% short arc rejected (coverage)",
         fmt("st=%s", detectStatusToString(st)));
}

// T29：2 m 直线段 → 可被拟成大圆但只占小角段/或超尺寸窗，必须拒绝（防墙缝/管道误检）
static void t29_straight_segment_rejected() {
  std::mt19937 rng(103);
  const Vec3 drone = vec3(0, 0, 1.2), c = vec3(3.5, 0, 1.2), n = vec3(1, 0, 0);
  auto pts = makeEdgeCloud(c, n, 2.0, 2.0, /*edge=*/0, 1200, 0.05, 0.008, rng);
  FrameDetector detector;
  FrameDetection out;
  const DetectStatus st = detector.detect(pts, drone, out);
  report(st != DetectStatus::OK, "T29 straight segment rejected (not a circle/frame)",
         fmt("st=%s", detectStatusToString(st)));
}

// ---------------- 骨架环检测 T35–T36（方向 A：形状无关 + 边中段缺口桥接）----------------

// T35：正六边形框（1.4 m 边长外接圆）——任何闭合细环都是框，形状/尺寸窗免疫
static void t35_hexagon_frame() {
  std::mt19937 rng(301);
  const Vec3 drone = vec3(0, 0, 1.2), c = vec3(4, 0, 1.2), n = vec3(1, 0, 0);
  const Vec3 nrm = normalized(n);
  const Vec3 u = anyPerpendicular(nrm), v = normalized(cross(nrm, u));
  std::vector<double> pts;
  std::uniform_real_distribution<double> uni(-0.5, 0.5);
  std::normal_distribution<double> g(0.0, 0.008);
  const double R = 0.7;  // 外接圆半径 → 边长 0.7，洞内切圆 0.606
  for (int e = 0; e < 6; ++e) {
    const double a0 = e * kPi / 3.0, a1 = (e + 1) * kPi / 3.0;
    for (int i = 0; i < 250; ++i) {
      const double t = uni(rng) + 0.5;
      const double a = a0 + (a1 - a0) * t;
      const Vec3 p = c + u * (std::cos(a) * (R + uni(rng) * 0.04)) +
                     v * (std::sin(a) * (R + uni(rng) * 0.04)) + nrm * g(rng);
      pts.push_back(p[0]); pts.push_back(p[1]); pts.push_back(p[2]);
    }
  }
  FrameDetector detector;
  FrameDetection out;
  const DetectStatus st = detector.detect(pts, drone, out);
  const double cerr = norm(out.center - c);
  const double size = 0.5 * (out.width + out.height);
  report(st == DetectStatus::OK && cerr < 0.10 && std::fabs(size - 1.4) < 0.3,
         "T35 hexagon frame detected (shape-agnostic)",
         fmt("st=%s cerr=%.3f size=%.3f", detectStatusToString(st), cerr, size));
}

// T36：方框一边中段 0.15 m 缺口（稀疏漏采）——骨架端点桥接闭合后检出；
//      旧路径只会四方向"整边缺失"封口，边中段缺口会渗漏
static void t36_mid_edge_gap() {
  std::mt19937 rng(302);
  const Vec3 drone = vec3(0, 0, 1.2), c = vec3(4, 0, 1.2), n = vec3(1, 0, 0);
  const Vec3 nrm = normalized(n);
  const Vec3 u = anyPerpendicular(nrm), v = normalized(cross(nrm, u));
  std::vector<double> pts;
  std::uniform_real_distribution<double> uni(-0.5, 0.5);
  std::normal_distribution<double> g(0.0, 0.008);
  auto addEdge = [&](double t0, double t1, int which) {
    for (int i = 0; i < 300; ++i) {
      const double t = t0 + (t1 - t0) * (uni(rng) + 0.5);
      Vec3 p;
      if (which < 2)  // 上/下边（沿 u）
        p = c + u * t + v * ((which == 0 ? 0.5 : -0.5) + uni(rng) * 0.02);
      else            // 左/右边（沿 v）
        p = c + u * ((which == 2 ? 0.5 : -0.5) + uni(rng) * 0.02) + v * t;
      p = p + nrm * g(rng);
      pts.push_back(p[0]); pts.push_back(p[1]); pts.push_back(p[2]);
    }
  };
  addEdge(-0.5, -0.08, 0);   // 上边：中段缺口 [-0.08, +0.08]（0.16 m 漏采）
  addEdge(0.08, 0.5, 0);
  addEdge(-0.5, 0.5, 1);     // 下边
  addEdge(-0.5, 0.5, 2);
  addEdge(-0.5, 0.5, 3);
  FrameDetector detector;
  FrameDetection out;
  const DetectStatus st = detector.detect(pts, drone, out);
  const double cerr = norm(out.center - c);
  report(st == DetectStatus::OK && cerr < 0.10,
         "T36 mid-edge gap bridged by skeleton endpoints",
         fmt("st=%s cerr=%.3f", detectStatusToString(st), cerr));
}

// ---------------- 圆环斜视角精化 T37/T38（2026-09-27：PCA 法向弧段偏置）----------------

// T37：环法向偏离视线 ~39°（斜视角）+ 80% 弧 → 中心与法向都要准
//（PCA 平面在弧段覆盖不对称时法向有偏，Kåsa 对开弧中心拉偏——精化前中心/法向漂移）
static void t37_oblique_ring() {
  std::mt19937 rng(20260927u);
  const Vec3 drone = vec3(0, 0, 1.2), c = vec3(4, 0, 1.2);
  const Vec3 n = normalized(vec3(1, 0, 0.8));  // 视线偏离环法向 ~38.7°
  auto pts = makeRingCloud(c, n, 2.15, 1600, 0.05, 0.008, rng, 0.8 * 2.0 * kPi);
  FrameDetector detector;
  FrameDetection out;
  const DetectStatus st = detector.detect(pts, drone, out);
  const double cerr = norm(out.center - c);
  const double ang = angleBetween(out.normal, n) * 180.0 / kPi;
  report(st == DetectStatus::OK && cerr < 0.08 && ang < 8.0,
         "T37 oblique ring center+normal refined",
         fmt("st=%s cerr=%.3f ang=%.1fdeg", detectStatusToString(st), cerr, ang));
}

// T38：更强斜视角 ~51° + 70% 弧（中距视场裁切）
static void t38_steep_oblique_ring() {
  std::mt19937 rng(20260928u);
  const Vec3 drone = vec3(0, 0, 1.2), c = vec3(4, 0, 1.2);
  const Vec3 n = normalized(vec3(1, 0, 1.25));  // ~51.3°
  auto pts = makeRingCloud(c, n, 2.15, 1600, 0.05, 0.008, rng, 0.7 * 2.0 * kPi);
  FrameDetector detector;
  FrameDetection out;
  const DetectStatus st = detector.detect(pts, drone, out);
  const double cerr = norm(out.center - c);
  const double ang = angleBetween(out.normal, n) * 180.0 / kPi;
  report(st == DetectStatus::OK && cerr < 0.12 && ang < 10.0,
         "T38 steep oblique ring refined",
         fmt("st=%s cerr=%.3f ang=%.1fdeg", detectStatusToString(st), cerr, ang));
}

// T39：斜视角短弧 + 偏置密度（用户报"歪圆框约束不精准"的最小复现）：
// 51° 视角 + 55% 弧 + 一侧密一侧疏——骨架洞包围盒畸变（w=1.30/h=2.15）致中心
// 偏 0.45 m（精化前实测）；圆约束精化后应 <0.05 m。自由拟合即达标
static void t39_skewed_short_arc_refined() {
  std::mt19937 rng(7);
  const Vec3 drone = vec3(0, 0, 1.2), c = vec3(4, 0, 1.2);
  const Vec3 n = normalized(vec3(1, 0, 1.25));
  const Vec3 u = anyPerpendicular(n), v = normalized(cross(n, u));
  std::uniform_real_distribution<double> uni(0.0, 1.0), j(-0.5, 0.5);
  std::normal_distribution<double> g(0.0, 0.03), go(0.0, 0.25);
  std::vector<double> pts;
  for (int i = 0; i < 500; ++i) {
    const double th = (std::pow(uni(rng), 2.0) - 0.5) * 0.55 * 2.0 * kPi;
    const double r = 2.15 / 2.0 + j(rng) * 0.05;
    const Vec3 p = c + u * (std::cos(th) * r) + v * (std::sin(th) * r) + n * g(rng);
    pts.insert(pts.end(), {p[0], p[1], p[2]});
  }
  for (int i = 0; i < 60; ++i) {
    const Vec3 p = c + u * (uni(rng) * 2.5 - 1.25) + v * (uni(rng) * 2.5 - 1.25) + n * go(rng);
    pts.insert(pts.end(), {p[0], p[1], p[2]});
  }
  FrameDetector detector;
  FrameDetection out;
  const DetectStatus st = detector.detect(pts, drone, out);
  const double cerr = norm(out.center - c);
  const double ang = angleBetween(out.normal, n) * 180.0 / kPi;
  report(st == DetectStatus::OK && cerr < 0.05 && ang < 5.0,
         "T39 skewed short-arc ring refined (was 0.45 m)",
         fmt("st=%s cerr=%.3f ang=%.1f", detectStatusToString(st), cerr, ang));
}


// ---- 预测模型（Holt 阻尼趋势，2026-09-28）----

static void t40_prediction_convergence() {
  // 收敛序列：真实框心固定在 c_true，锁定后观测从偏差位置逐帧收敛
  //（模拟锁定初期斜视角拟合的中心滑动）。趋势预测应比纯 EMA 提前贴到真值，
  // 且三航点保持组刚性（P1/P3 与 c 距离 = approach/exit）
  const Vec3 c_true = {{4.0, 1.0, 1.5}};
  const Vec3 n = {{1, 0, 0}};
  StabilizerParams sp_ema;                       // beta=0 纯 EMA 基线
  StabilizerParams sp_holt = sp_ema;
  sp_holt.trend_beta = 0.3;
  sp_holt.trend_damping = 0.85;
  FrameStabilizer s_ema(sp_ema), s_holt(sp_holt);
  Vec3 c_est = c_true + vec3(0, 0, -1.0);        // 初始偏 1 m
  for (int i = 0; i < 8; ++i) {
    const FrameDetection d = makeDet(c_est, n, 1.0, 1.0);
    s_ema.onDetection(d);
    s_holt.onDetection(d);
    c_est = c_true + (c_est - c_true) * 0.6;     // 观测每帧向真值收敛
  }
  const double e_ema = norm(s_ema.frame().center - c_true);
  const double e_holt = norm(s_holt.frame().center - c_true);
  report(s_holt.locked() && e_holt < e_ema,
         "T40 holt converges faster than ema",
         fmt("e_ema=%.3f e_holt=%.3f", e_ema, e_holt));
  const auto wps = s_holt.waypoints();
  const FrameDetection& f = s_holt.frame();
  const bool rigid = norm(wps[0] - (f.center - f.normal * sp_holt.approach_dist)) < 1e-9 &&
                     norm(wps[1] - f.center) < 1e-9 &&
                     norm(wps[2] - (f.center + f.normal * sp_holt.exit_dist)) < 1e-9;
  report(rigid, "T40b waypoints stay group-rigid under prediction", "");
}

static void t41_prediction_degenerates_and_residual() {
  // beta=0 分支与旧 EMA 逐位一致；beta>0 时一步预测残差可获取且收敛后趋小
  const Vec3 c = {{2.0, 0.0, 1.0}};
  const Vec3 n = {{0, 1, 0}};
  FrameStabilizer s_ref, s_off;                  // 两者都 beta=0
  for (int i = 0; i < 5; ++i) {
    s_ref.onDetection(makeDet(c, n, 1.0, 1.0));
    s_off.onDetection(makeDet(c, n, 1.0, 1.0));
  }
  s_ref.onDetection(makeDet(c + vec3(0.1, 0, 0), n, 1.0, 1.0));
  s_off.onDetection(makeDet(c + vec3(0.1, 0, 0), n, 1.0, 1.0));
  const bool identical = norm(s_ref.frame().center - s_off.frame().center) < 1e-12;
  report(identical, "T41 beta=0 identical to legacy ema", "");

  StabilizerParams sp = s_off.params();
  sp.trend_beta = 0.3;
  FrameStabilizer s_h(sp);
  for (int i = 0; i < 5; ++i) s_h.onDetection(makeDet(c, n, 1.0, 1.0));  // 锁定（SEARCHING 分支无预测）
  s_h.onDetection(makeDet(c + vec3(0.2, 0, 0), n, 1.0, 1.0));            // 锁后首帧：一步预测残差生效
  const double r0 = s_h.lastPredResidual();
  for (int i = 0; i < 10; ++i) s_h.onDetection(makeDet(c, n, 1.0, 1.0));  // 静止真值
  const double r1 = s_h.lastPredResidual();
  report(r0 >= 0.0 && r0 < 0.3 && r1 < 0.02,
         "T41b pred residual available and shrinks on steady target",
         fmt("r0=%.3f r1=%.4f", r0, r1));
}

int main() {
  t1_ideal();
  t2_noise_outliers();
  t3_tilted();
  t4_missing_edge();
  t5_board();
  t6_nonsquare();
  t6b_rectangle_accepted();
  t7_small();
  t8_wall();
  t9_oblique();
  t10_lock();
  t11_waypoints();
  t12_unlock();
  t13_false_detection();
  t14_jitter();
  t15_ambiguous();
  t16_timeout();
  t17_freeze();
  t18_circle();
  t19_accumulate();
  t20_parallel_wall();
  t21_glued_post();
  t22_deepest_failure();
  t23_wall_window();
  t24_triangle_enclosure();
  t25_pillar_supported();
  t26_missing_top();
  t27_open_arc_detected();
  t28_short_arc_rejected();
  t29_straight_segment_rejected();
  t35_hexagon_frame();
  t36_mid_edge_gap();
  t37_oblique_ring();
  t38_steep_oblique_ring();
  t39_skewed_short_arc_refined();
  t40_prediction_convergence();
  t41_prediction_degenerates_and_residual();
  std::printf("\n%d passed, %d failed\n", g_pass, g_fail);
  return g_fail == 0 ? 0 : 1;
}
