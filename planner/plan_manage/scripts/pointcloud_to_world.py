#!/usr/bin/env python3
# 作用：订阅雷达原始点云，用 tf2 变换到目标坐标系(默认 world)，可选距离/Z轴/体素/点数滤波后发布。
# 仿真链路：Gazebo Mid360 插件 /livox/lidar -> 本节点 -> /livox/lidar_world -> ego_planner/grid_map/cloud
# 所需 TF 链：world -> map(静态) -> base_link(mavros local_position tf) -> livox_link(静态外参)

import ctypes
import glob
import os

import rospy
import tf2_ros
import numpy as np
import tf.transformations as tft
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2 as pc2
from std_msgs.msg import Header


def _load_array_backend():
    """优先 CuPy（CUDA 加速），不可用则回退 numpy。
    nvidia-*.whl 装的 CUDA 库不在系统路径，先以 ctypes 绝对路径 RTLD_GLOBAL 预加载，
    cupy 随后 dlopen 同名库即命中，无需 LD_LIBRARY_PATH 或 launch 改动。"""
    try:
        roots = [os.path.expanduser('~/.local/lib/python3.8/site-packages/nvidia'),
                 '/usr/local/cuda/lib64']
        lib_dirs = []
        for root in roots:
            lib_dirs += sorted(glob.glob(os.path.join(root, '*', 'lib')))
        order = ['cuda_runtime', 'nvjitlink', 'cublas', 'cufft', 'curand',
                 'cusparse', 'cusolver', 'cuda_nvrtc']
        for name in order:
            for d in lib_dirs:
                if name in d:
                    for lib in sorted(glob.glob(os.path.join(d, '*.so*'))):
                        try:
                            ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
                        except OSError:
                            pass
        import cupy as cp
        _ = float(cp.arange(8).sum())  # 触发真实内核验证可用
        return cp, 'cupy(GPU)'
    except Exception as e:  # noqa: BLE001 任何失败静默回退
        rospy.logwarn('CUDA/CuPy unavailable (%s), fallback to numpy', e)
        return np, 'numpy(CPU)'


_CP, BACKEND = _load_array_backend()

# PointField.datatype → numpy 格式串（小端）
_PF_FMT = {
    1: 'i1', 2: 'u1', 3: 'i2', 4: 'u2', 5: 'i4', 6: 'u4', 7: 'f4', 8: 'f8',
}


def cloud_to_struct_array(cloud_msg):
    """PointCloud2 → numpy 结构化数组（跳过 x/y/z 含 NaN 的点）。
    Noetic 的 pc2.read_points 是纯 Python 生成器，6 万点逐点遍历数百毫秒；
    这里直接按 point_step 布局视图解析，列访问为 C 级速度。"""
    names = [f.name for f in cloud_msg.fields]
    try:
        dt = np.dtype({
            'names': names,
            'formats': [_PF_FMT[f.datatype] for f in cloud_msg.fields],
            'offsets': [f.offset for f in cloud_msg.fields],
            'itemsize': cloud_msg.point_step,
        })
    except KeyError:
        return None  # 含不支持的字段类型，调用方走旧生成器路径
    count = int(cloud_msg.width) * int(cloud_msg.height)
    if count == 0 or len(cloud_msg.data) == 0:
        return np.array([], dtype=dt)
    arr = np.frombuffer(bytes(cloud_msg.data), dtype=dt, count=count)
    if any(n in names for n in ('x', 'y', 'z')):
        ok = np.isfinite(arr['x']) & np.isfinite(arr['y']) & np.isfinite(arr['z'])
        arr = arr[ok]
    return arr


class PointCloudToWorld:
    def __init__(self):
        self.target_frame = rospy.get_param('~target_frame', 'world')
        self.source_topic = rospy.get_param('~source_topic', '/livox/lidar')
        self.output_topic = rospy.get_param('~output_topic', '/livox/lidar_world')
        self.source_frame = rospy.get_param('~source_frame', 'livox_link')
        self.lookup_timeout = rospy.get_param('~lookup_timeout', 0.1)
        self.use_latest_on_extrapolation = rospy.get_param('~use_latest_on_extrapolation', True)
        self.filter_enable = rospy.get_param('~filter_enable', True)
        self.voxel_leaf_size = rospy.get_param('~voxel_leaf_size', 0.08)
        # 孤立噪点滤除（防幽灵体素封孔卡死）：近距范围内 26 邻域无任何占据
        # 邻居的孤立体素视为配准残差/噪点丢弃——grid_map 的射线清除只清"有
        # 射线穿过"的格子，孔后方无结构时孔内幽灵点一旦入图永不消失，
        # inflation 直接封孔（2026-09-27 仿真倒数第二框卡死现场）。只在近距
        # 过滤：远距稀疏点要留给检测器累积
        self.outlier_remove = rospy.get_param('~outlier_remove', True)
        self.outlier_min_neighbors = int(rospy.get_param('~outlier_min_neighbors', 1))
        self.outlier_remove_range = float(rospy.get_param('~outlier_remove_range', 3.5))
        # 距离裁剪默认关闭，仅在 launch 显式设置 min_range/max_range 时启用
        self.min_range = rospy.get_param('~min_range', None)
        self.max_range = rospy.get_param('~max_range', None)
        self.min_z = rospy.get_param('~min_z', None)
        self.max_z = rospy.get_param('~max_z', None)
        self.max_points = rospy.get_param('~max_points', 200000)
        self.cloud_timeout = rospy.get_param('~cloud_timeout', 0.5)
        self.log_interval = rospy.get_param('~log_interval', 2.0)
        self.filter_log = rospy.get_param('~filter_log', True)
        # 位姿来源：
        #   tf          —— 按 header.stamp 查 TF（默认；依赖 mavros TF 实时性）
        #   odom_interp —— 直接订阅 odom 逐帧插值/恒速外推到点云时刻（仿真推荐：
        #                  mavros TF 常滞后于点云戳，旧版"退回最新 TF"在飞行中
        #                  每帧错位数厘米且随机，框壁被抹成多层壳=点云畸变主因）
        self.pose_source = rospy.get_param('~pose_source', 'tf')
        self.odom_topic = rospy.get_param('~odom_topic', '/mavros/local_position/odom')
        # 外推上限(s)：点云戳超前最新 odom 超过该值视为断流，退回 TF 路径
        self.max_extrapolation = rospy.get_param('~max_extrapolation', 0.3)

        self.odom_buf = []  # [(t, pos(3), quat(4)), ...] 世界系位姿，最新在尾
        if self.pose_source == 'odom_interp':
            self.odom_sub = rospy.Subscriber(
                self.odom_topic, Odometry, self._odom_cb, queue_size=100)
            rospy.loginfo('pointcloud_to_world pose_source: odom_interp(%s)', self.odom_topic)

        self.last_msg_time = None
        self.last_cloud_stamp = None
        self.tf_fail_count = 0
        self.tf_extrap_count = 0

        self.watchdog_timer = rospy.Timer(rospy.Duration(0.2), self._watchdog)

        self.buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
        self.listener = tf2_ros.TransformListener(self.buffer)

        self.pub = rospy.Publisher(self.output_topic, PointCloud2, queue_size=1)
        self.sub = rospy.Subscriber(self.source_topic, PointCloud2, self.callback, queue_size=1)
        rospy.loginfo('pointcloud_to_world backend: %s', BACKEND)

    def _lookup_transform(self, frame_id, stamp):
        """world→frame_id 位姿：odom_interp 模式下由里程计插值/外推构造，否则查 TF"""
        if self.pose_source != 'odom_interp':
            return self.buffer.lookup_transform(
                self.target_frame, frame_id, stamp, rospy.Duration(self.lookup_timeout))
        pose = self._odom_pose_at(stamp.to_sec())
        if pose is None:
            # 里程计断流/过旧：整体退回 TF 路径（含其日志计数）
            return self.buffer.lookup_transform(
                self.target_frame, frame_id, stamp, rospy.Duration(self.lookup_timeout))
        pos, quat = pose
        # base_link→livox_link 静态外参（不随时间变化，用最新 TF 取）
        tf_static = self.buffer.lookup_transform(
            'base_link', frame_id, rospy.Time(0), rospy.Duration(self.lookup_timeout))
        ts = tf_static.transform.translation
        qs = tf_static.transform.rotation
        pos_b = np.array([ts.x, ts.y, ts.z])
        q_b = np.array([qs.x, qs.y, qs.z, qs.w])
        # T_world_base · T_base_livox
        R_wb = np.array(tft.quaternion_matrix(quat)[:3, :3])
        pos = pos + R_wb @ pos_b
        quat = tft.quaternion_multiply(quat, q_b)
        out = TransformStamped()
        out.header.frame_id = self.target_frame
        out.header.stamp = stamp
        out.child_frame_id = frame_id
        out.transform.translation.x, out.transform.translation.y, out.transform.translation.z = pos
        out.transform.rotation.x, out.transform.rotation.y = quat[0], quat[1]
        out.transform.rotation.z, out.transform.rotation.w = quat[2], quat[3]
        return out

    def _odom_cb(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self.odom_buf.append((
            msg.header.stamp.to_sec(),
            np.array([p.x, p.y, p.z], dtype=np.float64),
            np.array([q.x, q.y, q.z, q.w], dtype=np.float64),
        ))
        # 保留 10 s 足够覆盖插值窗
        if len(self.odom_buf) > 400:
            self.odom_buf = self.odom_buf[-300:]

    def _odom_pose_at(self, t):
        """里程计位姿在时刻 t 的估计：区间内 lerp+slerp，超前尾部恒速外推"""
        buf = self.odom_buf
        if len(buf) < 2:
            return None
        if t < buf[0][0]:
            return None  # 早于缓冲区：交回 TF 路径
        if t <= buf[-1][0]:
            for i in range(len(buf) - 1):
                if buf[i][0] <= t <= buf[i + 1][0]:
                    t0, p0, q0 = buf[i]
                    t1, p1, q1 = buf[i + 1]
                    if t1 - t0 < 1e-6:
                        return p0, q0
                    a = (t - t0) / (t1 - t0)
                    return p0 + a * (p1 - p0), tft.quaternion_slerp(q0, q1, a)
        # 超前最新里程计：位置按差分速度外推，姿态沿用最新四元数
        # （0.1 s 级间隙内姿态变化可忽略；位置误差 v·Δt 才是畸变主项）
        t0, p0, _q0 = buf[-2]
        t1, p1, q1 = buf[-1]
        dt_gap = t - t1
        if dt_gap > self.max_extrapolation or t1 - t0 < 1e-6:
            return None
        v = (p1 - p0) / (t1 - t0)
        return p1 + v * dt_gap, q1

    def callback(self, msg):
        self.last_msg_time = rospy.Time.now()
        self.last_cloud_stamp = msg.header.stamp
        frame_id = msg.header.frame_id.strip('/') if msg.header.frame_id else self.source_frame
        # 处理 Gazebo 命名空间（例如 "Mid360::livox_link" -> "livox_link"）
        if '::' in frame_id:
            frame_id = frame_id.split('::')[-1]
        stamp = msg.header.stamp if msg.header.stamp != rospy.Time() else rospy.Time(0)

        try:
            transform = self._lookup_transform(frame_id, stamp)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException) as exc:
            self.tf_fail_count += 1
            rospy.logwarn_throttle(
                self.log_interval,
                "TF lookup failed: %s (fail_count=%d, cloud_stamp=%.3f)",
                exc, self.tf_fail_count, stamp.to_sec())
            return
        except tf2_ros.ExtrapolationException as exc:
            if not self.use_latest_on_extrapolation:
                self.tf_fail_count += 1
                rospy.logwarn_throttle(
                    self.log_interval,
                    "TF lookup failed: %s (fail_count=%d, cloud_stamp=%.3f)",
                    exc, self.tf_fail_count, stamp.to_sec())
                return
            self.tf_extrap_count += 1
            rospy.logwarn_throttle(
                self.log_interval,
                "TF extrapolation: %s (extrap_count=%d, cloud_stamp=%.3f). Falling back to latest TF.",
                exc, self.tf_extrap_count, stamp.to_sec())
            try:
                transform = self._lookup_transform(frame_id, rospy.Time(0))
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as exc2:
                self.tf_fail_count += 1
                rospy.logwarn_throttle(
                    self.log_interval,
                    "TF lookup failed: %s (fail_count=%d)",
                    exc2, self.tf_fail_count)
                return

        cloud_out = self._transform_cloud_with_fields(msg, transform)
        cloud_out.header.frame_id = self.target_frame
        if not self.filter_enable:
            self.pub.publish(cloud_out)
            return

        # 距离滤波必须以传感器位置为参考（world 系），不能用 world 原点
        t = transform.transform.translation
        sensor_pos = np.array([t.x, t.y, t.z], dtype=np.float32)
        filtered_cloud = self._filter_cloud(cloud_out, sensor_pos)
        self.pub.publish(filtered_cloud)

    def _transform_cloud_with_fields(self, cloud_msg, transform):
        """手动变换点云，保留所有字段（如 intensity）"""
        t = transform.transform.translation
        q = transform.transform.rotation
        R = np.array([
            [1-2*(q.y*q.y+q.z*q.z), 2*(q.x*q.y-q.z*q.w), 2*(q.x*q.z+q.y*q.w)],
            [2*(q.x*q.y+q.z*q.w), 1-2*(q.x*q.x+q.z*q.z), 2*(q.y*q.z-q.x*q.w)],
            [2*(q.x*q.z-q.y*q.w), 2*(q.y*q.z+q.x*q.w), 1-2*(q.x*q.x+q.y*q.y)]
        ])
        translation = np.array([t.x, t.y, t.z])

        field_names = [f.name for f in cloud_msg.fields]

        points = cloud_to_struct_array(cloud_msg)
        if points is None:
            # 字段类型不受支持：退回生成器路径（慢但通用）
            points = np.asarray(
                list(pc2.read_points(cloud_msg, field_names=field_names, skip_nans=True)),
                dtype=[(n, '<f8') for n in field_names])
        if points.size == 0:
            header = Header(frame_id=cloud_msg.header.frame_id, stamp=cloud_msg.header.stamp)
            return pc2.create_cloud(header, cloud_msg.fields, [])

        try:
            x_idx = field_names.index("x")
            y_idx = field_names.index("y")
            z_idx = field_names.index("z")
        except ValueError:
            rospy.logwarn_throttle(
                self.log_interval,
                "PointCloud2 missing x/y/z fields, skip transform. fields=%s",
                field_names)
            return cloud_msg

        # 结构化数组按列取值再堆叠（列级 C 遍历），避免逐点 Python 索引
        xyz = np.column_stack([points['x'], points['y'], points['z']]).astype(np.float32)
        if _CP is not np:
            g = _CP.asarray(xyz)
            xyz_new = (_CP.asarray(R) @ g.T).T + _CP.asarray(translation)
            xyz_new = _CP.asnumpy(xyz_new)
        else:
            xyz_new = (R @ xyz.T).T + translation

        # 除 x/y/z 外的字段原样保留；逐点重组改为按列更新结构化数组后整体 tolist
        out = points.copy()
        out['x'] = xyz_new[:, 0]
        out['y'] = xyz_new[:, 1]
        out['z'] = xyz_new[:, 2]

        header = Header(frame_id=cloud_msg.header.frame_id, stamp=cloud_msg.header.stamp)
        return pc2.create_cloud(header, cloud_msg.fields, out.tolist())

    def _watchdog(self, _event):
        if self.last_msg_time is None:
            return

        dt = (rospy.Time.now() - self.last_msg_time).to_sec()
        if dt > float(self.cloud_timeout):
            stamp = self.last_cloud_stamp.to_sec() if self.last_cloud_stamp else 0.0
            rospy.logwarn_throttle(
                self.log_interval,
                "No pointcloud received for %.3fs (last_cloud_stamp=%.3f, tf_fail=%d, tf_extrap=%d)",
                dt, stamp, self.tf_fail_count, self.tf_extrap_count)

    def _filter_cloud(self, cloud_msg, sensor_pos=None):
        pts = cloud_to_struct_array(cloud_msg)
        if pts is not None and pts.size:
            pts_np = np.column_stack([pts['x'], pts['y'], pts['z']]).astype(np.float32)
        else:
            gen = pc2.read_points(cloud_msg, field_names=("x", "y", "z"), skip_nans=True)
            pts_np = np.array(list(gen), dtype=np.float32).reshape(-1, 3)
        initial_count = pts_np.shape[0]
        points = _CP.asarray(pts_np)

        if points.size == 0:
            if self.filter_log:
                rospy.logwarn_throttle(
                    self.log_interval,
                    "Pointcloud empty after NaN removal (frame=%s)",
                    cloud_msg.header.frame_id)
            header = Header(frame_id=self.target_frame, stamp=cloud_msg.header.stamp)
            return pc2.create_cloud_xyz32(header, [])

        range_count = initial_count
        if self.min_range is not None or self.max_range is not None:
            # 点云已变换到 world 系，距离要相对传感器位置计算
            if sensor_pos is not None:
                ranges = _CP.linalg.norm(points - _CP.asarray(np.asarray(sensor_pos)), axis=1)
            else:
                ranges = _CP.linalg.norm(points, axis=1)
            range_mask = _CP.ones(points.shape[0], dtype=bool)
            if self.min_range is not None:
                range_mask &= (ranges >= float(self.min_range))
            if self.max_range is not None:
                range_mask &= (ranges <= float(self.max_range))
            points = points[range_mask]
            range_count = points.shape[0]

        if points.size == 0:
            if self.filter_log:
                rospy.logwarn_throttle(
                    self.log_interval,
                    "Pointcloud filtered out by range (initial=%d, min=%.2f, max=%.2f)",
                    initial_count,
                    float(self.min_range) if self.min_range is not None else -1.0,
                    float(self.max_range) if self.max_range is not None else -1.0)
            header = Header(frame_id=self.target_frame, stamp=cloud_msg.header.stamp)
            return pc2.create_cloud_xyz32(header, [])

        z_count = points.shape[0]
        if self.min_z is not None:
            points = points[points[:, 2] >= float(self.min_z)]
        if self.max_z is not None:
            points = points[points[:, 2] <= float(self.max_z)]
        z_count = points.shape[0]

        if points.size == 0:
            if self.filter_log:
                rospy.logwarn_throttle(
                    self.log_interval,
                    "Pointcloud filtered out by z (initial=%d, range=%d, min_z=%.2f, max_z=%.2f)",
                    initial_count,
                    range_count,
                    float(self.min_z) if self.min_z is not None else -1.0,
                    float(self.max_z) if self.max_z is not None else -1.0)
            header = Header(frame_id=self.target_frame, stamp=cloud_msg.header.stamp)
            return pc2.create_cloud_xyz32(header, [])

        sample_count = points.shape[0]
        if self.max_points and points.shape[0] > int(self.max_points):
            idx = np.random.choice(points.shape[0], int(self.max_points), replace=False)
            points = points[_CP.asarray(idx)]
            sample_count = points.shape[0]

        voxel_count = points.shape[0]
        leaf = 0.0
        if self.voxel_leaf_size and float(self.voxel_leaf_size) > 0.0:
            leaf = float(self.voxel_leaf_size)
            # int64 键体素去重（GPU 友好；坐标 ±100 m 内编码安全）
            v = _CP.floor(points / leaf).astype(_CP.int64)
            key = ((v[:, 0] + 4096) * 8192 + (v[:, 1] + 4096)) * 8192 + (v[:, 2] + 4096)
            _, unique_indices = _CP.unique(key, return_index=True)
            unique_indices = _CP.sort(unique_indices)
            points = points[unique_indices]
            voxel_count = points.shape[0]

        # ---- 孤立噪点滤除：体素域 26 邻域计数，近距孤立体素（配准残差）丢弃。
        # searchsorted 成员判定，numpy/cupy 双后端通用 ----
        outlier_count = points.shape[0]
        if self.outlier_remove and self.outlier_remove_range > 0.0 and points.shape[0] > 0:
            leaf_o = leaf if leaf > 0.0 else 0.05
            vo = _CP.floor(points / leaf_o).astype(_CP.int64)
            ko = ((vo[:, 0] + 4096) * 8192 + (vo[:, 1] + 4096)) * 8192 + (vo[:, 2] + 4096)
            if sensor_pos is not None:
                d = _CP.linalg.norm(points - _CP.asarray(np.asarray(sensor_pos, dtype=np.float32)), axis=1)
                near = d <= float(self.outlier_remove_range)
            else:
                near = _CP.ones(points.shape[0], dtype=bool)
            ks = _CP.sort(ko)
            nb = _CP.zeros(points.shape[0], dtype=_CP.int32)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        if dx == 0 and dy == 0 and dz == 0:
                            continue
                        nk = ((vo[:, 0] + dx + 4096) * 8192 + (vo[:, 1] + dy + 4096)) * 8192 + (vo[:, 2] + dz + 4096)
                        idx = _CP.clip(_CP.searchsorted(ks, nk), 0, ks.shape[0] - 1)
                        nb += (ks[idx] == nk).astype(_CP.int32)
            keep = (nb >= int(self.outlier_min_neighbors)) | (~near)
            dropped = int(points.shape[0] - _CP.asnumpy(keep).sum() if _CP is not np else points.shape[0] - keep.sum())
            if dropped > 0:
                points = points[keep]
                outlier_count = points.shape[0]
                if self.filter_log:
                    rospy.loginfo_throttle(
                        self.log_interval,
                        "outlier remove: dropped %d isolated voxels (near-only, leaf=%.2f)",
                        dropped, leaf_o)
        points = _CP.asnumpy(points) if _CP is not np else points

        if points.size == 0 and self.filter_log:
            rospy.logwarn_throttle(
                self.log_interval,
                "Pointcloud empty after filtering (initial=%d, range=%d, z=%d, sample=%d, voxel=%d, leaf=%.3f)",
                initial_count,
                range_count,
                z_count,
                sample_count,
                voxel_count,
                float(self.voxel_leaf_size) if self.voxel_leaf_size is not None else -1.0)

        header = Header(frame_id=self.target_frame, stamp=cloud_msg.header.stamp)
        return pc2.create_cloud_xyz32(header, points.tolist())


if __name__ == '__main__':
    rospy.init_node('pointcloud_to_world')
    PointCloudToWorld()
    rospy.spin()
