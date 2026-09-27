#!/usr/bin/env python3
# mission_state —— 
#
# 架构：显式 MissionState 枚举 + 状态处理函数表（HANDLERS）+ 进入钩子（on_enter）。
#   · 回调只收数据（检测状态/里程计/飞控状态），不改状态——状态迁移只发生在
#     spin 循环里由当前状态的处理函数决定，消除了 if/elif 链里的隐式迁移；
#   · 每个状态一个 _handle_<STATE>()，返回下一状态或 None（保持）；
#   · 迁移统一走 transition()：进入钩子 + ~state 发布 + 日志，可观测。
#
# 状态图（对齐《无人机竞速赛-技术说明》§5 race_manager 第一阶段子集）：
#   IDLE ──起飞──> RACING ──done≥target──┬─(return_home)─> RECALL ──回原点悬停，人工接管降落
#        └──────────────其余任意态 OFFBOARD 丢失───────────────> ABORTED      │
#   LAND_NOW ──已落地──> DONE（一次性汇总）<──────────────────────────────────┘
#   超时/无进展保护（默认禁用）与 se3 已在降落（围栏/断流/人工切 AUTO.LAND）
#   从 RACING/RECALL 强制进入 LAND_NOW。
#
# 输入：/window_detector/status、/path_manager/frame_count（框心穿越计数）、
#       /window_detector/stable_frame、/flight_state(se3)、
#       /mavros/state、/mavros/local_position/odom
# 输出：~state(std_msgs/String)、/path_manager/recall(回溯触发,携带 home 原点)
# 服务：/land(std_srvs/SetBool，缺省回退 mavros set_mode AUTO.LAND)

import math
from enum import Enum

import rospy
from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import State
from nav_msgs.msg import Odometry
from std_msgs.msg import String, Int8, Int32
from std_srvs.srv import SetBool, Trigger

# se3_hof_ctrl.h FlightState 枚举
FS_TAKEOFF = 2
FS_MISSION = 3
FS_LANDING = 4
FS_LANDED = 5
FS_EMERGENCY = 6


def dist2d(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def dist3d(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


class MissionState(Enum):
    TAKEOFF = 'TAKEOFF'            # 等待起飞
    RACING = 'RACING'        # 穿框竞赛
    RECALL = 'RECALL'        # 回溯：反转存储的注视航点倒数框返回起飞原点
    LAND_NOW = 'LAND_NOW'    # 触发降落并等待落地
    DONE = 'DONE'            # 已落地，一次性汇总
    ABORTED = 'ABORTED'      # 人工接管，停手


class MissionStateMachine:
    def __init__(self):
        self.target_frames = int(rospy.get_param('~target_frames', 9))
        self.total_frames = int(rospy.get_param('~total_frames', 9))
        # 超时降落默认取消（0=禁用）；需要恢复保护时 launch 传正值（如 420/90）
        self.mission_timeout = float(rospy.get_param('~mission_timeout', 0.0))
        self.no_progress_timeout = float(rospy.get_param('~no_progress_timeout', 0.0))
        self.return_home = bool(rospy.get_param('~return_home', True))
        # 返航目标（回溯 carrot 队尾接的点）。无自动降落：RECALL 到位后悬停，
        # 人工接管降落（OFFBOARD 断开即 ABORTED 停手；se3 急停保护不受影响）
        self.home_x = float(rospy.get_param('~home_x', 0.0))
        self.home_y = float(rospy.get_param('~home_y', 0.0))
        self.home_z = float(rospy.get_param('~home_z', 1.2))
        self.goal_topic = rospy.get_param('~goal_topic', '/move_base_simple/goal')
        # 里程计来源：仿真 mavros / 实机链 fastlio /Odometry（pos 与回溯 home 判定用）
        self.odom_topic = rospy.get_param('~odom_topic', '/mavros/local_position/odom')

        # 穿框提交（方案A §4.4）：锁定且距框进入 [commit_min_dist, commit_max_dist]
        # 时调 /window_detector/commit_crossing 冻结框参数——EGO 沿冻结法向直穿，
        # 规避 <2 m 近距检测盲区的解锁/漂移（本机实测根因）
        self.commit_enabled = bool(rospy.get_param('~commit_enabled', True))
        self.commit_min_dist = float(rospy.get_param('~commit_min_dist', 1.2))
        self.commit_max_dist = float(rospy.get_param('~commit_max_dist', 4.5))
        self.commit_retry_interval = float(rospy.get_param('~commit_retry_interval', 3.0))

        # 最小搜索行为（技术说明 §5.1 SEARCH 的降级实现）：穿越计数停滞且
        # 已失锁超过 search_nudge_after 秒时，向四周发短距探路点改变观察几何
        self.search_nudge_enabled = bool(rospy.get_param('~search_nudge_enabled', True))
        self.search_nudge_after = float(rospy.get_param('~search_nudge_after', 10.0))
        self.search_nudge_interval = float(rospy.get_param('~search_nudge_interval', 8.0))
        self.search_nudge_radius = float(rospy.get_param('~search_nudge_radius', 2.5))

        # ---- 感知输入（回调只写这里，不改状态）----
        self.done = 0
        self.done_times = []            # 每框穿越时刻（相对起飞）
        self.detector_locked = False
        self.detector_frozen = False
        self.frame_center = None
        self.frame_msg_time = None
        self.commit_frame = None        # 已提交框中心（防重复提交）
        self.commit_request_time = rospy.Time(0)
        self.last_nudge_time = rospy.Time(0)
        self.nudge_angle = 0.0
        self.flight_state = -1
        self.mode = ''
        self.armed = False
        self.pos = None

        # ---- 状态机上下文 ----
        self.state = MissionState.TAKEOFF
        self.takeoff_time = None        # 任务时钟零点（起飞/任务态首次出现）
        self.last_progress_time = None  # 最近一次穿越计数增长
        self.finish_reason = ''
        self.land_request_time = rospy.Time(0)
        self.final_summary = None       # DONE 一次性汇总标记
        self._land_service = None       # None=未探测, False=无 /land, 有值=se3 服务代理

        self.state_pub = rospy.Publisher('~state', String, queue_size=5, latch=True)
        self.recall_pub = rospy.Publisher('/path_manager/recall', PoseStamped, queue_size=1)
        self.recall_pub_time = rospy.Time(0)
        self.goal_pub = rospy.Publisher(self.goal_topic, PoseStamped, queue_size=1)
        rospy.Subscriber('/window_detector/status', String, self.status_cb, queue_size=1)
        rospy.Subscriber('/path_manager/frame_count', Int32, self.frame_count_cb, queue_size=1)
        rospy.Subscriber('/window_detector/stable_frame', PoseStamped, self.frame_cb, queue_size=1)
        rospy.Subscriber('/flight_state', Int8, self.flight_state_cb, queue_size=1)
        rospy.Subscriber('/mavros/state', State, self.mavros_state_cb, queue_size=1)
        rospy.Subscriber(self.odom_topic, Odometry, self.odom_cb, queue_size=1)

        rospy.loginfo('mission_state: target=%d/%d frames, timeout=%.0fs, return_home=%s',
                      self.target_frames, self.total_frames, self.mission_timeout, self.return_home)
        self.publish_state()

    # ---------------- 感知回调（只收数据） ----------------

    def status_cb(self, msg):
        # 格式：LOCKED|detect=...|...|frozen=N|...|done=N（window_detector publishStatus）
        self.detector_locked = msg.data.startswith('LOCKED')
        self.detector_frozen = '|frozen=1' in msg.data
        for part in msg.data.split('|'):
            if part.startswith('done='):
                try:
                    self._bump_done(int(part[len('done='):]), 'detector')
                except ValueError:
                    continue

    def frame_count_cb(self, msg):
        # path_manager 框心穿越计数（centers_ 已越过数，Int32 latch）：判定 =
        # 路径进度越过已登记框心 0.5 m，与穿越瞬间的检测器锁定状态解耦——
        # 锁丢失/幻影重锁时检测器 done 会漏计（实测 6-7/9 穿 9 不达标），
        # 由此计数补齐；与检测器计数取最大（_bump_done 单调）
        self._bump_done(msg.data, 'centers')

    def _bump_done(self, new_done, source):
        """穿越计数单调抬升（检测器事件 / 框心存储两源取最大）；source 入日志"""
        if new_done <= self.done:
            return
        now = rospy.Time.now()
        if self.takeoff_time is not None:
            t_rel = (now - self.takeoff_time).to_sec()
            self.done_times.append(t_rel)
            rospy.loginfo('mission_state: frame %d/%d TRAVERSED at t=%.1fs (%s)',
                          new_done, self.target_frames, t_rel, source)
        else:
            rospy.logwarn('mission_state: done=%d but takeoff not seen yet (%s)', new_done, source)
        self.done = new_done
        self.last_progress_time = now
        self.commit_frame = None  # 穿越完成，允许提交下一个框

    def frame_cb(self, msg):
        # stable_frame：位置=锁定框中心（latch，解锁时发布空消息）
        p = msg.pose.position
        if not (p.x == 0.0 and p.y == 0.0 and p.z == 0.0):
            self.frame_center = (p.x, p.y, p.z)
            self.frame_msg_time = rospy.Time.now()

    def flight_state_cb(self, msg):
        self.flight_state = msg.data
        # 任务节点晚于 se3 启动（如中途重启）时，看到 MISSION 也要起表：
        # 时钟只用于超时保护，晚起表只会让保护更宽松，不会误杀
        if self.takeoff_time is None and self.flight_state in (FS_TAKEOFF, FS_MISSION):
            self.takeoff_time = rospy.Time.now()
            self.last_progress_time = self.takeoff_time
            rospy.loginfo('mission_state: flight active (fs=%d), mission clock started', self.flight_state)

    def mavros_state_cb(self, msg):
        self.mode = msg.mode
        self.armed = msg.armed

    def odom_cb(self, msg):
        p = msg.pose.pose.position
        self.pos = (p.x, p.y, p.z)

    # ---------------- 状态机框架 ----------------

    def publish_state(self):
        s = String()
        s.data = '%s|done=%d|target=%d' % (self.state.value, self.done, self.target_frames)
        self.state_pub.publish(s)

    def transition(self, new_state):
        if new_state is None or new_state == self.state:
            return
        rospy.loginfo('mission_state: %s -> %s (%s)', self.state.value, new_state.value,
                      self.finish_reason or '-')
        self.state = new_state
        self.publish_state()

    def finish(self, to_state):
        """收尾入口。RECALL=回溯返航（return_home 开启时把 home 发给
        path_manager 反转存储航点原路返回，到位后悬停等人工接管）；否则
        LAND_NOW 仅由 se3 急停跟随路径进入（任务侧无自动降落）"""
        if self.state in (MissionState.RECALL, MissionState.LAND_NOW, MissionState.DONE, MissionState.ABORTED):
            return None
        if to_state == MissionState.RECALL:
            return MissionState.RECALL
        return MissionState.LAND_NOW

    # ---------------- 各状态处理（返回下一状态或 None 保持） ----------------

    def _handle_IDLE(self):
        if self.flying():
            if self.takeoff_time is None:
                self.takeoff_time = rospy.Time.now()
                self.last_progress_time = self.takeoff_time
                rospy.loginfo('mission_state: flight active, mission clock started')
            return MissionState.RACING
        return None

    def _handle_RACING(self):
        # 遥控/地面站接管：OFFBOARD 丢失即停手（不替人做降落决策）
        if self.mode not in ('OFFBOARD', 'AUTO.LAND'):
            self.finish_reason = 'manual takeover (%s)' % self.mode
            return MissionState.ABORTED
        # 距框进入提交窗 → 冻结框参数直穿（每周期检查，内部限频）
        self.try_commit()
        # 穿越计数停滞且失锁 → 探路（内部有冻结/锁定保护与限频）
        self.try_search_nudge()
        if self.done >= self.target_frames:
            self.finish_reason = 'target reached (%d frames)' % self.done
            if self.return_home:
                return self.finish(MissionState.RECALL)
            return MissionState.LAND_NOW
        if self.takeoff_time is not None:
            # 超时保护可整体关闭（参数 <=0 即禁用，用户 2026-09-26 取消超时降落）
            elapsed = (rospy.Time.now() - self.takeoff_time).to_sec()
            if self.mission_timeout > 0.0 and elapsed > self.mission_timeout:
                self.finish_reason = 'mission timeout %.0fs' % self.mission_timeout
                return MissionState.LAND_NOW
            if (self.no_progress_timeout > 0.0 and self.last_progress_time is not None and
                    (rospy.Time.now() - self.last_progress_time).to_sec() > self.no_progress_timeout):
                self.finish_reason = 'no progress for %.0fs' % self.no_progress_timeout
                return MissionState.LAND_NOW
        return None

    def _handle_RECALL(self):
        # 周期重发回溯触发（path_manager 侧幂等，重复触发安全）：携带 home，
        # path_manager 反转飞行过程中存储的注视航点，carrot 倒数框原路返回。
        # 无自动降落：到位后悬停，遥控器切 POSITION 即接管（PX4 闭环，任务
        # 节点状态不再影响飞行）
        now = rospy.Time.now()
        if (now - self.recall_pub_time).to_sec() > 2.0:
            self.recall_pub_time = now
            self.recall_pub.publish(self.home_pose())
        return None

    def _handle_LAND_NOW(self):
        # se3 收到 /land 后会切 AUTO.LAND；2s 重试直到看到 LANDING
        if (self.flight_state not in (FS_LANDING, FS_LANDED) and self.mode != 'AUTO.LAND'
                and (rospy.Time.now() - self.land_request_time).to_sec() > 2.0):
            self.land_request_time = rospy.Time.now()
            self.call_land()
        if self.landed():
            return MissionState.DONE
        return None

    def _handle_DONE(self):
        if self.final_summary is None:
            self.log_summary()
        return None

    def _handle_ABORTED(self):
        return None  # 停手，交还操控权

    HANDLERS = {
        MissionState.TAKEOFF: _handle_IDLE,
        MissionState.RACING: _handle_RACING,
        MissionState.RECALL: _handle_RECALL,
        MissionState.LAND_NOW: _handle_LAND_NOW,
        MissionState.DONE: _handle_DONE,
        MissionState.ABORTED: _handle_ABORTED,
    }

    # ---------------- 行为原语 ----------------

    def try_commit(self):
        """锁定稳定 + 距框进入提交窗 → 冻结框参数直穿（方案A §4.4）"""
        if not self.commit_enabled or self.state != MissionState.RACING:
            return
        if not self.detector_locked or self.pos is None or self.frame_center is None:
            return
        if self.frame_msg_time is None or (rospy.Time.now() - self.frame_msg_time).to_sec() > 1.0:
            return  # stable_frame 太旧，可能已解锁
        # 已提交过这个框（距提交中心 <1.5 m 视为同一框），穿越完成后才允许提交下一个
        if self.commit_frame is not None and dist2d(self.commit_frame, self.frame_center) < 1.5:
            return
        d = dist3d(self.pos, self.frame_center)
        if not (self.commit_min_dist <= d <= self.commit_max_dist):
            return
        if (rospy.Time.now() - self.commit_request_time).to_sec() < self.commit_retry_interval:
            return
        self.commit_request_time = rospy.Time.now()
        try:
            rospy.wait_for_service('/window_detector/commit_crossing', timeout=1.0)
            srv = rospy.ServiceProxy('/window_detector/commit_crossing', Trigger)
            resp = srv()
            if resp.success:
                self.commit_frame = self.frame_center
                rospy.loginfo('mission_state: COMMIT crossing at dist %.2f m, center (%.2f, %.2f, %.2f)',
                              d, *self.frame_center)
            else:
                rospy.logwarn('mission_state: commit rejected: %s', resp.message)
        except (rospy.ServiceException, rospy.ROSException) as e:
            rospy.logwarn('mission_state: commit_crossing call failed: %s', e)

    def try_search_nudge(self):
        """穿越计数停滞且失锁 → 发短距探路点改变观察几何（侧视方框是窄线、近环
        点云稀疏，悬停等不来有效锁定；锁定抖动会不停重置"无锁计时"，故以 done
        停滞为触发）。探点绕当前位置 60° 步进旋转，高度夹在 [1.2, 2.6] m。"""
        if not self.search_nudge_enabled or self.pos is None or self.takeoff_time is None:
            return
        if self.detector_frozen:
            return  # 冻结穿越进行中，不干扰直穿
        if self.detector_locked:
            return  # 锁定跟飞中不发探路点：goal 会整体替换 path_manager 的
            # 前视路径任务（[当前目标, 下一框穿出点]），把队列尾巴打掉
        now = rospy.Time.now()
        base = self.last_progress_time or self.takeoff_time
        if (now - base).to_sec() < self.search_nudge_after:
            return
        if (now - self.last_nudge_time).to_sec() < self.search_nudge_interval:
            return
        self.last_nudge_time = now
        gx = self.pos[0] + self.search_nudge_radius * math.cos(self.nudge_angle)
        gy = self.pos[1] + self.search_nudge_radius * math.sin(self.nudge_angle)
        gz = min(max(self.pos[2], 1.2), 2.6)
        self.nudge_angle += math.radians(60.0)
        goal = PoseStamped()
        goal.header.stamp = now
        goal.header.frame_id = 'world'
        goal.pose.position.x = gx
        goal.pose.position.y = gy
        goal.pose.position.z = gz
        goal.pose.orientation.w = 1.0
        self.goal_pub.publish(goal)
        rospy.loginfo('mission_state: SEARCH nudge -> (%.2f, %.2f, %.2f)', gx, gy, gz)

    def home_pose(self):
        goal = PoseStamped()
        goal.header.stamp = rospy.Time.now()
        goal.header.frame_id = 'world'
        goal.pose.position.x = self.home_x
        goal.pose.position.y = self.home_y
        goal.pose.position.z = self.home_z
        goal.pose.orientation.w = 1.0
        return goal

    def call_land(self):

        if self._land_service is None:
            try:
                rospy.wait_for_service('/land', timeout=0.5)
                self._land_service = rospy.ServiceProxy('/land', SetBool)
            except rospy.ROSException:
                self._land_service = False
        if self._land_service:
            try:
                resp = self._land_service(True)
                rospy.loginfo('mission_state: /land called, success=%s', resp.success)
                return
            except rospy.ServiceException as e:
                rospy.logwarn('mission_state: /land call failed: %s', e)
        try:
            from mavros_msgs.srv import SetMode
            rospy.wait_for_service('/mavros/set_mode', timeout=1.0)
            srv = rospy.ServiceProxy('/mavros/set_mode', SetMode)
            resp = srv(custom_mode='AUTO.LAND')
            rospy.loginfo('mission_state: mavros AUTO.LAND sent, mode_sent=%s', resp.mode_sent)
        except (rospy.ServiceException, rospy.ROSException) as e:
            rospy.logwarn('mission_state: set_mode AUTO.LAND failed: %s', e)

    def log_summary(self):
        total = ''
        if self.takeoff_time is not None:
            total = ', total %.1f s' % (rospy.Time.now() - self.takeoff_time).to_sec()
        crossings = ', '.join('%.1fs' % t for t in self.done_times) or '-'
        self.final_summary = ('STAGE1 DONE: %d/%d frames traversed [%s]%s, reason=%s'
                              % (self.done, self.target_frames, crossings, total, self.finish_reason))
        rospy.loginfo('mission_state: %s', self.final_summary)
        s = String()
        s.data = self.final_summary
        self.state_pub.publish(s)

    # ---------------- 飞行状态推断 ----------------

    def flying(self):
        """飞行中判定：se3 链看 /flight_state；无 se3（ctrl_v1 链）按 mavros 状态推断"""
        if self.flight_state in (FS_TAKEOFF, FS_MISSION):
            return True
        return (self.armed and self.mode == 'OFFBOARD' and self.pos is not None
                and self.pos[2] > 0.5)

    def landed(self):
        return self.flight_state == FS_LANDED or (not self.armed and self.pos is not None
                                                  and self.pos[2] < 0.15)

    # ---------------- 主循环 ----------------

    def spin(self):
        rate = rospy.Rate(5)
        while not rospy.is_shutdown():
            # se3/飞控已进入收尾（围栏、传感器断流、人工切 AUTO.LAND）：任务停手跟随
            if self.state not in (MissionState.LAND_NOW, MissionState.DONE, MissionState.ABORTED):
                if self.flight_state in (FS_LANDING, FS_LANDED, FS_EMERGENCY) or self.mode == 'AUTO.LAND':
                    self.finish_reason = self.finish_reason or 'se3 landing already active'
                    self.transition(MissionState.LAND_NOW)

            handler = self.HANDLERS[self.state]
            nxt = handler(self)
            if nxt is not None:
                self.transition(nxt)
            rate.sleep()


if __name__ == '__main__':
    rospy.init_node('mission_state')
    MissionStateMachine().spin()
