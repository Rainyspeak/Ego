#!/usr/bin/env python3
# mission_state —— 第一阶段穿框任务状态机（技术说明 §5 race_manager 子集）
#
# 架构：显式 MissionState 枚举 + 状态处理函数表（HANDLERS）。
#   · 感知回调只收数据不改状态；状态迁移只发生在 spin 循环里，由当前状态
#     的 _handle_<STATE>() 返回下一状态（None = 保持）；
#   · 迁移统一走 transition()（日志 + ~state 发布），可观测。
#
# 状态图（当前语义，无任务侧自动降落）：
#   TAKEOFF ──飞行中──> RACING ──done≥target──┬─(return_home)─> RECALL（回 home 悬停，
#     │                                        │                 遥控器切 POSITION 人工接管）
#     │  OFFBOARD 丢失（人工接管）             └─(不开回溯)────> LAND_NOW
#     └──────────────> ABORTED（停手）
#   LAND_NOW ──已落地──> DONE（一次性汇总）
#   另：se3 已在降落（围栏/断流/人工切 AUTO.LAND）时，任意非终态强制 LAND_NOW 跟随。
#   超时/无进展保护默认禁用（参数 ≤0）。
#
# 穿框计数（done）：双源取最大、单调递增（_bump_done）
#   · detector——/window_detector/status 的 done=N（穿越事件，要求穿越瞬间仍锁定）；
#   · centers——/path_manager/frame_count（路径进度越过已登记框心 +0.5 m，与锁定
#     状态解耦；锁丢失/幻影重锁时 detector 会漏计，此源补齐）。
#
# 输入：/window_detector/status、/path_manager/frame_count、
#       /window_detector/stable_frame、/flight_state(se3)、/mavros/state、odom(~odom_topic)
# 输出：~state(String，latch)、/path_manager/recall(PoseStamped=home，2 s 周期重发)、
#       ~goal_topic(仅搜索探路点)
# 服务：/window_detector/commit_crossing(Trigger)、/land(SetBool，缺省回退 mavros
#       set_mode AUTO.LAND)

import math
from enum import Enum

import rospy
from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import State
from nav_msgs.msg import Odometry
from std_msgs.msg import String, Int8, Int32
from std_srvs.srv import SetBool, Trigger

# se3_hof_ctrl.h FlightState 枚举值（/flight_state）
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
    TAKEOFF = 'TAKEOFF'      # 等待起飞（初始态）
    RACING = 'RACING'        # 穿框竞赛
    RECALL = 'RECALL'        # 回溯：反转存储的注视航点倒数框回 home，到位悬停
    LAND_NOW = 'LAND_NOW'    # 触发/跟随降落并等待落地（任务侧不再主动进入）
    DONE = 'DONE'            # 已落地，一次性汇总
    ABORTED = 'ABORTED'      # 人工接管，停手


class MissionStateMachine:
    def __init__(self):
        self._read_params()

        # ---- 感知输入（回调只写这里，不改状态）----
        self.done = 0
        self.done_times = []            # 每框穿越时刻（相对起飞，汇总用）
        self.detector_locked = False
        self.detector_frozen = False
        self.frame_center = None        # 最近有效 stable_frame 框心
        self.frame_msg_time = None
        self.commit_frame = None        # 已提交框中心（防同框重复提交）
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
        self.recall_pub_time = rospy.Time(0)
        self.final_summary = None       # DONE 一次性汇总标记
        self._land_service = None       # None=未探测, False=无 /land, 有值=服务代理

        # ---- ROS 接口 ----
        self.state_pub = rospy.Publisher('~state', String, queue_size=5, latch=True)
        self.recall_pub = rospy.Publisher('/path_manager/recall', PoseStamped, queue_size=1)
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

    def _read_params(self):
        """集中读参数：任务 / 返航 / 穿框提交 / 搜索探路"""
        # 任务
        self.target_frames = int(rospy.get_param('~target_frames', 9))
        self.total_frames = int(rospy.get_param('~total_frames', 9))
        # 超时保护默认取消（0=禁用）；需要恢复时 launch 传正值
        self.mission_timeout = float(rospy.get_param('~mission_timeout', 0.0))
        self.no_progress_timeout = float(rospy.get_param('~no_progress_timeout', 0.0))
        self.return_home = bool(rospy.get_param('~return_home', True))

        # 返航目标（回溯 carrot 队尾接的点）；无自动降落，到位悬停人工接管
        self.home_x = float(rospy.get_param('~home_x', 0.0))
        self.home_y = float(rospy.get_param('~home_y', 0.0))
        self.home_z = float(rospy.get_param('~home_z', 1.2))

        # 话题
        self.goal_topic = rospy.get_param('~goal_topic', '/move_base_simple/goal')
        self.odom_topic = rospy.get_param('~odom_topic', '/mavros/local_position/odom')

        # 穿框提交（方案A §4.4）：锁定且距框进入 [min,max] 时调 commit_crossing
        # 冻结框参数直穿——EGO 沿冻结法向直穿，规避 <2 m 近距盲区的解锁/漂移
        self.commit_enabled = bool(rospy.get_param('~commit_enabled', True))
        self.commit_min_dist = float(rospy.get_param('~commit_min_dist', 1.2))
        self.commit_max_dist = float(rospy.get_param('~commit_max_dist', 4.5))
        self.commit_retry_interval = float(rospy.get_param('~commit_retry_interval', 3.0))

        # 搜索探路（§5.1 SEARCH 降级实现）：计数停滞且失锁时发短距探路点
        # 改变观察几何（侧视方框是窄线、近环点云稀疏，悬停等不来锁定）
        self.search_nudge_enabled = bool(rospy.get_param('~search_nudge_enabled', True))
        self.search_nudge_after = float(rospy.get_param('~search_nudge_after', 10.0))
        self.search_nudge_interval = float(rospy.get_param('~search_nudge_interval', 8.0))
        self.search_nudge_radius = float(rospy.get_param('~search_nudge_radius', 2.5))

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
        # path_manager 框心穿越计数（Int32 latch）：路径进度越过已登记框心 0.5 m，
        # 与穿越瞬间的锁定状态解耦——detector 漏计（实测 6-7/9）时由此补齐
        self._bump_done(msg.data, 'centers')

    def _bump_done(self, new_done, source):
        """穿越计数单调抬升（detector / centers 两源取最大）；source 入日志"""
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
        # stable_frame：位置=锁定框中心（latch，解锁时发布空消息→全零坐标跳过）
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
        """任务收尾入口：RACING 达标后选 RECALL（回溯返航）或 LAND_NOW（不开
        return_home 时）。LAND_NOW 仅两种来源：此处不开回溯、或 spin 里跟随
        se3 急停降落——任务侧没有其它自动降落路径"""
        if self.state in (MissionState.RECALL, MissionState.LAND_NOW, MissionState.DONE, MissionState.ABORTED):
            return None
        if to_state == MissionState.RECALL:
            return MissionState.RECALL
        return MissionState.LAND_NOW

    # ---------------- 各状态处理（返回下一状态或 None 保持） ----------------

    def _handle_TAKEOFF(self):
        if self.flying():
            if self.takeoff_time is None:
                self.takeoff_time = rospy.Time.now()
                self.last_progress_time = self.takeoff_time
                rospy.loginfo('mission_state: flight active, mission clock started')
            return MissionState.RACING
        return None

    def _handle_RACING(self):
        if self._manual_takeover():
            self.finish_reason = 'manual takeover (%s)' % self.mode
            return MissionState.ABORTED
        self.try_commit()          # 距框进提交窗 → 冻结直穿（内部限频）
        self.try_search_nudge()    # 计数停滞且失锁 → 探路（内部限频）
        if self.done >= self.target_frames:
            self.finish_reason = 'target reached (%d frames)' % self.done
            return self.finish(MissionState.RECALL if self.return_home else MissionState.LAND_NOW)
        reason = self._timeout_reason()
        if reason:
            self.finish_reason = reason
            return MissionState.LAND_NOW
        return None

    def _handle_RECALL(self):
        # 周期重发回溯触发（path_manager 侧幂等，重复触发安全）：携带 home，
        # carrot 沿反转队列倒数框返回；到位后悬停，遥控器切 POSITION 即接管
        now = rospy.Time.now()
        if (now - self.recall_pub_time).to_sec() > 2.0:
            self.recall_pub_time = now
            self.recall_pub.publish(self.home_pose())
        return None

    def _handle_LAND_NOW(self):
        # 请求降落（2 s 重试）直到 se3 报 LANDING/LANDED 或飞控已切 AUTO.LAND
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
        MissionState.TAKEOFF: _handle_TAKEOFF,
        MissionState.RACING: _handle_RACING,
        MissionState.RECALL: _handle_RECALL,
        MissionState.LAND_NOW: _handle_LAND_NOW,
        MissionState.DONE: _handle_DONE,
        MissionState.ABORTED: _handle_ABORTED,
    }

    # ---- RACING 判定（保持 handler 主体一行一事）----

    def _manual_takeover(self):
        """OFFBOARD 丢失 = 遥控/地面站接管（不替人做降落决策）"""
        return self.mode not in ('OFFBOARD', 'AUTO.LAND')

    def _timeout_reason(self):
        """超时/无进展保护（均可参数禁用）；超时返回原因串，否则空串"""
        if self.takeoff_time is None:
            return ''
        if self.mission_timeout > 0.0 and \
                (rospy.Time.now() - self.takeoff_time).to_sec() > self.mission_timeout:
            return 'mission timeout %.0fs' % self.mission_timeout
        if self.no_progress_timeout > 0.0 and self.last_progress_time is not None and \
                (rospy.Time.now() - self.last_progress_time).to_sec() > self.no_progress_timeout:
            return 'no progress for %.0fs' % self.no_progress_timeout
        return ''

    # ---------------- 行为原语 ----------------

    def try_commit(self):
        """锁定稳定 + 距框进入 [commit_min, commit_max] → 冻结框参数直穿"""
        if not self.commit_enabled or self.state != MissionState.RACING:
            return
        if not self.detector_locked or self.pos is None or self.frame_center is None:
            return
        if self.frame_msg_time is None or (rospy.Time.now() - self.frame_msg_time).to_sec() > 1.0:
            return  # stable_frame 太旧，可能已解锁
        if self.commit_frame is not None and dist2d(self.commit_frame, self.frame_center) < 1.5:
            return  # 已提交过这个框（距提交中心 <1.5 m 视为同一框）
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
        """计数停滞且失锁 → 发短距探路点改变观察几何。探点绕当前位置 60° 步进
        旋转，高度夹在 [1.2, 2.6] m。以 done 停滞（而非无锁计时）为触发——
        锁定抖动会不停重置无锁计时"""
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
        """请求降落：优先 se3 的 /land 服务；不存在（如 ctrl_v1 链）则直接调
        mavros set_mode 切 AUTO.LAND"""
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
        """飞行中判定：se3 链看 /flight_state；无 se3（ctrl_v1 链）按 mavros 推断"""
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
            # se3/飞控已进入收尾（围栏、断流、人工切 AUTO.LAND）：任务停手跟随
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
