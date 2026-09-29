#!/usr/bin/env python3
# real_mission —— 实机链（ctrl_v1 控制器）任务状态机（planner_ctrl 自有文件，
# 不占用 mission_state.py——那是仿真 se3 链的原版，保持不动）。
# 与 mission_state.py 的差异（实机化）：
#   · 无 /flight_state、无 se3 /land 服务：飞行判定按 mavros 推断
#    （armed + OFFBOARD + 高度），降落直接 mavros set_mode AUTO.LAND；
#   · 达标先调 /forecast_searching/pause 关检测器任务开关（检测器自治，
#     不关会继续锁下一框入队，飞机停不下来）；
#   · 收尾按 return_home 开关：true→RECALL 回 home（到位悬停人工接管），
#     false→当前位悬停 goal 收口 carrot 后原地降落 AUTO.LAND（默认）。
#
# 架构：显式 MissionState 枚举 + 状态处理函数表（HANDLERS）。
#   · 感知回调只收数据不改状态；状态迁移只发生在 spin 循环里，由当前状态
#     的 _handle_<STATE>() 返回下一状态（None = 保持）；
#   · 迁移统一走 transition()（日志 + ~state 发布），可观测。
#
# 状态图：
#   TAKEOFF ──飞行中──> RACING ──done≥target──┬─(return_home=true)─> RECALL（关检测器，
#     │                                        │    回溯航点返 home，到位悬停切 POSITION 接管）
#     │  OFFBOARD 丢失（人工接管）             └─(false)─> LAND_NOW（关检测器+悬停收口，
#     └──────────────> ABORTED（停手）                    原地降落 AUTO.LAND，落地→DONE）
#   LAND_NOW（超时或飞控已切 AUTO.LAND）──已落地──> DONE（一次性汇总）
#   超时/无进展保护默认禁用（参数 ≤0）。
#
# 穿框计数（done）：双源取最大、单调递增（_bump_done）
#   · detector——/forecast_searching/status 的 done=N（穿越事件，要求穿越瞬间仍锁定）；
#   · centers——/path_manager/frame_count（路径进度越过已登记框心 +0.5 m，与锁定
#     状态解耦；锁丢失/幻影重锁时 detector 会漏计，此源补齐）。
#
# 输入：/forecast_searching/status、/path_manager/frame_count、
#       /forecast_searching/stable_frame、/mavros/state、odom(~odom_topic)
# 输出：~state(String，latch)、/path_manager/recall(PoseStamped=home，RECALL 周期重发)、
#       ~goal_topic(搜索探路点 + 达标悬停收口点)
# 服务：/forecast_searching/commit_crossing(Trigger)、/forecast_searching/pause(Trigger)

import math
from enum import Enum

import rospy
from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import State
from nav_msgs.msg import Odometry
from std_msgs.msg import String, Int32
from std_srvs.srv import Trigger


def dist2d(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def dist3d(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


class MissionState(Enum):
    TAKEOFF = 'TAKEOFF'      # 等待起飞（初始态）
    RACING = 'RACING'        # 穿框竞赛
    RECALL = 'RECALL'        # 回溯（return_home 开）：反转航点队列回 home，到位悬停
    LAND_NOW = 'LAND_NOW'    # 跟随降落并等待落地（超时、达标不开回溯或飞控已切 AUTO.LAND）
    DONE = 'DONE'            # 已落地，一次性汇总
    ABORTED = 'ABORTED'      # 人工接管，停手


class RealMissionStateMachine:
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
        self.mode = ''
        self.armed = False
        self.pos = None

        # ---- 状态机上下文 ----
        self.state = MissionState.TAKEOFF
        self.takeoff_time = None        # 任务时钟零点（起飞首次判定）
        self.last_progress_time = None  # 最近一次穿越计数增长
        self.finish_reason = ''
        self.land_request_time = rospy.Time(0)
        self.recall_pub_time = rospy.Time(0)
        self.final_summary = None       # DONE 一次性汇总标记

        # ---- ROS 接口 ----
        self.state_pub = rospy.Publisher('~state', String, queue_size=5, latch=True)
        self.recall_pub = rospy.Publisher('/path_manager/recall', PoseStamped, queue_size=1)
        self.goal_pub = rospy.Publisher(self.goal_topic, PoseStamped, queue_size=1)
        rospy.Subscriber('/forecast_searching/status', String, self.status_cb, queue_size=1)
        rospy.Subscriber('/path_manager/frame_count', Int32, self.frame_count_cb, queue_size=1)
        rospy.Subscriber('/forecast_searching/stable_frame', PoseStamped, self.frame_cb, queue_size=1)
        rospy.Subscriber('/mavros/state', State, self.mavros_state_cb, queue_size=1)
        rospy.Subscriber(self.odom_topic, Odometry, self.odom_cb, queue_size=1)

        rospy.loginfo('real_mission: target=%d/%d frames, timeout=%.0fs, return_home=%s',
                      self.target_frames, self.total_frames, self.mission_timeout, self.return_home)
        self.publish_state()

    def _read_params(self):
        """集中读参数：任务 / 收尾开关 / 穿框提交 / 搜索探路"""
        # 任务
        self.target_frames = int(rospy.get_param('~target_frames', 3))
        self.total_frames = int(rospy.get_param('~total_frames', 3))
        # 超时保护默认取消（0=禁用）；需要恢复时 launch 传正值
        self.mission_timeout = float(rospy.get_param('~mission_timeout', 0.0))
        self.no_progress_timeout = float(rospy.get_param('~no_progress_timeout', 0.0))

        # 收尾开关：true=达标后 RECALL（反转航点队列回 home，到位悬停人工接管）；
        # false=达标原地降落（悬停收口 carrot 后 AUTO.LAND）
        self.return_home = bool(rospy.get_param('~return_home', False))

        # 回溯终点（return_home=true 时 recall carrot 队尾接的点）
        self.home_x = float(rospy.get_param('~home_x', 0.0))
        self.home_y = float(rospy.get_param('~home_y', 0.0))
        self.home_z = float(rospy.get_param('~home_z', 1.0))

        # 话题
        self.goal_topic = rospy.get_param('~goal_topic', '/move_base_simple/goal')
        self.odom_topic = rospy.get_param('~odom_topic', '/Odometry')

        # 穿框提交（方案A §4.4）：锁定且距框进入 [min,max] 时调 commit_crossing
        # 冻结框参数直穿——EGO 沿冻结法向直穿，规避 <2 m 近距盲区的解锁/漂移
        self.commit_enabled = bool(rospy.get_param('~commit_enabled', True))
        self.commit_min_dist = float(rospy.get_param('~commit_min_dist', 1.2))
        self.commit_max_dist = float(rospy.get_param('~commit_max_dist', 4.5))
        self.commit_retry_interval = float(rospy.get_param('~commit_retry_interval', 3.0))

        # 搜索探路（SEARCH 降级实现）：计数停滞且失锁时发短距探路点
        # 改变观察几何（侧视方框是窄线、近环点云稀疏，悬停等不来锁定）
        self.search_nudge_enabled = bool(rospy.get_param('~search_nudge_enabled', True))
        self.search_nudge_after = float(rospy.get_param('~search_nudge_after', 10.0))
        self.search_nudge_interval = float(rospy.get_param('~search_nudge_interval', 8.0))
        self.search_nudge_radius = float(rospy.get_param('~search_nudge_radius', 2.5))

    # ---------------- 感知回调（只收数据） ----------------

    def status_cb(self, msg):
        # 格式：LOCKED|detect=...|...|frozen=N|...|done=N（forecast_searching publishStatus）
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
        # 与穿越瞬间的锁定状态解耦——detector 漏计时由此补齐
        self._bump_done(msg.data, 'centers')

    def _bump_done(self, new_done, source):
        """穿越计数单调抬升（detector / centers 两源取最大）；source 入日志"""
        if new_done <= self.done:
            return
        now = rospy.Time.now()
        if self.takeoff_time is not None:
            t_rel = (now - self.takeoff_time).to_sec()
            self.done_times.append(t_rel)
            rospy.loginfo('real_mission: frame %d/%d TRAVERSED at t=%.1fs (%s)',
                          new_done, self.target_frames, t_rel, source)
        else:
            rospy.logwarn('real_mission: done=%d but takeoff not seen yet (%s)', new_done, source)
        self.done = new_done
        self.last_progress_time = now
        self.commit_frame = None  # 穿越完成，允许提交下一个框

    def frame_cb(self, msg):
        # stable_frame：位置=锁定框中心（latch，解锁时发布空消息→全零坐标跳过）
        p = msg.pose.position
        if not (p.x == 0.0 and p.y == 0.0 and p.z == 0.0):
            self.frame_center = (p.x, p.y, p.z)
            self.frame_msg_time = rospy.Time.now()

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
        rospy.loginfo('real_mission: %s -> %s (%s)', self.state.value, new_state.value,
                      self.finish_reason or '-')
        self.state = new_state
        self.publish_state()

    # ---------------- 各状态处理（返回下一状态或 None 保持） ----------------

    def _handle_TAKEOFF(self):
        if self.flying():
            if self.takeoff_time is None:
                self.takeoff_time = rospy.Time.now()
                self.last_progress_time = self.takeoff_time
                rospy.loginfo('real_mission: flight active, mission clock started')
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
            self._pause_detector()  # 两种收尾都关检测器（停新框入队，防干扰收尾）
            if self.return_home:
                return MissionState.RECALL
            self._hover_goal()      # carrot 立即收口，随即原地降落
            return MissionState.LAND_NOW
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
        # 请求降落（2 s 重试）直到飞控已切 AUTO.LAND
        if (self.mode != 'AUTO.LAND'
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

    # ---- RACING 判定 ----

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

    def _pause_detector(self):
        """关检测器任务开关（pause 服务）：停止锁定与新框入队——检测器是自治的，
        不关的话 carrot 会被下一框继续引导飞下去。reset 服务重新打开"""
        try:
            rospy.wait_for_service('/forecast_searching/pause', timeout=1.0)
            srv = rospy.ServiceProxy('/forecast_searching/pause', Trigger)
            resp = srv()
            rospy.loginfo('real_mission: detector paused (success=%s)', resp.success)
        except (rospy.ServiceException, rospy.ROSException) as e:
            rospy.logwarn('real_mission: pause detector failed: %s', e)

    def _hover_goal(self):
        """当前位悬停 goal：整体替换 path_manager 任务，carrot 立即收口——
        降落请求重试期间（最长 2 s）飞机不再跟随旧队列飞；随即切降落模式"""
        if self.pos is None:
            return
        goal = PoseStamped()
        goal.header.stamp = rospy.Time.now()
        goal.header.frame_id = 'world'
        goal.pose.position.x = self.pos[0]
        goal.pose.position.y = self.pos[1]
        goal.pose.position.z = self.pos[2]
        goal.pose.orientation.w = 1.0
        self.goal_pub.publish(goal)
        rospy.loginfo('real_mission: hover goal -> (%.2f, %.2f, %.2f)',
                      self.pos[0], self.pos[1], self.pos[2])

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
            rospy.wait_for_service('/forecast_searching/commit_crossing', timeout=1.0)
            srv = rospy.ServiceProxy('/forecast_searching/commit_crossing', Trigger)
            resp = srv()
            if resp.success:
                self.commit_frame = self.frame_center
                rospy.loginfo('real_mission: COMMIT crossing at dist %.2f m, center (%.2f, %.2f, %.2f)',
                              d, *self.frame_center)
            else:
                rospy.logwarn('real_mission: commit rejected: %s', resp.message)
        except (rospy.ServiceException, rospy.ROSException) as e:
            rospy.logwarn('real_mission: commit_crossing call failed: %s', e)

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
        rospy.loginfo('real_mission: SEARCH nudge -> (%.2f, %.2f, %.2f)', gx, gy, gz)

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
        """请求降落（ctrl_v1 链）：直接 mavros set_mode 切 AUTO.LAND"""
        try:
            from mavros_msgs.srv import SetMode
            rospy.wait_for_service('/mavros/set_mode', timeout=1.0)
            srv = rospy.ServiceProxy('/mavros/set_mode', SetMode)
            resp = srv(custom_mode='AUTO.LAND')
            rospy.loginfo('real_mission: mavros AUTO.LAND sent, mode_sent=%s', resp.mode_sent)
        except (rospy.ServiceException, rospy.ROSException) as e:
            rospy.logwarn('real_mission: set_mode AUTO.LAND failed: %s', e)

    def log_summary(self):
        total = ''
        if self.takeoff_time is not None:
            total = ', total %.1f s' % (rospy.Time.now() - self.takeoff_time).to_sec()
        crossings = ', '.join('%.1fs' % t for t in self.done_times) or '-'
        self.final_summary = ('STAGE1 DONE: %d/%d frames traversed [%s]%s, reason=%s'
                              % (self.done, self.target_frames, crossings, total, self.finish_reason))
        rospy.loginfo('real_mission: %s', self.final_summary)
        s = String()
        s.data = self.final_summary
        self.state_pub.publish(s)

    # ---------------- 飞行状态推断 ----------------

    def flying(self):
        """飞行中判定（ctrl_v1 链）：解锁 + OFFBOARD + 高度"""
        return (self.armed and self.mode == 'OFFBOARD' and self.pos is not None
                and self.pos[2] > 0.5)

    def landed(self):
        return not self.armed and self.pos is not None and self.pos[2] < 0.15

    # ---------------- 主循环 ----------------

    def spin(self):
        rate = rospy.Rate(5)
        while not rospy.is_shutdown():
            # 飞控已切 AUTO.LAND（遥控/地面站触发）：任务停手跟随
            if self.state not in (MissionState.LAND_NOW, MissionState.DONE, MissionState.ABORTED):
                if self.mode == 'AUTO.LAND':
                    self.finish_reason = self.finish_reason or 'AUTO.LAND already active'
                    self.transition(MissionState.LAND_NOW)

            handler = self.HANDLERS[self.state]
            nxt = handler(self)
            if nxt is not None:
                self.transition(nxt)
            rate.sleep()


if __name__ == '__main__':
    rospy.init_node('real_mission')
    RealMissionStateMachine().spin()
