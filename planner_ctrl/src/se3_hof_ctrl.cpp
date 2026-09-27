// 移植自 OpenDrone(Tfly6, Gen3) se3_hopf/src/se3_ctrl.cpp
// 话题接口：订阅 odom_topic(参数，默认 /mavros/local_position/odom；实机链用
//           fastlio /Odometry)、/mavros/imu/data、/mavros/state、/planner/output；
//           发布 /mavros/setpoint_raw/attitude、/flight_state；服务 /land、/mavros/set_mode、/mavros/cmd/arming
#include "planner_ctrl/se3_hof_ctrl.h"
#include "planner_ctrl/planner_output_utils.h"

Se3HofCtrl::Se3HofCtrl(const ros::NodeHandle &nh, const ros::NodeHandle &private_nh)
    : nh_(nh), private_nh_(private_nh), dynamic_tune_server_(private_nh)
{
    cmd_pub_ = nh_.advertise<mavros_msgs::AttitudeTarget>("/mavros/setpoint_raw/attitude", 10);
    local_pos_pub_ = nh_.advertise<geometry_msgs::PoseStamped>("/mavros/setpoint_position/local", 10);

    set_mode_client_ = nh_.serviceClient<mavros_msgs::SetMode>("/mavros/set_mode");
    arming_client_ = nh_.serviceClient<mavros_msgs::CommandBool>("/mavros/cmd/arming");
    land_service_ = nh_.advertiseService("/land", &Se3HofCtrl::landCallback, this);

    // 里程计来源参数化（订阅前先读）：仿真 mavros / 实机链 fastlio /Odometry
    private_nh_.param<std::string>("odom_topic", odom_topic_, "/mavros/local_position/odom");
    odom_sub_ = nh_.subscribe<nav_msgs::Odometry>(odom_topic_, 10, &Se3HofCtrl::OdomCallback, this);
    imu_sub_ = nh_.subscribe<sensor_msgs::Imu>("/mavros/imu/data", 10, &Se3HofCtrl::IMUCallback, this);
    state_sub_ = nh_.subscribe<mavros_msgs::State>("/mavros/state", 10, &Se3HofCtrl::StateCallback, this);
    plannerOutput_sub_ = nh_.subscribe<planner_ctrl::PlannerOutput>("/planner/output", 10, &Se3HofCtrl::plannerOutputCallback, this);

    exec_timer_ = nh_.createTimer(ros::Duration(0.01), &Se3HofCtrl::execFSMCallback, this);

    flight_state_pub_ = nh_.advertise<std_msgs::Int8>("/flight_state", 10);

    private_nh_.param<bool>("enable_auto_offboard", enable_auto_offboard_, true);
    private_nh_.param<bool>("enable_auto_arm", enable_auto_arm_, true);
    private_nh_.param<bool>("use_dynamic_reconfigure", use_dynamic_reconfigure_, false);
    private_nh_.param<int>("offboard_warmup_count", offboard_warmup_count_, 80);
    private_nh_.param<double>("request_interval", request_interval_, 1.0);
    private_nh_.param<double>("msg_expire_time", msg_expire_time_, 0.01);
    private_nh_.param<double>("planner_timeout", planner_timeout_, 0.5);
    private_nh_.param<double>("planner_lost_land_time", planner_lost_land_time_, 0.0);
    private_nh_.param<double>("imu_timeout", imu_timeout_, 0.5);
    private_nh_.param<double>("odom_timeout", odom_timeout_, 0.5);
    private_nh_.param<double>("odom_vel_threshold", odom_vel_threshold_, 3.0);
    private_nh_.param<bool>("auto_takeoff", auto_takeoff_, true);
    private_nh_.param<double>("takeoff_height", takeoff_height_, 2.0);
    private_nh_.param<double>("geo_fence/x", geo_fence_[0], 10.0);
    private_nh_.param<double>("geo_fence/y", geo_fence_[1], 10.0);
    private_nh_.param<double>("geo_fence/z", geo_fence_[2], 4.0);

    enu_frame_ = true;
    vel_in_body_ = true;

    init_pose_ << 0, 0, 0.5;
    flightState_ = WAITING_FOR_CONNECTED;
    prev_flightState_ = flightState_;
    if (offboard_warmup_count_ < 1) {
        offboard_warmup_count_ = 1;
    }
    if (request_interval_ < 0.1) {
        request_interval_ = 0.1;
    }
    if (msg_expire_time_ < 0.001) {
        msg_expire_time_ = 0.001;
    }
    if (planner_timeout_ < 0.1) {
        planner_timeout_ = 0.1;
    }
    if (imu_timeout_ < 0.1) {
        imu_timeout_ = 0.1;
    }
    if (odom_timeout_ < 0.1) {
        odom_timeout_ = 0.1;
    }
    last_mode_request_ = ros::Time(0);
    last_arm_request_ = ros::Time(0);
    last_land_request_ = ros::Time(0);

    private_nh_.param<double>("hover_percent", hover_percent_, 0.25);
    private_nh_.param<double>("max_hover_percent", max_hover_percent_, 0.75);
    // 悬停油门用于初始化 T_a_ 推力归一化常数(=g/hover_percent)，稳态由 estimateTa 在线估计修正；
    // 值域非法时回退默认，避免 T_a_ 初值发散
    if (hover_percent_ <= 0.0 || hover_percent_ > 1.0) {
        hover_percent_ = 0.25;
    }
    if (max_hover_percent_ > 1.0 || max_hover_percent_ < hover_percent_) {
        max_hover_percent_ = 0.75;
    }

    se3_hof_.init(hover_percent_, max_hover_percent_, enu_frame_, vel_in_body_);
    if (use_dynamic_reconfigure_) {
        dynamic_tune_cb_type_ = boost::bind(&Se3HofCtrl::DynamicTuneCallback, this, _1, _2);
        dynamic_tune_server_.setCallback(dynamic_tune_cb_type_);
        ROS_INFO("se3_hof using dynamic_reconfigure for tuning parameters.");
    } else {
        loadStaticTuneConfig();
        ROS_INFO("se3_hof using static ROS parameters for tuning parameters.");
    }
}

void Se3HofCtrl::applyTuneConfig(const planner_ctrl::se3_hof_tuneConfig &config) {
    kp_p_ << config.kp_px, config.kp_py, config.kp_pz;
    kp_v_ << config.kp_vx, config.kp_vy, config.kp_vz;
    kp_a_ << config.kp_ax, config.kp_ay, config.kp_az;
    kp_q_ << config.kp_qx, config.kp_qy, config.kp_qz;
    kp_w_ << config.kp_wx, config.kp_wy, config.kp_wz;

    kd_p_ << config.kd_px, config.kd_py, config.kd_pz;
    kd_v_ << config.kd_vx, config.kd_vy, config.kd_vz;
    kd_a_ << config.kd_ax, config.kd_ay, config.kd_az;
    kd_q_ << config.kd_qx, config.kd_qy, config.kd_qz;
    kd_w_ << config.kd_wx, config.kd_wy, config.kd_wz;

    limit_err_p_ = config.limit_err_p;
    limit_err_v_ = config.limit_err_v;
    limit_err_a_ = config.limit_err_a;
    limit_d_err_p_ = config.limit_d_err_p;
    limit_d_err_v_ = config.limit_d_err_v;
    limit_d_err_a_ = config.limit_d_err_a;

    se3_hof_.setup(kp_p_, kp_v_, kp_a_, kp_q_, kp_w_,
                    kd_p_, kd_v_, kd_a_, kd_q_, kd_w_,
                    limit_err_p_, limit_err_v_, limit_err_a_,
                    limit_d_err_p_, limit_d_err_v_, limit_d_err_a_);
}

void Se3HofCtrl::loadStaticTuneConfig() {
    planner_ctrl::se3_hof_tuneConfig config;
    private_nh_.param("kp_px", config.kp_px, 0.85);
    private_nh_.param("kp_py", config.kp_py, 0.85);
    private_nh_.param("kp_pz", config.kp_pz, 1.5);
    private_nh_.param("kp_vx", config.kp_vx, 1.5);
    private_nh_.param("kp_vy", config.kp_vy, 1.5);
    private_nh_.param("kp_vz", config.kp_vz, 1.5);
    private_nh_.param("kp_ax", config.kp_ax, 1.5);
    private_nh_.param("kp_ay", config.kp_ay, 1.5);
    private_nh_.param("kp_az", config.kp_az, 1.5);
    private_nh_.param("kp_qx", config.kp_qx, 5.5);
    private_nh_.param("kp_qy", config.kp_qy, 5.5);
    private_nh_.param("kp_qz", config.kp_qz, 0.1);
    private_nh_.param("kp_wx", config.kp_wx, 1.5);
    private_nh_.param("kp_wy", config.kp_wy, 1.5);
    private_nh_.param("kp_wz", config.kp_wz, 0.1);
    private_nh_.param("kd_px", config.kd_px, 0.1);
    private_nh_.param("kd_py", config.kd_py, 0.1);
    private_nh_.param("kd_pz", config.kd_pz, 0.0);
    private_nh_.param("kd_vx", config.kd_vx, 0.0);
    private_nh_.param("kd_vy", config.kd_vy, 0.0);
    private_nh_.param("kd_vz", config.kd_vz, 0.0);
    private_nh_.param("kd_ax", config.kd_ax, 0.0);
    private_nh_.param("kd_ay", config.kd_ay, 0.0);
    private_nh_.param("kd_az", config.kd_az, 0.0);
    private_nh_.param("kd_qx", config.kd_qx, 0.0);
    private_nh_.param("kd_qy", config.kd_qy, 0.0);
    private_nh_.param("kd_qz", config.kd_qz, 0.0);
    private_nh_.param("kd_wx", config.kd_wx, 0.0);
    private_nh_.param("kd_wy", config.kd_wy, 0.0);
    private_nh_.param("kd_wz", config.kd_wz, 0.0);
    private_nh_.param("limit_err_p", config.limit_err_p, 3.0);
    private_nh_.param("limit_err_v", config.limit_err_v, 2.0);
    private_nh_.param("limit_err_a", config.limit_err_a, 1.0);
    private_nh_.param("limit_d_err_p", config.limit_d_err_p, 3.5);
    private_nh_.param("limit_d_err_v", config.limit_d_err_v, 1.0);
    private_nh_.param("limit_d_err_a", config.limit_d_err_a, 1.0);
    applyTuneConfig(config);
}


void Se3HofCtrl::execFSMCallback(const ros::TimerEvent &e){
    std_msgs::Int8 flight_state_msg;
    flight_state_msg.data = static_cast<int8_t>(flightState_);
    flight_state_pub_.publish(flight_state_msg);

    if (flightState_ != prev_flightState_) {
        ROS_WARN_STREAM("State changed from " << state2string(prev_flightState_) << " to " << state2string(flightState_));
        prev_flightState_ = flightState_;
    }

    // 规划器断流保护：收到过规划消息后，超过 planner_timeout 没有新消息，
    // 冻结当前位置悬停（继续跟踪冻结的期望速度会沿旧方向过冲甚至飞走）
    const ros::Time now = ros::Time::now();
    if (planner_msg_received_ &&
        (flightState_ == TAKEOFF || flightState_ == MISSION_EXECUTION) &&
        (now - last_planner_msg_time_).toSec() > planner_timeout_) {
        ROS_WARN_STREAM("se3_hof: planner stream lost for "
                        << (now - last_planner_msg_time_).toSec() << " s, hold at current position.");
        desired_state_.p = odom_data_.p;
        desired_state_.v.setZero();
        desired_state_.a.setZero();
        desired_state_.j.setZero();
        desired_state_.yaw = utils::fromQuaternion2yaw(odom_data_.q);
        desired_state_.yaw_rate = 0.0;
        desired_state_.q = planner_ctrl::planner_output::QuaternionFromYaw(desired_state_.yaw);
        planner_lost_enter_time_ = now;
        flightState_ = PLANNER_LOST;
    }

    // 传感器断流与定位合理性检测（超时阈值参照 px4ctrl 实机配置 msg_timeout=0.5s）：
    // IMU/odom 断流或里程计数据异常时，机载侧状态估计已不可信，本机没有遥控兜底，直接紧急降落。
    // 流"从未到达过"同样按断流处理，防止起飞阶段就带着死传感器上天
    if (flightState_ == TAKEOFF || flightState_ == MISSION_EXECUTION || flightState_ == PLANNER_LOST) {
        if (!imu_msg_received_ || (now - last_imu_msg_time_).toSec() > imu_timeout_) {
            ROS_ERROR_STREAM("se3_hof: imu stream lost for "
                             << (imu_msg_received_ ? (now - last_imu_msg_time_).toSec() : -1.0)
                             << " s, emergency land.");
            flightState_ = EMERGENCY;
        } else if (!odom_msg_received_ || (now - last_odom_msg_time_).toSec() > odom_timeout_) {
            ROS_ERROR_STREAM("se3_hof: odom stream lost for "
                             << (odom_msg_received_ ? (now - last_odom_msg_time_).toSec() : -1.0)
                             << " s, emergency land.");
            flightState_ = EMERGENCY;
        } else if (!odom_data_.p.allFinite() || !odom_data_.v.allFinite() ||
                   odom_data_.v.norm() > odom_vel_threshold_) {
            ROS_ERROR_STREAM("se3_hof: odom data fault (|v|=" << odom_data_.v.norm()
                             << " m/s), localization may be wrong, emergency land.");
            flightState_ = EMERGENCY;
        }
    }

    switch (flightState_)
    {
    case WAITING_FOR_CONNECTED:{
        ROS_INFO_ONCE("Waiting for FCU connection...");
        if(currState_.connected){
            ROS_INFO("connected!");
            offboard_warmup_counter_ = 0;
            flightState_ = WAITING_FOR_OFFBOARD;
        }
        break;
    }
    case WAITING_FOR_OFFBOARD:{
        ROS_INFO_ONCE("Waiting for OFFBOARD mode and arming...");
        Controller_Output_t init_output;
        init_output.thrust = 0.6;
        send_cmd(init_output, true); // send a zero command to initialize the offboard mode
        ++offboard_warmup_counter_;
        TrySetOffboard(now);
        TryArm(now);
        if(currState_.mode == "OFFBOARD" && currState_.armed){
            if(auto_takeoff_){
                ROS_INFO("Offboard and armed! Taking off...");
                desired_state_.p(0) = 0.0;
                desired_state_.p(1) = 0.0;
                desired_state_.p(2) = takeoff_height_;
                desired_state_.yaw = 0.0;
                takeoff_reached_ = false;
                flightState_ = TAKEOFF;
            }else{
                takeoff_reached_ = true;  // 无自动起飞：直接进任务态，当前高度即有效
                flightState_ = MISSION_EXECUTION;
            }
        }
        break;
    }
    case TAKEOFF:{
        ROS_INFO_ONCE("Auto Taking off...");
        Controller_Output_t output;
        if(se3_hof_.calControl(odom_data_, imu_data_, desired_state_, output)){
            send_cmd(output, true);
            se3_hof_.estimateTa(imu_data_.a);
        }
        if(fabs(odom_data_.p(2) - takeoff_height_) < 0.1){
            ROS_INFO("TakeOff Complete");
            takeoff_reached_ = true;
            flightState_ = MISSION_EXECUTION;
        }
        break;
    }

    case MISSION_EXECUTION:{
        ROS_INFO_ONCE("Mission execution...");
        Controller_Output_t output;
        if(se3_hof_.calControl(odom_data_, imu_data_, desired_state_, output)){
            send_cmd(output, true);
            se3_hof_.estimateTa(imu_data_.a);
        }
        break;
    }
    case PLANNER_LOST: {
        ROS_INFO_ONCE("Planner stream lost, holding position...");
        Controller_Output_t output;
        if(se3_hof_.calControl(odom_data_, imu_data_, desired_state_, output)){
            send_cmd(output, true);
            se3_hof_.estimateTa(imu_data_.a);
        }
        if((now - last_planner_msg_time_).toSec() < planner_timeout_){
            ROS_WARN("se3_hof: planner stream resumed, back to MISSION_EXECUTION.");
            flightState_ = MISSION_EXECUTION;
        } else if(planner_lost_land_time_ > 0.0 &&
                  (now - planner_lost_enter_time_).toSec() > planner_lost_land_time_){
            ROS_WARN_STREAM("se3_hof: planner lost over " << planner_lost_land_time_ << " s, landing.");
            flightState_ = LANDING;
        }
        break;
    }
    case LANDING: {
        landing_locked_ = true;
        // 限频请求，避免 100Hz 阻塞式服务调用卡住单线程 spinner（进而拖停全部回调）
        if ((now - last_land_request_).toSec() >= request_interval_) {
            last_land_request_ = now;
            mavros_msgs::SetMode land_set_mode;
            land_set_mode.request.custom_mode = "AUTO.LAND";
            if(set_mode_client_.call(land_set_mode) && land_set_mode.response.mode_sent){
                flightState_ = LANDED;
                ROS_INFO("land enabled");
            }
        }
        // ros::spinOnce();
        break;
    }
    case LANDED:
        if(!currState_.armed){
            ROS_INFO("Landed. Please set to position control and disarm.");
            exec_timer_.stop();
        }
        // ros::spinOnce();
        break;
    case EMERGENCY:
         ROS_ERROR("Emergency state! Please check the system.");
         flightState_ = LANDING;
         break;
    default:
        break;
    }
}

void Se3HofCtrl::send_cmd(const Controller_Output_t &output, bool angle){
    mavros_msgs::AttitudeTarget cmd;
    cmd.header.stamp = ros::Time::now();
    cmd.body_rate.x = output.bodyrates(0);
    cmd.body_rate.y = output.bodyrates(1);
    cmd.body_rate.z = output.bodyrates(2);
    cmd.orientation.w = output.q.w();
    cmd.orientation.x = output.q.x();
    cmd.orientation.y = output.q.y();
    cmd.orientation.z = output.q.z();
    cmd.thrust = output.thrust;
    if(angle){
        cmd.type_mask = mavros_msgs::AttitudeTarget::IGNORE_ROLL_RATE +
                        mavros_msgs::AttitudeTarget::IGNORE_PITCH_RATE +
                        mavros_msgs::AttitudeTarget::IGNORE_YAW_RATE;
    }else{
        cmd.type_mask = mavros_msgs::AttitudeTarget::IGNORE_ATTITUDE;
    }
    cmd_pub_.publish(cmd);
}

void Se3HofCtrl::pubLocalPose(const Eigen::Vector3d &pose)
{
    geometry_msgs::PoseStamped msg;
    msg.header.stamp = ros::Time::now();
    msg.header.frame_id = "map";
    msg.pose.position.x = pose[0];
    msg.pose.position.y = pose[1];
    msg.pose.position.z = pose[2];
    // 偏航保持当前值（fastlio 系 yaw 投影）——全零四元数非法，PX4 会拒绝设定点
    const Eigen::Quaterniond &q = odom_data_.q;
    const double yaw = std::atan2(2.0 * (q.w() * q.z() + q.x() * q.y()),
                                  1.0 - 2.0 * (q.y() * q.y() + q.z() * q.z()));
    msg.pose.orientation.w = std::cos(yaw * 0.5);
    msg.pose.orientation.z = std::sin(yaw * 0.5);

    local_pos_pub_.publish(msg);
}

bool Se3HofCtrl::landCallback(std_srvs::SetBool::Request &request, std_srvs::SetBool::Response &response) {
    ROS_INFO("trigger land!");
    flightState_ = LANDING;
    response.success = true;      // 不填时默认 False，mission_state 日志会误报失败
    response.message = "landing triggered";
    return true;
}

bool Se3HofCtrl::msgExpired(const ros::Time &stamp, const std::string &what) {
    if (stamp.isZero()) {
        // 无时间戳无法判定新旧，放行并提示（打戳是上游的义务）
        ROS_WARN_STREAM_THROTTLE(5.0, "se3_hof: " << what << " msg has no stamp, skip expiry check.");
        return false;
    }
    const double age = (ros::Time::now() - stamp).toSec();
    if (age > msg_expire_time_) {
        ROS_WARN_STREAM_THROTTLE(1.0, "se3_hof: drop expired " << what
                               << " msg, age " << age * 1000.0
                               << " ms > " << msg_expire_time_ * 1000.0 << " ms.");
        return true;
    }
    return false;
}

void Se3HofCtrl::OdomCallback(const nav_msgs::Odometry::ConstPtr &msg){
    if (msgExpired(msg->header.stamp, "odom")) {
        return;
    }
    last_odom_msg_time_ = ros::Time::now();
    odom_msg_received_ = true;
    odom_data_.feed(msg, enu_frame_, vel_in_body_);
    bool judge_x = ((odom_data_.p(0) >= geo_fence_[0]) || (odom_data_.p(0) <= -geo_fence_[0]));
    bool judge_y = ((odom_data_.p(1) >= geo_fence_[1]) || (odom_data_.p(1) <= -geo_fence_[1]));
    bool judge_z = (odom_data_.p(2) >= geo_fence_[2]);
    bool judge = (judge_x || judge_y || judge_z);
    if(judge && currState_.mode != mavros_msgs::State::MODE_PX4_LAND && flightState_ != LANDING && flightState_ != LANDED){
        flightState_ = EMERGENCY;
    }
}

void Se3HofCtrl::IMUCallback(const sensor_msgs::Imu::ConstPtr &msg){
    if (msgExpired(msg->header.stamp, "imu")) {
        return;
    }
    last_imu_msg_time_ = ros::Time::now();
    imu_msg_received_ = true;
    imu_data_.feed(msg, enu_frame_);
}

void Se3HofCtrl::StateCallback(const mavros_msgs::State::ConstPtr &msg){
    currState_ = *msg;
    if ((flightState_ == MISSION_EXECUTION || flightState_ == PLANNER_LOST) && !currState_.armed) {
        flightState_ = EMERGENCY;
        landing_locked_ = true;
        ROS_ERROR("se3_hof: unexpected disarm during mission.");
    }
    if (currState_.mode == "AUTO.LAND" && !landing_locked_) {
        landing_locked_ = true;
        ROS_WARN("se3_hof landing lock enabled (AUTO.LAND detected).");
    }
}

void Se3HofCtrl::TrySetOffboard(const ros::Time &now) {
    if (landing_locked_ || !enable_auto_offboard_) {
        return;
    }
    if (currState_.mode == "OFFBOARD") {
        return;
    }
    if (offboard_warmup_counter_ < offboard_warmup_count_) {
        return;
    }
    if ((now - last_mode_request_).toSec() < request_interval_) {
        return;
    }

    mavros_msgs::SetMode offb_set_mode;
    offb_set_mode.request.custom_mode = "OFFBOARD";
    if (set_mode_client_.call(offb_set_mode) && offb_set_mode.response.mode_sent) {
        offboard_triggered_ = true;
        ROS_INFO_THROTTLE(2.0, "se3_hof requested OFFBOARD mode.");
    } else {
        ROS_WARN_THROTTLE(2.0, "se3_hof failed to request OFFBOARD mode.");
    }
    last_mode_request_ = now;
}

void Se3HofCtrl::TryArm(const ros::Time &now) {
    if (landing_locked_ || !enable_auto_arm_) {
        return;
    }
    if (currState_.armed) {
        return;
    }
    if (enable_auto_offboard_ && currState_.mode != "OFFBOARD") {
        return;
    }
    if ((now - last_arm_request_).toSec() < request_interval_) {
        return;
    }

    arm_cmd.request.value = true;
    if (arming_client_.call(arm_cmd) && arm_cmd.response.success) {
        arm_triggered_ = true;
        ROS_INFO_THROTTLE(2.0, "se3_hof requested arming.");
    } else {
        ROS_WARN_THROTTLE(2.0, "se3_hof failed to arm.");
    }
    last_arm_request_ = now;
}

void Se3HofCtrl::plannerOutputCallback(const planner_ctrl::PlannerOutput::ConstPtr &msg)
{
    // 起飞未达预定高度前不接受规划器航点：地面/爬升段检测器就可能锁框发 goal，
    // 提前喂入会把 desired_state_ 从起飞目标拉走，导致爬升轨迹被干扰
    if (!takeoff_reached_) {
        ROS_WARN_STREAM_THROTTLE(3.0, "se3_hof: planner output ignored before takeoff height reached.");
        return;
    }
    if (msgExpired(msg->header.stamp, "planner_output")) {
        return;
    }
    if (msg->points.empty()) {
        ROS_WARN("Received empty planner output message");
        return;
    }
    last_planner_msg_time_ = ros::Time::now();
    planner_msg_received_ = true;
    const planner_ctrl::PlannerOutputPoint &pt = msg->points[0];

    desired_state_.p = planner_ctrl::planner_output::SelectPosition(pt, desired_state_.p);

    desired_state_.v = planner_ctrl::planner_output::SelectVelocity(pt);

    desired_state_.a = planner_ctrl::planner_output::SelectAcceleration(pt);
    desired_state_.j.setZero();

    desired_state_.yaw = planner_ctrl::planner_output::SelectYaw(pt, desired_state_.yaw);
    desired_state_.yaw_rate = planner_ctrl::planner_output::SelectYawRate(pt);
    desired_state_.q = planner_ctrl::planner_output::QuaternionFromYaw(desired_state_.yaw);
}
