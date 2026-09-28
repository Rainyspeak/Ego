#ifndef PLANNER_CTRL_SE3_HOF_HPP
#define PLANNER_CTRL_SE3_HOF_HPP


// SE3 几何控制器(基于 Hopf 纤维化姿态解算)，

#include <queue>
#include <Eigen/Dense>
#include <sensor_msgs/Imu.h>
#include <nav_msgs/Odometry.h>
#include "planner_ctrl/se3_utils.hpp"

// #define AIRSIM

struct Odom_Data_t{

	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	Eigen::Vector3d p;
	Eigen::Vector3d v;
	Eigen::Quaterniond q;
	Eigen::Vector3d w;

	nav_msgs::Odometry msg;
	ros::Time rcv_stamp;
	bool recv_new_msg;

	Odom_Data_t(){
		recv_new_msg = false;
	}
	void feed(nav_msgs::Odometry::ConstPtr pMsg, bool enu_frame, bool vel_in_body){
		msg = *pMsg;
		rcv_stamp = ros::Time::now();
		recv_new_msg = true;

		p(0) = msg.pose.pose.position.x;
		p(1) = msg.pose.pose.position.y;
		p(2) = msg.pose.pose.position.z;

		v(0) = msg.twist.twist.linear.x;
		v(1) = msg.twist.twist.linear.y;
		v(2) = msg.twist.twist.linear.z;

		q.w() = msg.pose.pose.orientation.w;
		q.x() = msg.pose.pose.orientation.x;
		q.y() = msg.pose.pose.orientation.y;
		q.z() = msg.pose.pose.orientation.z;

		w(0) = msg.twist.twist.angular.x;
		w(1) = msg.twist.twist.angular.y;
		w(2) = msg.twist.twist.angular.z;

		if(!enu_frame){
			Eigen::Matrix3d R_mid;
			R_mid << 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0;
			Eigen::Quaterniond q_mid(R_mid);

			p = q_mid.toRotationMatrix() * p;
			v = q_mid.toRotationMatrix() * v;
			q = q_mid * q * q_mid;
			q.normalize();
			w = q_mid.toRotationMatrix() * w;
		}

		if(vel_in_body)
			v = q.toRotationMatrix() * v;
	}
};

struct Desired_State_t
{
	Eigen::Vector3d p;
	Eigen::Vector3d v;
	Eigen::Vector3d a;
	Eigen::Vector3d j;
	Eigen::Quaterniond q;
	double yaw;
	double yaw_rate;

	Desired_State_t(){
		p = Eigen::Vector3d::Zero();
		v = Eigen::Vector3d::Zero();
		a = Eigen::Vector3d::Zero();
		j = Eigen::Vector3d::Zero();
		q.w() = 1;
		q.x() = 0;
		q.y() = 0;
		q.z() = 0;
		yaw = 0;
		yaw_rate = 0;
	}

	Desired_State_t(Odom_Data_t odom){
		p = odom.p;
		v = Eigen::Vector3d::Zero();
		a = Eigen::Vector3d::Zero();
		j = Eigen::Vector3d::Zero();
		q = odom.q.normalized();
		yaw = utils::fromQuaternion2yaw(q);
		yaw_rate = 0;
	}
};

struct Controller_Output_t
{
	// Eigen::Vector3d v;

	// Orientation of the body frame with respect to the world frame
	Eigen::Quaterniond q;

	// Body rates in body frame
	Eigen::Vector3d bodyrates; // [rad/s]

	// Collective mass normalized thrust
	double thrust;

	// Eigen 固定尺寸类型默认构造不清零：不显式初始化时 q/bodyrates 是栈上随机内存，
	// 会被整包发往 FCU（预热路径曾把未初始化四元数发出）
	Controller_Output_t()
		: q(1.0, 0.0, 0.0, 0.0), bodyrates(Eigen::Vector3d::Zero()), thrust(0.0) {}
};

struct Imu_Data_t{
	Eigen::Quaterniond q;
	Eigen::Vector3d w;
	Eigen::Vector3d a;

	sensor_msgs::Imu msg;
	ros::Time rcv_stamp;
	bool recv_new_msg;

	Imu_Data_t(){
		recv_new_msg = false;
	}
	void feed(sensor_msgs::ImuConstPtr pMsg, bool enu_frame){
		msg = *pMsg;
		rcv_stamp = ros::Time::now();
		recv_new_msg = true;

		a(0) = msg.linear_acceleration.x;
		a(1) = msg.linear_acceleration.y;
		a(2) = msg.linear_acceleration.z;

		q.x() = msg.orientation.x;
		q.y() = msg.orientation.y;
		q.z() = msg.orientation.z;
		q.w() = msg.orientation.w;

		w(0) = msg.angular_velocity.x;
		w(1) = msg.angular_velocity.y;
		w(2) = msg.angular_velocity.z;

		if(!enu_frame){
			Eigen::Matrix3d R_mid;
			R_mid << 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0;
			Eigen::Quaterniond q_mid(R_mid);

			a = q_mid.toRotationMatrix() * a;
			q = q_mid * q * q_mid;
			q.normalize();
			w = q_mid.toRotationMatrix() * w;
		}
	}
};

class SE3_HOF_CONTROLLER
{
private:
	Eigen::Vector3d Kp_p_, Kp_v_, Kp_a_, Kp_q_, Kp_w_, Kd_p_, Kd_v_, Kd_a_, Kd_q_, Kd_w_;
	double limit_err_p_, limit_err_v_, limit_err_a_, limit_d_err_p_, limit_d_err_v_, limit_d_err_a_;
	bool have_last_err_, enu_frame_, vel_in_body_;

	double hover_percent_, max_hover_percent_;
	double T_a_; // normalization constant
	double ta_min_, ta_max_;           // T_a_ 物理边界
	double P_ = 50.0;                  // RLS 协方差初值；原 1e6 使首拍增益 K≈5，单步可把 T_a_ 拉偏 ±15
	const double ta_step_max_ = 0.5;   // 单次更新允许的 T_a_ 变化上限
	const double ta_innov_gate_ = 5.0; // 新息门限(m/s^2)，超出视为瞬态/振动等无效配对
	const double rho_ = 0.998; // confidence
	const double gravity_ = 9.81;
	static constexpr double kAlmostZeroValueThreshold_ = 0.001;
	Eigen::Vector3d grav_vec_, last_err_p_, last_err_v_, last_err_a_, last_err_q_, last_err_w_;
	std::queue<std::pair<ros::Time, double>> timed_thrust_;


   //xyz 三轴解姿态
	void computeFlatInput(Desired_State_t desired_state, Odom_Data_t &desired_odom){

		desired_odom.p = desired_state.p;
		desired_odom.v = desired_state.v;

		// TODO check
		Eigen::Vector3d xc(cos(desired_state.yaw), sin(desired_state.yaw), 0);
		Eigen::Vector3d yc(-sin(desired_state.yaw), cos(desired_state.yaw), 0);

		Eigen::Vector3d zb = desired_state.a.normalized();
		Eigen::Vector3d xb = yc.cross(zb);
		xb.normalize();
		Eigen::Vector3d yb = zb.cross(xb);
		yb.normalize();

		Eigen::Matrix3d rotM;
		rotM << xb, yb, zb;
		desired_odom.q = Eigen::Quaterniond(rotM);

		double a_zb = zb.dot(desired_state.a);
		desired_odom.w(0) = -yb.dot(desired_state.j) / a_zb;
		desired_odom.w(1) = xb.dot(desired_state.j) / a_zb;
		desired_odom.w(2) = desired_state.yaw_rate * xc.dot(xb) + yc.dot(zb) * desired_odom.w(1);
		desired_odom.w(2) /= (yc.cross(zb)).norm();
	}
   //球体化解算姿态
	void computeFlatInput_Hopf_Fibration(Desired_State_t desired_state, Odom_Data_t &desired_odom){
		Eigen::Vector3d abc = desired_state.a.normalized();
		double a = abc(0), b = abc(1), c = abc(2);
		Eigen::Vector3d abc_dot = (desired_state.a.dot(desired_state.a) * Eigen::MatrixXd::Identity(3, 3) - desired_state.a * desired_state.a.transpose()) / desired_state.a.norm() / desired_state.a.squaredNorm() * desired_state.j;
		double a_dot = abc_dot(0), b_dot = abc_dot(1), c_dot = abc_dot(2);
		double yaw = desired_state.yaw;
		double yaw_dot = desired_state.yaw_rate;
		double syaw = sin(yaw), cyaw = cos(yaw);

		if(c > 0){
			double norm = sqrt(2 * (1 + c));
			Eigen::Quaterniond q((1 + c) / norm, -b / norm, a / norm, 0);
			Eigen::Quaterniond q_yaw(cos(yaw / 2), 0, 0, sin(yaw / 2));
			desired_odom.q = q * q_yaw;
			desired_odom.w(0) = syaw * a_dot - cyaw * b_dot - (a * syaw - b * cyaw) * c_dot / (c + 1);
			desired_odom.w(1) = cyaw * a_dot + syaw * b_dot - (a * cyaw + b * syaw) * c_dot / (c + 1);
			desired_odom.w(2) = (b * a_dot - a * b_dot) / (1 + c) + yaw_dot;
		}else{
			double norm = sqrt(2 * (1 - c));
			Eigen::Quaterniond q(-b / norm, (1 - c) / norm, 0, a / norm);
			yaw += 2 * atan2(a, b);
			Eigen::Quaterniond q_yaw(cos(yaw / 2), 0, 0, sin(yaw / 2));
			desired_odom.q = q * q_yaw;
			syaw = sin(yaw);
			cyaw = cos(yaw);
			yaw_dot = yaw_dot;// + atan2_dot(a, b, a_dot, b_dot);

			desired_odom.w(0) = syaw * a_dot + cyaw * b_dot - (a * syaw + b * cyaw) * c_dot / (c - 1);
			desired_odom.w(1) = cyaw * a_dot - syaw * b_dot - (a * cyaw - b * syaw) * c_dot / (c - 1);
			desired_odom.w(2) = (b * a_dot - a * b_dot) / (c - 1) + yaw_dot;
		}
	}

	void limitErr(Eigen::Vector3d &err, double low, double upper){
		err(0) = std::max(std::min(err(0), upper), low);
		err(1) = std::max(std::min(err(1), upper), low);
		err(2) = std::max(std::min(err(2), upper), low);
	}

	double limitYaw(double yaw_curr, double yaw_des, double yaw_limit){
		double err = yaw_des - yaw_curr;
		yaw_limit = std::abs(yaw_limit);
		if(err < -yaw_limit){
			return yaw_curr - yaw_limit;
		}else if (err > yaw_limit){
			return yaw_curr + yaw_limit;
		}else{
			return yaw_des;
		}
	}

public:
	SE3_HOF_CONTROLLER(){};
    ~SE3_HOF_CONTROLLER(){};

	void init(double hover_percent, double max_hover_percent, bool enu_frame, bool vel_in_body){

		hover_percent_ = hover_percent;
		max_hover_percent_ = max_hover_percent;
		enu_frame_ = enu_frame;
		vel_in_body_ = vel_in_body;
		T_a_ = gravity_ / hover_percent_;
		ta_min_ = gravity_ / max_hover_percent_;
		ta_max_ = 1.6 * gravity_ / hover_percent_;
		if (ta_max_ <= ta_min_) {
			ta_max_ = 1.5 * ta_min_;
		}
		grav_vec_ << 0.0, 0.0, gravity_;

		last_err_p_ = Eigen::Vector3d::Zero();
		last_err_v_ = Eigen::Vector3d::Zero();
		last_err_a_ = Eigen::Vector3d::Zero();
		last_err_q_ = Eigen::Vector3d::Zero();
		last_err_w_ = Eigen::Vector3d::Zero();

		have_last_err_ = false;
	}

	void setup(Eigen::Vector3d kp_p, Eigen::Vector3d kp_v, Eigen::Vector3d kp_a, Eigen::Vector3d kp_q, Eigen::Vector3d kp_w,
				Eigen::Vector3d kd_p, Eigen::Vector3d kd_v, Eigen::Vector3d kd_a, Eigen::Vector3d kd_q, Eigen::Vector3d kd_w,
				double limit_err_p, double limit_err_v, double limit_err_a,
				double limit_d_err_p, double limit_d_err_v, double limit_d_err_a){
		Kp_p_ = kp_p;
		Kp_v_ = kp_v;
		Kp_a_ = kp_a;
		Kp_q_ = kp_q; 
		Kp_w_ = kp_w;
		Kd_p_ = kd_p;
		Kd_v_ = kd_v;
		Kd_a_ = kd_a;
		Kd_q_ = kd_q;
		Kd_w_ = kd_w;
		limit_err_p_ = limit_err_p;
		limit_err_v_ = limit_err_v;
		limit_err_a_ = limit_err_a;
		limit_d_err_p_ = limit_d_err_p;
		limit_d_err_v_ = limit_d_err_v;
		limit_d_err_a_ = limit_d_err_a;
	}

	bool calControl(Odom_Data_t odom_data, Imu_Data_t imu_data, Desired_State_t desired_state, Controller_Output_t &output){

		//保险时间，杜绝过期的周期指令
		if((ros::Time::now() - odom_data.rcv_stamp).toSec() > 0.1){
			// std::cout << "odom not rcv" << std::endl;
			return false;
		}
		// desired_state.yaw = limitYaw(fromQuaternion2yaw(odom_data.q), desired_state.yaw, M_PI_2 / 3);
		Eigen::Vector3d err_p = odom_data.p - desired_state.p;
		limitErr(err_p, -limit_err_p_, limit_err_p_);
		if(have_last_err_ == false)
			last_err_p_ = err_p;
		Eigen::Vector3d d_err_p = err_p - last_err_p_;

		limitErr(d_err_p, -limit_d_err_p_, limit_d_err_p_);
		desired_state.v = desired_state.v - Kp_p_.asDiagonal() * err_p - Kd_p_.asDiagonal() * d_err_p;


		Eigen::Vector3d err_v = odom_data.v - desired_state.v;
		limitErr(err_v, -limit_err_v_, limit_err_v_);
		if(have_last_err_ == false)
			last_err_v_ = err_v;
		Eigen::Vector3d d_err_v = err_v - last_err_v_;
		limitErr(d_err_v, -limit_d_err_v_, limit_d_err_v_);
		desired_state.a = desired_state.a - Kp_v_.asDiagonal() * err_v - Kd_v_.asDiagonal() * d_err_v + grav_vec_;

		// std::cout << "err_p: " << err_p.transpose() << std::endl;
		// std::cout << "err_v: " << err_v.transpose() << std::endl;
		// std::cout << "imu_data.a: " << imu_data.a.transpose() << std::endl;
		// std::cout << "odom_data.v: " << odom_data.v.transpose() << std::endl;

		Eigen::Vector3d a_world = odom_data.q.toRotationMatrix() * imu_data.a;
		Eigen::Vector3d err_a = a_world - desired_state.a;
		limitErr(err_a, -limit_err_a_, limit_err_a_);
		if(have_last_err_ == false)
			last_err_a_ = err_a;
		Eigen::Vector3d d_err_a = err_a - last_err_a_;
		limitErr(d_err_a, -limit_d_err_a_, limit_d_err_a_);
		desired_state.j = desired_state.j - Kp_a_.asDiagonal() * err_a - Kd_a_.asDiagonal() * d_err_a;

		last_err_p_ = err_p;
		last_err_v_ = err_v;
		last_err_a_ = err_a;



		//解算推力，修正期望加速度
		double thr = desired_state.a.transpose() * (odom_data.q * Eigen::Vector3d::UnitZ());
		output.thrust = thr / T_a_;
		// ROS_INFO_STREAM_THROTTLE(1.0, "T_a_: " << T_a_);
		// std::cout << std::endl << "desired_state.a: " << desired_state.a.transpose() << std::endl;
		// std::cout << "odom_v: " << odom_data.v.transpose() << std::endl;


		//修补姿态位置源定位的偏差
		Odom_Data_t desired_odom;
		// computeFlatInput(desired_state, desired_odom);
		computeFlatInput_Hopf_Fibration(desired_state, desired_odom);
		output.q = imu_data.q * odom_data.q.inverse() * desired_odom.q; // Align with FCU frame, from odom to imu frame

		// printf("desired q: (%lf,%lf,%lf,%lf)\n", desired_odom.q.w(), desired_odom.q.x(), desired_odom.q.y(), desired_odom.q.z());
		// std::cout << "desired q: " << desired_state.a.transpose() << std::endl;
		
		Eigen::Quaterniond err_q = odom_data.q.inverse() * desired_odom.q;

		//前馈修正 加速度
		Eigen::Vector3d err_br;
		if (err_q.w() >= 0){
			err_br.x() = Kp_q_(0) * err_q.x();
			err_br.y() = Kp_q_(1) * err_q.y();
			err_br.z() = Kp_q_(2) * err_q.z();
		}
		else{
			err_br.x() = -Kp_q_(0) * err_q.x();
			err_br.y() = -Kp_q_(1) * err_q.y();
			err_br.z() = -Kp_q_(2) * err_q.z();
		}

		output.bodyrates = desired_odom.w + err_br;

		if(!enu_frame_){
			Eigen::Matrix3d R_mid;
			R_mid << 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0;
			Eigen::Quaterniond q_mid(R_mid.inverse());
			output.q = q_mid * output.q * q_mid;
			output.bodyrates = q_mid * output.bodyrates;
		}

		//推送输出
		timed_thrust_.push(std::pair<ros::Time, double>(ros::Time::now(), output.thrust));
		while (timed_thrust_.size() > 100)
			timed_thrust_.pop();
		return true;
	}

	// 在线估计推力归一化常数 T_a_。模型 est_a(2) = thr * T_a_，est_a 为机体系 z 轴比力。
	// 加固(9-28 冲顶复盘)：地面反力使 est_a 与推力解耦、大倾角/剧烈瞬态时 35~45ms 延迟
	// 配对失真，原实现(P_=1e6、无门限、仅下界钳位)会在这类数据上把 T_a_ 砸到下界 13，
	// 之后 u = a_cmd/T_a_ 全部过热直至饱和。现在：未离地/大倾角/垂直速度异常时冻结，
	// 新息超门限丢弃本次更新，单步限幅 + 上下界。
	bool estimateTa(const Eigen::Vector3d &est_a, const Odom_Data_t &odom){
		ros::Time t_now = ros::Time::now();
		while (timed_thrust_.size() >= 1)
		{
			// Choose data before 35~45ms ago
			std::pair<ros::Time, double> t_t = timed_thrust_.front();
			double time_passed = (t_now - t_t.first).toSec();
			if (time_passed > 0.045){ // 45ms
				timed_thrust_.pop();
				continue;
			}
			if (time_passed < 0.035){ // 35ms
				return false;
			}

			double thr = t_t.second;
			timed_thrust_.pop();

			// 冻结条件：未离地(高度<0.15m，地面反力污染模型)、倾角>35°、|vz|>3(冲顶/坠落瞬态)
			Eigen::Vector3d body_z = odom.q.normalized().toRotationMatrix().col(2);
			if (odom.p(2) < 0.15 ||
				body_z(2) < 0.819 /* cos(35°) */ ||
				std::abs(odom.v(2)) > 3.0){
				return false;
			}

			double innov = est_a(2) - thr * T_a_;
			if (!(std::abs(innov) <= ta_innov_gate_)){ // 同时拦截 NaN
				return false;
			}

			/***********************************************************/
			/* Recursive least squares algorithm with vanishing memory */
			/***********************************************************/

			/***********************************/
			/* Model: est_a(2) = T_a_ * thr    */
			/***********************************/

			double gamma = 1 / (rho_ + thr * P_ * thr);
			double K = gamma * P_ * thr;
			double T_a_new = T_a_ + K * innov;
			T_a_new = std::min(T_a_new, T_a_ + ta_step_max_);
			T_a_new = std::max(T_a_new, T_a_ - ta_step_max_);
			T_a_ = std::min(std::max(T_a_new, ta_min_), ta_max_);

			P_ = (1 - K * thr) * P_ / rho_;
			P_ = std::min(std::max(P_, 1e-3), 50.0); // 防增益塌缩致死或再度爆炸
			return true;
		}
		return false;
	}
};

#endif
