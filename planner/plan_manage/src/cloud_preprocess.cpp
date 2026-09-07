#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/PointCloud2.h>

#include <Eigen/Core>
#include <pcl/filters/crop_box.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

#include <string>
#include <vector>

class CloudPreprocess {
 public:
  CloudPreprocess() : nh_(), pnh_("~"), has_odom_(false), current_pos_(Eigen::Vector3d::Zero()) {
    pnh_.param<std::string>("odom_topic", odom_topic_, "/Odometry");
    pnh_.param<std::string>("input_cloud_topic", input_cloud_topic_, "/cloud_registered");
    pnh_.param<std::string>("output_cloud_topic", output_cloud_topic_, "/cloud_ego_registered");
    pnh_.param<std::string>("output_frame_id", output_frame_id_, "");

    pnh_.param("crop_half_x", crop_half_x_, 0.3);
    pnh_.param("crop_half_y", crop_half_y_, 0.3);
    pnh_.param("crop_half_z", crop_half_z_, 0.2);
    pnh_.param("voxel_leaf_size", voxel_leaf_size_, 0.15);

    odom_sub_ = nh_.subscribe(odom_topic_, 50, &CloudPreprocess::odomCallback, this);
    cloud_sub_ = nh_.subscribe(input_cloud_topic_, 10, &CloudPreprocess::cloudCallback, this);
    cloud_pub_ = nh_.advertise<sensor_msgs::PointCloud2>(output_cloud_topic_, 10);

    ROS_INFO_STREAM("cloud_preprocess_node ready."
                    << " odom=" << odom_topic_
                    << " input_cloud=" << input_cloud_topic_
                    << " output_cloud=" << output_cloud_topic_
                    << " crop=(" << crop_half_x_ << ", " << crop_half_y_ << ", " << crop_half_z_ << ")"
                    << " voxel=" << voxel_leaf_size_);
  }

 private:
  void odomCallback(const nav_msgs::OdometryConstPtr& msg) {
    current_pos_(0) = msg->pose.pose.position.x;
    current_pos_(1) = msg->pose.pose.position.y;
    current_pos_(2) = msg->pose.pose.position.z;
    has_odom_ = true;
  }

  void cloudCallback(const sensor_msgs::PointCloud2ConstPtr& msg) {
    if (!has_odom_) {
      return;
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::fromROSMsg(*msg, *cloud_in);

    std::vector<int> valid_indices;
    pcl::removeNaNFromPointCloud(*cloud_in, *cloud_in, valid_indices);
    if (cloud_in->empty()) {
      return;
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_body_filtered(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::CropBox<pcl::PointXYZ> crop_filter;
    crop_filter.setMin(Eigen::Vector4f(current_pos_(0) - crop_half_x_, current_pos_(1) - crop_half_y_,
                                       current_pos_(2) - crop_half_z_, 1.0f));
    crop_filter.setMax(Eigen::Vector4f(current_pos_(0) + crop_half_x_, current_pos_(1) + crop_half_y_,
                                       current_pos_(2) + crop_half_z_, 1.0f));
    crop_filter.setNegative(true);
    crop_filter.setInputCloud(cloud_in);
    crop_filter.filter(*cloud_body_filtered);

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_downsampled(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::VoxelGrid<pcl::PointXYZ> voxel_filter;
    voxel_filter.setLeafSize(voxel_leaf_size_, voxel_leaf_size_, voxel_leaf_size_);
    voxel_filter.setInputCloud(cloud_body_filtered);
    voxel_filter.filter(*cloud_downsampled);

    sensor_msgs::PointCloud2 cloud_out;
    pcl::toROSMsg(*cloud_downsampled, cloud_out);
    cloud_out.header = msg->header;
    if (!output_frame_id_.empty()) {
      cloud_out.header.frame_id = output_frame_id_;
    }

    cloud_pub_.publish(cloud_out);
  }

  ros::NodeHandle nh_;
  ros::NodeHandle pnh_;
  ros::Subscriber odom_sub_;
  ros::Subscriber cloud_sub_;
  ros::Publisher cloud_pub_;

  std::string odom_topic_;
  std::string input_cloud_topic_;
  std::string output_cloud_topic_;
  std::string output_frame_id_;

  double crop_half_x_;
  double crop_half_y_;
  double crop_half_z_;
  double voxel_leaf_size_;

  bool has_odom_;
  Eigen::Vector3d current_pos_;
};

int main(int argc, char** argv) {
  ros::init(argc, argv, "cloud_preprocess_node");
  CloudPreprocess node;
  ros::spin();
  return 0;
}
