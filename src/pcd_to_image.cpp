#include <ros/ros.h>
//Image message
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
//pcl::toROSMsg
//conversions from PCL custom types
#include <pcl_conversions/pcl_conversions.h>
//stl stuff
#include <string>

class PointCloudToImage
{
public:
  void
  cloud_cb (const sensor_msgs::PointCloud2ConstPtr& cloud)
  {
    if (cloud->height <= 1)
    {
      ROS_ERROR("Input point cloud is not organized, ignoring!");
      return;
    }
    try
    {
      pcl::toROSMsg (*cloud, image_); //convert the cloud
      image_.header = cloud->header;
      image_pub_.publish (image_); //publish our cloud image
    }
    catch (std::runtime_error &e)
    {
      ROS_ERROR_STREAM("Error in converting cloud to image message: "
                        << e.what());
    }
  }
  PointCloudToImage () : cloud_topic_("lidar_top"),image_topic_("output")
  {
    sub_ = nh_.subscribe (cloud_topic_, 30,
                          &PointCloudToImage::cloud_cb, this);
    image_pub_ = nh_.advertise<sensor_msgs::Image> (image_topic_, 30);

    //print some info about the node
    std::string r_ct = nh_.resolveName (cloud_topic_);
    std::string r_it = nh_.resolveName (image_topic_);
    ROS_INFO_STREAM("Listening for incoming data on topic " << r_ct );
    ROS_INFO_STREAM("Publishing image on topic " << r_it );
  }
private:
  ros::NodeHandle nh_;
  sensor_msgs::Image image_; //cache the image message
  std::string cloud_topic_; //default input
  std::string image_topic_; //default output
  ros::Subscriber sub_; //cloud subscriber
  ros::Publisher image_pub_; //image message publisher
};

int
main (int argc, char **argv)
{
  ros::init (argc, argv, "convert_pointcloud_to_image");
  PointCloudToImage pci; //this loads up the node
  ros::spin (); //where she stops nobody knows
  return 0;
}
