#include <pcl/common/io.h>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/ros.h>

#include "include/point_pillars.h"
#include "include/visualize.h"

void initDevice(int devNum) {
    int dev = devNum;
    cudaDeviceProp deviceProp;
    GPU_CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using device %d: %s\n", dev, deviceProp.name);
    GPU_CHECK(cudaSetDevice(dev));
}


void publishCloud(
    const ros::Publisher *in_publisher, std_msgs::Header header,
    const pcl::PointCloud<pcl::PointXYZI>::Ptr in_cloud_to_publish_ptr) {
    sensor_msgs::PointCloud2 cloud_msg;
    pcl::toROSMsg(*in_cloud_to_publish_ptr, cloud_msg);
    cloud_msg.header = header;
    in_publisher->publish(cloud_msg);
}

class TestPointPillars {
  public:
    TestPointPillars(ros::NodeHandle nh);
    ~TestPointPillars(){};

    bool Process();

  private:
    std::string label_str[3] = {"Car", "Pedestrian", "Cyclist"};
    ros::Publisher pub_in_;
    ros::Publisher pub_detect_visualize_markers_;
    VisualizeDetectedObjects vdo_;
    ros::NodeHandle nh_;

    cudaStream_t stream = NULL;

    std::unique_ptr<PointPillars> pointpillars_ptr_;

    void CloudToArray(const pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_ptr,
                      float *out_points_array, const float normalizing_factor);
    void Callback(const sensor_msgs::PointCloud2::Ptr &msg);
    void pubDetectedObject(std::vector<Bndbox> obstacle_objects,
                           std_msgs::Header in_header);
};

TestPointPillars::TestPointPillars(ros::NodeHandle nh) : nh_(nh) {
    GPU_CHECK(cudaStreamCreate(&stream));
    pointpillars_ptr_.reset(
        new PointPillars(stream)); // 外部定义调用不了cuda函数

    pointpillars_ptr_->Init();
}

void TestPointPillars::pubDetectedObject(std::vector<Bndbox> obstacle_objects,
                                         std_msgs::Header in_header) {
    DetectedObjectArray objects;
    objects.header = in_header;
    for (size_t i = 0; i < obstacle_objects.size(); i++) {
        DetectedObject object;
        object.header = in_header;
        object.valid = true;
        object.pose_reliable = true;

        object.pose.position.x = obstacle_objects[i].x;
        object.pose.position.y = obstacle_objects[i].y;
        object.pose.position.z = obstacle_objects[i].z;
        object.dimensions.x = obstacle_objects[i].w;
        object.dimensions.y = obstacle_objects[i].l;
        object.dimensions.z = obstacle_objects[i].h;

        float yaw = obstacle_objects[i].rt;
        geometry_msgs::Quaternion q = tf::createQuaternionMsgFromYaw(yaw);
        object.pose.orientation = q;

        object.label = label_str[obstacle_objects[i].id - 1];

        objects.objects.push_back(object);
    }

    std::cout << "objects :" << objects.objects.size() << std::endl;

    // 可视化
    visualization_msgs::MarkerArray visualize_markers;
    vdo_.GetVisualizeMarkers(objects, visualize_markers);
    pub_detect_visualize_markers_.publish(visualize_markers);
}

bool TestPointPillars::Process() {

    ros::Subscriber sub =
        nh_.subscribe("/velodyne_points", 1, &TestPointPillars::Callback, this);
    pub_in_ = nh_.advertise<sensor_msgs::PointCloud2>("/points_in", 1);
    pub_detect_visualize_markers_ =
        nh_.advertise<visualization_msgs::MarkerArray>("visualize/objects", 1);
    ros::spin();
}

void TestPointPillars::CloudToArray(
    const pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_ptr,
    float *out_points_array, const float normalizing_factor) {
    for (size_t i = 0; i < cloud_ptr->size(); i++) {
        pcl::PointXYZI point = cloud_ptr->at(i);
        out_points_array[i * 4 + 0] = point.x;
        out_points_array[i * 4 + 1] = point.y;
        out_points_array[i * 4 + 2] = point.z;

        out_points_array[i * 4 + 3] = point.intensity / normalizing_factor;
    }
}

void TestPointPillars::Callback(const sensor_msgs::PointCloud2::Ptr &msg) {
    int64_t tm0 = gtm();
    pcl::PointCloud<pcl::PointXYZI>::Ptr in_cloud_ptr(
        new pcl::PointCloud<pcl::PointXYZI>);
    pcl::fromROSMsg(*msg, *in_cloud_ptr);

    int num_points = in_cloud_ptr->size();
    float *points_array = new float[num_points * 255];
    CloudToArray(in_cloud_ptr, points_array, 255);

    std::vector<Bndbox> objects;
    pointpillars_ptr_->Detect(points_array, num_points, objects);
    int64_t tm1 = gtm();
    ROS_INFO("point_pillars_detection cost time:%ld ms", (tm1 - tm0) / 1000);

    publishCloud(&pub_in_, msg->header, in_cloud_ptr);
    pubDetectedObject(objects, msg->header);

    delete[] points_array;
}

int main(int argc, char **argv) {
    ros::init(argc, argv, "pre_process");
    initDevice(1);
    ros::NodeHandle nh;
    TestPointPillars test_point_pillars_(nh);
    test_point_pillars_.Process();

    return 0;
}