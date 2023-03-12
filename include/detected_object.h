#pragma once

#include <string>
#include <vector>
#include <map>

#include <ros/types.h>

#include <std_msgs/Header.h>
#include <std_msgs/ColorRGBA.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/Vector3.h>
#include <geometry_msgs/Vector3.h>
#include <geometry_msgs/Twist.h>
#include <geometry_msgs/Twist.h>
#include <sensor_msgs/PointCloud2.h>
#include <geometry_msgs/PolygonStamped.h>
#include <sensor_msgs/Image.h>


template <class T>
struct DetectedObject_
{
  typedef DetectedObject_<T> Type;

  DetectedObject_()
    : header()
    , id(0)
    , label()
    , score(0.0)
    , color()
    , valid(false)
    , space_frame()
    , pose()
    , dimensions()
    , variance()
    , velocity()
    , acceleration()
    , pointcloud()
    , convex_hull()
    , pose_reliable(false)
    , velocity_reliable(false)
    , acceleration_reliable(false)
    , image_frame()
    , x(0)
    , y(0)
    , width(0)
    , height(0)
    , angle(0.0)
    , roi_image()
    , indicator_state(0)
    , behavior_state(0)
    , user_defined_info()  {
    }
  DetectedObject_(const T& _alloc)
    : header(_alloc)
    , id(0)
    , label(_alloc)
    , score(0.0)
    , color(_alloc)
    , valid(false)
    , space_frame(_alloc)
    , pose(_alloc)
    , dimensions(_alloc)
    , variance(_alloc)
    , velocity(_alloc)
    , acceleration(_alloc)
    , pointcloud(_alloc)
    , convex_hull(_alloc)
    , pose_reliable(false)
    , velocity_reliable(false)
    , acceleration_reliable(false)
    , image_frame(_alloc)
    , x(0)
    , y(0)
    , width(0)
    , height(0)
    , angle(0.0)
    , roi_image(_alloc)
    , indicator_state(0)
    , behavior_state(0)
    , user_defined_info(_alloc)  {
  (void)_alloc;
    }

   typedef  ::std_msgs::Header_<T>  _header_type;
  _header_type header;

   typedef uint32_t _id_type;
  _id_type id;

   typedef std::basic_string<char, std::char_traits<char>, typename T::template rebind<char>::other >  _label_type;
  _label_type label;

   typedef float _score_type;
  _score_type score;

   typedef  ::std_msgs::ColorRGBA_<T>  _color_type;
  _color_type color;

   typedef uint8_t _valid_type;
  _valid_type valid;

   typedef std::basic_string<char, std::char_traits<char>, typename T::template rebind<char>::other >  _space_frame_type;
  _space_frame_type space_frame;

  typedef geometry_msgs::Pose_<T> _pose_type;
  _pose_type pose;

  typedef geometry_msgs::Vector3_<T> _dimensions_type;
  _dimensions_type dimensions;

  typedef geometry_msgs::Vector3_<T> _variance_type;
  _variance_type variance;

  typedef geometry_msgs::Twist_<T> _velocity_type;
  _velocity_type velocity;

  typedef geometry_msgs::Twist_<T> _acceleration_type;
  _acceleration_type acceleration;

  typedef sensor_msgs::PointCloud2_<T> _pointcloud_type;
  _pointcloud_type pointcloud;

  typedef geometry_msgs::PolygonStamped_<T> _convex_hull_type;
  _convex_hull_type convex_hull;

   typedef uint8_t _pose_reliable_type;
  _pose_reliable_type pose_reliable;

   typedef uint8_t _velocity_reliable_type;
  _velocity_reliable_type velocity_reliable;

   typedef uint8_t _acceleration_reliable_type;
  _acceleration_reliable_type acceleration_reliable;

   typedef std::basic_string<char, std::char_traits<char>, typename T::template rebind<char>::other >  _image_frame_type;
  _image_frame_type image_frame;

   typedef int32_t _x_type;
  _x_type x;

   typedef int32_t _y_type;
  _y_type y;

   typedef int32_t _width_type;
  _width_type width;

   typedef int32_t _height_type;
  _height_type height;

   typedef float _angle_type;
  _angle_type angle;

   typedef  ::sensor_msgs::Image_<T>  _roi_image_type;
  _roi_image_type roi_image;

   typedef uint8_t _indicator_state_type;
  _indicator_state_type indicator_state;

   typedef uint8_t _behavior_state_type;
  _behavior_state_type behavior_state;

   typedef std::vector<std::basic_string<char, std::char_traits<char>, typename T::template rebind<char>::other > , typename T::template rebind<std::basic_string<char, std::char_traits<char>, typename T::template rebind<char>::other > >::other >  _user_defined_info_type;
  _user_defined_info_type user_defined_info;



  typedef boost::shared_ptr< DetectedObject_<T> > Ptr;
  typedef boost::shared_ptr< DetectedObject_<T> const> ConstPtr;

};

typedef DetectedObject_<std::allocator<void> > DetectedObject;

typedef boost::shared_ptr< DetectedObject > DetectedObjectPtr;
typedef boost::shared_ptr< DetectedObject const> DetectedObjectConstPtr;


template<typename T>
std::ostream& operator<<(std::ostream& s, const DetectedObject_<T> & v)
{
ros::message_operations::Printer<DetectedObject_<T> >::stream(s, "", v);
return s;
}


template<typename T1, typename T2>
bool operator==(const DetectedObject_<T1> & lhs, const DetectedObject_<T2> & rhs)
{
  return lhs.header == rhs.header &&
    lhs.id == rhs.id &&
    lhs.label == rhs.label &&
    lhs.score == rhs.score &&
    lhs.color == rhs.color &&
    lhs.valid == rhs.valid &&
    lhs.space_frame == rhs.space_frame &&
    lhs.pose == rhs.pose &&
    lhs.dimensions == rhs.dimensions &&
    lhs.variance == rhs.variance &&
    lhs.velocity == rhs.velocity &&
    lhs.acceleration == rhs.acceleration &&
    lhs.pointcloud == rhs.pointcloud &&
    lhs.convex_hull == rhs.convex_hull &&
    lhs.pose_reliable == rhs.pose_reliable &&
    lhs.velocity_reliable == rhs.velocity_reliable &&
    lhs.acceleration_reliable == rhs.acceleration_reliable &&
    lhs.image_frame == rhs.image_frame &&
    lhs.x == rhs.x &&
    lhs.y == rhs.y &&
    lhs.width == rhs.width &&
    lhs.height == rhs.height &&
    lhs.angle == rhs.angle &&
    lhs.roi_image == rhs.roi_image &&
    lhs.indicator_state == rhs.indicator_state &&
    lhs.behavior_state == rhs.behavior_state &&
    lhs.user_defined_info == rhs.user_defined_info;
}

template<typename T1, typename T2>
bool operator!=(const DetectedObject_<T1> & lhs, const DetectedObject_<T2> & rhs)
{
  return !(lhs == rhs);
}
