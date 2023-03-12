#pragma once

#include <string>
#include <vector>
#include <map>

#include <ros/types.h>
#include <std_msgs/Header.h>

#include "detected_object.h"

template <class T>
struct DetectedObjectArray_
{
  typedef DetectedObjectArray_<T> Type;

  DetectedObjectArray_(): header(), objects()  {}
  DetectedObjectArray_(const T& _alloc): header(_alloc), objects(_alloc)  {
    (void)_alloc;
  }

  typedef  ::std_msgs::Header_<T>  _header_type;
  _header_type header;

  typedef std::vector<DetectedObject_<T> , typename T::template rebind<DetectedObject_<T> >::other >  _objects_type;
  _objects_type objects;


  typedef boost::shared_ptr<DetectedObjectArray_<T> > Ptr;
  typedef boost::shared_ptr<DetectedObjectArray_<T> const> ConstPtr;

}; // struct DetectedObjectArray_

typedef DetectedObjectArray_<std::allocator<void> > DetectedObjectArray;

typedef boost::shared_ptr<DetectedObjectArray > DetectedObjectArrayPtr;
typedef boost::shared_ptr<DetectedObjectArray const> DetectedObjectArrayConstPtr;


template<typename T>
std::ostream& operator<<(std::ostream& s, const DetectedObjectArray_<T> & v)
{
ros::message_operations::Printer<DetectedObjectArray_<T> >::stream(s, "", v);
return s;
}


template<typename T1, typename T2>
bool operator==(const DetectedObjectArray_<T1> & lhs, const DetectedObjectArray_<T2> & rhs)
{
  return lhs.header == rhs.header &&
    lhs.objects == rhs.objects;
}

template<typename T1, typename T2>
bool operator!=(const DetectedObjectArray_<T1> & lhs, const DetectedObjectArray_<T2> & rhs)
{
  return !(lhs == rhs);
}
