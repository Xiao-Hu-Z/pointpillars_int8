#pragma once

#include <cmath>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#include <ros/ros.h>
#include <std_msgs/Header.h>
#include <tf/transform_datatypes.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include "detected_object.h"
#include "detected_object_array.h"

class VisualizeDetectedObjects {
  private:
    const double arrow_height_;
    const double label_height_;
    const double object_max_linear_size_ = 50.;
    double object_speed_threshold_;
    double arrow_speed_threshold_;
    double marker_display_duration_;

    int marker_id_;

    std_msgs::ColorRGBA label_color_, cube_color_, box_color_, hull_color_,
        arrow_color_, centroid_color_;

    visualization_msgs::MarkerArray
    ObjectsToLabels(const DetectedObjectArray &in_objects);

    visualization_msgs::MarkerArray
    ObjectsToArrows(const DetectedObjectArray &in_objects);

    visualization_msgs::MarkerArray
    ObjectsToCubes(const DetectedObjectArray &in_objects);

    visualization_msgs::MarkerArray
    ObjectsToBoxes(const DetectedObjectArray &in_objects);

    visualization_msgs::MarkerArray
    ObjectsToHulls(const DetectedObjectArray &in_objects);

    visualization_msgs::MarkerArray
    ObjectsToCentroids(const DetectedObjectArray &in_objects);

    bool IsObjectValid(const DetectedObject &in_object);

    float CheckColor(double value);

    float CheckAlpha(double value);

    std_msgs::ColorRGBA ParseColor(const std::vector<double> &in_color);

  public:
    VisualizeDetectedObjects();

    void GetVisualizeMarkers(
        const DetectedObjectArray &in_objects,
        visualization_msgs::MarkerArray &visualization_markers);
};
