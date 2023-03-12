#include "visualize.h"

VisualizeDetectedObjects::VisualizeDetectedObjects()
    : arrow_height_(0.5), label_height_(1.0) {
    object_speed_threshold_ = 0.1;
    arrow_speed_threshold_ = 0.25;
    marker_display_duration_ = 0.2;

    std::vector<double> label_color{255., 255., 255., 1.0};
    label_color_ = ParseColor(label_color);

    std::vector<double> arrow_color{0., 255., 0., 0.8};
    arrow_color_ = ParseColor(arrow_color);

    std::vector<double> hull_color{51., 204., 51., 0.8};
    hull_color_ = ParseColor(hull_color);

    std::vector<double> box_color{255.0f, 0.0f, 0.0f, 1.0};
    box_color_ = ParseColor(box_color);

    std::vector<double> cube_color{51., 128., 204., 0.8};
    cube_color_ = ParseColor(cube_color);

    std::vector<double> centroid_color{77., 121., 255., 0.8};
    centroid_color_ = ParseColor(centroid_color);
}

float VisualizeDetectedObjects::CheckColor(double value) {
    float final_value;
    if (value > 255.)
        final_value = 1.f;
    else if (value < 0)
        final_value = 0.f;
    else
        final_value = value / 255.f;
    return final_value;
}

float VisualizeDetectedObjects::CheckAlpha(double value) {
    float final_value;
    if (value > 1.)
        final_value = 1.f;
    else if (value < 0.1)
        final_value = 0.1f;
    else
        final_value = value;
    return final_value;
}

std_msgs::ColorRGBA
VisualizeDetectedObjects::ParseColor(const std::vector<double> &in_color) {
    std_msgs::ColorRGBA color;
    float r, g, b, a;
    if (in_color.size() == 4) // r,g,b,a
    {
        color.r = CheckColor(in_color[0]);
        color.g = CheckColor(in_color[1]);
        color.b = CheckColor(in_color[2]);
        color.a = CheckAlpha(in_color[3]);
    }
    return color;
}

visualization_msgs::MarkerArray VisualizeDetectedObjects::ObjectsToCentroids(
    const DetectedObjectArray &in_objects) {
    visualization_msgs::MarkerArray centroid_markers;
    for (auto const &object : in_objects.objects) {
        visualization_msgs::Marker centroid_marker;
        centroid_marker.lifetime = ros::Duration(marker_display_duration_);
        centroid_marker.header = in_objects.header;
        centroid_marker.type = visualization_msgs::Marker::SPHERE;
        centroid_marker.action = visualization_msgs::Marker::ADD;
        centroid_marker.pose = object.pose;
        centroid_marker.ns = "centroid_markers";

        centroid_marker.scale.x = 0.5;
        centroid_marker.scale.y = 0.5;
        centroid_marker.scale.z = 0.5;

        if (object.color.a == 0) {
            centroid_marker.color = centroid_color_;
        } else {
            centroid_marker.color = object.color;
        }
        centroid_marker.id = marker_id_++;
        centroid_markers.markers.push_back(centroid_marker);
    }
    return centroid_markers;
}

visualization_msgs::MarkerArray VisualizeDetectedObjects::ObjectsToCubes(
    const DetectedObjectArray &in_objects) {
    visualization_msgs::MarkerArray object_boxes;

    for (auto const &object : in_objects.objects) {
        if ((object.dimensions.x + object.dimensions.y + object.dimensions.z) <
            object_max_linear_size_) {
            visualization_msgs::Marker cube;
            cube.lifetime = ros::Duration(marker_display_duration_);
            cube.header = in_objects.header;
            cube.type = visualization_msgs::Marker::CUBE;
            cube.action = visualization_msgs::Marker::ADD;
            cube.ns = "cube_markers";
            cube.id = marker_id_++;
            cube.scale = object.dimensions;
            cube.pose.position = object.pose.position;

            cube.pose.orientation = object.pose.orientation;

            if (object.color.a == 0) {
                cube.color = cube_color_;
            } else {
                cube.color = object.color;
            }

            object_boxes.markers.push_back(cube);
        }
    }
    return object_boxes;
}

visualization_msgs::MarkerArray VisualizeDetectedObjects::ObjectsToBoxes(
    const DetectedObjectArray &in_objects) {
    visualization_msgs::MarkerArray object_boxes;
    for (auto const &object : in_objects.objects) {
        if ((object.dimensions.x + object.dimensions.y + object.dimensions.z) <
            object_max_linear_size_) {
            visualization_msgs::Marker box;
            box.lifetime = ros::Duration(
                marker_display_duration_); // marker_display_duration_
            box.header = in_objects.header;
            box.type = visualization_msgs::Marker::LINE_STRIP;
            box.action = visualization_msgs::Marker::ADD;
            box.ns = "box_markers";
            box.id = marker_id_++;
            box.scale.x = 0.15;
            box.color = box_color_;

            float length = object.dimensions.x;
            float width = object.dimensions.y;
            float height = object.dimensions.z;
            geometry_msgs::Point p[8];
            p[0].x = length / 2, p[0].y = width / 2, p[0].z = height / 2;
            p[1].x = length / 2, p[1].y = -width / 2, p[1].z = height / 2;
            p[2].x = length / 2, p[2].y = -width / 2, p[2].z = -height / 2;
            p[3].x = length / 2, p[3].y = width / 2, p[3].z = -height / 2;
            p[4].x = -length / 2, p[4].y = width / 2, p[4].z = -height / 2;
            p[5].x = -length / 2, p[5].y = -width / 2, p[5].z = -height / 2;
            p[6].x = -length / 2, p[6].y = -width / 2, p[6].z = height / 2;
            p[7].x = -length / 2, p[7].y = width / 2, p[7].z = height / 2;

            float x = object.pose.orientation.x;
            float y = object.pose.orientation.y;
            float z = object.pose.orientation.z;
            float w = object.pose.orientation.w;
            float RT[12] = {
                1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * z * w,
                2 * x * z + 2 * y * w,     float(object.pose.position.x),
                2 * x * y + 2 * z * w,     1 - 2 * x * x - 2 * z * z,
                2 * y * z - 2 * x * w,     float(object.pose.position.y),
                2 * x * z - 2 * y * w,     2 * y * z + 2 * x * w,
                1 - 2 * x * x - 2 * y * y, float(object.pose.position.z)};
            for (int i = 0; i < 8; i++) {
                // 待调试
                // p[i].x =
                //     RT[0] * p[i].x + RT[1] * p[i].y + RT[2] * p[i].z + RT[3];
                // p[i].y =
                //     RT[4] * p[i].x + RT[5] * p[i].y + RT[6] * p[i].z + RT[7];
                // p[i].z =
                //     RT[8] * p[i].x + RT[9] * p[i].y + RT[10] * p[i].z +
                //     RT[11];
                float x =
                    RT[0] * p[i].x + RT[1] * p[i].y + RT[2] * p[i].z + RT[3];
                float y =
                    RT[4] * p[i].x + RT[5] * p[i].y + RT[6] * p[i].z + RT[7];
                float z =
                    RT[8] * p[i].x + RT[9] * p[i].y + RT[10] * p[i].z + RT[11];
                p[i].x = x;
                p[i].y = y;
                p[i].z = z;
                // box.points.push_back(p[i]);
            }

            for (size_t i = 0; i < 8; i++) {
                box.points.push_back(p[i]);
            }
            box.points.push_back(p[0]);
            box.points.push_back(p[3]);
            box.points.push_back(p[2]);
            box.points.push_back(p[5]);
            box.points.push_back(p[6]);
            box.points.push_back(p[1]);
            box.points.push_back(p[0]);
            box.points.push_back(p[7]);
            box.points.push_back(p[4]);

            object_boxes.markers.push_back(box);
        }
    }
    return object_boxes;
}

visualization_msgs::MarkerArray VisualizeDetectedObjects::ObjectsToHulls(
    const DetectedObjectArray &in_objects) {
    visualization_msgs::MarkerArray polygon_hulls;

    for (auto const &object : in_objects.objects) {
        if (!object.convex_hull.polygon.points.empty()) {
            visualization_msgs::Marker hull;
            hull.lifetime = ros::Duration(marker_display_duration_);
            hull.header = in_objects.header;
            hull.type = visualization_msgs::Marker::LINE_STRIP;
            hull.action = visualization_msgs::Marker::ADD;
            hull.ns = "hull_markers";
            hull.id = marker_id_++;
            hull.scale.x = 0.15;

            std::vector<geometry_msgs::Point32> points =
                object.convex_hull.polygon.points;
            size_t len = points.size() / 2;
            geometry_msgs::Point pre_bottom_points;
            geometry_msgs::Point pre_top_point;
            geometry_msgs::Point cur_bottom_point;
            geometry_msgs::Point cur_top_point;
            // pre bottom
            pre_bottom_points.x = points[2 * (len - 1)].x;
            pre_bottom_points.y = points[2 * (len - 1)].y;
            pre_bottom_points.z = points[2 * (len - 1)].z;
            // pre top
            pre_top_point.x = points[2 * (len - 1) + 1].x;
            pre_top_point.y = points[2 * (len - 1) + 1].y;
            pre_top_point.z = points[2 * (len - 1) + 1].z;
            // current bottom
            cur_bottom_point.x = points[0].x;
            cur_bottom_point.y = points[0].y;
            cur_bottom_point.z = points[0].z;
            // current top
            cur_top_point.x = points[1].x;
            cur_top_point.y = points[1].y;
            cur_top_point.z = points[1].z;
            hull.points.push_back(cur_bottom_point);
            hull.points.push_back(pre_bottom_points);
            hull.points.push_back(pre_top_point);
            hull.points.push_back(cur_top_point);
            hull.points.push_back(cur_bottom_point);

            for (size_t i = 1; i < len; i++) {
                geometry_msgs::Point tmp_pre_bottom_points;
                geometry_msgs::Point tmp_pre_top_point;
                geometry_msgs::Point tmp_cur_bottom_point;
                geometry_msgs::Point tmp_cur_top_point;
                // pre bottom
                tmp_pre_bottom_points = hull.points.at(hull.points.size() - 1);
                // pre top
                tmp_pre_top_point = hull.points.at(hull.points.size() - 2);
                // current bottom
                tmp_cur_bottom_point.x = points[2 * i].x;
                tmp_cur_bottom_point.y = points[2 * i].y;
                tmp_cur_bottom_point.z = points[2 * i].z;
                // current top
                tmp_cur_top_point.x = points[2 * i + 1].x;
                tmp_cur_top_point.y = points[2 * i + 1].y;
                tmp_cur_top_point.z = points[2 * i + 1].z;

                hull.points.push_back(tmp_cur_bottom_point);
                hull.points.push_back(tmp_pre_bottom_points);
                hull.points.push_back(tmp_pre_top_point);
                hull.points.push_back(tmp_cur_top_point);
                hull.points.push_back(tmp_cur_bottom_point);
            }

            if (object.color.a == 0) {
                hull.color = hull_color_;
            } else {
                hull.color = object.color;
            }

            polygon_hulls.markers.push_back(hull);
        }
    }
    return polygon_hulls;
}

visualization_msgs::MarkerArray VisualizeDetectedObjects::ObjectsToArrows(
    const DetectedObjectArray &in_objects) {
    visualization_msgs::MarkerArray arrow_markers;
    for (auto const &object : in_objects.objects) {
        double velocity = object.velocity.linear.x;

        if (abs(velocity) >= arrow_speed_threshold_) {
            visualization_msgs::Marker arrow_marker;
            arrow_marker.lifetime = ros::Duration(marker_display_duration_);

            tf::Quaternion q(
                object.pose.orientation.x, object.pose.orientation.y,
                object.pose.orientation.z, object.pose.orientation.w);
            double roll, pitch, yaw;

            tf::Matrix3x3(q).getRPY(roll, pitch, yaw);

            // in the case motion model fit opposite direction
            if (velocity < -0.1) {
                yaw += M_PI;
                // normalize angle
                while (yaw > M_PI)
                    yaw -= 2. * M_PI;
                while (yaw < -M_PI)
                    yaw += 2. * M_PI;
            }

            tf::Matrix3x3 obs_mat;
            tf::Quaternion q_tf;

            obs_mat.setEulerYPR(yaw, 0, 0); // yaw, pitch, roll
            obs_mat.getRotation(q_tf);

            arrow_marker.header = in_objects.header;
            arrow_marker.ns = "arrow_markers";
            arrow_marker.action = visualization_msgs::Marker::ADD;
            arrow_marker.type = visualization_msgs::Marker::ARROW;

            // green
            if (object.color.a == 0) {
                arrow_marker.color = arrow_color_;
            } else {
                arrow_marker.color = object.color;
            }
            arrow_marker.id = marker_id_++;

            // Set the pose of the marker.  This is a full 6DOF pose
            // relative to the frame/time specified in the header
            arrow_marker.pose.position.x = object.pose.position.x;
            arrow_marker.pose.position.y = object.pose.position.y;
            arrow_marker.pose.position.z = arrow_height_;

            arrow_marker.pose.orientation.x = q_tf.getX();
            arrow_marker.pose.orientation.y = q_tf.getY();
            arrow_marker.pose.orientation.z = q_tf.getZ();
            arrow_marker.pose.orientation.w = q_tf.getW();

            // Set the scale of the arrow -- 1x1x1 here means 1m on a side
            arrow_marker.scale.x = 3;
            arrow_marker.scale.y = 0.1;
            arrow_marker.scale.z = 0.1;

            arrow_markers.markers.push_back(arrow_marker);
        }
    }
    return arrow_markers;
}

visualization_msgs::MarkerArray VisualizeDetectedObjects::ObjectsToLabels(
    const DetectedObjectArray &in_objects) {
    visualization_msgs::MarkerArray label_markers;
    for (auto const &object : in_objects.objects) {
        visualization_msgs::Marker label_marker;
        label_marker.lifetime = ros::Duration(marker_display_duration_);
        label_marker.header = in_objects.header;
        label_marker.ns = "label_markers";
        label_marker.action = visualization_msgs::Marker::ADD;
        label_marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
        label_marker.scale.x = 1.5;
        label_marker.scale.y = 1.5;
        label_marker.scale.z = 1.5;

        label_marker.color = label_color_;

        label_marker.id = marker_id_++;

        if (!object.label.empty() && object.label != "unknown")
            label_marker.text = object.label + " ";

        std::stringstream distance_stream;
        distance_stream
            << std::fixed << std::setprecision(1)
            << sqrt((object.pose.position.x * object.pose.position.x) +
                    (object.pose.position.y * object.pose.position.y));
        std::string distance_str = distance_stream.str() + " m";
        label_marker.text += distance_str;

        if (object.velocity_reliable) {
            double velocity = object.velocity.linear.x;
            if (velocity < -0.1) {
                velocity *= -1;
            }

            if (abs(velocity) < object_speed_threshold_) {
                velocity = 0.0;
            }

            tf::Quaternion q(
                object.pose.orientation.x, object.pose.orientation.y,
                object.pose.orientation.z, object.pose.orientation.w);

            double roll, pitch, yaw;
            tf::Matrix3x3(q).getRPY(roll, pitch, yaw);

            // convert m/s to km/h
            std::stringstream kmh_velocity_stream;
            kmh_velocity_stream << std::fixed << std::setprecision(1)
                                << (velocity * 3.6);
            std::string text = "\n<" + std::to_string(object.id) + "> " +
                               kmh_velocity_stream.str() + " km/h";
            label_marker.text += text;
        }

        label_marker.pose.position.x = object.pose.position.x;
        label_marker.pose.position.y = object.pose.position.y;
        label_marker.pose.position.z = label_height_;
        label_marker.scale.z = 1.0;
        if (!label_marker.text.empty())
            label_markers.markers.push_back(label_marker);
    }

    return label_markers;
}

bool VisualizeDetectedObjects::IsObjectValid(const DetectedObject &in_object) {
    if (!in_object.valid || std::isnan(in_object.pose.orientation.x) ||
        std::isnan(in_object.pose.orientation.y) ||
        std::isnan(in_object.pose.orientation.z) ||
        std::isnan(in_object.pose.orientation.w) ||
        std::isnan(in_object.pose.position.x) ||
        std::isnan(in_object.pose.position.y) ||
        std::isnan(in_object.pose.position.z) ||
        (in_object.pose.position.x == 0.) ||
        (in_object.pose.position.y == 0.) || (in_object.dimensions.x <= 0.) ||
        (in_object.dimensions.y <= 0.) || (in_object.dimensions.z <= 0.)) {
        return false;
    }
    return true;
}

void VisualizeDetectedObjects::GetVisualizeMarkers(
    const DetectedObjectArray &in_objects,
    visualization_msgs::MarkerArray &visualization_markers) {
    visualization_msgs::MarkerArray label_markers, arrow_markers,
        centroid_markers, polygon_hulls, bounding_cubes, bounding_boxes,
        object_models, polygon_freespace;
    marker_id_ = 0;

    label_markers = ObjectsToLabels(in_objects);
    arrow_markers = ObjectsToArrows(in_objects);
    polygon_hulls = ObjectsToHulls(in_objects);
    bounding_cubes = ObjectsToCubes(in_objects);
    bounding_boxes = ObjectsToBoxes(in_objects);
    centroid_markers = ObjectsToCentroids(in_objects);

    visualization_markers.markers.insert(visualization_markers.markers.end(),
                                         label_markers.markers.begin(),
                                         label_markers.markers.end());
    visualization_markers.markers.insert(visualization_markers.markers.end(),
                                         arrow_markers.markers.begin(),
                                         arrow_markers.markers.end());
    visualization_markers.markers.insert(visualization_markers.markers.end(),
                                         polygon_hulls.markers.begin(),
                                         polygon_hulls.markers.end());
    visualization_markers.markers.insert(visualization_markers.markers.end(),
                                         bounding_cubes.markers.begin(),
                                         bounding_cubes.markers.end());
    visualization_markers.markers.insert(visualization_markers.markers.end(),
                                         bounding_boxes.markers.begin(),
                                         bounding_boxes.markers.end());
    visualization_markers.markers.insert(visualization_markers.markers.end(),
                                         centroid_markers.markers.begin(),
                                         centroid_markers.markers.end());
}