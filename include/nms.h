#pragma once

#include <kernel.h>
#include <vector>
#include <cmath>
#include <algorithm>

const float ThresHold = 1e-8;
/*
box_encodings: (B, N, 7 + C) or (N, 7 + C) [x, y, z, dx, dy, dz, heading or
*[cos, sin], ...] anchors: (B, N, 7 + C) or (N, 7 + C) [x, y, z, dx, dy, dz,
heading, ...]
*/
struct Bndbox {
    float x;
    float y;
    float z;
    float w;
    float l;
    float h;
    float rt;
    int id;
    float score;
    Bndbox(){};
    Bndbox(float x_, float y_, float z_, float w_, float l_, float h_,
           float rt_, int id_, float score_)
        : x(x_), y(y_), z(z_), w(w_), l(l_), h(h_), rt(rt_), id(id_),
          score(score_) {}
};

inline float cross(const float2 p1, const float2 p2, const float2 p0);

inline int check_box2d(const Bndbox box, const float2 p);

bool intersection(const float2 p1, const float2 p0, const float2 q1,const float2 q0, float2 &ans);

inline void rotate_around_center(const float2 &center, const float angle_cos, const float angle_sin, float2 &p);

inline float box_overlap(const Bndbox &box_a, const Bndbox &box_b);

int nms_cpu(std::vector<Bndbox> bndboxes, const float nms_thresh,std::vector<Bndbox> &nms_pred);