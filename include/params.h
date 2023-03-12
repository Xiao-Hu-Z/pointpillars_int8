/*
 * @Author: xiaohu 
 * @Date: 2022-08-26 07:22:04 
 * @Last Modified by:   xiaohu 
 * @Last Modified time: 2022-08-26 07:22:04 
 */
#pragma once

#include <vector>

#define POINT_FEATURES_NUM 4
#define VOXEL_SIZE_X 0.16
#define VOXEL_SIZE_Y 0.16
#define VOXEL_SIZE_Z 4.0
#define X_MIN 0.0
#define X_MAX 69.12
#define Y_MIN -39.68
#define Y_MAX 39.68
#define Z_MIN -3.0
#define Z_MAX 1.0
#define GRID_X_SIZE 432
#define GRID_Y_SIZE 496
#define GRID_Z_SIZE 1
#define NUM_CLASS 3
#define MAX_VOXEL_NUM 40000 // 003341.bin voxel个数大于20000
#define MAX_POINT_NUM 150000
#define MAX_POINT_NUM_PER_VOXEL 32
#define NUM_BEV_FEATURES 64
#define NUM_THREADS 64
#define NUM_INDS_FOR_SCAN 512
#define NUM_POINT_OUT_FEATURE 10
#define NUM_OUTPUT_BOX_FEATURE 7
#define NUM_ANCHOR_X_INDS 216
#define NUM_ANCHOR_Y_INDS 248  
#define NUM_ANCHOR_R_INDS 2  
#define NUM_ANCHOR (216 * 248 * 2 * 3)
#define PFE_OUT_SIZE (20000* 64)
#define RPN_INPUT_SIZE (64 * 496 * 432)
#define RPN_BOX_OUT_SIZE (216 * 248 * 2 * 3 * 7) // 2249856
#define RPN_CLS_OUT_SIZE (216 * 248 * 2 * 3 * 3) // 964224
#define RPN_DIR_OUT_SIZE (216 * 248 * 2 * 3 * 2) // 642816


class Params {
  public:
    static const int num_anchors = NUM_CLASS * 2;
    static const int len_per_anchor = 4;
    const float anchors[num_anchors * len_per_anchor] = {
        3.9, 1.6, 1.56, 0.0,  3.9,  1.6, 1.56, 1.57, 0.8,  0.6, 1.73, 0.0,
        0.8, 0.6, 1.73, 1.57, 1.76, 0.6, 1.73, 0.0,  1.76, 0.6, 1.73, 1.57,
    };

    const float anchor_bottom_heights[NUM_CLASS] = {
        -1.78,
        -0.6,
        -0.6,
    };

    const float score_thresh = 0.1;
    const float nms_thresh = 0.01;
    const int num_box_values = 7;
    const float dir_offset = 0.78539;

    Params() = default;
    ~Params() = default;
}; // class Params
