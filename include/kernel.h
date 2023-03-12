/*
 * @Author: xiaohu
 * @Date: 2022-10-17 15:07:03
 * @Last Modified by: xiaohu
 * @Last Modified time: 2022-10-17 15:07:03
 */

#pragma once

#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <string>
#include <vector>
#include <cmath>

#include "cuda_runtime_api.h"
#include "params.h"

const int WARPS_PER_BLOCK = 4; // four warp for one block
const int WARP_SIZE = 32; // one warp(32 threads) for one pillar
const int FEATURES_SIZE = 10; // features maps number depands on "params.h"

// need to be changed when num_threads_ is changed

#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))

#define GPU_CHECK(ans)                                                         \
    { GPUAssert((ans), __FILE__, __LINE__); }
inline void GPUAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
                line);
        if (abort)
            exit(code);
    }
};

cudaError_t generateVoxels_random_launch(float *points, size_t points_size,
        float min_x_range, float max_x_range,
        float min_y_range, float max_y_range,
        float min_z_range, float max_z_range,
        float pillar_x_size, float pillar_y_size, float pillar_z_size,
        int grid_y_size, int grid_x_size,
        unsigned int *mask, float *voxels,
        cudaStream_t stream = 0);

cudaError_t generateBaseFeatures_launch(unsigned int *mask, float *voxels,
        int grid_y_size, int grid_x_size,
        unsigned int *pillar_num,
        float *voxel_features,
        unsigned int *voxel_num,
        unsigned int *voxel_idxs,
        cudaStream_t stream = 0);

cudaError_t generateFeatures_launch(float* voxel_features,
    unsigned int *voxel_num,
    unsigned int *voxel_idxs,
    unsigned int *params,
    float voxel_x, float voxel_y, float voxel_z,
    float range_min_x, float range_min_y, float range_min_z,
    float* features,
    cudaStream_t stream = 0);

cudaError_t scatter(unsigned int *coords, float *pfe_output,
                    float *scattered_feature, const int grid_x_size,
                    const int grid_y_size, int voxel_count, int num_threads,
                    cudaStream_t stream);

cudaError_t postprocess_launch(
    const float *cls_input, float *box_input, const float *dir_cls_input,
    float *anchors, float *anchor_bottom_heights, float *bndbox_output,
    int *object_counter, const float min_x_range, const float max_x_range,
    const float min_y_range, const float max_y_range, const int feature_x_size,
    const int feature_y_size, const int num_anchors, const int num_classes,
    const int num_box_values, const float score_thresh, const float dir_offset,
    cudaStream_t stream);
