#include "preprocess_points_cuda.h"

PreProcessCuda::PreProcessCuda(cudaStream_t stream) {
    stream_ = stream;

    unsigned int mask_size =
        GRID_Z_SIZE * GRID_Y_SIZE * GRID_X_SIZE * sizeof(unsigned int);
    unsigned int voxels_size = GRID_Z_SIZE * GRID_Y_SIZE * GRID_X_SIZE *
                               MAX_POINT_NUM_PER_VOXEL * POINT_FEATURES_NUM *
                               sizeof(float);
    GPU_CHECK(cudaMallocManaged((void **)&mask_, mask_size));
    GPU_CHECK(cudaMallocManaged((void **)&voxels_, voxels_size));

    GPU_CHECK(cudaMemsetAsync(mask_, 0, mask_size, stream_));
    GPU_CHECK(cudaMemsetAsync(voxels_, 0, voxels_size, stream_));
}

PreProcessCuda::~PreProcessCuda() {
    GPU_CHECK(cudaFree(mask_));
    GPU_CHECK(cudaFree(voxels_));
    return;
}

int PreProcessCuda::generateVoxels(float *points, size_t points_size,
                                   unsigned int *voxel_count,
                                   float *voxel_features,
                                   unsigned int *voxel_point_num,
                                   unsigned int *voxel_coords) {
    GPU_CHECK(generateVoxels_random_launch(
        points, points_size, X_MIN, X_MAX, Y_MIN, Y_MAX, Z_MIN, Z_MAX,
        VOXEL_SIZE_X, VOXEL_SIZE_Y, VOXEL_SIZE_Z, GRID_Y_SIZE, GRID_X_SIZE,
        mask_, voxels_, stream_));

    generateBaseFeatures_launch(mask_, voxels_, GRID_Y_SIZE, GRID_X_SIZE,
                                voxel_count, voxel_features, voxel_point_num,
                                voxel_coords, stream_);
    return 0;
}

int PreProcessCuda::generateFeatures(float *voxel_features,
                                     unsigned int *voxel_point_num,
                                     unsigned int *voxel_coords,
                                     unsigned int *voxel_count,
                                     float *features) {
    GPU_CHECK(generateFeatures_launch(voxel_features, voxel_point_num,
                                      voxel_coords, voxel_count, VOXEL_SIZE_X,
                                      VOXEL_SIZE_Y, VOXEL_SIZE_Z, X_MIN, Y_MIN,
                                      Z_MIN, features, stream_));

    return 0;
}

int points_to_voxel(const float *points, int point_size, float *voxels_feature,
                    int *coords, int *voxel_point_num, int *coor_to_voxelidx,
                    int max_point_num_per_voxel, int max_voxel_num,
                    int point_feature_num, int *grid_size, float *voxel_size,
                    float *point_range) {
    int voxel_num = 0;
    bool failed = false;
    int coor[3];
    int c;

    int voxelidx, num;
    for (int i = 0; i < point_size; ++i) {
        failed = false;
        for (int j = 0; j < 3; ++j) {
            c = floor((points[i * point_feature_num + j] - point_range[j]) /
                      voxel_size[j]);
            if ((c < 0 || c >= grid_size[j])) { // 超出坐标范围
                failed = true;
                break;
            }
            coor[2 - j] = c; // z,y,x
        }
        if (failed) // 该点超出范围
            continue;
        int index = coor[0] * grid_size[0] * grid_size[1] +
                    coor[1] * grid_size[1] + coor[2];
        voxelidx = coor_to_voxelidx[index];
        if (voxelidx == -1) {
            voxelidx = voxel_num; // voxel索引，代表第几个voxel
            if (voxel_num >= max_voxel_num)
                continue;
            voxel_num += 1;
            coor_to_voxelidx[index] = voxelidx;
            for (int k = 0; k < 3; ++k) {
                coords[voxelidx * 4 + k + 1] = coor[k]; // z,y,x voxel网格坐标
            }
        }
        num = voxel_point_num[voxelidx];     // voxel的点个数，初始为0
        if (num < max_point_num_per_voxel) { // 10
            // voxel_point_mask[voxelidx * max_point_num_per_voxel_ + num] = 1;
            for (int k = 0; k < point_feature_num; ++k) { // 特征维度遍历
                voxels_feature[voxelidx * max_point_num_per_voxel *
                                   point_feature_num +
                               num * point_feature_num + k] =
                    points[i * point_feature_num + k]; // voxel特征[60000,10,5]
            }
            voxel_point_num[voxelidx] += 1;
        }
    }
    for (int i = 0; i < voxel_num; ++i) {
        int coord_z = coords[i * 4 + 1];
        int coord_y = coords[i * 4 + 2];
        int coord_x = coords[i * 4 + 3];
        int id = coord_z * grid_size[0] * grid_size[1] +
                 coord_y * grid_size[1] + coord_x;
        coor_to_voxelidx[id] = -1; // 对存在的voxel网格坐标取 -1
    }
    return voxel_num;
}
