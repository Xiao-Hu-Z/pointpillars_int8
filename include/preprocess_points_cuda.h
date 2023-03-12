#pragma once

#include "common.h"
#include "kernel.h"
#include "params.h"

class PreProcessCuda
{
private:
  unsigned int *mask_;
  float *voxels_;
  float *params_cuda_;
  cudaStream_t stream_ = 0;

public:
  PreProcessCuda(cudaStream_t stream_ = 0);
  ~PreProcessCuda();

  void PreprocessCPU(const float *in_points_array,
                                         int in_num_points,
                                         float *voxel_features, int *coords,
                                         int *host_voxel_count);

  // points cloud -> voxels (BEV) -> feature*4
  int generateVoxels(float *points, size_t points_size,
                     unsigned int *pillar_num, float *voxel_features,
                     unsigned int *voxel_num, unsigned int *voxel_idxs);

  // feature*4 -> feature * 10
  int generateFeatures(float *voxel_features, unsigned int *voxel_num,
                       unsigned int *voxel_idxs, unsigned int *params,
                       float *features);
};
