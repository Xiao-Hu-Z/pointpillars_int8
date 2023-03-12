#pragma once

#include "kernel.h"
#include "params.h"

class ScatterCuda {
  private:
    cudaStream_t stream_;
  public:
    ScatterCuda(cudaStream_t stream = 0);
    ~ScatterCuda(){};

    void DoScatterCuda(const int voxel_count, unsigned int *coords,
                       float *pfe_output, float *scattered_feature);
};
