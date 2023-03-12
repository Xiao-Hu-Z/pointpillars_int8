#include "scatter_cuda.h"

ScatterCuda::ScatterCuda(cudaStream_t stream) { stream_ = stream; }

void ScatterCuda::DoScatterCuda(const int voxel_count, unsigned int *coords,
                                float *pfe_output, float *scattered_feature) {
    GPU_CHECK(scatter(coords, pfe_output, scattered_feature, GRID_X_SIZE,
                      GRID_Y_SIZE, voxel_count, NUM_THREADS, stream_));
}
