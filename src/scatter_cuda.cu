#include "kernel.h"

__global__ void scatter_kernel(unsigned int *coords, float *pfe_output,
                               float *scattered_feature, const int grid_x_size,
                               const int grid_y_size) {
    int i_voxel = blockIdx.x;
    int i_feature = threadIdx.x;
    unsigned int x_ind = coords[i_voxel * 4 + 3];
    unsigned int y_ind = coords[i_voxel * 4 + 2];
    float feature = pfe_output[i_voxel * 64 + i_feature];
    scattered_feature[i_feature * grid_y_size * grid_x_size +
                      y_ind * grid_x_size + x_ind] = feature;
}

cudaError_t scatter(unsigned int *coords, float *pfe_output,
                    float *scattered_feature, const int grid_x_size,
                    const int grid_y_size, int voxel_count,
                    int num_threads,cudaStream_t stream) {
    scatter_kernel<<<voxel_count, num_threads, 0, stream>>>(
        coords, pfe_output, scattered_feature, grid_x_size, grid_y_size);
    cudaError_t err = cudaGetLastError();
    return err;
}
