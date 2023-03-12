#include "postprocess_cuda.h"

PostprocessCuda::PostprocessCuda(cudaStream_t stream) {
    stream_ = stream;
    GPU_CHECK(cudaMalloc((void **)&anchors_, params_.num_anchors * params_.len_per_anchor *sizeof(float)));
    GPU_CHECK(cudaMalloc((void **)&anchor_bottom_heights_, NUM_CLASS * sizeof(float)));
    GPU_CHECK(cudaMalloc((void **)&object_counter_, sizeof(int)));

    GPU_CHECK(cudaMemcpyAsync(anchors_, params_.anchors,
                              params_.num_anchors * params_.len_per_anchor *
                                  sizeof(float),
                              cudaMemcpyDefault, stream_));
    GPU_CHECK(cudaMemcpyAsync(anchor_bottom_heights_, params_.anchor_bottom_heights,
                        NUM_CLASS * sizeof(float), cudaMemcpyDefault, stream_));
    GPU_CHECK(cudaMemsetAsync(object_counter_, 0, sizeof(int), stream_));
}

int PostprocessCuda::doPostprocessCuda(const float *cls_input, float *box_input,
                                       const float *dir_cls_input,
                                       float *bndbox_output) {
    GPU_CHECK(cudaMemsetAsync(object_counter_, 0, sizeof(int)));
    GPU_CHECK(postprocess_launch(
        cls_input, box_input, dir_cls_input, anchors_, anchor_bottom_heights_,
        bndbox_output, object_counter_, X_MIN, X_MAX, Y_MIN, Y_MAX,
        NUM_ANCHOR_X_INDS, NUM_ANCHOR_Y_INDS, params_.num_anchors, NUM_CLASS,
        params_.num_box_values, params_.score_thresh, params_.dir_offset,
        stream_));
    return 0;
}

PostprocessCuda::~PostprocessCuda() {
    GPU_CHECK(cudaFree(object_counter_));
    GPU_CHECK(cudaFree(anchors_));
    GPU_CHECK(cudaFree(anchor_bottom_heights_));
}
