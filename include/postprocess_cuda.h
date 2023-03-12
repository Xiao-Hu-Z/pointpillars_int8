#pragma once

#include <memory>
#include "kernel.h"
#include "params.h"


class PostprocessCuda {
  private:
    Params params_;
    float *anchors_;
    float *anchor_bottom_heights_;
    int *object_counter_;
    cudaStream_t stream_ = 0;

    float *dev_score_thresholds_;
    float *dev_inv_score_thresholds_;
    std::vector<float> inv_score_thresholds_;
    std::vector<float> score_threshold_vecs;

  public:
    PostprocessCuda(cudaStream_t stream = 0);

    ~PostprocessCuda();

    int doPostprocessCuda(const float *cls_input,
                                        float *box_input,
                                        const float *dir_cls_input,
                                        float *bndbox_output) ;
};
