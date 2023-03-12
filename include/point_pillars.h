#pragma once

#include <algorithm>
#include <assert.h>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <stdio.h>
#include <string.h>
#include <string>
#include <vector>
#include <yaml-cpp/yaml.h>

#include "NvInfer.h"

#include "common.h"
#include "nms.h"
#include "params.h"
#include "perception_config.h"
#include "postprocess_cuda.h"
#include "preprocess_points_cuda.h"
#include "scatter_cuda.h"
#include "timer.h"

// Logger for TensorRT info/warning/errors
class Logger : public nvinfer1::ILogger {
  public:
    Logger(Severity severity = Severity::kWARNING)
        : reportableSeverity(severity) {}

    void log(Severity severity, char const *msg) noexcept {
        // suppress messages with severity enum value greater than the
        // reportable
        if (severity > reportableSeverity)
            return;

        switch (severity) {
        case Severity::kINTERNAL_ERROR:
            std::cerr << "INTERNAL_ERROR: ";
            break;
        case Severity::kERROR:
            std::cerr << "ERROR: ";
            break;
        case Severity::kWARNING:
            std::cerr << "WARNING: ";
            break;
        case Severity::kINFO:
            std::cerr << "INFO: ";
            break;
        default:
            std::cerr << "UNKNOWN: ";
            break;
        }
        std::cerr << msg << std::endl;
    }
    Severity reportableSeverity;
};

class TRT {
  private:
    Params params_;

    Logger gLogger_;
    nvinfer1::ICudaEngine *engine_ = nullptr;

    cudaStream_t stream_ = 0;

  public:
    TRT(std::string modelFile, cudaStream_t stream = 0);
    ~TRT(void);

    int doinfer(void **buffers);
};

class PointPillars {
  private:
    Params params_;
    Logger g_logger_;
    cudaStream_t stream_;

    bool preprocess_mode_;
    float score_threshold_;
    float nms_overlap_threshold_;
    std::string pfe_file_;
    std::string rpn_file_;

    float *voxel_features_ = nullptr;
    unsigned int *coords_ = nullptr;
    unsigned int *voxel_point_num_ = nullptr;
    unsigned int *voxel_count_ = nullptr;

    float *pfe_input_ = nullptr;
    float *pfe_output_ = nullptr;

    float *scattered_feature_ = nullptr;

    float *cls_output_ = nullptr;
    float *box_output_ = nullptr;
    float *dir_cls_output_ = nullptr;

    float *bndbox_output_;
    unsigned int bndbox_size_ = 0;
    std::vector<Bndbox> res_;

    // std::shared_ptr<TRT> trt_;
    std::shared_ptr<PreProcessCuda> pre_;
    std::shared_ptr<ScatterCuda> scatter_;
    std::shared_ptr<PostprocessCuda> post_;

    void deviceMemoryMalloc();

  public:
    void initTRT();
    void EngineToTRTModel(const std::string &engine_file,
                          nvinfer1::ICudaEngine **engine_ptr);

    std::string Name() const { return "PointPillarsDetector"; }

  public:
    nvinfer1::ICudaEngine *pfe_engine_;
    nvinfer1::ICudaEngine *rpn_engine_;
    nvinfer1::IExecutionContext *pfe_context_;
    nvinfer1::IExecutionContext *rpn_context_;

    PointPillars(cudaStream_t stream = 0);
    ~PointPillars();

    bool Init();
    void DoInference(float *points_data, unsigned points_size,
                     std::vector<Bndbox> &nms_pred);

    bool Detect(float *points_data, unsigned int points_size,
                std::vector<Bndbox> &objects);
};
