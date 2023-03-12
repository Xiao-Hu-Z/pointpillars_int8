#include "point_pillars.h"
#include <iostream>

// TRT::~TRT(void) {
//     delete (engine_);
//     return;
// }

// TRT::TRT(std::string modelFile, cudaStream_t stream) : stream_(stream) {
//     std::fstream trtCache(modelFile, std::ifstream::in);
//     std::cout << "load TRT cache." << std::endl;
//     char *data;
//     unsigned int length;

//     // get length of file:
//     trtCache.seekg(0, trtCache.end);
//     length = trtCache.tellg();
//     trtCache.seekg(0, trtCache.beg);

//     data = (char *)malloc(length);
//     if (data == NULL) {
//         std::cout << "Can't malloc data.\n";
//         exit(-1);
//     }

//     trtCache.read(data, length);
//     // create context
//     auto runtime = nvinfer1::createInferRuntime(gLogger_);

//     if (runtime == nullptr) {
//         std::cout << "load TRT cache0." << std::endl;
//         std::cerr << ": runtime null!" << std::endl;
//         exit(-1);
//     }
//     // plugin_ = nvonnxparser::createPluginFactory(gLogger_);
//     engine_ = (runtime->deserializeCudaEngine(data, length, 0));
//     if (engine_ == nullptr) {
//         std::cerr << ": engine null!" << std::endl;
//         exit(-1);
//     }

//     free(data);
//     trtCache.close();

//     context_ = engine_->createExecutionContext();
//     return;
// }

// int TRT::doinfer(void **buffers) {
//     int status;

//     status = context_->enqueueV2(buffers, stream_, nullptr);

//     if (!status) {
//         return -1;
//     }

//     return 0;
// }

bool PointPillars::Init() {
    YAML::Node config;

    try {
        config = YAML::LoadFile("../config/point_pillars_detection.yaml");
    } catch (YAML::Exception &ex) {
        std::cerr << "point_pillars_cuda read yaml failed";
        return false;
    }

    std::string root_path = ROOT_PATH;
    pfe_file_ = root_path + "/" + config["model_file1"].as<std::string>();
    rpn_file_ = root_path + "/" + config["model_file2"].as<std::string>();
    preprocess_mode_ = config["preprocess_mode"].as<bool>();
    score_threshold_ = config["score_threshold"].as<float>();
    nms_overlap_threshold_ = config["nms_overlap_threshold"].as<float>();

    pre_.reset(new PreProcessCuda(stream_));
    scatter_.reset(new ScatterCuda(stream_));
    post_.reset(new PostprocessCuda(stream_));

    deviceMemoryMalloc();
    initTRT();
    return true;
}

PointPillars::PointPillars(cudaStream_t stream) : stream_(stream) {}

PointPillars::~PointPillars() {
    pre_.reset();
    scatter_.reset();
    post_.reset();
    GPU_CHECK(cudaFree(voxel_features_));
    GPU_CHECK(cudaFree(coords_));
    GPU_CHECK(cudaFree(voxel_point_num_));
    GPU_CHECK(cudaFree(voxel_count_));

    GPU_CHECK(cudaFree(pfe_input_));
    GPU_CHECK(cudaFree(pfe_output_));

    GPU_CHECK(cudaFree(scattered_feature_));

    GPU_CHECK(cudaFree(cls_output_));
    GPU_CHECK(cudaFree(box_output_));
    GPU_CHECK(cudaFree(dir_cls_output_));

    GPU_CHECK(cudaFree(bndbox_output_));
}

void PointPillars::deviceMemoryMalloc() {
    // generate feature
    GPU_CHECK(cudaMallocManaged((void **)&voxel_features_,
                                MAX_VOXEL_NUM * MAX_POINT_NUM_PER_VOXEL * 4 *
                                    sizeof(float)));
    GPU_CHECK(cudaMallocManaged((void **)&coords_,
                                MAX_VOXEL_NUM * 4 * sizeof(unsigned int)));
    GPU_CHECK(cudaMallocManaged((void **)&voxel_point_num_,
                                MAX_VOXEL_NUM * sizeof(unsigned int)));
    GPU_CHECK(cudaMallocManaged((void **)&voxel_count_, sizeof(unsigned int)));

    // pfe
    GPU_CHECK(cudaMallocManaged((void **)&pfe_input_,
                                MAX_VOXEL_NUM * MAX_POINT_NUM_PER_VOXEL *
                                    NUM_POINT_OUT_FEATURE * sizeof(float)));
    GPU_CHECK(
        cudaMallocManaged((void **)&pfe_output_,
                          MAX_VOXEL_NUM * NUM_BEV_FEATURES * sizeof(float)));

    // scatter
    GPU_CHECK(cudaMallocManaged((void **)&scattered_feature_,
                                RPN_INPUT_SIZE * sizeof(float)));

    // rpn
    GPU_CHECK(cudaMallocManaged((void **)&cls_output_,
                                RPN_CLS_OUT_SIZE * sizeof(float)));
    GPU_CHECK(cudaMallocManaged((void **)&box_output_,
                                RPN_BOX_OUT_SIZE * sizeof(float)));
    GPU_CHECK(cudaMallocManaged((void **)&dir_cls_output_,
                                RPN_DIR_OUT_SIZE * sizeof(float)));

    // head
    bndbox_size_ = (GRID_X_SIZE * GRID_Y_SIZE * params_.num_anchors * 9 + 1) *
                   sizeof(float);
    GPU_CHECK(cudaMallocManaged((void **)&bndbox_output_, bndbox_size_));

    res_.reserve(100);
}

void PointPillars::initTRT() {
    EngineToTRTModel(pfe_file_, &pfe_engine_);
    EngineToTRTModel(rpn_file_, &rpn_engine_);
    if (pfe_engine_ == nullptr || rpn_engine_ == nullptr) {
        std::cerr << "Failed to load trt file.";
    }

    // 生成 Context 并通过Context进行推理
    pfe_context_ = pfe_engine_->createExecutionContext();
    rpn_context_ = rpn_engine_->createExecutionContext();
    
    if (pfe_context_ == nullptr || rpn_context_ == nullptr) {
        std::cerr << "Failed to create TensorRT Execution Context.";
    }
}

void PointPillars::EngineToTRTModel(const std::string &engine_file,
                                    nvinfer1::ICudaEngine **engine_ptr) {

    int verbosity = static_cast<int>(nvinfer1::ILogger::Severity::kWARNING);

    std::fstream trtCache(engine_file, std::ifstream::in);
    // get length of file
    trtCache.seekg(0, trtCache.end);
    unsigned int length = trtCache.tellg();
    trtCache.seekg(0, trtCache.beg);

    char *data = (char *)malloc(length);
    trtCache.read(data, length);
    // create context
    auto runtime = nvinfer1::createInferRuntime(g_logger_);
    if (runtime == nullptr) {
        std::cerr << "load TRT cache0." << std::endl;
    }
    *engine_ptr = (runtime->deserializeCudaEngine(data, length, 0));
    free(data);
    trtCache.close();
}

void PointPillars::DoInference(float *points_data, unsigned int points_size,
                               std::vector<Bndbox> &nms_pred) {
    // 必须放在这里
    GPU_CHECK(cudaMemsetAsync(
        voxel_features_, 0,
        MAX_VOXEL_NUM * MAX_POINT_NUM_PER_VOXEL * 4 * sizeof(float), stream_));
    GPU_CHECK(cudaMemsetAsync(
        coords_, 0, MAX_VOXEL_NUM * 4 * sizeof(unsigned int), stream_));
    GPU_CHECK(cudaMemsetAsync(voxel_point_num_, 0,
                              MAX_VOXEL_NUM * sizeof(unsigned int), stream_));
    GPU_CHECK(cudaMemsetAsync(voxel_count_, 0, sizeof(unsigned int), stream_));
    GPU_CHECK(cudaMemsetAsync(pfe_input_, 0,
                              MAX_VOXEL_NUM * MAX_POINT_NUM_PER_VOXEL *
                                  NUM_POINT_OUT_FEATURE * sizeof(float),
                              stream_));

    pre_->generateVoxels((float *)points_data, points_size, voxel_count_,
                         voxel_features_, voxel_point_num_, coords_);
    // GPU_CHECK(cudaStreamSynchronize(stream_));
    // printf("voxel_count_:%d \n", voxel_count_[0]);
    pre_->generateFeatures(voxel_features_, voxel_point_num_, coords_,
                           voxel_count_, pfe_input_);
    // 必须，不然后面voxel_count_[0]可能为0
    GPU_CHECK(cudaStreamSynchronize(stream_));

    GPU_CHECK(cudaMemsetAsync(pfe_output_, 0,
                              MAX_VOXEL_NUM * NUM_BEV_FEATURES * sizeof(float),
                              stream_));
    void *pfe_buffers_[] = {pfe_input_, pfe_output_};
    pfe_context_->enqueueV2(pfe_buffers_, stream_, nullptr);

    GPU_CHECK(cudaMemsetAsync(scattered_feature_, 0,
                              RPN_INPUT_SIZE * sizeof(float), stream_));
    scatter_->DoScatterCuda(voxel_count_[0], coords_,
                            reinterpret_cast<float *>(pfe_output_),
                            scattered_feature_);

    GPU_CHECK(cudaMemsetAsync(cls_output_, 0, RPN_CLS_OUT_SIZE * sizeof(float),
                              stream_));
    GPU_CHECK(cudaMemsetAsync(box_output_, 0, RPN_BOX_OUT_SIZE * sizeof(float),
                              stream_));
    GPU_CHECK(cudaMemsetAsync(dir_cls_output_, 0,
                              RPN_DIR_OUT_SIZE * sizeof(float), stream_));

    void *rpn_buffers_[] = {scattered_feature_, cls_output_, box_output_,
                            dir_cls_output_};
    rpn_context_->enqueueV2(rpn_buffers_, stream_, nullptr);

    GPU_CHECK(cudaMemsetAsync(bndbox_output_, 0, bndbox_size_, stream_));

    post_->doPostprocessCuda(cls_output_, box_output_, dir_cls_output_,
                             bndbox_output_);

    GPU_CHECK(cudaDeviceSynchronize());
    float obj_count = bndbox_output_[0];

    int num_obj = static_cast<int>(obj_count);
    auto output = bndbox_output_ + 1;
    for (int i = 0; i < num_obj; i++) {
        auto Bb =
            Bndbox(output[i * 9], output[i * 9 + 1], output[i * 9 + 2],
                   output[i * 9 + 3], output[i * 9 + 4], output[i * 9 + 5],
                   output[i * 9 + 6], static_cast<int>(output[i * 9 + 7]),
                   output[i * 9 + 8]);
        res_.push_back(Bb);
    }

    nms_cpu(res_, params_.nms_thresh, nms_pred);
    res_.clear();
}

bool PointPillars::Detect(float *points_data, unsigned points_size,
                          std::vector<Bndbox> &objects) {
    std::vector<Bndbox> nms_pred;
    DoInference(points_data, points_size, objects);

    return true;
}