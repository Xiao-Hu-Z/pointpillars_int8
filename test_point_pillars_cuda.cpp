#include <chrono>
#include <fstream>
#include <iostream>
#include <string>

#include <pcl/common/io.h>
#include <pcl/conversions.h>

#include "include/point_pillars.h"

// 注意数据集的路径正确
std::string data_package = "/kitti/training/velodyne/";
std::string save_dir = "../eval/kitti/object/pre_kitti/fp32/";


void Getinfo(void) {
    cudaDeviceProp prop;

    int count = 0;
    cudaGetDeviceCount(&count);
    printf("\nGPU has cuda devices: %d\n", count);
    for (int i = 0; i < count; ++i) {
        cudaGetDeviceProperties(&prop, i);
        printf("----device id: %d info----\n", i);
        printf("  GPU : %s \n", prop.name);
        printf("  Capbility: %d.%d\n", prop.major, prop.minor);
        printf("  Global memory: %luMB\n", prop.totalGlobalMem >> 20);
        printf("  Const memory: %luKB\n", prop.totalConstMem >> 10);
        printf("  SM in a block: %luKB\n", prop.sharedMemPerBlock >> 10);
        printf("  warp size: %d\n", prop.warpSize);
        printf("  threads in a block: %d\n", prop.maxThreadsPerBlock);
        printf("  block dim: (%d,%d,%d)\n", prop.maxThreadsDim[0],
               prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("  grid dim: (%d,%d,%d)\n", prop.maxGridSize[0],
               prop.maxGridSize[1], prop.maxGridSize[2]);
    }
    printf("\n");
}

int loadData(const char *file, void **data, unsigned int *length) {
    std::fstream dataFile(file, std::ifstream::in);

    if (!dataFile.is_open()) {
        std::cout << "Can't open files: " << file << std::endl;
        return -1;
    }

    // get length of file:
    unsigned int len = 0;
    dataFile.seekg(0, dataFile.end);
    len = dataFile.tellg();
    dataFile.seekg(0, dataFile.beg);

    // allocate memory:
    char *buffer = new char[len];
    if (buffer == NULL) {
        std::cout << "Can't malloc buffer." << std::endl;
        dataFile.close();
        exit(-1);
    }

    // read data as a block:
    dataFile.read(buffer, len);
    dataFile.close();

    *data = (void *)buffer;
    *length = len;
    return 0;
}

void readBinFile(std::string &filename, void *&bufPtr, int &pointNum,
                 int pointDim) {
    // open the file:
    std::streampos fileSize;
    std::ifstream file(filename, std::ios::binary);

    if (!file) {
        std::cerr << "[Error] Open file " << filename << " failed" << std::endl;
        return;
    }

    file.seekg(0, std::ios::end);
    fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    bufPtr = malloc(fileSize);
    if (bufPtr == nullptr) {
        std::cerr << "[Error] Malloc Memory Failed! Size: " << fileSize
                  << std::endl;
        return;
    }
    // read the data:
    file.read((char *)bufPtr, fileSize);
    file.close();

    pointNum = fileSize / sizeof(float) / pointDim;
    if (fileSize / sizeof(float) % pointDim != 0) {
        std::cerr << "[Error] File Size Error! " << fileSize << std::endl;
    }
    std::cout << "[INFO] pointNum : " << pointNum << std::endl;
}

void SaveBoxPred(std::vector<Bndbox> boxes, std::string file_name) {
    std::ofstream ofs;
    // 打开文件，既可读取其内容，也可向其写入数据。如果文件本来就存在，则打开时清除原来的内容；如果文件不存在，则新建该文件
    ofs.open(file_name, std::ios::in | std::ios::out | std::ios::trunc);
    if (ofs.is_open()) {
        for (const auto box : boxes) {
            ofs << box.x << " ";
            ofs << box.y << " ";
            ofs << box.z << " ";
            ofs << box.w << " ";
            ofs << box.l << " ";
            ofs << box.h << " ";
            ofs << box.rt << " ";
            ofs << box.id << " ";
            ofs << box.score << " ";
            ofs << "\n";
        }
    } else {
        std::cerr << "Output file cannot be opened!" << std::endl;
    }
    ofs.close();
    // std::cout << "Saved prediction in: " << file_name << std::endl;
    return;
};

int main(int argc, char **argv) {
    Getinfo();
    cudaEvent_t start, stop;
    cudaStream_t stream = NULL;
    GPU_CHECK(cudaEventCreate(&start));
    GPU_CHECK(cudaEventCreate(&stop));
    GPU_CHECK(cudaStreamCreate(&stream));

    float elapsedTime = 0.0f;

    PointPillars pointpillars_ptr_(stream);
    pointpillars_ptr_.Init();
    std::ifstream in_file("../eval/val.txt");
    if (!in_file) {
        std::cout << "data_package is not exist  " << std::endl;
        return false;
    }

    std::string name;
    std::vector<Bndbox> objects;
    // reserve是容器预留空间，但在空间内不真正创建元素对象
    objects.reserve(100);

    float *points_data = nullptr;
    GPU_CHECK(cudaMallocManaged((void **)&points_data,
                                MAX_POINT_NUM * 4 * sizeof(float)));

    int count = 0;
    double sum_time = 0.0;
    while (in_file >> name) {
        count++;
        std::string index_str = name.substr(0, 6);
        // std::cout << "index_str " << index_str << std::endl;

        std::string data_file = data_package + name;
        data_file += ".bin";

        // load points cloud
        unsigned int length = 0;
        void *data = NULL;
        std::shared_ptr<char> buffer((char *)data,
                                     std::default_delete<char[]>());
        loadData(data_file.data(), &data, &length);
        buffer.reset((char *)data);
        float *points = (float *)buffer.get();
        size_t points_size = length / sizeof(float) / 4;

        unsigned int points_data_size = points_size * 4 * sizeof(float);
        GPU_CHECK(cudaMemset(points_data, 0, points_data_size));
        GPU_CHECK(cudaMemcpy(points_data, points, points_data_size,
                             cudaMemcpyDefault));
        GPU_CHECK(cudaDeviceSynchronize());

        cudaEventRecord(start, stream);
        pointpillars_ptr_.Detect(points_data, points_size, objects);
        cudaEventRecord(stop, stream);

        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop);
        sum_time += elapsedTime;
        std::cout << "pointpillars kitti " << name
                  << " frame cost time :" << elapsedTime << std::endl;

        // std::cout << "Bndbox objs: " << objects.size() << std::endl;
        std::string save_file_name = save_dir + index_str + ".txt";
        SaveBoxPred(objects, save_file_name);

        objects.clear();
    }

    std::cout << "pointpillars kitti val average  cost time :"
              << sum_time / count << " ms." << std::endl;

    in_file.close();

    GPU_CHECK(cudaEventDestroy(start));
    GPU_CHECK(cudaEventDestroy(stop));
    GPU_CHECK(cudaFree(points_data));
    GPU_CHECK(cudaStreamDestroy(stream));

    return 0;
}
