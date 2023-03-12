#pragma once

#include <Eigen/Core>
#include <assert.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdio.h>
#include <string.h>
#include <string>
#include <vector>

#include "cuda_runtime_api.h"

template <typename T>
void HOST_SAVE(T *array, int size, std::string filename,
               std::string root = "../data", std::string postfix = ".txt") {
    std::string filepath = root + "/" + filename + postfix;
    if (postfix == ".bin") {
        std::fstream file(filepath, std::ios::out | std::ios::binary);
        file.write(reinterpret_cast<char *>(array), sizeof(size * sizeof(T)));
        file.close();
        std::cout << "|>>>|  Data has been written in " << filepath << "  |<<<|"
                  << std::endl;
        return;
    } else if (postfix == ".txt") {
        std::ofstream file(filepath, std::ios::out);
        for (int i = 0; i < size; ++i)
            file << array[i] << " ";
        file.close();
        std::cout << "|>>>|  Data has been written in " << filepath << "  |<<<|"
                  << std::endl;
        return;
    }
};

template <typename T>
void DEVICE_SAVE(T *array, int size, std::string filename,
                 std::string root = "../data", std::string postfix = ".txt") {
    T *temp_ = new T[size];
    cudaMemcpy(temp_, array, size * sizeof(T), cudaMemcpyDeviceToHost);
    HOST_SAVE<T>(temp_, size, filename, root, postfix);
    delete[] temp_;
};

template <typename T>
void HOST_SAVE(T *array, int size, std::string filename, int channel,
               std::string root = "../data", std::string postfix = ".txt") {
    std::string filepath = root + "/" + filename + postfix;
    if (postfix == ".txt") {
        std::ofstream file(filepath, std::ios::out);
        file << std::setiosflags(std::ios::fixed) << std::setprecision(5);
        for (int k = 0; k < size / channel; k++) {
            for (int i = 0; i < channel; ++i)
                file << array[k * channel + i] << " ";
            file << std::endl;
        }

        file.close();
        std::cout << "|>>>|  Data has been written in " << filepath << "  |<<<|"
                  << std::endl;
        return;
    }
};

template <typename T>
void DEVICE_SAVE(T *array, int size, std::string filename, int channel,
                 std::string root = "../data", std::string postfix = ".txt") {
    T *temp_ = new T[size];
    cudaMemcpy(temp_, array, size * sizeof(T), cudaMemcpyDeviceToHost);
    HOST_SAVE<T>(temp_, size, filename, channel, root, postfix);
    delete[] temp_;
};
