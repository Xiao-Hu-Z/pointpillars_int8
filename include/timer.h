#pragma once

#include <chrono>
#include <ctime>
#include <stdio.h>
#include <sys/time.h>

inline int64_t gtm() {
    struct timeval tm;
    gettimeofday(&tm, 0);
    int64_t re = (((int64_t)tm.tv_sec) * 1000 * 1000 + tm.tv_usec);
    return re;
}

class Timer
{
public:
    Timer(const char *nameIn)
    {
        name = nameIn;
        Tic();
    }

    void Tic()
    {
        start = std::chrono::system_clock::now();
    }

    double Toc()
    {
        end = std::chrono::system_clock::now();
        std::chrono::duration<double> dt = end - start;
        return dt.count() * 1000;//输出毫秒
    }

    void TicToc()
    {
        std::cout <<"The process " << name <<"takes " << Toc() <<" ms" << std::endl;
    }

private:
    const char* name;
    std::chrono::time_point<std::chrono::system_clock> start, end;
};