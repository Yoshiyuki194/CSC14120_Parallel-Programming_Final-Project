#ifndef SRC_GPUTIMER_H_
#define SRC_GPUTIMER_H_

#include <cuda_runtime.h>

class GpuTimer()
{
private:
    cudaEvent_t start;
    cudaEvent_t stop;

public:
    GpuTimer()
    {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    ~GpuTimer()
    {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void start()
    {
        cudaEventRecord(start, 0);
    }

    void stop()
    {
        cudaEventRecord(stop, 0);
    }

    float elapsed()
    {
        float elapsed;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        return elapsed;
    }
};