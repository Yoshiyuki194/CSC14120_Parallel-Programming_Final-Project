#ifndef SRC_LAYER_DEVICE_H_
#define SRC_LAYER_DEVICE_H_
#pragma once

#include <stdio.h>
#include <cuda_runtime.h>
// Macro for checking cuda errors following a cuda launch or api call
#define CHECK(call)\
{\
	const cudaError_t error = call;\
	if (error != cudaSuccess)\
	{\
		fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);\
		fprintf(stderr, "code: %d, reason: %s\n", error,\
				cudaGetErrorString(error));\
		exit(EXIT_FAILURE);\
	}\
}

// Class for timing cuda events and calculating elapsed time
class GpuTimer
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

    void Start()
    {
        cudaEventRecord(start, 0);
    }

    void Stop()
    {
        cudaEventRecord(stop, 0);
    }

    float Elapsed()
    {
        float elapsed;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        return elapsed;
    }
};

// Class for getting device information
class Device
{
private:
    cudaDeviceProp prop;
public:
    void print_device_info();
};

#endif /* SRC_LAYER_DEVICE_H_ */