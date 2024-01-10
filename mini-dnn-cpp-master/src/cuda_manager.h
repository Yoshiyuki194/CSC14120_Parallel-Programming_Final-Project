#ifndef SRC_LAYER_CUDA_CUDA_MANAGER_H_
#define SRC_LAYER_CUDA_CUDA_MANAGER_H_
#pragma once

#include <stdio.h>
#include <cuda_runtime.h>

class cuda_manager
{
public:
    void conv_forward(const float* in, float* out, const float* weight, const int n_samples, 
        const int channel_in, const int channel_out, 
        const int height_in, const int width_in, const int kernel_width, const int n_streams=1, const int version=0)
    {
        if (version == 0) 
            basic_forward(in, out, weight, n_samples, channel_in, channel_out, height_in, width_in, kernel_width, n_streams);
        else if (version == 1 || version == 3)
            smem_forward(in, out, weight, n_samples, channel_in, channel_out, height_in, width_in, kernel_width, n_streams);
        // else if (version == ?) //if have any optimizer more
    }

    void basic_forward(const float* in, float* out, const float* weight, const int n_samples, 
        const int channel_in, const int channel_out, 
        const int height_in, const int width_in, const int kernel_width, const int n_streams=1);

    void smem_forward(const float* in, float* out, const float* weight, const int n_samples, 
        const int channel_in, const int channel_out, 
        const int height_in, const int width_in, const int kernel_width, const int n_streams=1);
};


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

class cuda_helper
{
private:
    cudaDeviceProp prop;
public:
    void print_device_info()
    {
      CHECK(cudaGetDeviceProperties(&prop, 0));
      printf("**********GPU Device Properties**********\n");
      printf("Name: %s\n", prop.name);
      printf("Compute capability: %d.%d\n", prop.major, prop.minor);
      printf("Number of SMs: %d\n", prop.multiProcessorCount);
      printf("Threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
      printf("Warps per SM: %d\n", prop.maxThreadsPerMultiProcessor / prop.warpSize);
      printf("GMEM: %zu bytes\n", prop.totalGlobalMem);
      printf("SMEM per SM: %zu bytes\n", prop.sharedMemPerMultiprocessor);
      printf("SMEM per Block: %zu bytes\n", prop.sharedMemPerBlock);
      printf("*****************************************\n");
    }
};

#endif // SRC_LAYER_CUDA_CUDA_MANAGER_H_