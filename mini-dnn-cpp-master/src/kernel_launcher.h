#ifndef SRC_KERNEL_LAUNCHER_H_
#define SRC_KERNEL_LAUNCHER_H_
#pragma once

#include <stdio.h>
#include <cuda_runtime.h>
#include "./layer/cuda_functions/device.h"

/**
 * @brief The KernelLauncher class is responsible for launching kernel functions for convolutional forward pass.
 * 
 * This class provides methods for performing convolutional forward pass using different optimization techniques.
 * It includes a basic_forward method and a smem_forward method.
 * The conv_forward method selects the appropriate forward pass method based on the specified parameters.
 * The execution time of each convolutional layer is measured using a GpuTimer object from CUDA device class.
 */
class KernelLauncher
{
public:
    /**
     * @brief Performs convolutional forward pass using the basic method.
     * 
     * @param in Pointer to the input data.
     * @param out Pointer to the output data.
     * @param weight Pointer to the weight data.
     * @param n_samples Number of input samples.
     * @param channel_in Number of input channels.
     * @param channel_out Number of output channels.
     * @param height_in Height of the input data.
     * @param width_in Width of the input data.
     * @param kernel_width Width of the convolutional kernel.
     * @param n_streams Number of streams for parallel execution (default is 1).
     */
    void basic_forward(const float *in, float *out, const float *weight, const int n_samples,
                       const int channel_in, const int channel_out,
                       const int height_in, const int width_in, const int kernel_width, const int n_streams = 1);

    /**
     * @brief Performs convolutional forward pass using the shared memory (smem) method.
     * 
     * @param in Pointer to the input data.
     * @param out Pointer to the output data.
     * @param weight Pointer to the weight data.
     * @param n_samples Number of input samples.
     * @param channel_in Number of input channels.
     * @param channel_out Number of output channels.
     * @param height_in Height of the input data.
     * @param width_in Width of the input data.
     * @param kernel_width Width of the convolutional kernel.
     * @param n_streams Number of streams for parallel execution (default is 1).
     */
    void smem_forward(const float *in, float *out, const float *weight, const int n_samples,
                      const int channel_in, const int channel_out,
                      const int height_in, const int width_in, const int kernel_width, const int n_streams = 1);

    /**
     * @brief Performs convolutional forward pass using the appropriate method based on the specified parameters.
     * 
     * @param in Pointer to the input data.
     * @param out Pointer to the output data.
     * @param weight Pointer to the weight data.
     * @param n_samples Number of input samples.
     * @param channel_in Number of input channels.
     * @param channel_out Number of output channels.
     * @param height_in Height of the input data.
     * @param width_in Width of the input data.
     * @param kernel_width Width of the convolutional kernel.
     * @param n_streams Number of streams for parallel execution (default is 1).
     * @param use_smem Flag indicating whether to use shared memory optimization (default is 0).
     */
    void conv_forward(const float *in, float *out, const float *weight, const int n_samples,
                      const int channel_in, const int channel_out,
                      const int height_in, const int width_in, const int kernel_width, const int n_streams = 1, const int use_smem = 0)
    {
        GpuTimer timer;
        timer.Start();
        if (!use_smem)
            basic_forward(in, out, weight, n_samples, channel_in, channel_out, height_in, width_in, kernel_width, n_streams);
        else if (use_smem)
            smem_forward(in, out, weight, n_samples, channel_in, channel_out, height_in, width_in, kernel_width, n_streams);
        // else if (... == ?) // if have any more optimizer 
        timer.Stop();
        if (channel_out == 6)
            printf("C1 layer time: %f s\n", timer.Elapsed() / 1000);
        else if (channel_out == 16)
            printf("C3 layer time: %f s\n", timer.Elapsed() / 1000);
    }
};

#endif // SRC_KERNEL_LAUNCHER_H_