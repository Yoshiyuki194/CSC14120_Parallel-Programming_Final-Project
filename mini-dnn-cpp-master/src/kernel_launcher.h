#ifndef SRC_KERNEL_LAUNCHER_H_
#define SRC_KERNEL_LAUNCHER_H_
#pragma once

#include <stdio.h>
#include <cuda_runtime.h>
#include "./layer/cuda_functions/device.h"

class KernelLauncher
{
public:
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

    void basic_forward(const float *in, float *out, const float *weight, const int n_samples,
                       const int channel_in, const int channel_out,
                       const int height_in, const int width_in, const int kernel_width, const int n_streams = 1);

    void smem_forward(const float *in, float *out, const float *weight, const int n_samples,
                      const int channel_in, const int channel_out,
                      const int height_in, const int width_in, const int kernel_width, const int n_streams = 1);
};

#endif // SRC_KERNEL_LAUNCHER_H_