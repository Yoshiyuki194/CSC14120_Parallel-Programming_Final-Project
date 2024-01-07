#include "cuda_manager.h"

#define TILE_WIDTH 16

// Convolution forward kernel: Naive implementation
__global__ void conv_forward_kernel(const float *in, float *out, const float *weight,
                                    const int channel_in, const int channel_out,
                                    const int height_in, const int width_in, const int kernel_width)
{
    const int height_out = height_in - kernel_width + 1;
    const int width_out = width_in - kernel_width + 1;

    int height_grid = (height_out - 1) / TILE_WIDTH + 1;
    int width_grid = (width_out - 1) / TILE_WIDTH + 1;

    int sample_idx = blockIdx.z;
    int map_idx = blockIdx.x;
    int row = (blockIdx.y / width_grid) * TILE_WIDTH + threadIdx.y;
    int col = (blockIdx.y % width_grid) * TILE_WIDTH + threadIdx.x;

    float accum = 0;

    if (row >= height_out || col >= width_out)
        return;

    int hw_in = height_in * width_in;
    int hw_out = height_out * width_out;

    for (int i = 0; i < channel_in; i++)
    {
        for (int j = 0; j < kernel_width; j++)
        {
            for (int k = 0; k < kernel_width; k++)
            {
                int pixel_row = row + j;
                int pixel_col = col + k;
                accum += in[sample_idx * channel_in * hw_in + i * hw_in +
                            pixel_row * width_in + pixel_col] *
                        weight[map_idx * channel_in * kernel_width * kernel_width +
                            i * kernel_width * kernel_width + j * kernel_width + k];
            }
        }
    }
    out[sample_idx * channel_out * hw_out + map_idx * hw_out + row * width_out + col] = accum;
}

// Convolution forward kernel: Shared memory implementation
__global__ void conv_forward_kernel_1(const float *in, float *out, const float *weight,
                                      const int channel_in, const int channel_out,
                                      const int height_in, const int width_in, const int kernel_width)
{
    const int height_out = height_in - kernel_width + 1;
    const int width_out = width_in - kernel_width + 1;

    int height_grid = (height_out - 1) / TILE_WIDTH + 1;
    int width_grid = (width_out - 1) / TILE_WIDTH + 1;

    int sample_idx = blockIdx.z;
    int map_idx = blockIdx.x;
    int row = (blockIdx.y / width_grid) * TILE_WIDTH + threadIdx.y;
    int col = (blockIdx.y % width_grid) * TILE_WIDTH + threadIdx.x;

    __shared__ float in_shared[TILE_WIDTH][TILE_WIDTH];
    __shared__ float weight_shared[TILE_WIDTH][TILE_WIDTH];

    float accum = 0;

    if (row >= height_out || col >= width_out)
        return;

    int hw_in = height_in * width_in;
    int hw_out = height_out * width_out;

    for (int i = 0; i < channel_in; i++)
    {
        for (int j = 0; j < kernel_width; j++)
        {
            for (int k = 0; k < kernel_width; k++)
            {
                int pixel_row = row + j;
                int pixel_col = col + k;
                in_shared[threadIdx.y][threadIdx.x] = in[sample_idx * channel_in * hw_in + i * hw_in +
                                                         pixel_row * width_in + pixel_col];
                weight_shared[threadIdx.y][threadIdx.x] = weight[map_idx * channel_in * kernel_width * kernel_width +
                                                                 i * kernel_width * kernel_width + j * kernel_width + k];
                __syncthreads();
                accum += in_shared[threadIdx.y][threadIdx.x] * weight_shared[threadIdx.y][threadIdx.x];
                __syncthreads();
            }
        }
    }
    out[sample_idx * channel_out * hw_out + map_idx * hw_out + row * width_out + col] = accum;
}

__host__ void cuda_manager::conv_forward(const float *in, float *out, const float *weight,
                                         const int n_samples, const int channel_in, const int channel_out,
                                         const int height_in, const int width_in, const int kernel_width, const int n_streams, const int version)
{
    int height_out = height_in - kernel_width + 1;
    int width_out = width_in - kernel_width + 1;
    int size_in = n_samples * channel_in * height_in * width_in;
    int size_out = n_samples * channel_out * height_out * width_out;
    int size_weight = channel_out * channel_in * kernel_width * kernel_width;

    float *d_in;
    float *d_out;
    float *d_weight;
    CHECK(cudaMalloc((void **)&d_in, size_in * sizeof(float)));
    CHECK(cudaMalloc((void **)&d_out, size_out * sizeof(float)));
    CHECK(cudaMalloc((void **)&d_weight, size_weight * sizeof(float)));
    CHECK(cudaMemcpy(d_in, in, size_in * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_weight, weight, size_weight * sizeof(float), cudaMemcpyHostToDevice));

    // Create "nStreams" device streams
    cudaStream_t *streams = (cudaStream_t *)malloc(n_streams * sizeof(cudaStream_t));
    for (int i = 0; i < n_streams; i++)
        CHECK(cudaStreamCreate(&(streams[i])));
    
    int n_samples_per_stream = (n_samples + n_streams - 1) / n_streams;
    int size_in_per_stream = n_samples_per_stream * channel_in * height_in * width_in;
    int size_out_per_stream = n_samples_per_stream * channel_out * height_out * width_out;
    int size_weight_per_stream = channel_out * channel_in * kernel_width * kernel_width;
    int size_in_per_sample = channel_in * height_in * width_in;
    int size_out_per_sample = channel_out * height_out * width_out;
    int size_weight_per_sample = channel_out * channel_in * kernel_width * kernel_width;

    // Set grid and block dimensions and launch the kernel
    int height_grid = (height_out - 1) / TILE_WIDTH + 1;
    int width_grid = (width_out - 1) / TILE_WIDTH + 1;
    int z = height_grid * width_grid;

    for (int i = 0; i < n_streams; i++)
    {
        int offset = i * n_samples_per_stream;
        int n_samples_per_stream_ = min(n_samples - offset, n_samples_per_stream);
        int size_in_per_stream = n_samples_per_stream_ * channel_in * height_in * width_in;
        int size_out_per_stream = n_samples_per_stream_ * channel_out * height_out * width_out;
        int size_in_per_sample = channel_in * height_in * width_in;
        int size_out_per_sample = channel_out * height_out * width_out;

        CHECK(cudaMemcpyAsync(d_in + offset * size_in_per_sample, in + offset * size_in_per_sample, size_in_per_stream * sizeof(float), cudaMemcpyHostToDevice, streams[i]));
        CHECK(cudaMemcpyAsync(d_weight, weight, size_weight * sizeof(float), cudaMemcpyHostToDevice, streams[i]));

        dim3 dimGrid(channel_out, z, n_samples_per_stream_);
        dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
        if (version == 0)
            conv_forward_kernel<<<dimGrid, dimBlock, 0, streams[i]>>>(d_in + offset * size_in_per_sample, d_out + offset * size_out_per_sample, d_weight, channel_in, channel_out, height_in, width_in, kernel_width);
        else if (version == 1 || version == 3)
            conv_forward_kernel_1<<<dimGrid, dimBlock, 0, streams[i]>>>(d_in + offset * size_in_per_sample, d_out + offset * size_out_per_sample, d_weight, channel_in, channel_out, height_in, width_in, kernel_width);
        CHECK(cudaMemcpyAsync(out + offset * size_out_per_sample, d_out + offset * size_out_per_sample, size_out_per_stream * sizeof(float), cudaMemcpyDeviceToHost, streams[i]));
    }
    // Destroy device streams
    for (int i = 0; i < n_streams; i++)
    {
        CHECK(cudaStreamSynchronize(streams[i]));
        CHECK(cudaStreamDestroy(streams[i]));
    }

    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());
    
    CHECK(cudaMemcpy(out, d_out, size_out * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(d_in));
    CHECK(cudaFree(d_out));
    CHECK(cudaFree(d_weight));
    free(streams);
}