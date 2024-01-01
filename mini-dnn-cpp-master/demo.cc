#include <Eigen/Dense>
#include <algorithm>
#include <iostream>

#include "src/layer.h"
#include "src/layer/conv.h"
#include "src/layer/conv_gpu.h"
#include "src/layer/fully_connected.h"
#include "src/layer/max_pooling.h"
#include "src/layer/relu.h"
#include "src/layer/sigmoid.h"
#include "src/layer/softmax.h"
#include "src/loss.h"
#include "src/loss/mse_loss.h"
#include "src/loss/cross_entropy_loss.h"
#include "src/mnist.h"
#include "src/network.h"
#include "src/optimizer.h"
#include "src/optimizer/sgd.h"
#include "src/layer/cuda/helper.h"
#include "src/layer/cuda/cuda_manager.h"

int main()
{
    cuda_helper cuda_helper;
    cuda_helper.print_device_info();

    MNIST dataset("../data/fashion/");
    dataset.read();
    int n_train = dataset.train_data.cols();
    int dim_in = dataset.train_data.rows();

    std::cout << "Fashion-mnist training samples: " << n_train << std::endl;
    std::cout << "Fashion-mnist test samples: " << dataset.test_labels.cols() << std::endl;
    std::cout << "--------------------------------------------" << std::endl;

    float acc = 0;
    std::cout << "CPU version:" << std::endl;
    Network cpu_dnn;
    Layer *cpu_conv1 = new Conv(1, 28, 28, 6, 5, 5);
    Layer *cpu_pool1 = new MaxPooling(6, 24, 24, 2, 2, 2);
    Layer *cpu_conv2 = new Conv(6, 12, 12, 16, 5, 5);
    Layer *cpu_pool2 = new MaxPooling(16, 8, 8, 2, 2, 2);
    Layer *cpu_fc1 = new FullyConnected(cpu_pool2->output_dim(), 120);
    Layer *cpu_fc2 = new FullyConnected(120, 84);
    Layer *cpu_fc3 = new FullyConnected(84, 10);
    Layer *cpu_relu_conv1 = new ReLU;
    Layer *cpu_relu_conv2 = new ReLU;
    Layer *cpu_relu_fc1 = new ReLU;
    Layer *cpu_relu_fc2 = new ReLU;
    Layer *cpu_softmax = new Softmax;
    cpu_dnn.add_layer(cpu_conv1);
    cpu_dnn.add_layer(cpu_relu_conv1);
    cpu_dnn.add_layer(cpu_pool1);
    cpu_dnn.add_layer(cpu_conv2);
    cpu_dnn.add_layer(cpu_relu_conv2);
    cpu_dnn.add_layer(cpu_pool2);
    cpu_dnn.add_layer(cpu_fc1);
    cpu_dnn.add_layer(cpu_relu_fc1);
    cpu_dnn.add_layer(cpu_fc2);
    cpu_dnn.add_layer(cpu_relu_fc2);
    cpu_dnn.add_layer(cpu_fc3);
    cpu_dnn.add_layer(cpu_softmax);
    Loss *cpu_loss = new CrossEntropy;
    cpu_dnn.add_loss(cpu_loss);
    // Load parameters
    cpu_dnn.load_parameters("../model/params.txt");
    GpuTimer timer;
    timer.Start();
    cpu_dnn.forward(dataset.test_data);
    timer.Stop();
    std::cout << "CPU forward time: " << timer.Elapsed() / 1000 << " secs" << std::endl;
    acc = compute_accuracy(cpu_dnn.output(), dataset.test_labels);
    std::cout << "CPU accuracy: " << acc << std::endl;
    std::cout << "-----------------------------------------" << std::endl;

    std::cout << "GPU version:" << std::endl;
    Network gpu_dnn;
    Layer *gpu_conv1 = new Conv_gpu(1, 28, 28, 6, 5, 5);
    Layer *gpu_pool1 = new MaxPooling(6, 24, 24, 2, 2, 2);
    Layer *gpu_conv2 = new Conv_gpu(6, 12, 12, 16, 5, 5);
    Layer *gpu_pool2 = new MaxPooling(16, 8, 8, 2, 2, 2);
    Layer *gpu_fc1 = new FullyConnected(gpu_pool2->output_dim(), 120);
    Layer *gpu_fc2 = new FullyConnected(120, 84);
    Layer *gpu_fc3 = new FullyConnected(84, 10);
    Layer *gpu_relu_conv1 = new ReLU;
    Layer *gpu_relu_conv2 = new ReLU;
    Layer *gpu_relu_fc1 = new ReLU;
    Layer *gpu_relu_fc2 = new ReLU;
    Layer *gpu_softmax = new Softmax;
    gpu_dnn.add_layer(gpu_conv1);
    gpu_dnn.add_layer(gpu_relu_conv1);
    gpu_dnn.add_layer(gpu_pool1);
    gpu_dnn.add_layer(gpu_conv2);
    gpu_dnn.add_layer(gpu_relu_conv2);
    gpu_dnn.add_layer(gpu_pool2);
    gpu_dnn.add_layer(gpu_fc1);
    gpu_dnn.add_layer(gpu_relu_fc1);
    gpu_dnn.add_layer(gpu_fc2);
    gpu_dnn.add_layer(gpu_relu_fc2);
    gpu_dnn.add_layer(gpu_fc3);
    gpu_dnn.add_layer(gpu_softmax);
    Loss *gpu_loss = new CrossEntropy;
    gpu_dnn.add_loss(gpu_loss);
    // Load parameters
    gpu_dnn.load_parameters("../model/params.txt");
    timer.Start();
    gpu_dnn.forward(dataset.test_data);
    timer.Stop();
    std::cout << "GPU forward time: " << timer.Elapsed() / 1000 << " secs" << std::endl;
    acc = compute_accuracy(gpu_dnn.output(), dataset.test_labels);
    std::cout << "GPU accuracy: " << acc << std::endl;
    std::cout << "-----------------------------------------" << std::endl;

    return 0;
}