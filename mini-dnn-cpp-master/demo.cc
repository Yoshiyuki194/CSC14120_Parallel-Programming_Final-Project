#include <Eigen/Dense>
#include <algorithm>
#include <iostream>
#include <vector>
#include <string>

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

int main(int argc, char* argv[])
{
    int use_gpu = 0;
    int n_streams = 1;
    int version = 0;
    GpuTimer timer;
    const std::vector<std::string> version_names = {"Basic", "Shared memory", "Multi-stream", "Multi-stream with shared memory"};

    // Parse command-line arguments
    if (argc > 1) 
        use_gpu = std::atoi(argv[1]);
    if (argc > 2) 
        n_streams = std::atoi(argv[2]);
    if (argc > 3) 
        version = std::atoi(argv[3]);
    
    if (use_gpu < 0 || use_gpu > 1 || n_streams < 1 || version < 0 || version >= version_names.size()) {
        std::cerr << "Invalid command-line arguments." << std::endl;
        return 1;
    }
    std::cout << "Use GPU: " << (bool)use_gpu << std::endl;
    std::cout << "Number of streams: " << n_streams << std::endl;
    std::cout << "Version: " << version_names[version] << std::endl;
    std::cout << "-----------------------------------------" << std::endl;

    MNIST dataset("../data/fashion/");
    dataset.read();
    int n_train = dataset.train_data.cols();
    int dim_in = dataset.train_data.rows();

    std::cout << "Fashion-mnist training samples: " << n_train << std::endl;
    std::cout << "Fashion-mnist test samples: " << dataset.test_labels.cols() << std::endl;
    std::cout << "-----------------------------------------" << std::endl;

    float acc = 0;

    // Create GPU network if use_gpu is true
    if (use_gpu)
    {
        cuda_helper cuda_helper;
        cuda_helper.print_device_info();
        std::cout << version_names[version] << " GPU version:" << std::endl;
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
        gpu_dnn.load_parameters("../model/params-10eps.txt");
        timer.Start();
        gpu_dnn.forward(dataset.test_data);
        timer.Stop();
        std::cout << version_names[version] << " GPU forward time: " << timer.Elapsed() / 1000 << " secs" << std::endl;
        acc = compute_accuracy(gpu_dnn.output(), dataset.test_labels);
        std::cout << version_names[version] << " GPU accuracy: " << acc << std::endl;
        std::cout << "-----------------------------------------" << std::endl;
    }
    else
    {
        // Create CPU network
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
        cpu_dnn.load_parameters("../model/params-10eps.txt");
        timer.Start();
        cpu_dnn.forward(dataset.test_data);
        timer.Stop();
        std::cout << "CPU forward time: " << timer.Elapsed() / 1000 << " secs" << std::endl;
        acc = compute_accuracy(cpu_dnn.output(), dataset.test_labels);
        std::cout << "CPU accuracy: " << acc << std::endl;
        std::cout << "-----------------------------------------" << std::endl;
    }

    return 0;
}

// // int main()
// // {
// //     cuda_helper cuda_helper;
// //     cuda_helper.print_device_info();

// //     MNIST dataset("../data/fashion/");
// //     dataset.read();
// //     int n_train = dataset.train_data.cols();
// //     int dim_in = dataset.train_data.rows();

// //     std::cout << "Fashion-mnist training samples: " << n_train << std::endl;
// //     std::cout << "Fashion-mnist test samples: " << dataset.test_labels.cols() << std::endl;
// //     std::cout << "-----------------------------------------" << std::endl;

// //     float acc = 0;
// //     std::cout << "CPU version:" << std::endl;
// //     Network cpu_dnn;
// //     Layer *cpu_conv1 = new Conv(1, 28, 28, 6, 5, 5);
// //     Layer *cpu_pool1 = new MaxPooling(6, 24, 24, 2, 2, 2);
// //     Layer *cpu_conv2 = new Conv(6, 12, 12, 16, 5, 5);
// //     Layer *cpu_pool2 = new MaxPooling(16, 8, 8, 2, 2, 2);
// //     Layer *cpu_fc1 = new FullyConnected(cpu_pool2->output_dim(), 120);
// //     Layer *cpu_fc2 = new FullyConnected(120, 84);
// //     Layer *cpu_fc3 = new FullyConnected(84, 10);
// //     Layer *cpu_relu_conv1 = new ReLU;
// //     Layer *cpu_relu_conv2 = new ReLU;
// //     Layer *cpu_relu_fc1 = new ReLU;
// //     Layer *cpu_relu_fc2 = new ReLU;
// //     Layer *cpu_softmax = new Softmax;
// //     cpu_dnn.add_layer(cpu_conv1);
// //     cpu_dnn.add_layer(cpu_relu_conv1);
// //     cpu_dnn.add_layer(cpu_pool1);
// //     cpu_dnn.add_layer(cpu_conv2);
// //     cpu_dnn.add_layer(cpu_relu_conv2);
// //     cpu_dnn.add_layer(cpu_pool2);
// //     cpu_dnn.add_layer(cpu_fc1);
// //     cpu_dnn.add_layer(cpu_relu_fc1);
// //     cpu_dnn.add_layer(cpu_fc2);
// //     cpu_dnn.add_layer(cpu_relu_fc2);
// //     cpu_dnn.add_layer(cpu_fc3);
// //     cpu_dnn.add_layer(cpu_softmax);
// //     Loss *cpu_loss = new CrossEntropy;
// //     cpu_dnn.add_loss(cpu_loss);
// //     // Load parameters
// //     cpu_dnn.load_parameters("../model/params-10eps.txt");
// //     GpuTimer timer;
// //     timer.Start();
// //     cpu_dnn.forward(dataset.test_data);
// //     timer.Stop();
// //     std::cout << "CPU forward time: " << timer.Elapsed() / 1000 << " secs" << std::endl;
// //     acc = compute_accuracy(cpu_dnn.output(), dataset.test_labels);
// //     std::cout << "CPU accuracy: " << acc << std::endl;
// //     std::cout << "-----------------------------------------" << std::endl;

// //     std::cout << "GPU version:" << std::endl;
// //     Network gpu_dnn;
// //     Layer *gpu_conv1 = new Conv_gpu(1, 28, 28, 6, 5, 5);
// //     Layer *gpu_pool1 = new MaxPooling(6, 24, 24, 2, 2, 2);
// //     Layer *gpu_conv2 = new Conv_gpu(6, 12, 12, 16, 5, 5);
// //     Layer *gpu_pool2 = new MaxPooling(16, 8, 8, 2, 2, 2);
// //     Layer *gpu_fc1 = new FullyConnected(gpu_pool2->output_dim(), 120);
// //     Layer *gpu_fc2 = new FullyConnected(120, 84);
// //     Layer *gpu_fc3 = new FullyConnected(84, 10);
// //     Layer *gpu_relu_conv1 = new ReLU;
// //     Layer *gpu_relu_conv2 = new ReLU;
// //     Layer *gpu_relu_fc1 = new ReLU;
// //     Layer *gpu_relu_fc2 = new ReLU;
// //     Layer *gpu_softmax = new Softmax;
// //     gpu_dnn.add_layer(gpu_conv1);
// //     gpu_dnn.add_layer(gpu_relu_conv1);
// //     gpu_dnn.add_layer(gpu_pool1);
// //     gpu_dnn.add_layer(gpu_conv2);
// //     gpu_dnn.add_layer(gpu_relu_conv2);
// //     gpu_dnn.add_layer(gpu_pool2);
// //     gpu_dnn.add_layer(gpu_fc1);
// //     gpu_dnn.add_layer(gpu_relu_fc1);
// //     gpu_dnn.add_layer(gpu_fc2);
// //     gpu_dnn.add_layer(gpu_relu_fc2);
// //     gpu_dnn.add_layer(gpu_fc3);
// //     gpu_dnn.add_layer(gpu_softmax);
// //     Loss *gpu_loss = new CrossEntropy;
// //     gpu_dnn.add_loss(gpu_loss);
// //     // Load parameters
// //     gpu_dnn.load_parameters("../model/params-10eps.txt");
// //     timer.Start();
// //     gpu_dnn.forward(dataset.test_data);
// //     timer.Stop();
// //     std::cout << "GPU forward time: " << timer.Elapsed() / 1000 << " secs" << std::endl;
// //     acc = compute_accuracy(gpu_dnn.output(), dataset.test_labels);
// //     std::cout << "GPU accuracy: " << acc << std::endl;
//     std::cout << "-----------------------------------------" << std::endl;

//     std::cout << "Multi-stream version:" << std::endl;
//     Network multi_dnn;
//     Layer *multi_conv1 = new Conv_gpu(1, 28, 28, 6, 5, 5, 1, 0, 0, 3);
//     Layer *multi_pool1 = new MaxPooling(6, 24, 24, 2, 2, 2);
//     Layer *multi_conv2 = new Conv_gpu(6, 12, 12, 16, 5, 5, 1, 0, 0, 3);
//     Layer *multi_pool2 = new MaxPooling(16, 8, 8, 2, 2, 2);
//     Layer *multi_fc1 = new FullyConnected(multi_pool2->output_dim(), 120);
//     Layer *multi_fc2 = new FullyConnected(120, 84);
//     Layer *multi_fc3 = new FullyConnected(84, 10);
//     Layer *multi_relu_conv1 = new ReLU;
//     Layer *multi_relu_conv2 = new ReLU;
//     Layer *multi_relu_fc1 = new ReLU;
//     Layer *multi_relu_fc2 = new ReLU;
//     Layer *multi_softmax = new Softmax;
//     multi_dnn.add_layer(multi_conv1);
//     multi_dnn.add_layer(multi_relu_conv1);
//     multi_dnn.add_layer(multi_pool1);
//     multi_dnn.add_layer(multi_conv2);
//     multi_dnn.add_layer(multi_relu_conv2);
//     multi_dnn.add_layer(multi_pool2);
//     multi_dnn.add_layer(multi_fc1);
//     multi_dnn.add_layer(multi_relu_fc1);
//     multi_dnn.add_layer(multi_fc2);
//     multi_dnn.add_layer(multi_relu_fc2);
//     multi_dnn.add_layer(multi_fc3);
//     multi_dnn.add_layer(multi_softmax);
//     Loss *multi_loss = new CrossEntropy;
//     multi_dnn.add_loss(multi_loss);
//     // Load parameters
//     multi_dnn.load_parameters("../model/params-10eps.txt");
//     timer.Start();
//     multi_dnn.forward(dataset.test_data);
//     timer.Stop();
//     std::cout << "Multi-stream forward time: " << timer.Elapsed() / 1000 << " secs" << std::endl;
//     acc = compute_accuracy(multi_dnn.output(), dataset.test_labels);
//     std::cout << "Multi-stream accuracy: " << acc << std::endl;
//     std::cout << "-----------------------------------------" << std::endl;

//     std::cout << "Multi-stream version with shared memory:" << std::endl;
//     Network multi_shared_dnn;
//     Layer *multi_shared_conv1 = new Conv_gpu(1, 28, 28, 6, 5, 5, 1, 0, 0, 3, 1);
//     Layer *multi_shared_pool1 = new MaxPooling(6, 24, 24, 2, 2, 2);
//     Layer *multi_shared_conv2 = new Conv_gpu(6, 12, 12, 16, 5, 5, 1, 0, 0, 3, 1);
//     Layer *multi_shared_pool2 = new MaxPooling(16, 8, 8, 2, 2, 2);
//     Layer *multi_shared_fc1 = new FullyConnected(multi_shared_pool2->output_dim(), 120);
//     Layer *multi_shared_fc2 = new FullyConnected(120, 84);
//     Layer *multi_shared_fc3 = new FullyConnected(84, 10);
//     Layer *multi_shared_relu_conv1 = new ReLU;
//     Layer *multi_shared_relu_conv2 = new ReLU;
//     Layer *multi_shared_relu_fc1 = new ReLU;
//     Layer *multi_shared_relu_fc2 = new ReLU;
//     Layer *multi_shared_softmax = new Softmax;
//     multi_shared_dnn.add_layer(multi_shared_conv1);
//     multi_shared_dnn.add_layer(multi_shared_relu_conv1);
//     multi_shared_dnn.add_layer(multi_shared_pool1);
//     multi_shared_dnn.add_layer(multi_shared_conv2);
//     multi_shared_dnn.add_layer(multi_shared_relu_conv2);
//     multi_shared_dnn.add_layer(multi_shared_pool2);
//     multi_shared_dnn.add_layer(multi_shared_fc1);
//     multi_shared_dnn.add_layer(multi_shared_relu_fc1);
//     multi_shared_dnn.add_layer(multi_shared_fc2);
//     multi_shared_dnn.add_layer(multi_shared_relu_fc2);
//     multi_shared_dnn.add_layer(multi_shared_fc3);
//     multi_shared_dnn.add_layer(multi_shared_softmax);
//     Loss *multi_shared_loss = new CrossEntropy;
//     multi_shared_dnn.add_loss(multi_shared_loss);
//     // Load parameters
//     multi_shared_dnn.load_parameters("../model/params-10eps.txt");
//     timer.Start();
//     multi_shared_dnn.forward(dataset.test_data);
//     timer.Stop();
//     std::cout << "Multi-stream with shared memory forward time: " << timer.Elapsed() / 1000 << " secs" << std::endl;
//     acc = compute_accuracy(multi_shared_dnn.output(), dataset.test_labels);
//     std::cout << "Multi-stream with shared memory accuracy: " << acc << std::endl;
//     std::cout << "-----------------------------------------" << std::endl;

//     return 0;
// }