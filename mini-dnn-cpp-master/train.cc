/*
 * CNN demo for MNIST dataset
 * Author: Kai Han (kaihana@163.com)
 * Details in https://github.com/iamhankai/mini-dnn-cpp
 * Copyright 2018 Kai Han
 */
#include <Eigen/Dense>
#include <algorithm>
#include <iostream>

#include "src/layer.h"
#include "src/layer/conv.h"
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
#include "src/layer/conv_gpu.h"

int main()
{
    // data
    MNIST dataset("../data/fashion/");
    dataset.read();
    int n_train = dataset.train_data.cols();
    int dim_in = dataset.train_data.rows();
    std::cout << "mnist train number: " << n_train << std::endl;
    std::cout << "mnist test number: " << dataset.test_labels.cols() << std::endl;
    // dnn
    Network dnn;
    Layer *conv1 = new Conv(1, 28, 28, 6, 5, 5);
    Layer *pool1 = new MaxPooling(6, 24, 24, 2, 2, 2);
    Layer *conv2 = new Conv(6, 12, 12, 16, 5, 5);
    Layer *pool2 = new MaxPooling(16, 8, 8, 2, 2, 2);
    Layer *fc1 = new FullyConnected(pool2->output_dim(), 120);
    Layer *fc2 = new FullyConnected(120, 84);
    Layer *fc3 = new FullyConnected(84, 10);
    Layer *relu_conv1 = new ReLU;
    Layer *relu_conv2 = new ReLU;
    Layer *relu_fc1 = new ReLU;
    Layer *relu_fc2 = new ReLU;
    Layer *softmax = new Softmax;
    dnn.add_layer(conv1);
    dnn.add_layer(relu_conv1);
    dnn.add_layer(pool1);
    dnn.add_layer(conv2);
    dnn.add_layer(relu_conv2);
    dnn.add_layer(pool2);
    dnn.add_layer(fc1);
    dnn.add_layer(relu_fc1);
    dnn.add_layer(fc2);
    dnn.add_layer(relu_fc2);
    dnn.add_layer(fc3);
    dnn.add_layer(softmax);
    // loss
    Loss *loss = new CrossEntropy;
    dnn.add_loss(loss);
    // train & test
    SGD opt(0.001, 5e-4, 0.9, true);
    // SGD opt(0.001);
    const int n_epoch = 5;
    const int batch_size = 128;
    for (int epoch = 0; epoch < n_epoch; epoch++)
    {
        // GpuTimer timer;
        // timer.Start();
        shuffle_data(dataset.train_data, dataset.train_labels);
        for (int start_idx = 0; start_idx < n_train; start_idx += batch_size)
        {
            int ith_batch = start_idx / batch_size;
            Matrix x_batch = dataset.train_data.block(0, start_idx, dim_in,
                                                      std::min(batch_size, n_train - start_idx));
            Matrix label_batch = dataset.train_labels.block(0, start_idx, 1,
                                                            std::min(batch_size, n_train - start_idx));
            Matrix target_batch = one_hot_encode(label_batch, 10);
            if (false && ith_batch % 10 == 1)
            {
                std::cout << ith_batch << "-th grad: " << std::endl;
                dnn.check_gradient(x_batch, target_batch, 10);
            }
            dnn.forward(x_batch);
            dnn.backward(x_batch, target_batch);
            // display
            if (ith_batch % 50 == 0)
            {
                std::cout << ith_batch << "-th batch, loss: " << dnn.get_loss()
                          << std::endl;
            }
            // optimize
            dnn.update(opt);
        }
        // timer.Stop();

        // long long elapsed = (long long)timer.Elapsed(); // get elapsed time in milliseconds
        // int hours = (int)(elapsed / (1000 * 60 * 60));
        // int mins = (int)((elapsed % (1000 * 60 * 60)) / (1000 * 60));
        // double secs = ((elapsed % (1000 * 60)) / 1000.0);

        // printf("Training Time: %dh %dm %.4fs\n", hours, mins, secs);

        // printf("Training Time: %.3f ms\n", timer.Elapsed());
        //  test
        dnn.forward(dataset.test_data);
        float acc = compute_accuracy(dnn.output(), dataset.test_labels);
        std::cout << std::endl;
        std::cout << epoch + 1 << "-th epoch, test acc: " << acc << std::endl;
        std::cout << std::endl;
    }
    dnn.save_parameters("../model/params.txt");
    return 0;
}