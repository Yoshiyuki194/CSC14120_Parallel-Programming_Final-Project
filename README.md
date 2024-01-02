# CSC14120_Parallel-Programming_Final-Project
In this final project, students will be implementing and optimizing the forward-pass of a convolutional layer using CUDA.
We will use [mini-dnn-cpp](https://github.com/iamhankai/mini-dnn-cpp) (Mini-DNN) framework as a starting point to implement the LeNet-5. This framework is a C++ demo of deep neural networks implementation, purely used C++ language with the support of Eigen library for matrix and vector representation.

## Tasks
- Run and explore the starter project on the classic MNIST.
- Make the host code: modify the code to change the network architecture to
LeNet-5 and run on Fashion MNIST.
- Implement a basic GPU Convolution kernel.
- Optimize your GPU Convolution kernel. Some ideas to optimize:
  - Tiled shared memory convolution.
  - Shared memory matrix multiplication and input matrix unrolling.
  - Kernel fusion for unrolling and matrix-multiplication.
  - Weight matrix (kernel values) in constant memory.
  - Tuning with restrict and loop unrolling.
  - Sweeping various parameters to find best values (block sizes, amount of
  thread coarsening).
  - Multiple kernel implementations for different layer sizes.
  - Input channel reduction: tree.
  - Input channel reduction: atomics.
  - Fixed point (FP16) arithmetic.
  - Using Streams to overlap computation with data transfer.
  - An advanced matrix multiplication algorithm.
## How to run the project in a CUDA-enabled environment
```
mkdir build
cd build
cmake .. -Wno-dev
make
```
## References
1. Kirk, D. & Hwu, W. (2016). Programming Massively Parallel Processors: A Hands-on Approach (3rd ed.). Morgan Kaufmann. Chapter 16, Application case study—machine learning.
2. https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-convolutional-neural-networks
3. dProgrammer lopez. (2019). C++ Convolutional Neural Network Tutorial 2019 [YouTube playlist]. YouTube. Retrieved January 2, 2024, from https://youtube.com/playlist?list=PL3MCKCM5GS4UGRk9wUD5HaNKTwm71fNZx
