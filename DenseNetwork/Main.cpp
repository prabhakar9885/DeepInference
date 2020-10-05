#include "NN/NN.cuh"
#include "NN/Layer/DenseLayer.cuh"
#include "NN/Layer/ConvLayer.cuh"
#include "NN/Layer/Flatten.cuh"
#include <iostream>

void main()
{
    NN nn;
/*
    // Input image spec: 16 x 16 x 3 (RGB)

    // 5 Kernals of size (H,W,C) = (3, 3, 3)    => output: (15, 15, 5)
    Layer* conv1 = new ConvLayer(3, 5, 3, 3, 0, 0, Activation::ReLU); 
    nn.pushLayer(conv1);

    // 4 Kernals of size (H,W,C) = (3, 3, 5)    => output: (14, 14, 4)
    Layer* conv2 = new ConvLayer(5, 4, 3, 3, 0, 0, Activation::ReLU);
    nn.pushLayer(conv2);

    // 3 Kernals of size (H,W,C) = (3, 5, 5)    => output: (10, 10, 3)
    Layer* conv3 = new ConvLayer(4, 3, 3, 3, 0, 0, Activation::ReLU);
    nn.pushLayer(conv3);

    // Size of the flattened layer will be 3*H*W*C = 900
    Layer* flatten = new Flatten();
    nn.pushLayer(flatten);

    Layer* layer1 = new DenseLayer(32, Activation::SIGMOID);
    nn.pushLayer(layer1);

    Layer* layer2 = new DenseLayer(16, Activation::SIGMOID);
    nn.pushLayer(layer2);

    Layer* layer3 = new DenseLayer(8, Activation::SIGMOID);
    nn.pushLayer(layer3);

    std::vector<std::vector<float>> weight_bias(5, std::vector<float>(5, 1));
    nn.init(weight_bias);
    
    std::vector<float> input(32, 1);
    float out = nn.forward(input);
    std::cout << out << "\n";
*/

    //Layer* layer0 = new DenseLayer(3, Activation::NONE); // Input layer
    //nn.pushLayer(layer0);

    Layer* layer1 = new DenseLayer(5, Activation::SIGMOID, 3);
    nn.pushLayer(layer1);

    Layer* layer2 = new DenseLayer(4, Activation::SIGMOID);
    nn.pushLayer(layer2);

    Layer* layer3 = new DenseLayer(3, Activation::SIGMOID);
    nn.pushLayer(layer3);

    std::vector<std::vector<float>> weight_bias;
    weight_bias.push_back(std::vector<float>(5*3, 1));
    weight_bias.push_back(std::vector<float>(5, 1));
    weight_bias.push_back(std::vector<float>(4*5, 1));
    weight_bias.push_back(std::vector<float>(4, 1));
    weight_bias.push_back(std::vector<float>(3*4, 1));
    weight_bias.push_back(std::vector<float>(3, 1));
    nn.init(weight_bias);

    std::vector<float> input(3, 1);
    const float* out = nn.forward(input);
    std::cout << *out << "\n";
}