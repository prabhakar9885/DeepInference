#include "NN/NN.cuh"
#include "NN/Layer/DenseLayer.cuh"
#include "NN/Layer/ConvLayer.cuh"
#include "NN/Layer/Flatten.cuh"
#include <iostream>
#include <iomanip>

void main()
{
    NN nn;
    std::vector<std::vector<float>> weight_bias;

    // Input image spec: 7 x 8 x 3 (RGB)
    // Conv1(N, C, H, W) = (3, 3, 5, 5)    => output(C, H, W): (3, 3, 4)
    Layer* conv1 = new ConvLayer(3, 3, 5, 5, 1, 0, 1, Activation::ReLU, ConvInputLayerDims{ 1,3,7,8 });
    nn.pushLayer(conv1);
    {
        std::vector<float>&& wt_3_3_5_5 = std::initializer_list<float>({
            // Kernel-00
            1, 1, 0, 1, 1,
            1, 1, 0, 1, 1,
            1, 1, 0, 1, 1,
            1, 1, 0, 1, 1,
            1, 1, 0, 1, 1,
            // Kernel-01
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1,
            0, 0, 0, 0, 0,
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1,
            // Kernel-02
            1, 1, 0, 1, 1,
            1, 1, 0, 1, 1,
            0, 0, 0, 0, 0,
            1, 1, 0, 1, 1,
            1, 1, 0, 1, 1,
            // Kernel-10
            0, 0, 1, 0, 0,
            0, 0, 1, 0, 0,
            0, 0, 1, 0, 0,
            0, 0, 1, 0, 0,
            0, 0, 1, 0, 0,
            // Kernel-11
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            1, 1, 1, 1, 1,
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            // Kernel-12
            0, 0, 1, 0, 0,
            0, 0, 1, 0, 0,
            1, 1, 1, 1, 1,
            0, 0, 1, 0, 0,
            0, 0, 1, 0, 0,
            // Kernel-20
            1, 0, 0, 0, 1,
            0, 1, 0, 1, 0,
            0, 0, 1, 0, 0,
            0, 1, 0, 1, 0,
            1, 0, 0, 0, 1,
            // Kernel-21
            -1,  0,  0,  0, -1,
             0, -1,  0, -1,  0,
             0,  0, -1,  0,  0,
             0, -1,  0, -1,  0,
            -1,  0,  0,  0, -1,
            // Kernel-22
             0,  0, -1,  0,  0,
             0,  0, -1,  0,  0,
            -1, -1, -1, -1, -1,
             0,  0, -1,  0,  0,
             0,  0, -1,  0,  0,
            });
        std::vector<float> bias_3(3, 0);
        weight_bias.push_back(std::move(wt_3_3_5_5));
        weight_bias.push_back(std::move(bias_3));
    }

    // Conv2(N, C, H, W) = (2, 3, 3, 3)    => output(C, H, W): (2, 1, 2)
    Layer* conv2 = new ConvLayer(3, 2, 3, 3, 1, 0, 1, Activation::ReLU);
    nn.pushLayer(conv2);
    {
        std::vector<float>&& wt_2_3_3_3 = std::initializer_list<float>({
            // Kernel-00
            1, 0, 1,
            1, 0, 1,
            1, 0, 1,
            // Kernel-01
            0, 1, 0,
            0, 1, 0,
            0, 1, 0,
            // Kernel-02
            1, 0, 1,
            0, 1, 0,
            1, 0, 1,
            // Kernel-10
            1, 0, 1,
            1, 0, 1,
            1, 0, 1,
            // Kernel-11
            0, 1, 0,
            0, 1, 0,
            0, 1, 0,
            // Kernel-12
            1, 0, 1,
            0, 1, 0,
            1, 0, 1
            });
        std::vector<float> bias_2(2, 0);
        weight_bias.push_back(std::move(wt_2_3_3_3));
        weight_bias.push_back(std::move(bias_2));
    }

    // Size of the flattened layer will be 3*H*W*C = 4
    Layer* flatten = new Flatten();
    nn.pushLayer(flatten);

    Layer* layer1 = new DenseLayer(5, Activation::SIGMOID);
    nn.pushLayer(layer1);
    weight_bias.push_back(std::move(std::vector<float>(5 * 54, 1)));
    weight_bias.push_back(std::move(std::vector<float>(5, 1)));

    Layer* layer2 = new DenseLayer(4, Activation::SIGMOID);
    nn.pushLayer(layer2);
    weight_bias.push_back(std::move(std::vector<float>(4 * 5, 1)));
    weight_bias.push_back(std::move(std::vector<float>(4, 1)));

    Layer* layer3 = new DenseLayer(3, Activation::SIGMOID);
    nn.pushLayer(layer3);
    weight_bias.push_back(std::move(std::vector<float>(3*4, 1)));
    weight_bias.push_back(std::move(std::vector<float>(3, 1)));

    nn.init(weight_bias);

    // Size of input (C, H, W): (3, 7, 8)
    std::vector<float>&& input_3_7_8 = std::initializer_list<float>({
                // Channel-0
               1, 1, 1, 1, 1, 1, 1, 1,
               1, 0, 1, 0, 1, 0, 1, 0,
               0, 0, 0, 0, 0, 0, 0, 0,
               0, 1, 0, 1, 0, 1, 0, 1,
               1, 1, 1, 1, 1, 1, 1, 1,
               1, 0, 1, 0, 1, 0, 1, 0,
               1, 1, 1, 1, 1, 1, 1, 1,
               // Channel-1
               1, 0, 1, 0, 1, 0, 1, 0,
               1, 1, 1, 1, 1, 1, 1, 1,
               0, 0, 0, 0, 0, 0, 0, 0,
               1, 0, 1, 0, 1, 0, 1, 0,
               1, 1, 1, 1, 1, 1, 1, 1,
               0, 1, 0, 1, 0, 1, 0, 1,
               1, 1, 1, 1, 1, 1, 1, 1,
               // Channel-2
               1, 0, 1, 0, 1, 0, 1, 0,
               0, 0, 0, 0, 0, 0, 0, 0,
               1, 1, 1, 1, 1, 1, 1, 1,
               1, 0, 1, 0, 1, 0, 1, 0,
               0, 0, 0, 0, 0, 0, 0, 0,
               0, 1, 0, 1, 0, 1, 0, 1,
               1, 1, 1, 1, 1, 1, 1, 1,
        });
    const float* out = nn.forward(input_3_7_8);
    std::cout << *out << "\n";
}