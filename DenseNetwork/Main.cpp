#include "NN/NN.cuh"
#include "Layer/DenseLayer.cuh"
#include <iostream>

void main()
{
    NN nn;

    Layer* layer1 = new DenseLayer();
    nn.pushLayer(layer1);

    Layer* layer2 = new DenseLayer();
    nn.pushLayer(layer2);

    Layer* layer3 = new DenseLayer();
    nn.pushLayer(layer3);

    std::vector<std::vector<float>> weight_bias(5, std::vector<float>(5, 10));
    nn.init(weight_bias);
    float out = nn.forward(std::vector<float>(5, 10));
    std::cout << out << "\n";
}