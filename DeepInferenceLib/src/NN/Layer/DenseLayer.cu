#ifndef DENSELAYER_CU
#define DENSELAYER_CU

#include "DenseLayer.cuh"

DenseLayer::DenseLayer(int size, Activation activation) : size{ size }, activation{ activation }
{
}

DenseLayer::~DenseLayer()
{
    if(cuDenseLayer)
        delete(cuDenseLayer);
    std::cout << "\nDenseLayer freed";
}

void DenseLayer::init(const std::vector<float>& weight, const std::vector<float>& bias)
{
    int sizeOfPreviousLayer;
    if (weight.size() % this->size != 0)
        throw "Dimensionality of weights is not compatible with the layer-size";

    this->cuDenseLayer = new CuDenseLayer(this->size, this->activation);
    this->cuDenseLayer->init(weight.data(), weight.size(), bias.data(), bias.size());
}

bool DenseLayer::canBeStackedOn(const Layer* prevLayer) const
{
    LayerType typeOfPrevLayer = Utills::Layers::getLayerType(prevLayer);
    bool canBeStacked = false;
    if (typeOfPrevLayer == LayerType::INPUT_LAYER)
    {
        const InputLayer* inputLayer = dynamic_cast<const InputLayer*>(prevLayer);
        canBeStacked = inputLayer->getLayerType() == LayerType::DENSE || inputLayer->getLayerType() == LayerType::FLATTEN;
    }
    else
        canBeStacked = (typeOfPrevLayer == LayerType::DENSE || typeOfPrevLayer == LayerType::FLATTEN);
    return canBeStacked;
}

float* DenseLayer::forward(const float* input) const
{
    return this->cuDenseLayer->compute(input);
}

#endif // !DENSELAYER_CU
