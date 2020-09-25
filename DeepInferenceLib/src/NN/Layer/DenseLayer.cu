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

void DenseLayer::initAsInputLayer()
{
    this->cuDenseLayer = new CuDenseLayer(this->size, this->activation);
    this->cuDenseLayer->initAsInputLayer();
}

bool DenseLayer::canBeStackedOn(const Layer* prevLayer) const
{
    LayerType typeOfPrevLayer = Utills::Layers::getLayerType(prevLayer);
    return  (typeOfPrevLayer == LayerType::DENSE || typeOfPrevLayer == LayerType::FLATTEN);
}

float* DenseLayer::forward(const float* input) const
{
    return this->cuDenseLayer->compute(input);
}

int DenseLayer::getSize() const
{
    return this->size;
}

void* DenseLayer::getOutput() const
{
	return nullptr;
}

#endif // !DENSELAYER_CU
