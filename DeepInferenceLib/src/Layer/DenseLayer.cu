#ifndef DENSELAYER_CU
#define DENSELAYER_CU

#include "DenseLayer.cuh"

DenseLayer::DenseLayer(int size, Activation activation) : size{ size }, activation{ activation }
{
}

void DenseLayer::init()
{
}

bool DenseLayer::canBeStackedOn(const Layer* prevLayer) const
{
    LayerType typeOfPrevLayer = Utills::Layers::getLayerType(prevLayer);
    bool canBeStacked = false;
    if (typeOfPrevLayer == LayerType::DENSE || typeOfPrevLayer == LayerType::FLATTEN)
    {
        canBeStacked = true;
    }
    return canBeStacked;
}

void DenseLayer::forward(const std::vector<int> &input) const
{
}

void* DenseLayer::getOutput() const
{
	return nullptr;
}

#endif // !DENSELAYER_CU
