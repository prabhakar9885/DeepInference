#ifndef FLATTEN_CU
#define FLATTEN_CU

#include "Flatten.cuh"

Flatten::Flatten()
{
}

bool Flatten::hasInputLayer() const
{
    return false;
}

void Flatten::init()
{
    ConvLayer* prevLayer = dynamic_cast<ConvLayer*>(this->prevLayer);
    if (prevLayer)
        this->cuFlattenedLayer = new CuFlattenedLayer(prevLayer->getPreviousCuLayer());
    else
        throw "Previous Layer Should be a ConvLayer";
}

void Flatten::init(const std::vector<float>& weight, const std::vector<float>& bias)
{
}

bool Flatten::canBeStackedOn(const Layer* prevLayer) const
{
    LayerType typeOfPrevLayer = Utills::Layers::getLayerType(prevLayer);
    return (typeOfPrevLayer == LayerType::CONV);
}

float* Flatten::forward(const float* input) const
{
    return this->cuFlattenedLayer->compute(input);
}

const CuFlattenedLayer* Flatten::getCuLayer() const
{
    return this->cuFlattenedLayer;
}

void* Flatten::getOutput() const
{
	return nullptr;
}

#endif // !FLATTEN_CU
