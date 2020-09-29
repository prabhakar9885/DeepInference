#include "Flatten.cuh"

Flatten::Flatten():size{0}, activation{Activation::NONE}
{
}

void Flatten::init()
{
}

bool Flatten::canBeStackedOn(const Layer* prevLayer) const
{
    LayerType typeOfPrevLayer = Utills::Layers::getLayerType(prevLayer);
    return (typeOfPrevLayer == LayerType::CONV);
}

float* Flatten::forward(const float* input) const
{
    return nullptr;
}