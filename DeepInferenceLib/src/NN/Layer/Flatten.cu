#ifndef FLATTEN_CU
#define FLATTEN_CU

#include "Flatten.cuh"

Flatten::Flatten():size{0}, activation{Activation::NONE}
{
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
    return nullptr;
}

void* Flatten::getOutput() const
{
	return nullptr;
}

#endif // !FLATTEN_CU
