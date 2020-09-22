#ifndef FLATTEN_CU
#define FLATTEN_CU

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
    bool canBeStacked = false;
    if (typeOfPrevLayer == LayerType::CONV)
    {
        canBeStacked = true;
    }
    return canBeStacked;
}

void Flatten::forward(const std::vector<int> &input) const
{
}

void* Flatten::getOutput() const
{
	return nullptr;
}

#endif // !FLATTEN_CU
