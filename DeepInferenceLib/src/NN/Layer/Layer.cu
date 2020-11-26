#ifndef LAYER_CU
#define LAYER_CU

#include "Layer.cuh"

Layer::~Layer()
{
}

void Layer::setPrevLayer(Layer* prevLayer)
{
    this->prevLayer = prevLayer;
}

#endif // !LAYER_CU
