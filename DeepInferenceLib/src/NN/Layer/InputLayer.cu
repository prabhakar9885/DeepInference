#include "InputLayer.cuh""

InputLayer::InputLayer(int size, Activation activation)
{
    this->layerType = LayerType::DENSE;
    this->denseInputLayer.currentLayerSize = size;
    this->cuLayer = new CuDenseLayer(size, Activation::NONE);
}

InputLayer::InputLayer(int N, int H, int W, int C)
{
    this->layerType = LayerType::CONV;
    this->convLayerDims.N = N;
    this->convLayerDims.H = H;
    this->convLayerDims.W = W;
    this->convLayerDims.C = C;
    //this->cuLayer = new CuConvLayer(N, H, W, C);
}

InputLayer::~InputLayer()
{
    delete this->cuLayer;
}

void InputLayer::init()
{
    switch (this->layerType)
    {
    case LayerType::DENSE:
        this->cuLayer->initAsInputLayer();
        break;
    case LayerType::CONV:
        break;
    }
}

bool InputLayer::canBeStackedOn(const Layer* prevLayer) const
{
    return false;
}

float* InputLayer::forward(const float* input) const
{
    return this->cuLayer->compute(input);
}

LayerType InputLayer::getLayerType() const
{
    return this->layerType;
}


