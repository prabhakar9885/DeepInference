#include "ConvLayer.cuh"

ConvLayer::ConvLayer(int inChannels, int outChannels, int H, int W, int stride, int padding, int dilation, Activation activation)
{
    this->convLayerDims = ConvLayerDims{ outChannels, inChannels , H, W };
    this->convAlgoSpecs = ConvAlgoSpecs{ stride, dilation, padding };
    this->activation = activation;
}

ConvLayer::ConvLayer(int inChannels, int outChannels, int H, int W, int stride, int padding, int dilation, Activation activation, ConvInputLayerDims&& convInputLayerDims) :ConvLayer(inChannels, outChannels, H, W, stride, padding, dilation, activation)
{
    this->convInputLayerDims = std::move(convInputLayerDims);
    this->inputSizeIsSet = true;
}

bool ConvLayer::canBeStackedOn(const Layer* prevLayer) const
{
    LayerType typeOfPrevLayer = Utills::Layers::getLayerType(prevLayer);
    bool canBeStacked = false;
    if (typeOfPrevLayer == LayerType::CONV)
    {
        const ConvLayer* prevConvLayer = static_cast<const ConvLayer*>(prevLayer);
        canBeStacked = prevConvLayer->outChannels == this->inChannels;
    }
    return canBeStacked;
}

bool ConvLayer::hasInputLayer() const
{
    return this->inputSizeIsSet;
}

void ConvLayer::init(const std::vector<float> &weight, const std::vector<float> &bias)
{
}

float* ConvLayer::forward(const float* input) const
{
    return nullptr;
}

ConvLayerDims ConvLayer::getSize() const
{
    return this->convLayerDims;
}

void* ConvLayer::getOutput() const
{
    return nullptr;
}
