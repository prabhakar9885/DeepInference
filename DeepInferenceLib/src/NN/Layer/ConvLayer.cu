#include "ConvLayer.cuh"

ConvLayer::ConvLayer(int inChannels, int outChannels, int H, int W, int padding, int dilation, Activation activation) : inChannels{ inChannels }, outChannels{ outChannels }, H{ H }, W{ W }, padding{ padding }, dilation{ dilation }, activation{ activation }
{
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

void ConvLayer::init()
{
}

void ConvLayer::forward(const std::vector<int> &input) const
{
}

void* ConvLayer::getOutput() const
{
    return nullptr;
}
