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

ConvLayer::~ConvLayer()
{
}

bool ConvLayer::canBeStackedOn(const Layer* prevLayer) const
{
    LayerType typeOfPrevLayer = Utills::Layers::getLayerType(prevLayer);
    bool canBeStacked = false;
    if (typeOfPrevLayer == LayerType::CONV)
    {
        const ConvLayer* prevConvLayer = static_cast<const ConvLayer*>(prevLayer);
        canBeStacked = prevConvLayer->convLayerDims.N == this->convLayerDims.C;
    }
    return canBeStacked;
}

bool ConvLayer::hasInputLayer() const
{
    return this->inputSizeIsSet;
}

void ConvLayer::init(const std::vector<float> &weight, const std::vector<float> &bias)
{
    ConvLayerDims prevLayerDims;
    ConvLayer* prevLayer = dynamic_cast<ConvLayer*>(this->prevLayer);
    if (prevLayer)
    {
        prevLayerDims = prevLayer->convLayerDims;
        this->cuConvLayer = new CuConvLayer(this->convLayerDims.C, this->convLayerDims.N, this->convLayerDims.W, this->convLayerDims.H, this->activation, prevLayer->cuConvLayer);
    }
    else
    {
        prevLayerDims = ConvLayerDims{
                            this->convInputLayerDims.batchSize,
                            this->convInputLayerDims.channelsPerImage,
                            this->convInputLayerDims.imageHeight,
                            this->convInputLayerDims.imageWidth
                        };
        this->cuConvLayer = new CuConvLayer(this->convLayerDims.C, this->convLayerDims.N, this->convLayerDims.W, this->convLayerDims.H, this->activation);
    }
    if (prevLayer && this->convLayerDims.N * this->convLayerDims.C * this->convLayerDims.H * this->convLayerDims.W != weight.size())
        throw "WeightDimensionsInvalid: ";
    this->cuConvLayer->setSizeOfInput(prevLayerDims.N, prevLayerDims.C, prevLayerDims.H, prevLayerDims.W);
    this->cuConvLayer->allocMemForLayer();
    this->cuConvLayer->init(weight.data(), weight.size(), bias.data(), bias.size());
}

float* ConvLayer::forward(const float* input) const
{
    return this->cuConvLayer->compute(input);
}

ConvLayerDims ConvLayer::getSize() const
{
    return this->convLayerDims;
}

void* ConvLayer::getOutput() const
{
    return nullptr;
}
