#include "ConvLayer.cuh"

ConvLayer::ConvLayer(int inChannels, int outChannels, int H, int W, int stride, int padding, int dilation, Activation activation)
{
    this->convLayerDims = ConvLayerDims{ outChannels, inChannels , H, W };
    this->convAlgoSpecs = ConvAlgoSpecs{ stride, dilation, padding };
    this->activation = activation;
}

ConvLayer::ConvLayer(int inChannels, int outChannels, int H, int W, int stride, int padding, int dilation, Activation activation, ImageInputLayerDims&& convInputLayerDims) :
    ConvLayer(inChannels, outChannels, H, W, stride, padding, dilation, activation)
{
    if (convInputLayerDims.channelsPerImage != inChannels)
        throw "NumberOfChannels in input must match the numberOfChannels in each kernel";
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

void ConvLayer::init(const std::vector<float>& weight, const std::vector<float>& bias)
{
    if (this->hasInputLayer())
    {
        this->cuConvLayer = new CuConvLayer(
            this->convLayerDims.C,
            this->convLayerDims.N,
            this->convLayerDims.H,
            this->convLayerDims.W,
            this->convAlgoSpecs.padding,
            this->convAlgoSpecs.stride,
            this->convAlgoSpecs.dilation,
            this->convInputLayerDims.batchSize,
            this->convInputLayerDims.channelsPerImage,
            this->convInputLayerDims.imageHeight,
            this->convInputLayerDims.imageWidth,
            this->activation
        );
    }
    else
    {
        size_t numberOfWeightsReceived = (size_t)this->convLayerDims.N * this->convLayerDims.C * this->convLayerDims.H * this->convLayerDims.W;
        if (weight.size() != numberOfWeightsReceived)
            throw "WeightDimensionsInvalid";
        ConvLayer* prevLayer = dynamic_cast<ConvLayer*>(this->prevLayer);
        this->cuConvLayer = new CuConvLayer(
            this->convLayerDims.C,
            this->convLayerDims.N,
            this->convLayerDims.W,
            this->convLayerDims.H,
            this->convAlgoSpecs.padding,
            this->convAlgoSpecs.stride,
            this->convAlgoSpecs.dilation,
            prevLayer->getCuLayer(),
            this->activation
        );
    }
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

const CuConvLayer* ConvLayer::getCuLayer() const
{
    return this->cuConvLayer;
}

void* ConvLayer::getOutput() const
{
    return this->cuConvLayer->getOutputOnDevice().dataOnDevice;
}
