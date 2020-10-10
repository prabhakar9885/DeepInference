#include "CudaEngine/Layers/CuFlattenedLayer.cuh"


CuFlattenedLayer::CuFlattenedLayer(const CuConvLayer* prevLayer)
{
    this->isInputLayer = false;
    const Tensor4D& prevLayerOut = prevLayer->getOutputOnDevice();
    this->sizeOfCurrentLayer = prevLayerOut.batchSize * prevLayerOut.channelCount * prevLayerOut.height * prevLayerOut.width;
    this->outputOnDevice = prevLayerOut.dataOnDevice;
}

/// @brief Conversion of data's mem-layout happens here.
void CuFlattenedLayer::init()
{
}

void CuFlattenedLayer::init(const float* weights, const int numberOfWeights, const float* bias, const int numberOfBias)
{
}

float* CuFlattenedLayer::compute(const float* x)
{
    this->outputOnDevice = const_cast<float*>(x);
    return this->outputOnDevice;
}

std::vector<float>&& CuFlattenedLayer::getOutput() const
{
    return std::vector<float>();
}
