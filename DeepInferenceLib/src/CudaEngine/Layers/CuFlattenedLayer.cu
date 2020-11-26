#include "CudaEngine/Layers/CuFlattenedLayer.cuh"


CuFlattenedLayer::CuFlattenedLayer(const CuConvLayer* prevLayer)
{
    this->isInputLayer = false;
    this->prevLayer = prevLayer;
    const Tensor4D& prevLayerOut = prevLayer->getOutputOnDevice();
    this->sizeOfCurrentLayer = prevLayerOut.batchSize * prevLayerOut.channelCount * prevLayerOut.height * prevLayerOut.width;
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
    const Tensor4D& prevLayerOut = this->prevLayer->getOutputOnDevice();
    int N = prevLayerOut.batchSize;
    int C = prevLayerOut.channelCount;
    int H = prevLayerOut.height;
    int W = prevLayerOut.width;
    float* data = new float[(long long)N * C * H * W];
    int i = 0;
    for (int n = 0; n < N; n++)
    {
        for (int c = 0; c < C; c++)
        {
            for (int h = 0; h < H; h++)
            {
                for (int w = 0; w < W; w++)
                    data[i++] = x[c + (w + (h + (n)*H) * W) * C];
            }
        }
    }
    checkCUDA(cudaMallocManaged(&this->outputOnDevice, prevLayerOut.sizeInBytes));
    checkCUDA(cudaMemcpy(this->outputOnDevice, data, prevLayerOut.sizeInBytes, cudaMemcpyHostToDevice));
    delete(data);
    return this->outputOnDevice;
}

std::vector<float>&& CuFlattenedLayer::getOutput() const
{
    return std::vector<float>();
}
