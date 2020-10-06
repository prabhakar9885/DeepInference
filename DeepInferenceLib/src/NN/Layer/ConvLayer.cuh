#ifndef CONVLAYER_CUH
#define CONVLAYER_CUH

#include "Layer.cuh"
#include "Shared/Utills.cuh"
#include "Shared/DataStructs.cuh"
#include "CudaEngine/Layers/CuConvLayer.cuh"

class ConvLayer final : public Layer
{
private:
    ConvInputLayerDims convInputLayerDims;
    bool inputSizeIsSet = false;
    ConvLayerDims convLayerDims;
    ConvAlgoSpecs convAlgoSpecs;
    Activation activation;
    CuConvLayer* cuConvLayer = nullptr;
public:
    ConvLayer(int inChannels, int outChannels, int H, int W, int stride, int padding, int dilation, Activation activation);
    ConvLayer(int inChannels, int outChannels, int H, int W, int stride, int padding, int dilation, Activation activation, ConvInputLayerDims&& convInputLayerDims);
    bool canBeStackedOn(const Layer* prevLayer) const;
    bool hasInputLayer() const override;
    void init(const std::vector<float> &weight, const std::vector<float> &bias) override;
    float* forward(const float* input) const override;
    ConvLayerDims getSize() const;
    void* getOutput() const override;
};

#endif // !CONVLAYER_CUH