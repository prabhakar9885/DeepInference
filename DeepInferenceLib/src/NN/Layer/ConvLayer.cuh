#ifndef CONVLAYER_CUH
#define CONVLAYER_CUH

#include "Layer.cuh"
#include "Shared/Utills.cuh"
#include "Shared/DataStructs.cuh"
#include "CudaEngine/Layers/CuConvLayer.cuh"

class ConvLayer final : public Layer
{

private:
    ImageInputLayerDims convInputLayerDims;
    bool inputSizeIsSet = false;
    ConvLayerDims convLayerDims;
    ConvAlgoSpecs convAlgoSpecs;
    Activation activation;
    CuConvLayer* cuConvLayer = nullptr;

public:
    ConvLayer(int inChannels, int outChannels, int H, int W, int stride, int padding, int dilation, Activation activation);
    ConvLayer(int inChannels, int outChannels, int H, int W, int stride, int padding, int dilation, Activation activation, ImageInputLayerDims&& convInputLayerDims);
    ~ConvLayer();
    
    /* Layer specific methods */
    ConvLayerDims getSize() const;
    const CuConvLayer* getCuLayer() const;

    /*  Overriden methods */
    void init(const std::vector<float>& weight, const std::vector<float>& bias) override;
    bool canBeStackedOn(const Layer* prevLayer) const override;
    bool hasInputLayer() const override;
    float* forward(const float* input) const override;
    void* getOutput() const override;
};

#endif // !CONVLAYER_CUH