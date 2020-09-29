#ifndef INPUTLAYER_CUH
#define INPUTLAYER_CUH

#include "AbstractLayers/Layer.cuh"
#include "Shared/Utills.cuh"
#include "Shared/DataStructs.cuh"
#include "CudaEngine/Layers/CuLayer.cuh"
#include "CudaEngine/Layers/CuDenseLayer.cuh"
#include "AbstractLayers/PlaceholderLayer.cuh"


class InputLayer final : public PlaceholderLayer
{
private:
    LayerType layerType;
    DenseLayerWeightDims denseInputLayer;
    ConvLayerDims convLayerDims;
    CuLayer* cuLayer;
public:
    InputLayer(int size, Activation activation);    // Create a Dense input layer
    InputLayer(int N, int H, int W, int C);         // Create a Conv input layer
    ~InputLayer();
    void init() override;
    bool canBeStackedOn(const Layer* prevLayer) const override;
    float* forward(const float* input) const override;
    LayerType getLayerType() const;
};
#endif // !INPUTLAYER_CUH
