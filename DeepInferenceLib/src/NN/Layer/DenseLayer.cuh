#ifndef DENSELAYER_CUH
#define DENSELAYER_CUH

#include "Layer.cuh"
#include "Shared/Utills.cuh"
#include "CudaEngine/Layers/CuDenseLayer.cuh"

class DenseLayer final: public Layer
{
private:
    int size;
    int inputSize = -1;
    Activation activation;
    CuDenseLayer* cuDenseLayer = nullptr;
    bool inputSizeIsSet = false;
public:
    DenseLayer(int size, Activation activation);
    DenseLayer(int size, Activation activation, int inputSize);
    ~DenseLayer();
    void init(const std::vector<float>& weight, const std::vector<float>& bias) override;
    bool canBeStackedOn(const Layer* prevLayer) const override;
    bool hasInputLayer() const override;
    float* forward(const float* input) const override;
    int getSize() const;
    void* getOutput() const override;

};

#endif // !DENSELAYER_CUH