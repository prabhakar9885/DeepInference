#ifndef FLATTEN_CUH
#define FLATTEN_CUH

#include "NN/Layer/Layer.cuh"
#include "NN/Layer/DenseLayer.cuh"
#include "Shared/Utills.cuh"
#include "CudaEngine/Layers/CuFlattenedLayer.cuh"

class Flatten final : public Layer
{
private:
    float* dataOnDevice;
    int size;
    CuFlattenedLayer* cuFlattenedLayer = nullptr;
public:
    Flatten();
    bool hasInputLayer() const;
    void init();
    void init(const std::vector<float>& weight, const std::vector<float>& bias) override;
    bool canBeStackedOn(const Layer* prevLayer) const override;
    float* forward(const float* input) const override;
    const CuFlattenedLayer* getCuLayer() const;
    void* getOutput() const override;
};

#endif // !FLATTEN_CUH
