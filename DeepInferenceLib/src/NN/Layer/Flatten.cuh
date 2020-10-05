#ifndef FLATTEN_CUH
#define FLATTEN_CUH

#include "NN/Layer/Layer.cuh"
#include "NN/Layer/DenseLayer.cuh"
#include "Shared/Utills.cuh"

class Flatten final : public Layer
{
private:
    int size;
    Activation activation;
public:
    Flatten();
    void init(const std::vector<float>& weight, const std::vector<float>& bias) override;
    bool canBeStackedOn(const Layer* prevLayer) const override;
    float* forward(const float* input) const override;
    void* getOutput() const override;
};

#endif // !FLATTEN_CUH
