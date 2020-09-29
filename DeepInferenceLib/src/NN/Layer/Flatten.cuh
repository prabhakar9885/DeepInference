#ifndef FLATTEN_CUH
#define FLATTEN_CUH

#include "NN/Layer/AbstractLayers/Layer.cuh"
#include "NN/Layer/AbstractLayers/PlaceholderLayer.cuh"
#include "NN/Layer/DenseLayer.cuh"
#include "Shared/Utills.cuh"

class Flatten final : public PlaceholderLayer
{
private:
    int size;
    Activation activation;
public:
    Flatten();
    void init() override;
    bool canBeStackedOn(const Layer* prevLayer) const override;
    float* forward(const float* input) const override;
};

#endif // !FLATTEN_CUH
