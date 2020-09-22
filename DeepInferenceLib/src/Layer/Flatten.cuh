#ifndef FLATTEN_CUH
#define FLATTEN_CUH

#include "Layer/Layer.cuh"
#include "Layer/DenseLayer.cuh"
#include "Shared/Utills.cuh"

class Flatten final : public Layer
{
private:
    int size;
    Activation activation;
public:
    Flatten();
    void init() override;
    bool canBeStackedOn(const Layer* prevLayer) const override;
    void forward(const std::vector<int> &input) const override;
    void* getOutput() const override;
};

#endif // !FLATTEN_CUH
