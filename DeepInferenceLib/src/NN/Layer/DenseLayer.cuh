#ifndef DENSELAYER_CUH
#define DENSELAYER_CUH

#include "Layer.cuh"
#include "Shared/Utills.cuh"

class DenseLayer final: public Layer
{
private:
    int size;
    Activation activation;
public:
    DenseLayer(int size, Activation activation);
    void init() override ;
    bool canBeStackedOn(const Layer* prevLayer) const override;
    void forward(const std::vector<int> &input) const override;
    void* getOutput() const override;

};

#endif // !DENSELAYER_CUH