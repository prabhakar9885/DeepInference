#ifndef CONVLAYER_CUH
#define CONVLAYER_CUH

#include "Layer.cuh"
#include "Shared/Utills.cuh"

class ConvLayer final : public Layer
{
private:
    int padding, dilation;
    Activation activation;
public:
    int inChannels, outChannels, H, W;

    ConvLayer(int size, Activation activation) = delete;
    ConvLayer(int inChannels, int outChannels, int H, int W, int padding, int dilation, Activation activation);
    bool canBeStackedOn(const Layer* prevLayer) const;
    void init() override;
    void forward(const std::vector<int> &input) const override;
    void* getOutput() const override;
};

#endif // !CONVLAYER_CUH