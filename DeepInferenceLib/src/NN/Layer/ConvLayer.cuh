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
    void init(const std::vector<float> &weight, const std::vector<float> &bias) override;
    void initAsInputLayer() override;
    float* forward(const float* input) const override;
    void* getOutput() const override;
};

#endif // !CONVLAYER_CUH