#ifndef LAYER_CUH
#define LAYER_CUH

#include "Shared/Activation.cuh"
#include <vector>

enum class LayerType
{
    DENSE,
    CONV,
    FLATTEN
};

enum class MemLayout
{
    NHWC,
    NCHW
};

class Layer
{
protected:
    Layer* prevLayer = nullptr;
public:
    virtual ~Layer();
    virtual bool hasInputLayer() const = 0;
    virtual void init(const std::vector<float> &weight, const std::vector<float> &bias) = 0;
    virtual bool canBeStackedOn(const Layer* prevLayer) const = 0;
    void setPrevLayer(Layer* prevLayer);
    virtual float* forward(const float* input) const = 0;
    virtual void* getOutput() const = 0;
};

#endif // !LAYER_CUH