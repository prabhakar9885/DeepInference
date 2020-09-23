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
private:

public:
    virtual void init() = 0;
    virtual bool canBeStackedOn(const Layer* prevLayer) const = 0;
    virtual void forward(const std::vector<int> &input) const = 0;
    virtual void* getOutput() const = 0;
};

#endif // !LAYER_CUH