#ifndef LAYER_CUH
#define LAYER_CUH

#include "Shared/Activation.cuh"
#include <vector>

enum class LayerType
{
    INPUT_LAYER,
    DENSE,
    CONV,
    FLATTEN
};

enum class MemLayout
{
    NHWC,
    NCHW
};

/// @brief Base class for all the layers.
class Layer
{
private:

public:
    virtual ~Layer();
    /// @brief Check is a given type of layer can be cancatinated to the prevLayer.
    /// @param prevLayer 
    /// @return returns true if the layer can be cancatinated to the prevLayer.
    virtual bool canBeStackedOn(const Layer* prevLayer) const = 0;

    /// @brief In case of input-layer, this method will load the input from host to device and return a pointer pointing to the device-mem.
    /// And for other layers, this method will compute A*X+B and return a pointer pointing to the result on device-memory.
    /// @param input 
    /// @return pointer pointing to the data on device memory
    virtual float* forward(const float* input) const = 0;
};

#endif // !LAYER_CUH