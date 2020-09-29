#ifndef UTILLS_CUH
#define UTILLS_CUH

#include "NN/Layer/AbstractLayers/Layer.cuh"
#include "NN/Layer/InputLayer.cuh"
#include "NN/Layer/ConvLayer.cuh"
#include "NN/Layer/Flatten.cuh"
#include "NN/Layer/DenseLayer.cuh"
#include <typeinfo>

namespace Utills
{
    namespace Layers
    {
        LayerType getLayerType(const Layer* layer);
    }
}

#endif // !UTILLS_CUH
