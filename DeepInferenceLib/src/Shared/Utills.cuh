#ifndef UTILLS_CUH
#define UTILLS_CUH

#include "Layer/Layer.cuh"
#include "Layer/ConvLayer.cuh"
#include "Layer/Flatten.cuh"
#include "Layer/DenseLayer.cuh"
#include <typeinfo>

namespace Utills
{
    namespace Layers
    {
        LayerType getLayerType(const Layer* layer);
    }
}

#endif // !UTILLS_CUH
