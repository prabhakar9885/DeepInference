#ifndef UTILLS_CU
#define UTILLS_CU

#include "Utills.cuh"

LayerType Utills::Layers::getLayerType(const Layer* layer)
{
    LayerType type;
    const std::type_info& typeInfoOfLayer = typeid(*layer);
    if (typeInfoOfLayer == typeid(ConvLayer))
        type = LayerType::CONV;
    if (typeInfoOfLayer == typeid(Flatten))
        type = LayerType::FLATTEN;
    if (typeInfoOfLayer == typeid(DenseLayer))
        type = LayerType::DENSE;
    return type;
}

#endif // !UTILLS_CU
