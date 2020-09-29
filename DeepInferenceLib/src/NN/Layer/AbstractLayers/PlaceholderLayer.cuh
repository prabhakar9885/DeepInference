#ifndef PLACEHOLDER_CUH
#define PLACEHOLDER_CUH

#include "Layer.cuh"

/// @brief All the layers which don't have incoming-weighted-edges must implement this layer.
class PlaceholderLayer : public Layer
{
public:
    /// @brief The allocation of resources is done here.
    virtual void init() = 0;
};
#endif // !PLACEHOLDER_CUH
