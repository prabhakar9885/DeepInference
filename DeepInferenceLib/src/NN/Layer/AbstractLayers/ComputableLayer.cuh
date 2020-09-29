#ifndef COMPUTABLELAYER_CUH
#define COMPUTABLELAYER_CUH

#include "Layer.cuh"

/// @brief All the layers which have weighted-incoming-paths should inherit from ComputableLayer
class ComputableLayer : public Layer
{
public:
    /// @brief The weighted-incoming-paths should be initialized with the weight and bias, by taking the dimensions of the layer into consideration.
    /// @param weight 
    /// @param bias 
    virtual void init(const std::vector<float>& weight, const std::vector<float>& bias) = 0;
};
#endif // !COMPUTABLELAYER_CUH
