#ifndef UTILLS_CUH
#define UTILLS_CUH

#include "NN/Layer/Layer.cuh"
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

    namespace StringUtils
    {
        // trim from start (in place)
        void ltrim(std::string& s);

        // trim from end (in place)
        void rtrim(std::string& s);

        // trim from both ends (in place)
        void trim(std::string& s);

        // trim from start (copying)
        std::string ltrim_copy(std::string s);

        // trim from end (copying)
        std::string rtrim_copy(std::string s);

        // trim from both ends (copying)
        std::string trim_copy(std::string s);
    }
}

#endif // !UTILLS_CUH
