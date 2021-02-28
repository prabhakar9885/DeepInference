#ifndef CUFLATTENEDLAYER_CUH
#define CUFLATTENEDLAYER_CUH

#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include "CudaEngine/CuUtills.cuh"
#include "CudaEngine/Layers/CuLayer.cuh"
#include "CudaEngine/Layers/CuConvLayer.cuh"
#include "Shared/Activation.cuh"

#define checkCUDNN(expression)                               \
  {                                                          \
    cudnnStatus_t status = (expression);                     \
    if (status != CUDNN_STATUS_SUCCESS) {                    \
      std::cerr << "Error on line " << __LINE__ << ": "      \
                << cudnnGetErrorString(status) << std::endl; \
      std::exit(EXIT_FAILURE);                               \
    }                                                        \
  }

class CuFlattenedLayer : public CuLayer
{

private:
    float* outputOnDevice = nullptr;
    int sizeOfCurrentLayer = 0;
    const CuConvLayer* prevLayer;

public:
    static cublasHandle_t handle;

    CuFlattenedLayer(const CuConvLayer* prevLayer);

    /* Layer specific methods */

    /*  Overriden methods */
    float* compute(const float* x) override;
    void allocMemForLayer() override;
    void init(const float* weights, const int numberOfWeights, const float* bias, const int numberOfBias) override; /*weights is in RMO*/
    std::vector<float>&& getOutput() const override;
};


#endif // !CUFLATTENEDLAYER_CUH
