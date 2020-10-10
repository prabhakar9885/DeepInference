#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include "CudaEngine/CuUtills.cuh"
#include "CudaEngine/Layers/CuLayer.cuh"
#include "CudaEngine/Layers/CuConvLayer.cuh"
#include "Shared/Activation.cuh"

class CuFlattenedLayer : public CuLayer
{
private:
    float* outputOnDevice = nullptr;
    int sizeOfCurrentLayer = 0;
    CuConvLayer* prevLayer;
public:
    static cublasHandle_t handle;

    CuFlattenedLayer(const CuConvLayer* prevLayer);
    void init();
    void init(const float* weights, const int numberOfWeights, const float* bias, const int numberOfBias) override; /*weights is in RMO*/
    float* compute(const float* x);
    std::vector<float>&& getOutput() const override;
};

