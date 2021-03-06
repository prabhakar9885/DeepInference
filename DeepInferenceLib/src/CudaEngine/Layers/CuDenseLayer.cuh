#ifndef CUDENSELAYER_CUH
#define CUDENSELAYER_CUH

#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include "CudaEngine/CuUtills.cuh"
#include "CudaEngine/Layers/CuLayer.cuh"
#include "Shared/Activation.cuh"

class CuDenseLayer : public CuLayer
{

private:
    float* aDevice = nullptr;       /*Stores the weight matrix in CMO*/
    int aDeviceCount = 0;           /*Number of elements in aDevice[]*/
    float* bDevice = nullptr;       /*Stores the bias*/
    int sizeOfCurrentLayer = 0;     /*This will be same as the number of elements in bDevice[]*/
    int sizeOfInput = -1;           /*Size of input the current layer expects. Valied*/
    float* inputDevice = nullptr;
    Activation activation;

public:
    static cublasHandle_t handle;

    CuDenseLayer(int sizeOfCurrentLayer, Activation activation);
    CuDenseLayer(int sizeOfCurrentLayer, Activation activation, CuLayer* prevLayer);
    ~CuDenseLayer();

    /* Layer specific methods */
    void setSizeOfInput(int sizeOfInput);

    /*  Overriden methods */
    void allocMemForLayer() override;
    float* compute(const float* x) override;
    void init(const float* weights, const int numberOfWeights, const float* bias, const int numberOfBias) override; /*weights is in RMO*/
    std::vector<float>&& getOutput() const override;
};


#endif // !CUDENSELAYER_CUH
