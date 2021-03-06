#ifndef CUCONVLAYER_CUH
#define CUCONVLAYER_CUH

#define DEBUG

#include <iostream>
#include <iomanip>
#include <cudnn.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include "CudaEngine/CuUtills.cuh"
#include "CudaEngine/Layers/CuLayer.cuh"
#include "Shared/Activation.cuh"
#include "Shared/DataStructs.cuh"

#define checkCUDNN(expression)                               \
  {                                                          \
    cudnnStatus_t status = (expression);                     \
    if (status != CUDNN_STATUS_SUCCESS) {                    \
      std::cerr << "Error on line " << __LINE__ << ": "      \
                << cudnnGetErrorString(status) << std::endl; \
      std::exit(EXIT_FAILURE);                               \
    }                                                        \
  }

#define checkCUDA(expression)                               \
  {                                                          \
    cudaError_t status = (expression);                     \
    if (status != cudaSuccess) {                    \
      std::cerr << "Error on line " << __LINE__ << ": "      \
                << cudaGetErrorString(status) << std::endl; \
      std::cerr << status << std::endl; \
      std::exit(EXIT_FAILURE);                               \
    }                                                        \
  }

class CuConvLayer : public CuLayer
{
private:
    CuKernel cuKernel;
    Tensor4D cuInput;
    Tensor4D cuOutput;
    CuConvolution cuConvolution;
    CuWorkspace cuWorkspace;
    Activation activation;

    CuConvLayer(int inputChannelCount, int outputChannelCount, int heightOfChannels, int widthOfChannels,
        int padding, int stride, int dilation, Activation activation);

public:
    static cudnnHandle_t handle;

    CuConvLayer(int inputChannelCount, int outputChannelCount, int heightOfChannels, int widthOfChannels,
        int padding, int stride, int dilation, const CuConvLayer* prevLayer, Activation activation);
    CuConvLayer(int inputChannelCount, int outputChannelCount, int heightOfChannels, int widthOfChannels,
        int padding, int stride, int dilation,
        int inputImageBatchSize, int inputImageChannels, int inputImageHeight, int inputImageWidth, Activation activation);
    ~CuConvLayer();

    /* Layer specific methods */
    const Tensor4D& getOutputOnDevice() const;

    /*  Overriden methods */
    void allocMemForLayer() override;
    float* compute(const float* x) override;
    void init(const float* weights, const int numberOfWeights, const float* bias, const int numberOfBias) override; /*weights is in RMO*/
    std::vector<float>&& getOutput() const override;
};


#endif // !CUCONVLAYER_CUH