
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

class CuConvLayer : public CuLayer
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

    CuConvLayer(int inputChannelCount, int outputChannelCount, int widthOfChannels, int heightOfChannels, Activation activation);
    CuConvLayer(int inputChannelCount, int outputChannelCount, int widthOfChannels, int heightOfChannels, Activation activation, CuLayer* prevLayer);
    ~CuConvLayer();
    void setSizeOfInput(int sizeOfInput);
    void allocMemForLayer();
    void init(const float* weights, const int numberOfWeights, const float* bias, const int numberOfBias) override; /*weights is in RMO*/
    float* compute(const float* x);
    void releaseMem();
    std::vector<int>& getOutput() const;
};

