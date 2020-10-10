
#include "CudaEngine/Layers/CuDenseLayer.cuh"
#define IDX2C(i,j,ld) (j*ld + i)

cublasHandle_t CuDenseLayer::handle;

CuDenseLayer::CuDenseLayer(int sizeOfCurrentLayer, Activation activation) : sizeOfCurrentLayer{ sizeOfCurrentLayer }, activation{ activation }
{
    this->isInputLayer = true;
}

CuDenseLayer::CuDenseLayer(int sizeOfCurrentLayer, Activation activation, const CuLayer* prevLayer) : sizeOfCurrentLayer{ sizeOfCurrentLayer }, activation{ activation }
{
    this->prevLayer = prevLayer;
}

CuDenseLayer::~CuDenseLayer()
{
    std::cout << "\nCuDenseLayer->Destructor...";
    cudaError_t status;
    if (aDevice && (status = cudaFree(aDevice)) != cudaSuccess)
    {
        std::cerr << "Failed to release device memory. Status code: " << status;
    }
    if (bDevice && (status = cudaFree(bDevice)) != cudaSuccess)
    {
        std::cerr << "Failed to release device memory. Status code: " << status;
    }
    cublasStatus_t cublasStatus;
    if (this->hasInputLayer() && (cublasStatus = cublasDestroy(CuDenseLayer::handle)) != CUBLAS_STATUS_SUCCESS)
    {
        std::cerr << "Failed to destroy cublas-handle. Status code: " << cublasStatus;
    }
}

void CuDenseLayer::setSizeOfInput(int sizeOfInput)
{
    this->sizeOfInput = sizeOfInput;
}

void CuDenseLayer::allocMemForLayer()
{
    if (this->hasInputLayer())
    {
        if (cudaSuccess != cudaMallocManaged((void**)&inputDevice, this->sizeOfInput * sizeof(float)))
            throw "Unable to allocate Input memory";
        cublasCreate(&CuDenseLayer::handle);
    }
    this->aDeviceCount = this->sizeOfCurrentLayer * this->sizeOfInput;
    if (cudaSuccess != cudaMallocManaged((void**)&aDevice, this->aDeviceCount * sizeof(float)))
        throw "Unable to allocate memory";
    if (cudaSuccess != cudaMallocManaged((void**)&bDevice, this->sizeOfCurrentLayer * sizeof(float)))
        throw "Unable to allocate memory";
}

void CuDenseLayer::init(const float* weights, const int numberOfWeights, const float* bias, const int numberOfBias)
{
    if (this->sizeOfCurrentLayer != numberOfBias)
        throw "Size of bias and the number of nodes in the layer should match";
    if (this->aDeviceCount != numberOfWeights)
        throw " sizeOfPrevLayer * sizeOfCurrLayer != numberOfWeights";

    int wtIndx = 0;
    int sizeOfPreviousLayer = numberOfWeights / this->sizeOfCurrentLayer;
    for (int j = 0; j < sizeOfPreviousLayer; j++)
    {
        for (int i = 0; i < this->sizeOfCurrentLayer; i++)
        {
            this->aDevice[IDX2C(i, j, this->sizeOfCurrentLayer)] = weights[wtIndx];
            wtIndx++;
        }
    }
    for (int i = 0; i < this->sizeOfCurrentLayer; i++)
    {
        this->bDevice[i] = bias[i];
    }
}

float* CuDenseLayer::compute(const float* xDevice)
{
    float alpha = 1, beta = 1;
    if (this->hasInputLayer())
    {
        cublasSetVector(this->sizeOfInput, sizeof(float), xDevice, 1, this->inputDevice, 1);
        xDevice = this->inputDevice;
    }
    int sizeOfPreviousLayer = this->aDeviceCount / this->sizeOfCurrentLayer;
    cublasStatus_t status;
    status = cublasSgemv(handle, CUBLAS_OP_N, this->sizeOfCurrentLayer, sizeOfPreviousLayer, &alpha, this->aDevice, this->sizeOfCurrentLayer, xDevice, 1, &beta, this->bDevice, 1);
    if (status != CUBLAS_STATUS_SUCCESS)
        throw "cuBLAS operation failure";
    if (this->activation != Activation::NONE)
        CuUtills::computeActivation(this->bDevice, this->sizeOfCurrentLayer, this->activation);
    cudaDeviceSynchronize();
    return this->bDevice;
}

void CuDenseLayer::releaseMem()
{
    cudaFree(this->aDevice);
    cudaFree(this->bDevice);
}

std::vector<float>&& CuDenseLayer::getOutput() const
{
    return std::vector<float>(this->bDevice, this->bDevice + this->sizeOfCurrentLayer);
}

