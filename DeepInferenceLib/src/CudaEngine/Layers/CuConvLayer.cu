#include "CudaEngine/Layers/CuConvLayer.cuh"

cudnnHandle_t CuConvLayer::handle;

void CuConvLayer::allocMemForLayer()
{
    if (this->hasInputLayer()) {
        checkCUDNN(cudnnCreateTensorDescriptor(&this->cuInput.descriptor));
        checkCUDNN(cudnnSetTensor4dDescriptor(this->cuInput.descriptor,
            /*format=*/CUDNN_TENSOR_NHWC,
            /*dataType=*/CUDNN_DATA_FLOAT,
            /*batch_size=*/this->cuInput.batchSize,
            /*channels=*/this->cuInput.channelCount,
            /*image_height=*/this->cuInput.height,
            /*image_width=*/this->cuInput.width));
        cudaMallocManaged(&(this->cuInput.onDevice), this->cuInput.sizeInBytes);
    }

    checkCUDNN(cudnnCreateFilterDescriptor(&this->cuKernel.descriptor));
    checkCUDNN(cudnnSetFilter4dDescriptor(this->cuKernel.descriptor,
        /*dataType=*/CUDNN_DATA_FLOAT,
        /*format=*/CUDNN_TENSOR_NCHW,
        /*out_channels=*/ this->cuKernel.outChannels,
        /*in_channels=*/ this->cuKernel.inChannels,
        /*kernel_height=*/ this->cuKernel.heightOfChannel,
        /*kernel_width=*/ this->cuKernel.widthOfChannel));
    cudaMallocManaged(&(this->cuKernel.onDevice), this->cuKernel.sizeInBytes);

    checkCUDNN(cudnnCreateConvolutionDescriptor(&this->cuConvolution.descriptor));
    checkCUDNN(cudnnSetConvolution2dDescriptor(this->cuConvolution.descriptor,
        /*pad_height=*/this->cuConvolution.padding,
        /*pad_width=*/this->cuConvolution.padding,
        /*vertical_stride=*/this->cuConvolution.stride,
        /*horizontal_stride=*/this->cuConvolution.stride,
        /*dilation_height=*/this->cuConvolution.dilation,
        /*dilation_width=*/this->cuConvolution.dilation,
        /*mode=*/CUDNN_CROSS_CORRELATION,
        /*computeType=*/CUDNN_DATA_FLOAT));
    checkCUDNN(cudnnGetConvolution2dForwardOutputDim(this->cuConvolution.descriptor,
        this->cuInput.descriptor,
        this->cuKernel.descriptor,
        &(this->cuOutput.batchSize),
        &(this->cuOutput.channelCount),
        &(this->cuOutput.height),
        &(this->cuOutput.width)));

    Tensor4D& cuOut = this->cuOutput;
    checkCUDNN(cudnnCreateTensorDescriptor(&cuOut.descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(cuOut.descriptor,
        /*format=*/CUDNN_TENSOR_NHWC,
        /*dataType=*/CUDNN_DATA_FLOAT,
        /*batch_size=*/cuOut.batchSize,
        /*channels=*/cuOut.channelCount,
        /*image_height=*/cuOut.height,
        /*image_width=*/cuOut.width));
    cuOut.sizeInBytes = sizeof(float) * cuOut.batchSize * cuOut.channelCount * cuOut.height * cuOut.width;
    cudaMallocManaged(&(cuOut.onDevice), cuOut.sizeInBytes);
    cudaMemset(cuOut.onDevice, 0, cuOut.sizeInBytes);

    checkCUDNN(cudnnGetConvolutionForwardAlgorithm(CuConvLayer::handle,
        this->cuInput.descriptor,
        this->cuKernel.descriptor,
        this->cuConvolution.descriptor,
        this->cuOutput.descriptor,
        CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
        /*memoryLimitInBytes=*/0,
        &this->cuConvolution.algo));

    checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(CuConvLayer::handle,
        this->cuInput.descriptor,
        this->cuKernel.descriptor,
        this->cuConvolution.descriptor,
        this->cuOutput.descriptor,
        this->cuConvolution.algo,
        &(this->cuWorkspace.sizeInBytes)));
    cudaMallocManaged(&this->cuWorkspace.onDevice, this->cuWorkspace.sizeInBytes);
}


CuConvLayer::CuConvLayer(int inputChannelCount, int outputChannelCount, int widthOfChannels, int heightOfChannels,
    int padding, int stride, int dilation, Activation activation)
{
    this->cuKernel.outChannels = outputChannelCount;
    this->cuKernel.inChannels = inputChannelCount;
    this->cuKernel.heightOfChannel = heightOfChannels;
    this->cuKernel.widthOfChannel = widthOfChannels;
    this->cuKernel.sizeInBytes = sizeof(float) * outputChannelCount * inputChannelCount * heightOfChannels * widthOfChannels;

    this->cuConvolution.padding = padding;
    this->cuConvolution.stride = stride;
    this->cuConvolution.dilation = dilation;

    this->activation = activation;
}

/// @brief Use this for Creating Conv input-layer 
/// @param inputChannelCount 
/// @param outputChannelCount 
/// @param widthOfChannels 
/// @param heightOfChannels 
/// @param padding 
/// @param stride 
/// @param dilation 
/// @param inputImageBatchSize 
/// @param inputImageChannels 
/// @param inputImageHeight 
/// @param inputImageWidth 
/// @param activation 
CuConvLayer::CuConvLayer(int inputChannelCount, int outputChannelCount, int heightOfChannels, int widthOfChannels, 
    int padding, int stride, int dilation,
    int inputImageBatchSize, int inputImageChannels, int inputImageHeight, int inputImageWidth, Activation activation) : 
        CuConvLayer(inputChannelCount, outputChannelCount, widthOfChannels, heightOfChannels, padding, stride, dilation, activation)
{
    this->isInputLayer = true;
    cudaSetDevice(0);
    checkCUDNN(cudnnCreate(&CuConvLayer::handle));

    this->cuInput.batchSize = inputImageBatchSize;
    this->cuInput.channelCount = inputImageChannels;
    this->cuInput.height = inputImageHeight;
    this->cuInput.width = inputImageWidth;
    this->cuInput.sizeInBytes = sizeof(float) * inputImageBatchSize * inputImageChannels * inputImageHeight * inputImageWidth;
}

/// @brief Use if for creating hidden Conv input-layers
/// @param inputChannelCount 
/// @param outputChannelCount 
/// @param widthOfChannels 
/// @param heightOfChannels 
/// @param padding 
/// @param stride 
/// @param dilation 
/// @param prevLayer 
/// @param activation 
CuConvLayer::CuConvLayer(int inputChannelCount, int outputChannelCount, int widthOfChannels, int heightOfChannels,
    int padding, int stride, int dilation, 
    const CuConvLayer* prevLayer, Activation activation) :
        CuConvLayer(inputChannelCount, outputChannelCount, widthOfChannels, heightOfChannels, padding, stride, dilation, activation)
{
    this->cuInput = std::move(prevLayer->cuOutput);
}

CuConvLayer::~CuConvLayer()
{
}

/// @brief 
/// @param weights are in NCHW order
/// @param numberOfWeights 
/// @param bias it's not supported
/// @param numberOfBias it's not supported
void CuConvLayer::init(const float* weights, const int numberOfWeights, const float* bias, const int numberOfBias)
{
    CuKernel& cuKernel = this->cuKernel;
    if (numberOfWeights * sizeof(float) != cuKernel.sizeInBytes)
        throw "Number of weights received doesn't fit the expectation.";
    checkCUDA(cudaMemcpy(cuKernel.onDevice, weights, cuKernel.sizeInBytes, cudaMemcpyHostToDevice));
    cudaDeviceSynchronize();
}

float* CuConvLayer::compute(const float* x)
{
    const float alpha = 1.0f, beta = 0.0f;

    if(this->hasInputLayer())
        checkCUDA(cudaMemcpy(cuInput.onDevice, x, cuInput.sizeInBytes, cudaMemcpyHostToDevice));

    checkCUDNN(cudnnConvolutionForward(CuConvLayer::handle,
        &alpha,
        this->cuInput.descriptor,
        this->cuInput.onDevice,
        this->cuKernel.descriptor,
        this->cuKernel.onDevice,
        this->cuConvolution.descriptor,
        this->cuConvolution.algo,
        this->cuWorkspace.onDevice,
        this->cuWorkspace.sizeInBytes,
        &beta,
        this->cuOutput.descriptor,
        this->cuOutput.onDevice));
    cudaDeviceSynchronize();

    #ifdef DEBUG
        std::cout << "\n=x=x=x=x=x=x=x=x=x=x=x=x=x=x=";
        int N = cuOutput.batchSize;
        int C = cuOutput.channelCount;
        int H = cuOutput.height;
        int W = cuOutput.width;
        float* data = new float[(long long)N * C * H * W];
        checkCUDA(cudaMemcpy(data, cuOutput.onDevice, cuOutput.sizeInBytes, cudaMemcpyDeviceToHost));
        int i = 0;
        for (int n = 0; n < N; n++)
        {
            std::cout << "\nBatch: " << N;
            for (int c = 0; c < C; c++)
            {
                std::cout << "\nChannel: " << c << " => \n";
                for (int h = 0; h < H; h++)
                {
                    for (int w = 0; w < W; w++)
                        std::cout << std::setw(6) << data[c + (w + (h + (n)*H) * W) * C] << "/" << data[i++];
                    std::cout << "\n";
                }
            }
        }
        delete(data);
    #endif // DEBUG
        return cuOutput.onDevice;
}

const Tensor4D& CuConvLayer::getOutputOnDevice() const
{
    return this->cuOutput;
}

std::vector<float>&& CuConvLayer::getOutput() const
{
    const Tensor4D& output = this->cuOutput;
    return std::vector<float>(output.onDevice, output.onDevice + output.sizeInBytes / sizeof(float));
}

