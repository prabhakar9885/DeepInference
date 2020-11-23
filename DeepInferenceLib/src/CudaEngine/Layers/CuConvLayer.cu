#include "CudaEngine/Layers/CuConvLayer.cuh"

cudnnHandle_t CuConvLayer::handle;

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
    int inputImageBatchSize, int inputImageChannels, int inputImageHeight, int inputImageWidth, Activation activation)
{
    this->isInputLayer = true;
    cudaSetDevice(0);
    checkCUDNN(cudnnCreate(&CuConvLayer::handle));

    checkCUDNN(cudnnCreateTensorDescriptor(&this->cuInput.descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(this->cuInput.descriptor,
        /*format=*/CUDNN_TENSOR_NHWC,
        /*dataType=*/CUDNN_DATA_FLOAT,
        /*batch_size=*/inputImageBatchSize,
        /*channels=*/inputImageChannels,
        /*image_height=*/inputImageHeight,
        /*image_width=*/inputImageWidth));
    this->cuInput.sizeInBytes = sizeof(float) * inputImageBatchSize * inputImageChannels * inputImageHeight * inputImageWidth;
    cudaMallocManaged(&(this->cuInput.onDevice), this->cuInput.sizeInBytes);

    checkCUDNN(cudnnCreateFilterDescriptor(&this->cuKernel.descriptor));
    checkCUDNN(cudnnSetFilter4dDescriptor(this->cuKernel.descriptor,
        /*dataType=*/CUDNN_DATA_FLOAT,
        /*format=*/CUDNN_TENSOR_NCHW,
        /*out_channels=*/ (this->cuKernel.outChannels = outputChannelCount),
        /*in_channels=*/ (this->cuKernel.inChannels = inputChannelCount),
        /*kernel_height=*/ (this->cuKernel.heightOfChannel = heightOfChannels),
        /*kernel_width=*/ (this->cuKernel.widthOfChannel = widthOfChannels)));
    this->cuKernel.sizeInBytes = sizeof(float) * outputChannelCount * inputChannelCount * heightOfChannels * widthOfChannels;
    cudaMallocManaged(&(this->cuKernel.onDevice), this->cuKernel.sizeInBytes);

    checkCUDNN(cudnnCreateConvolutionDescriptor(&this->cuConvolution.descriptor));
    checkCUDNN(cudnnSetConvolution2dDescriptor(this->cuConvolution.descriptor,
        /*pad_height=*/padding,
        /*pad_width=*/padding,
        /*vertical_stride=*/stride,
        /*horizontal_stride=*/stride,
        /*dilation_height=*/dilation,
        /*dilation_width=*/dilation,
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

    this->activation = activation;
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
    const CuConvLayer* prevLayer, Activation activation)
{
    this->cuInput = std::move(prevLayer->cuOutput);

    checkCUDNN(cudnnCreateFilterDescriptor(&this->cuKernel.descriptor));
    checkCUDNN(cudnnSetFilter4dDescriptor(this->cuKernel.descriptor,
        /*dataType=*/CUDNN_DATA_FLOAT,
        /*format=*/CUDNN_TENSOR_NCHW,
        /*out_channels=*/ (this->cuKernel.outChannels = outputChannelCount),
        /*in_channels=*/ (this->cuKernel.inChannels = inputChannelCount),
        /*kernel_height=*/ (this->cuKernel.heightOfChannel = heightOfChannels),
        /*kernel_width=*/ (this->cuKernel.widthOfChannel = widthOfChannels)));
    this->cuKernel.sizeInBytes = sizeof(float) * outputChannelCount * inputChannelCount * heightOfChannels * widthOfChannels;
    checkCUDA(cudaMallocManaged(&(this->cuKernel.onDevice), this->cuKernel.sizeInBytes));

    checkCUDNN(cudnnCreateConvolutionDescriptor(&this->cuConvolution.descriptor));
    checkCUDNN(cudnnSetConvolution2dDescriptor(this->cuConvolution.descriptor,
        /*pad_height=*/padding,
        /*pad_width=*/padding,
        /*vertical_stride=*/stride,
        /*horizontal_stride=*/stride,
        /*dilation_height=*/dilation,
        /*dilation_width=*/dilation,
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
    checkCUDA(cudaMallocManaged(&(cuOut.onDevice), cuOut.sizeInBytes));
    checkCUDA(cudaMemset(cuOut.onDevice, 0, cuOut.sizeInBytes));

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
    checkCUDA(cudaMallocManaged(&this->cuWorkspace.onDevice, this->cuWorkspace.sizeInBytes));

    this->activation = activation;
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

