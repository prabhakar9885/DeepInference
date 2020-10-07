#include "CudaEngine/Layers/CuConvLayer.cuh"

cudnnHandle_t CuConvLayer::handle;

CuConvLayer::CuConvLayer(int inputChannelCount, int outputChannelCount, int widthOfChannels, int heightOfChannels, 
    int padding, int stride, int dilation,
    int inputImageBatchSize, int inputImageChannels, int inputImageHeight, int inputImageWidth, Activation activation)
{
    this->isInputLayer = true;
    checkCUDNN(cudnnCreateTensorDescriptor(this->cuInput.descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(*this->cuInput.descriptor,
        /*format=*/CUDNN_TENSOR_NHWC,
        /*dataType=*/CUDNN_DATA_FLOAT,
        /*batch_size=*/inputImageBatchSize,
        /*channels=*/inputImageChannels,
        /*image_height=*/inputImageHeight,
        /*image_width=*/inputImageWidth));
    this->cuInput.sizeInBytes = sizeof(float) * inputImageBatchSize * inputImageChannels * inputImageHeight * inputImageWidth;
    cudaMalloc(&(this->cuInput.onDevice), this->cuInput.sizeInBytes);

    checkCUDNN(cudnnCreateFilterDescriptor(this->cuKernel.descriptor));
    checkCUDNN(cudnnSetFilter4dDescriptor(*this->cuKernel.descriptor,
        /*dataType=*/CUDNN_DATA_FLOAT,
        /*format=*/CUDNN_TENSOR_NCHW,
        /*out_channels=*/outputChannelCount,
        /*in_channels=*/inputChannelCount,
        /*kernel_height=*/heightOfChannels,
        /*kernel_width=*/widthOfChannels));

    checkCUDNN(cudnnCreateConvolutionDescriptor(this->cuConvolution.descriptor));
    checkCUDNN(cudnnSetConvolution2dDescriptor(*this->cuConvolution.descriptor,
        /*pad_height=*/padding,
        /*pad_width=*/padding,
        /*vertical_stride=*/stride,
        /*horizontal_stride=*/stride,
        /*dilation_height=*/dilation,
        /*dilation_width=*/dilation,
        /*mode=*/CUDNN_CROSS_CORRELATION,
        /*computeType=*/CUDNN_DATA_FLOAT));
    checkCUDNN(cudnnGetConvolution2dForwardOutputDim(*this->cuConvolution.descriptor,
        *this->cuInput.descriptor,
        *this->cuKernel.descriptor,
        &(this->cuOutput.batchSize),
        &(this->cuOutput.channelCount),
        &(this->cuOutput.height),
        &(this->cuOutput.width)));

    Tensor4D& cuOut = this->cuOutput;
    checkCUDNN(cudnnCreateTensorDescriptor(cuOut.descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(*cuOut.descriptor,
        /*format=*/CUDNN_TENSOR_NHWC,
        /*dataType=*/CUDNN_DATA_FLOAT,
        /*batch_size=*/cuOut.batchSize,
        /*channels=*/cuOut.channelCount,
        /*image_height=*/cuOut.height,
        /*image_width=*/cuOut.width));
    cuOut.sizeInBytes = sizeof(float) * cuOut.batchSize * cuOut.channelCount * cuOut.height * cuOut.width;
    cudaMalloc(&(cuOut.onDevice), cuOut.sizeInBytes);
    cudaMemset(cuOut.onDevice, 0, cuOut.sizeInBytes);

    checkCUDNN(cudnnGetConvolutionForwardAlgorithm(CuConvLayer::handle,
        *this->cuInput.descriptor,
        *this->cuKernel.descriptor,
        *this->cuConvolution.descriptor,
        *this->cuOutput.descriptor,
        CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
        /*memoryLimitInBytes=*/0,
        this->cuConvolution.algo));

    checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(CuConvLayer::handle,
        *this->cuInput.descriptor,
        *this->cuKernel.descriptor,
        *this->cuConvolution.descriptor,
        *this->cuOutput.descriptor,
        *this->cuConvolution.algo,
        &(this->cuWorkspace.sizeInBytes)));
    cudaMalloc(&this->cuWorkspace.onDevice, this->cuWorkspace.sizeInBytes);

    this->activation = activation;
}

CuConvLayer::CuConvLayer(int inputChannelCount, int outputChannelCount, int widthOfChannels, int heightOfChannels,
    int padding, int stride, int dilation, 
    CuConvLayer* prevLayer, Activation activation)
{
    this->cuInput = std::move(prevLayer->cuOutput);

    checkCUDNN(cudnnCreateFilterDescriptor(this->cuKernel.descriptor));
    checkCUDNN(cudnnSetFilter4dDescriptor(*this->cuKernel.descriptor,
        /*dataType=*/CUDNN_DATA_FLOAT,
        /*format=*/CUDNN_TENSOR_NCHW,
        /*out_channels=*/outputChannelCount,
        /*in_channels=*/inputChannelCount,
        /*kernel_height=*/heightOfChannels,
        /*kernel_width=*/widthOfChannels));

    checkCUDNN(cudnnCreateConvolutionDescriptor(this->cuConvolution.descriptor));
    checkCUDNN(cudnnSetConvolution2dDescriptor(*this->cuConvolution.descriptor,
        /*pad_height=*/padding,
        /*pad_width=*/padding,
        /*vertical_stride=*/stride,
        /*horizontal_stride=*/stride,
        /*dilation_height=*/dilation,
        /*dilation_width=*/dilation,
        /*mode=*/CUDNN_CROSS_CORRELATION,
        /*computeType=*/CUDNN_DATA_FLOAT));
    checkCUDNN(cudnnGetConvolution2dForwardOutputDim(*this->cuConvolution.descriptor,
        *this->cuInput.descriptor,
        *this->cuKernel.descriptor,
        &(this->cuOutput.batchSize),
        &(this->cuOutput.channelCount),
        &(this->cuOutput.height),
        &(this->cuOutput.width)));

    Tensor4D& cuOut = this->cuOutput;
    checkCUDNN(cudnnCreateTensorDescriptor(cuOut.descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(*cuOut.descriptor,
        /*format=*/CUDNN_TENSOR_NHWC,
        /*dataType=*/CUDNN_DATA_FLOAT,
        /*batch_size=*/cuOut.batchSize,
        /*channels=*/cuOut.channelCount,
        /*image_height=*/cuOut.height,
        /*image_width=*/cuOut.width));
    cuOut.sizeInBytes = sizeof(float) * cuOut.batchSize * cuOut.channelCount * cuOut.height * cuOut.width;
    cudaMalloc(&(cuOut.onDevice), cuOut.sizeInBytes);
    cudaMemset(cuOut.onDevice, 0, cuOut.sizeInBytes);

    checkCUDNN(cudnnGetConvolutionForwardAlgorithm(CuConvLayer::handle,
        *this->cuInput.descriptor,
        *this->cuKernel.descriptor,
        *this->cuConvolution.descriptor,
        *this->cuOutput.descriptor,
        CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
        /*memoryLimitInBytes=*/0,
        this->cuConvolution.algo));

    checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(CuConvLayer::handle,
        *this->cuInput.descriptor,
        *this->cuKernel.descriptor,
        *this->cuConvolution.descriptor,
        *this->cuOutput.descriptor,
        *this->cuConvolution.algo,
        &(this->cuWorkspace.sizeInBytes)));
    cudaMalloc(&this->cuWorkspace.onDevice, this->cuWorkspace.sizeInBytes);

    this->activation = activation;
}

CuConvLayer::~CuConvLayer()
{
}
