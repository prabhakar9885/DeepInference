#ifndef DATASTRUCTS_CUH
#define DATASTRUCTS_CUH


#include <cudnn.h>
struct DenseLayerWeightDims {
    int currentLayerSize;
    int prevLayerSize;
};

struct ConvInputLayerDims
{
    int batchSize;          // N
    int channelsPerImage;   // C
    int imageHeight;        // H
    int imageWidth;         // W
};

struct ConvLayerDims
{
    int N;          //  Number of Output channels
    int C;          //  Number of Input channels
    int H;          //  Height of the channel
    int W;          //  Width of the channel
};

struct ConvAlgoSpecs
{
    int stride;
    int dilation;
    int padding;
};


struct CuKernel
{
    int outChannels;        //  (N) Number of Output channels
    int inChannels;         //  (C) Number of Input channels
    int heightOfChannel;    //  (H) Height of the channel
    int widthOfChannel;     //  (W) Width of the channel
    cudnnFilterDescriptor_t descriptor;
    float* onDevice;
    size_t sizeInBytes;
};

struct CuWorkspace
{
    size_t sizeInBytes;
    void* onDevice;
};

struct CuConvolution
{
    int padding, stride, dilation;
    cudnnConvolutionDescriptor_t descriptor;
    cudnnConvolutionFwdAlgo_t algo;
};

struct Tensor4D
{
    int batchSize;          // (N) Number of output-channels; In case of input-image, it's the batch-size
    int channelCount;       // (C) Number of input-channels;
    int height;             // (H) Height of output; In case of input-image, it's the height of the image
    int width;              // (W) Width of output; In case of input-image, it's the width of the image
    float* dataOnDevice;    // Image-data layed-out in NHWC format on device memory
    cudnnTensorDescriptor_t descriptor;
    float* onDevice;
    size_t sizeInBytes;
};
#endif // !DATASTRUCTS_CUH
