#ifndef DENSELAYER_CU
#define DENSELAYER_CU

#include "DenseLayer.cuh"

DenseLayer::DenseLayer(int size, Activation activation) : size{ size }, activation{ activation }
{
}

DenseLayer::DenseLayer(int size, Activation activation, int inputSize) : DenseLayer(size, activation)
{
    this->inputSizeIsSet = true;
    this->inputSize = inputSize;
}

DenseLayer::~DenseLayer()
{
    if(cuDenseLayer)
        delete(cuDenseLayer);
    std::cout << "\nDenseLayer freed";
}

void DenseLayer::init(const std::vector<float>& weight, const std::vector<float>& bias)
{
    int prevLayerSize;
    LayerType typeOfPrevLayer = Utills::Layers::getLayerType(prevLayer);
    if (typeOfPrevLayer == LayerType::DENSE)
    {
        DenseLayer* prevLayer = dynamic_cast<DenseLayer*>(this->prevLayer);
        prevLayerSize = prevLayer->size;
        this->cuDenseLayer = new CuDenseLayer(this->size, this->activation, prevLayer->cuDenseLayer);
    }
    else if (typeOfPrevLayer == LayerType::FLATTEN)
    {
        Flatten* prevLayer = dynamic_cast<Flatten*>(this->prevLayer);
        prevLayerSize = prevLayer->getSize();
        this->inputSizeIsSet = true;
        this->inputSize = prevLayerSize;
        this->cuDenseLayer = new CuDenseLayer(this->size, this->activation, (CuLayer*)prevLayer);
    }
    else
    {
        prevLayerSize = this->inputSize;
        this->cuDenseLayer = new CuDenseLayer(this->size, this->activation);
    }
    if (prevLayer && this->size * prevLayerSize != weight.size())
        throw "WeightDimensionsInvalid: inputSize * sizeOfLayerToWhichInputIsGiven != NumberOfWeightsThatIsFedToFirstLayer";
    this->cuDenseLayer->setSizeOfInput(prevLayerSize);
    this->cuDenseLayer->allocMemForLayer();
    this->cuDenseLayer->init(weight.data(), weight.size(), bias.data(), bias.size());
}

bool DenseLayer::canBeStackedOn(const Layer* prevLayer) const
{
    LayerType typeOfPrevLayer = Utills::Layers::getLayerType(prevLayer);
    return  (typeOfPrevLayer == LayerType::DENSE || typeOfPrevLayer == LayerType::FLATTEN);
}

bool DenseLayer::hasInputLayer() const
{
    return this->inputSizeIsSet;
}

float* DenseLayer::forward(const float* input) const
{
    return this->cuDenseLayer->compute(input);
}

int DenseLayer::getSize() const
{
    return this->size;
}

void* DenseLayer::getOutput() const
{
	return nullptr;
}

#endif // !DENSELAYER_CU
