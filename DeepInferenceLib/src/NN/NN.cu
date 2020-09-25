#include "NN.cuh"

bool NN::isNetworkValid() const
{
    if (layers.size() <= 1)
        throw "NN should have more than 1 layer";
    auto layer_iterator = this->layers.begin();
    while(true)
    {
        Layer* const currentLayer = *layer_iterator;
        layer_iterator++;
        if (layer_iterator == this->layers.end())
            break;
        Layer* const nextLayer = *layer_iterator;
        if (!nextLayer->canBeStackedOn(currentLayer))
            throw "Layer-{} is not compattible with Layer-{}";
    }
    return true;
}

NN::NN()
{
}

NN::~NN()
{
    std::cout << "\nNN -> Destructor";
    std::for_each(this->layers.begin(), this->layers.end(), [](auto iterator) {delete(iterator); std::cout << "\nNN -> Destructor -> layer freed"; });
}

void NN::pushLayer(Layer* layer)
{
    if (layer != nullptr)
        layers.push_back(layer);
    else
        throw "Layer can't be null";
}

void NN::init(const std::vector<std::vector<float>> &weightsAndBias) const
{
    if (isNetworkValid()) 
    {
        auto weightsAndBiasIterator = weightsAndBias.begin();
        auto layerIterator = this->layers.begin();
        Layer* inputLayer = *layerIterator;
        inputLayer->initAsInputLayer();
        layerIterator++;
        while (layerIterator != layers.end() && weightsAndBiasIterator != weightsAndBias.end())
        {
            Layer* currentLayer = *layerIterator;
            const std::vector<float> &weight = *weightsAndBiasIterator;
            const std::vector<float> &bias = *(weightsAndBiasIterator + 1);
            currentLayer->init(weight, bias);
            weightsAndBiasIterator += 2;
            layerIterator++;
        }
    }
    else
    {
        throw "InvalidNetworkException";
    }
}

const float* NN::forward(const std::vector<float>& input_sample) const
{
    auto currentLayerIterator = this->layers.begin();
    const float* input_data = input_sample.data();
    int layerIndex = 0;
    while (currentLayerIterator != this->layers.end())
    {
        Layer* currentLayer = *currentLayerIterator;
        input_data = currentLayer->forward(input_data);
        currentLayerIterator++;
        layerIndex++;
    }
    return input_data;
}