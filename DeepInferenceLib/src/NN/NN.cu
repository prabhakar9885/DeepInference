#include "NN.cuh"

bool NN::isNetworkValid() const
{
    if (layers.size() <= 1)
        throw "NN should have more than 1 layer";
    auto layer_iterator = this->layers.begin();
    int indexOfCurrentLayer = 0;
    Layer* nextLayer;
    while(true)
    {
        const Layer* const currentLayer = *layer_iterator;
        indexOfCurrentLayer++;
        layer_iterator++;
        if (layer_iterator == this->layers.end())
            break;
        const Layer* const nextLayer = *layer_iterator;
        if (!nextLayer->canBeStackedOn(currentLayer))
            throw "Layer-{} is not compattible with Layer-{}";
    }
    return true;
}

NN::NN()
{
}

void NN::pushLayer(const Layer* layer)
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

    }
    else
    {
        throw "InvalidNetworkException";
    }
}

float NN::forward(const std::vector<float>& input_sample) const
{
    return 0.0f;
}