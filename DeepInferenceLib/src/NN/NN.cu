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
    {
        if (this->layers.size() && layer->hasInputLayer())
            throw "Intermediate layers should not have an input-size set";
        else if (layer->hasInputLayer())
            layers.push_back(layer);
        else if (layer->canBeStackedOn(this->layers.back()))
        {
            layer->setPrevLayer(layers.back());
            layers.push_back(layer);
        }
        else
            throw "Can't stack the layer";
    }
    else
        throw "Layer can't be null";
}

void NN::init(const std::vector<std::vector<float>>& weightsAndBias) const
{
    auto weightsAndBiasIterator = weightsAndBias.begin();
    auto layerIterator = this->layers.begin();
    while (layerIterator != layers.end() && weightsAndBiasIterator != weightsAndBias.end())
    {
        Layer* currentLayer = *layerIterator;
        if (Utills::Layers::getLayerType(currentLayer) == LayerType::FLATTEN)
        {
            Flatten* flattened = dynamic_cast<Flatten*>(currentLayer);
            flattened->init();
        }
        else
        {
            const std::vector<float>& weight = *weightsAndBiasIterator;
            const std::vector<float>& bias = *(weightsAndBiasIterator + 1);
            currentLayer->init(weight, bias);
            weightsAndBiasIterator += 2;
        }
        layerIterator++;
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