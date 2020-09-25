
#include "CuLayer.cuh"

void CuLayer::initAsInputLayer()
{
    this->isInputLayer = true;
}

bool CuLayer::isAnInputLayer()
{
    return this->isInputLayer;
}
