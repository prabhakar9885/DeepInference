#ifndef CULAYER_CUH
#define CULAYER_CUH

#include <algorithm>
#include <vector>
#include <iostream>

class CuLayer
{
private:
protected:
    CuLayer* prevLayer = nullptr;
    bool isInputLayer = false;
public:
    virtual bool hasInputLayer() final;
    virtual void allocMemForLayer() = 0;
    virtual void init(const float* weights, const int numberOfWeights, const float* bias, const int numberOfBias) = 0;
    virtual float* compute(const float* x) = 0;
    virtual std::vector<float>&& getOutput() const = 0;
};

#endif // !CULAYER_CUH
