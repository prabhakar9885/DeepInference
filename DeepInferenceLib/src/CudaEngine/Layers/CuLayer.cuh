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
    virtual void init(const float* weights, const int numberOfWeights, const float* bias, const int numberOfBias) = 0;
    bool hasInputLayer();
    virtual float* compute(const float* x) = 0;
    virtual std::vector<float>&& getOutput() const = 0;
};
