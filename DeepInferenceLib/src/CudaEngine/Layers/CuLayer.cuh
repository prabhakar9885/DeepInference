#include <algorithm>
#include <vector>
#include <iostream>

class CuLayer
{
private:
    bool isInputLayer = false;
public:
    virtual void init(const float* weights, const int numberOfWeights, const float* bias, const int numberOfBias) = 0;
    virtual void initAsInputLayer();
    bool isAnInputLayer();
    virtual float* compute(const float* x) = 0;
    virtual void releaseMem() = 0;
    virtual std::vector<int> &getOutput() const = 0;
};
