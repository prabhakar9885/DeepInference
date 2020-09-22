#ifndef DENSELAYER_CUH
#define DENSELAYER_CUH

#include "Layer.cuh"

class DenseLayer: public Layer
{

public:
    void init() override ;
    void forward() override;
    void* getOutput() override;

};

#endif // !DENSELAYER_CUH