#ifndef LAYER_CUH
#define LAYER_CUH


class Layer
{
private:

public:
    virtual void init() = 0;
    virtual void forward() = 0;
    virtual void* getOutput() = 0;
};

#endif // !LAYER_CUH