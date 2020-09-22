#ifndef NN_CUH
#define NN_CUH

#include "Layer/Layer.cuh"
#include <list>
#include <memory>
#include <vector>

class NN {
private:
	std::list<Layer*> layers;
public:
	NN();
	void pushLayer(Layer* layer);
	void init(std::vector<std::vector<float>> weightsAndBias);
	float forward(std::vector<float>& input_sample);
};

#endif // !NN_CUH
