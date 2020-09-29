#ifndef NN_CUH
#define NN_CUH

#include "NN/Layer/AbstractLayers/Layer.cuh"
#include "NN/Layer/AbstractLayers/ComputableLayer.cuh"
#include "NN/Layer/InputLayer.cuh"
#include "Shared/Utills.cuh"
#include <algorithm>
#include <list>
#include <memory>
#include <vector>

class NN {
private:
	std::list<Layer*> layers;
	bool isNetworkValid() const;
public:
	NN();
	~NN();
	void pushLayer(Layer* layer);
	void init(const std::vector<std::vector<float>> &weightsAndBias) const;
	const float* forward(const std::vector<float>& input_sample) const;
};

#endif // !NN_CUH
