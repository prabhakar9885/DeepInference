#ifndef NN_CUH
#define NN_CUH

#include "NN/Layer/Layer.cuh"
#include "Shared/Utills.cuh"
#include <algorithm>
#include <list>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
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
	void init(std::string fileContaingWeightsAndBias) const;
	const float* forward(const std::vector<float>& input_sample) const;
};

#endif // !NN_CUH
