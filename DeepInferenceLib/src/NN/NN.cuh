#ifndef NN_CUH
#define NN_CUH

#include "Layer/Layer.cuh"
#include "Shared/Utills.cuh"
#include <list>
#include <memory>
#include <vector>

class NN {
private:
	std::list<const Layer*> layers;
	bool isNetworkValid() const;
public:
	NN();
	void pushLayer(const Layer* layer);
	void init(const std::vector<std::vector<float>> &weightsAndBias) const;
	float forward(const std::vector<float>& input_sample) const;
};

#endif // !NN_CUH
