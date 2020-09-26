
#include "CuUtills.cuh"
#include <cuda_device_runtime_api.h>

namespace kernals
{
	__global__ void sigmoid(float* inp, int N) {
		int indx = blockDim.x * blockIdx.x + threadIdx.x;
		if (indx < N)
			inp[indx] = 1 / (1 + exp(-inp[indx]));
	}

	__global__ void relu(float* inp, int N) {
		int indx = blockDim.x * blockIdx.x + threadIdx.x;
		if (indx < N)
			inp[indx] = 0 > inp[indx] ? 0 : inp[indx];
	}
}

void CuUtills::computeActivation(float*& x, int xSize, Activation activation)
{
	int blockSize = 1024;
	int blockCount = xSize / blockSize + (xSize % blockSize != 0);

	switch (activation)
	{
	case Activation::SIGMOID:
		kernals::sigmoid << < blockCount, blockSize >> > (x, xSize);
		break;
	case Activation::ReLU:
		kernals::relu << < blockCount, blockSize >> > (x, xSize);
		break;
	default:
		throw "Unidentified Activation type";
	}


}
