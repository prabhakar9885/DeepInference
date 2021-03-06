
#include "CuUtills.cuh"
#include <cuda_device_runtime_api.h>
#include "iostream"

namespace kernals
{
	__global__ void sigmoid(float* inp, int N) 
	{
		int indx = blockDim.x * blockIdx.x + threadIdx.x;
		if (indx < N)
			inp[indx] = 1 / (1 + exp(-inp[indx]));
	}

	__global__ void relu(float* inp, int N) 
	{
		int indx = blockDim.x * blockIdx.x + threadIdx.x;
		if (indx < N)
			inp[indx] = 0 > inp[indx] ? 0 : inp[indx];
	}

	__global__ void addBiasForNHWC(float* arr, float* bias, int N, int H, int W, int C, int* logs)
	{
		int phyIndex = blockIdx.x * blockDim.x + threadIdx.x;
		int sizeOfarr = N * H * W * C;
		if (phyIndex < sizeOfarr)
		{
			int n = phyIndex % C;
			arr[phyIndex] = arr[phyIndex] + bias[n];

			logs[phyIndex * 5 + 0] = -1 * phyIndex;
			logs[phyIndex * 5 + 1] = blockIdx.x;
			logs[phyIndex * 5 + 2] = threadIdx.x;
			logs[phyIndex * 5 + 3] = arr[phyIndex];
			logs[phyIndex * 5 + 4] = bias[n];
		}
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
	case Activation::NONE:
		break;
	default:
		throw "Unidentified Activation type";
	}
}

void CuUtills::addBiasForNHWC(float*& arr, float*& bias, int N, int H, int W, int C)
{
	int maxNumberOfThreadsPerBlock = 512;
	int blockCount = (N * H * W * C + maxNumberOfThreadsPerBlock - 1) / maxNumberOfThreadsPerBlock;
	
	int* logsDevice;	
	gpuErrchk(cudaMalloc((void**)&logsDevice, sizeof(int) * N * W * C * H * 5));
	gpuErrchk(cudaPeekAtLastError());

	kernals::addBiasForNHWC << <blockCount, maxNumberOfThreadsPerBlock >> > (arr, bias, N, H, W, C, logsDevice);
	cudaDeviceSynchronize();
	return;
}
