#ifndef CUUTILLS_CUH
#define CUUTILLS_CUH

#include "Shared/Activation.cuh"
#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "iostream"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        std::cout << "GPUassert: " << cudaGetErrorString(code) << " : " << file << " : " << line;
        if (abort) exit(code);
    }
}

class CuUtills
{
public:
	static void computeActivation(float*& x, int xSize, Activation activation);
    static void addBiasForNHWC(float*& arr, float*& bias, int N, int H, int W, int C);
};
#endif // !CUUTILLS_CUH
