#include "Shared/Activation.cuh"
#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

class CuUtills
{
public:
    static void computeActivation(float*& x, int xSize, Activation activation);
};