/**
 * Matthew Smith
 * github.com/mattonaise
 * Created on Oct 11, 2019
 *
 * kernel.cu
 **/

#include <stdio.h>
#include <cuda.h>

#include "cuda.hpp"

__global__ void kernel(float4* ptr, size_t size)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    ptr[index].x += 0.1f;
}
