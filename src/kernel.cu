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

__global__ void kernel()
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Hi, I'm thread %d\n", index);
}
