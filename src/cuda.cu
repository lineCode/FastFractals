/**
 * Matthew Smith
 * github.com/mattonaise
 * Created on Oct 05, 2019
 *
 * cuda.cu
 **/

#include <stdio.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_gl_interop.h>
#include <curand_kernel.h>

#include "cuda.hpp"
#include "mapping.hpp"

// defining kernel in seperate source file for clarity
#include "kernel.cu"

/* 
 * CUDA error-checking function and macro - from CUDA BY EXAMPLE
 */
static void HandleError(cudaError_t err, const char* file, int line,
    bool abort = true)
{
    if(err != cudaSuccess)
    {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
        if(abort)
            exit(EXIT_FAILURE);
    }
}
#define HANDLE_ERROR( err) (HandleError( err, __FILE__, __LINE__ ))

void cudaInit()
{
    printf("===CUDA INITIALIZATION===\n");

    // Select CUDA device with compute capability >=6.0
    // Compute capability 6.0 needed for unified memory
    int device;
    cudaDeviceProp prop;
    memset(&prop, 0, sizeof(cudaDeviceProp));
    prop.major = 6;
    prop.minor = 0;
    HANDLE_ERROR( cudaChooseDevice(&device, &prop) );

    cudaPrintDeviceProperties(device);
}

void cudaPrintDeviceProperties(int device)
{
    // Query device properties
    cudaDeviceProp prop;
    int driverVersion, runtimeVersion;
    HANDLE_ERROR( cudaGetDeviceProperties(&prop, device) );
    HANDLE_ERROR( cudaDriverGetVersion(&driverVersion) );
    HANDLE_ERROR( cudaRuntimeGetVersion(&runtimeVersion) );

    // Print device properties
    printf("\tDevice Name: %s\n", prop.name);
    printf("\tCUDA Driver Version / Runtime Version: %d.%d / %d.%d\n",
            driverVersion / 1000, (driverVersion % 100) / 10,
            runtimeVersion / 1000, (runtimeVersion % 100) / 10);
    printf("\tCompute Capability: %d.%d\n", prop.major, prop.minor);
    printf("\tTotal Global Memory: %ld bytes\n", prop.totalGlobalMem);
    printf("\tNumber of Multiprocessors: %d\n", prop.multiProcessorCount);
    printf("\tMaximum Threads per Multiprocessor: %d\n",
            prop.maxThreadsPerMultiProcessor);
    printf("\tTotal Number of Threads: %d\n", prop.multiProcessorCount *
            prop.maxThreadsPerMultiProcessor);
    printf("\tMaximum Threads per Block: %d\n", prop.maxThreadsPerBlock);
}

void* cudaRegisterBuffer(GLuint buf)
{
    cudaGraphicsResource* resource = nullptr;
    HANDLE_ERROR( cudaGraphicsGLRegisterBuffer(&resource, buf, 
            cudaGraphicsMapFlagsNone) );
    return resource;
}

void cudaUnregisterResource(void* resource)
{
    HANDLE_ERROR( cudaGraphicsUnregisterResource((cudaGraphicsResource*)
            resource) );
}

void cudaMapResource(void* resource, void** devicePtr, size_t* size)
{
    // map CUDA resource and get device pointer and size
    HANDLE_ERROR( cudaGraphicsMapResources(1, (cudaGraphicsResource**)
            &resource) );
    void* devicePtr_ = nullptr;
    size_t size_ = 0;
    HANDLE_ERROR( cudaGraphicsResourceGetMappedPointer(&devicePtr_, &size_,
            (cudaGraphicsResource*) resource) );

    // ensure devicePtr_ and size_ are valid
    assert(devicePtr_ != nullptr);
    assert(size_ != 0);
    
    // set pointer values
    *devicePtr = devicePtr_;
    *size = size_;
}

void cudaUnmapResource(void* resource)
{
    HANDLE_ERROR( cudaGraphicsUnmapResources(1, (cudaGraphicsResource**)
            &resource) );
}

void cudaAllocateMapping(mapping** mapping, int numMappings)
{
    HANDLE_ERROR( cudaMallocManaged(mapping, numMappings * sizeof(mapping)) );
}

void cudaDeallocateMapping(mapping* mapping)
{
    HANDLE_ERROR( cudaFree(mapping) );
}

void cudaRunKernel(void* d_pointData, int numPoints,
        mapping* d_mappings, int numMappings)
{
    // calculate block numbers and block size
    int blockSize = 256;
    //int iterations = 256;
    //int numBlocks = (numPoints + blockSize - 1) / (blockSize * iterations); 
    int numBlocks = 1;
    printf("CUDA: Running kernel (%d block(s), %d threads per block) - ",
            numBlocks, blockSize);
    
    // set up CUDA events for timing the kernel
    cudaEvent_t start, stop;
    HANDLE_ERROR( cudaEventCreate(&start) );
    HANDLE_ERROR( cudaEventCreate(&stop) );
    
    HANDLE_ERROR( cudaEventRecord(start) );
    kernel<<<numBlocks, blockSize, numMappings * sizeof(mapping)>>>
        ((float4*)d_pointData, numPoints, d_mappings, numMappings);
    HANDLE_ERROR( cudaEventRecord(stop) );
 
    // handle any synchronous and asynchronous kernel errors
    HANDLE_ERROR( cudaGetLastError() );
    HANDLE_ERROR( cudaDeviceSynchronize() );

    // record and print kernel timing
    HANDLE_ERROR( cudaEventSynchronize(stop) );
    float milliseconds = 0;
    HANDLE_ERROR( cudaEventElapsedTime(&milliseconds, start, stop) );
    printf("%f ms\n", milliseconds);
}

void cudaShutdown()
{
    printf("===CUDA SHUTDOWN===\n");
}
