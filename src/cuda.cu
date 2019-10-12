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

#include "cuda.hpp"

// defining kernel in seperate source file for clarity
#include "kernel.cu"

/* 
 * CUDA error-checking function and macro - from CUDA BY EXAMPLE
 */
static void HandleError(cudaError_t err, const char* file, int line)
{
    if(err != cudaSuccess)
    {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}
#define HANDLE_ERROR( err) (HandleError( err, __FILE__, __LINE__ ))

void cudaInit()
{
    printf("===CUDA INITIALIZATION===\n");

    // Select CUDA device with compute capability >=3.0
    int device;
    cudaDeviceProp prop;
    memset(&prop, 0, sizeof(cudaDeviceProp));
    prop.major = 3;
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
    printf("CUDA: Registering OpenGL buffer %d\n", buf);
    cudaGraphicsResource* resource = nullptr;
    HANDLE_ERROR( cudaGraphicsGLRegisterBuffer(&resource, buf, 
            cudaGraphicsMapFlagsNone) );
    return resource;
}

void cudaUnregisterResource(void* resource)
{
    printf("CUDA: Unregistering resource %p\n", resource);
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
    printf("CUDA: Mapped resource returned pointer %p with size %d\n",
            devicePtr_, size_);

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

void cudaRunKernel(void* devicePtr, size_t size)
{
    kernel<<<5,5>>>();

    // handle any synchronous and asynchronous kernel errors
    HANDLE_ERROR( cudaGetLastError() );
    HANDLE_ERROR( cudaDeviceSynchronize() );
}

void cudaShutdown()
{
    printf("===CUDA SHUTDOWN===\n");
}
