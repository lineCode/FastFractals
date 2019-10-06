/**
 * Matthew Smith
 * github.com/mattonaise
 * Created on Oct 05, 2019
 *
 * cuda.cu
 **/

#include <stdio.h>
#include <cuda.h>

#include "cuda.hpp"

/* CUDA error-checking function and macro - from CUDA BY EXAMPLE */
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

    /* Query device count */
    int nDevices;
    HANDLE_ERROR( cudaGetDeviceCount(&nDevices) );
    printf("Total devices Found: %d\n", nDevices);

    /* Print device info */
    for(int i = 0; i < nDevices; i++)
    {
        /* Query device properties */
        cudaDeviceProp prop;
        int driverVersion, runtimeVersion;
        HANDLE_ERROR( cudaGetDeviceProperties(&prop, i) );
        HANDLE_ERROR( cudaSetDevice(i) );
        HANDLE_ERROR( cudaDriverGetVersion(&driverVersion) );
        HANDLE_ERROR( cudaRuntimeGetVersion(&runtimeVersion) );

        /* Print device properties */
        printf("Device Number: %d\n", i);
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
}

void cudaShutdown()
{
    printf("===CUDA SHUTDOWN===\n");
}
