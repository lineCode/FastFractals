/**
 * Matthew Smith
 * github.com/mattonaise
 * Created on Oct 05, 2019
 *
 * cuda.hpp
 **/

#ifndef CUDA_HPP
#define CUDA_HPP

#include <cstddef>

#include <GL/gl.h>

void cudaInit();
void cudaPrintDeviceProperties(int device);
void* cudaRegisterBuffer(GLuint buf);
void cudaUnregisterResource(void* resource);
void cudaMapResource(void* resource, void** devicePtr, size_t* size);
void cudaUnmapResource(void* resource);
void cudaRunKernel(void* d_pointData, int numPoints, int numMappings);
void cudaShutdown();

#endif
