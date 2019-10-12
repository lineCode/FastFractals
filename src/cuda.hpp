/**
 * Matthew Smith
 * github.com/mattonaise
 * Created on Oct 05, 2019
 *
 * cuda.hpp
 **/

#ifndef CUDA_HPP
#define CUDA_HPP

#include <GL/gl.h>

void cudaInit();
void cudaPrintDeviceProperties(int device);
void* cudaRegisterBuffer(GLuint buf);
void cudaUnregisterResource(void* resource);
void* cudaMapResource(void* resource);
void cudaUnmapResource(void* resource);
void cudaRunKernel(void* devicePtr);
void cudaShutdown();

#endif
