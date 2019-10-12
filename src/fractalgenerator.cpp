/**
 * Matthew Smith
 * github.com/mattonaise
 * Created on Oct 05, 2019
 *
 * fractalgenerator.cpp
 **/

#include "fractalgenerator.hpp"
#include "cuda.hpp"

FractalGenerator::FractalGenerator(QObject* parent) : QObject(parent),
    m_cudaResource(0)
{
    cudaInit();
}

FractalGenerator::~FractalGenerator()
{
    cudaShutdown();
}

/*
 * Calls CUDA code to register OpenGL vertex buffer object - called by
 * FractalView constructor
 */
void FractalGenerator::registerGLBuffer(GLuint buf)
{
    m_cudaResource = cudaRegisterBuffer(buf);
}

/*
 * Calls CUDA kernel to generate fractal - called whenever current fractal
 * model is updated
 */
void FractalGenerator::generateFractal()
{
    void* devicePtr;
    size_t size;
    cudaMapResource(m_cudaResource, &devicePtr, &size);
    cudaRunKernel(devicePtr, size);
    cudaUnmapResource(m_cudaResource);
}

/*
 * Cleans up CUDA graphics resource - called on exit
 */
void FractalGenerator::cleanup()
{
    cudaUnregisterResource(m_cudaResource);
}
