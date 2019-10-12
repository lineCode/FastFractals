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
 * Calls CUDA code to register OpenGL vertex buffer object
 * Called by FractalView constructor
 */
void FractalGenerator::registerGLBuffer(GLuint buf)
{
    m_cudaResource = cudaRegisterBuffer(buf);
}

/*
 * Calls CUDA kernel to generate fractal and emits signal for redrawing
 * Called whenever current fractal model is updated
 */
void FractalGenerator::generateFractal()
{
    void* devicePtr;
    size_t size;
    cudaMapResource(m_cudaResource, &devicePtr, &size);
    cudaRunKernel(devicePtr, size);
    cudaUnmapResource(m_cudaResource);

    // emit signal to schedule a redraw in FractalView
    emit fractalUpdated();
}

/*
 * Cleans up CUDA graphics resource
 * Called on exit
 */
void FractalGenerator::cleanup()
{
    cudaUnregisterResource(m_cudaResource);
}
