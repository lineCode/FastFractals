/**
 * Matthew Smith
 * github.com/mattonaise
 * Created on Oct 05, 2019
 *
 * fractalgenerator.cpp
 **/

#include "fractalgenerator.hpp"
#include "cuda.hpp"
#include "fractalmodel.hpp"

FractalGenerator::FractalGenerator(FractalModel* model, QObject* parent) :
    QObject(parent), m_cudaResource(0), m_currentModel(model)
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
 * Updates m_curentModel pointer and regenerates fractal
 * Called by MainWidget UI elements
 */
void FractalGenerator::updateModel(FractalModel* newModel)
{
    m_currentModel = newModel;
    generateFractal();
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
    cudaRunKernel(devicePtr, m_currentModel->m_numPoints);
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
