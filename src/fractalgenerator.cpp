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

FractalGenerator::FractalGenerator(QObject* parent) :
    QObject(parent), m_cudaResource(0), m_currentModel(nullptr)
{
    cudaInit();
}

FractalGenerator::~FractalGenerator()
{
    cudaShutdown();
}

/*
 * Sets m_currentModel pointer without regenerating fractal
 * Called by MainWidget constructor to allow for cudaInit() to be called before
 * any FractalModel allocates device memory
 */
void FractalGenerator::setModel(FractalModel* model)
{
    m_currentModel = model;
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
 * Updates m_currentModel pointer and regenerates fractal
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
    cudaRunKernel(devicePtr, m_currentModel->m_numPoints,
            m_currentModel->m_mappingsPtr, m_currentModel->m_numMappings);
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
