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

void FractalGenerator::registerGLBuffer(GLuint buf)
{
    m_cudaResource = cudaRegisterBuffer(buf);
}

void FractalGenerator::cleanup()
{
    cudaUnregisterResource(m_cudaResource);
}
