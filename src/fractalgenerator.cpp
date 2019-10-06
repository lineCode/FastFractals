/**
 * Matthew Smith
 * github.com/mattonaise
 * Created on Oct 05, 2019
 *
 * fractalgenerator.cpp
 **/

#include "fractalgenerator.hpp"
#include "cuda.hpp"

/* PUBLIC */

FractalGenerator::FractalGenerator(QObject* parent) : QObject(parent)
{
    cudaInit();
}

FractalGenerator::~FractalGenerator()
{
    cudaShutdown();
}
