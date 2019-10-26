/**
 * Matthew Smith
 * github.com/mattonaise
 * Created on Oct 20, 2019
 *
 * fractalmodel.cpp
 **/

#include "fractalmodel.hpp"

#include "cuda.hpp"

FractalModel::FractalModel() : m_numPoints(DEFAULT_POINTS), m_numMappings(4),
    m_mappingsPtr(nullptr)
{
    cudaAllocateMapping(&m_mappingsPtr, m_numMappings);
    m_mappingsPtr[0] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.16f, 0.01f};
    m_mappingsPtr[1] = {0.0f, 1.6f, 0.85f, 0.04f, -0.04f, 0.85f, 0.85f};
    m_mappingsPtr[2] = {0.0f, 1.6f, 0.2f, -0.26f, 0.23f, 0.22f, 0.07f};
    m_mappingsPtr[3] = {0.0f, 0.44f, -0.15f, 0.28f, 0.26f, 0.24f, 0.07};

    float scalingValues[] = {0.35f, 0.0f, 0.0f, 0.2f};
    scalingMatrix = QMatrix2x2(scalingValues);
    translationVector = QVector2D(0.0f, -1.0f);
}

FractalModel::~FractalModel()
{
    cudaDeallocateMapping(m_mappingsPtr);
}
