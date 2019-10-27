/**
 * Matthew Smith
 * github.com/mattonaise
 * Created on Oct 20, 2019
 *
 * fractalmodel.cpp
 **/

#include <QFile>

#include "fractalmodel.hpp"

#include "cuda.hpp"

/*
 * TODO: this constructor can use more error checking in its file IO
 */
FractalModel::FractalModel(const char* fileName) : m_numPoints(DEFAULT_POINTS)
{
    QFile file(fileName);
    if(!file.open(QIODevice::ReadOnly | QIODevice::Text))
        qFatal("ERROR: file %s not found - exiting.", fileName);
    QTextStream in(&file);

    // Read in name of fractal
    name = in.readLine();
    
    // Read in number of mappings
    in >> m_numMappings;
    cudaAllocateMapping(&m_mappingsPtr, m_numMappings);

    // Read in all given mappings
    for(int i = 0; i < m_numMappings; i++)
    {
        float a, b, c, d, x, y, p;
        in >> a >> b >> c >> d >> x >> y >> p;
        m_mappingsPtr[i] = {x, y, a, b, c, d, p};
    }

    // Read in scaling and translation factors for rendering
    float scaleX, scaleY, translationX, translationY;
    in >> scaleX >> scaleY >> translationX >> translationY;
    float scalingValues[] = {scaleX, 0.0f, 0.0f, scaleY};
    scalingMatrix = QMatrix2x2(scalingValues);
    translationVector = QVector2D(translationX, translationY);
}

FractalModel::~FractalModel()
{
    cudaDeallocateMapping(m_mappingsPtr);
}
