/**
 * Matthew Smith
 * github.com/mattonaise
 * Created on Oct 20, 2019
 *
 * fractalmodel.hpp
 **/

#ifndef FRACTALMODEL_HPP
#define FRACTALMODEL_HPP

#include <QMatrix2x2>
#include <QVector2D>

#include "mapping.hpp"
#include "defaultvalues.hpp"

class FractalModel
{
    public:
        FractalModel(QString filename);
        ~FractalModel();

        int m_numThreads {DEFAULT_THREADS};
        float m_kernelRuntime;

        QString name;
        int m_numPoints {DEFAULT_POINTS};

        int m_numMappings;
        mapping* m_mappingsPtr;

        QMatrix2x2 scalingMatrix;
        QVector2D translationVector;
};

#endif
