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

#include "defaultvalues.hpp"
#include "mapping.hpp"

class FractalModel
{
    public:
        FractalModel();
        ~FractalModel();

        int m_numPoints;

        int m_numMappings;
        mapping* m_mappingsPtr;

        QMatrix2x2 scalingMatrix;
        QVector2D translationVector;
};

#endif
