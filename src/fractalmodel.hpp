/**
 * Matthew Smith
 * github.com/mattonaise
 * Created on Oct 20, 2019
 *
 * fractalmodel.hpp
 **/

#ifndef FRACTALMODEL_HPP
#define FRACTALMODEL_HPP

#include "defaultvalues.hpp"

class FractalModel
{
    public:
        FractalModel() : m_numPoints(DEFAULT_POINTS) {} 
        ~FractalModel() {}

        int m_numPoints;
};

#endif
