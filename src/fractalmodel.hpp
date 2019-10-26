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
#include "mapping.hpp"

class FractalModel
{
    public:
        FractalModel();
        ~FractalModel();

        int m_numPoints;
        int m_numMappings;
        mapping* m_mappingsPtr;
};

#endif
