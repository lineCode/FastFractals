/**
 * Matthew Smith
 * github.com/mattonaise
 * Created on Oct 05, 2019
 *
 * fractalgenerator.hpp
 **/

#ifndef FRACTALGENERATOR_HPP
#define FRACTALGENERATOR_HPP

#include <QObject>

class FractalGenerator : public QObject
{
    Q_OBJECT

    public:
        FractalGenerator(QObject* parent = nullptr);
        ~FractalGenerator();

    private:
        
};

#endif
