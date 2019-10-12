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
#include <QOpenGLFunctions>

class FractalGenerator : public QObject
{
    Q_OBJECT

    public:
        FractalGenerator(QObject* parent = nullptr);
        ~FractalGenerator();

        void cleanup();

    public slots:
        void registerGLBuffer(GLuint buf);
        void generateFractal();

    private:
        void* m_cudaResource;

};

#endif
