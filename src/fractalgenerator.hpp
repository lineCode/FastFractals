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

class FractalModel;

class FractalGenerator : public QObject
{
    Q_OBJECT

    public:
        FractalGenerator(QObject* parent = nullptr);

        void setModel(FractalModel* model);
        void cleanup();

    signals:
        void fractalUpdated();

    public slots:
        void registerGLBuffer(GLuint buf);
        void updateModel(FractalModel* newModel);
        void generateFractal();

    private:
        void* m_cudaResource;

        FractalModel* m_currentModel;
};

#endif
