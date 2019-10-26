/**
 * Matthew Smith
 * github.com/mattonaise
 * Created on Oct 01, 2019
 *
 * fractalview.hpp
 **/

#ifndef FRACTALVIEW_HPP
#define FRACTALVIEW_HPP

#include <QOpenGLWidget>
#include <QOpenGLFunctions>
#include <QOpenGLBuffer>
#include <QOpenGLVertexArrayObject>
#include <QOpenGLShaderProgram>

#include "fractalmodel.hpp"

class FractalView : public QOpenGLWidget, protected QOpenGLFunctions
{
    Q_OBJECT
    
    public:
        FractalView(FractalModel* model, QWidget* parent = nullptr);
        ~FractalView();

    signals:
        void glBufferCreated(GLuint buf);

    public slots:
        void updateModel(FractalModel* newModel) {m_currentModel = newModel;}

    protected:
        void initializeGL() override;
        void resizeGL(int w, int h) override;
        void paintGL() override;

    private:
        void cleanupGL();
        void printContextInfo();

        QOpenGLBuffer m_vbo;
        QOpenGLVertexArrayObject m_vao;
        QOpenGLShaderProgram* m_program;

        int m_scalingUniform;
        int m_translationUniform;

        FractalModel* m_currentModel;
};

#endif
