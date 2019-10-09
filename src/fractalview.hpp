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

class FractalView : public QOpenGLWidget, protected QOpenGLFunctions
{
    Q_OBJECT
    
    public:
        FractalView(QWidget* parent = nullptr);
        ~FractalView();

    signals:
        void glBufferCreated(GLuint buf);

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
};

#endif
