/**
 * Matthew Smith
 * github.com/mattonaise
 * Created on Oct 01, 2019
 *
 * fractalview.cpp
 **/

#include <QDebug>
#include <QString>

#include "fractalview.hpp"
#include "shaders.hpp"

/* TODO remove temp vertices */
static const float vertices[] = {
    -0.5f, -0.5f, 0.0f,     1.0f, 0.0f, 0.0f,
    0.5f, -0.5f, 0.0f,      0.0f, 1.0f, 0.0f,
    0.0f, 0.5f, 0.0f,       0.0f, 0.0f, 1.0f
};

/** PUBLIC **/

FractalView::FractalView(QWidget* parent) : QOpenGLWidget(parent),
    m_program(0)
{
    /* no OpenGL in constructor */
}

FractalView::~FractalView()
{
    makeCurrent();
    cleanupGL();
}

/** PROTECTED **/

/* Sets up OpenGL context - called by QT */
void FractalView::initializeGL()
{
    initializeOpenGLFunctions();
    printContextInfo();

    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

    /* Set up shaders */
    m_program = new QOpenGLShaderProgram();
    m_program->addShaderFromSourceFile(QOpenGLShader::Vertex,
            VERTEX_SHADER_FILE);
    m_program->addShaderFromSourceFile(QOpenGLShader::Fragment,
            FRAGMENT_SHADER_FILE);
    m_program->link();
    m_program->bind();

    /* Create VBO */
    m_vbo.create();
    m_vbo.bind();
    m_vbo.setUsagePattern(QOpenGLBuffer::StaticDraw);
    m_vbo.allocate(vertices, sizeof(vertices));

    /* Create VAO */
    m_vao.create();
    m_vao.bind();
    m_program->enableAttributeArray(0);
    m_program->enableAttributeArray(1);
    m_program->setAttributeBuffer(0, GL_FLOAT, 0, 3, 6 * sizeof(float));
    m_program->setAttributeBuffer(1, GL_FLOAT, 3 * sizeof(float), 3,
            6 * sizeof(float));

    /* Unbind all */
    m_vao.release();
    m_vbo.release();
    m_program->release();

    /* emit signal to let FractalGenerator register buffer */
    emit glBufferCreated(m_vbo.bufferId());
}

/* Resize OpenGL - called by QT when widget is resized */
void FractalView::resizeGL(int w, int h)
{
    /* Not using this function right now */
    (void) w;
    (void) h;
}

/* Redraw the scene - called by QT when redrawing widget */
void FractalView::paintGL()
{
    glClear(GL_COLOR_BUFFER_BIT);

    /* Render */
    m_program->bind();
    m_vao.bind();
    glDrawArrays(GL_TRIANGLES, 0, 3);
    m_vao.release();
    m_program->release();
}

/** PRIVATE **/

/* Clean up OpenGL - called in destructor */
void FractalView::cleanupGL()
{
    m_vao.destroy();
    m_vbo.destroy();
    delete m_program;
}

/* Prints OpenGL info - called in initializeGL() */
void FractalView::printContextInfo() 
{
    qDebug() << "===OPENGL INFO===";
    QString glType;
    QString glVersion;
    QString glProfile;

    glType = (context()->isOpenGLES()) ? "OpenGL ES" : "OpenGL";
    glVersion = reinterpret_cast<const char*>(glGetString(GL_VERSION));

    switch(format().profile())
    {
        case QSurfaceFormat::NoProfile :
            glProfile = "No Profile";
            break;
        case QSurfaceFormat::CoreProfile :
            glProfile = "Core Profile";
            break;
        case QSurfaceFormat::CompatibilityProfile :
            glProfile = "Compatibility Profile";
            break;
    }

    qDebug() << "\t" << qPrintable(glType) << qPrintable(glVersion) << "(" << 
            qPrintable(glProfile) << ")";
}
