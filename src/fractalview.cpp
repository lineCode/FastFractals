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
#include "defaultvalues.hpp"

FractalView::FractalView(FractalModel* model, QWidget* parent) :
    QOpenGLWidget(parent), m_program(0), m_currentModel(model)
{
    // no OpenGL in constructor
}

FractalView::~FractalView()
{
    // set current context and cleanup resources
    makeCurrent();
    cleanupGL();
}

/*
 * Sets up OpenGL context
 * Called by QT
 */
void FractalView::initializeGL()
{
    initializeOpenGLFunctions();
    printContextInfo();

    // OpenGL settings
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_PROGRAM_POINT_SIZE);

    // Set up shaders
    m_program = new QOpenGLShaderProgram();
    m_program->addShaderFromSourceFile(QOpenGLShader::Vertex,
            VERTEX_SHADER_FILE);
    m_program->addShaderFromSourceFile(QOpenGLShader::Fragment,
            FRAGMENT_SHADER_FILE);
    m_program->link();
    m_program->bind();

    // Cache uniform locations
    m_scalingUniform = m_program->uniformLocation("scaling");
    m_translationUniform = m_program->uniformLocation("translation");
    m_numMappingsUniform = m_program->uniformLocation("numMappings");

    // Create VBO
    m_vbo.create();
    m_vbo.bind();
    m_vbo.setUsagePattern(QOpenGLBuffer::StaticDraw);
    m_vbo.allocate(MAX_POINTS * 4 * sizeof(float));

    // Create VAO
    m_vao.create();
    m_vao.bind();
    m_program->enableAttributeArray(0);
    m_program->enableAttributeArray(1);
    m_program->setAttributeBuffer(0, GL_FLOAT, 0, 2, 4 * sizeof(float));
    m_program->setAttributeBuffer(1, GL_FLOAT, 2 * sizeof(float), 2, 
            4 * sizeof(float));

    // Unbind all
    m_vao.release();
    m_vbo.release();
    m_program->release();

    // emit signal to let FractalGenerator register buffer
    emit glBufferCreated(m_vbo.bufferId());
}

/*
 * Resize OpenGL
 * Called by QT when widget is resized
 */
void FractalView::resizeGL(int w, int h)
{
    // Not using this function right now
    (void) w;
    (void) h;
}

/* 
 * Redraw the scene
 * Called by QT when redrawing widget
 */
void FractalView::paintGL()
{
    glClear(GL_COLOR_BUFFER_BIT);

    m_program->bind();
    m_program->setUniformValue(m_scalingUniform,
            m_currentModel->scalingMatrix);
    m_program->setUniformValue(m_translationUniform,
            m_currentModel->translationVector);
    m_program->setUniformValue(m_numMappingsUniform,
            m_currentModel->m_numMappings);

    m_vao.bind();
    glDrawArrays(GL_POINTS, 0, m_currentModel->m_numPoints);
    m_vao.release();

    m_program->release();
}

/* 
 * Clean up OpenGL
 * Called in destructor
 */
void FractalView::cleanupGL()
{
    m_vao.destroy();
    m_vbo.destroy();
    delete m_program;
}

/* 
 * Prints OpenGL info
 * Called in initializeGL() 
 */
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
