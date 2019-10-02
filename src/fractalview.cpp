#include <QDebug>
#include <QString>

#include "fractalview.hpp"

/** PUBLIC **/

FractalView::FractalView(QWidget* parent) : QOpenGLWidget(parent)
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

    /* OpenGL Settings */
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
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
}

/* Clean up OpenGL - called in destructor */
void FractalView::cleanupGL()
{

}

/** PRIVATE **/

/* Prints OpenGL info - called on initializeGL() */
void FractalView::printContextInfo()
{
    QString glType;
    QString glVersion;
    QString glProfile;

    glType = (context()->isOpenGLES()) ? "OpenGL ES" : "OpenGL";
    glVersion = reinterpret_cast<const char*>(glGetString(GL_VERSION));

    switch(format().profile())
    {
        case QSurfaceFormat::NoProfile :
            glProfile = "NoProfile";
            break;
        case QSurfaceFormat::CoreProfile :
            glProfile = "CoreProfile";
            break;
        case QSurfaceFormat::CompatibilityProfile :
            glProfile = "CompatibilityProfile";
            break;
    }

    qDebug() << qPrintable(glType) << qPrintable(glVersion) << "(" << 
            qPrintable(glProfile) << ")";
}
