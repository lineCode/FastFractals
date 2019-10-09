/**
 * Matthew Smith
 * github.com/mattonaise
 * Created on Oct 01, 2019
 *
 * mainwidget.cpp
 **/

#include <QtWidgets>
#include <QHBoxLayout>
#include <QTextEdit>

#include "mainwidget.hpp"
#include "fractalview.hpp"
#include "fractalgenerator.hpp"

MainWidget::MainWidget(QWidget* parent) : QWidget(parent)
{
    m_fractalView = new FractalView(this);
    m_fractalGenerator = new FractalGenerator();

    /* connect signals and slots between view and generator */
    connect(m_fractalView, &FractalView::glBufferCreated,
            m_fractalGenerator, &FractalGenerator::registerGLBuffer);

    /* Set up horizontal layout */
    QHBoxLayout* hLayout = new QHBoxLayout(this);
    hLayout->addWidget(m_fractalView);
    hLayout->setStretch(0, 3);
    setLayout(hLayout);
    
    /* TODO remove temporary widget */
    QTextEdit* textEdit = new QTextEdit(this);
    hLayout->addWidget(textEdit);
    hLayout->setStretch(1, 1);
}

MainWidget::~MainWidget()
{
    /* No need to delete fractalView - parented to this widget */
    delete m_fractalGenerator;
}

/*
 * Overriding closeEvent function to allow for unregistering of CUDA resource
 * in the generator before the OpenGL context is lost
 */
void MainWidget::closeEvent(QCloseEvent* event)
{
    m_fractalGenerator->cleanup();
    event->accept();
}
