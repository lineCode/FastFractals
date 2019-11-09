/**
 * Matthew Smith
 * github.com/mattonaise
 * Created on Oct 01, 2019
 *
 * mainwidget.cpp
 **/

#include <QtWidgets>
#include <QDirIterator>

#include "mainwidget.hpp"
#include "fractalview.hpp"
#include "fractalgenerator.hpp"
#include "fractalmodel.hpp"
#include "defaultvalues.hpp"

MainWidget::MainWidget(QWidget* parent) : QWidget(parent)
{
    // Construct FractalGenerator first to allow for CUDA initialization needed
    // before FractalModels allocate device memory
    m_fractalGenerator = new FractalGenerator();
 
    // Read all fractal files and generate FractalModels
    QDirIterator it(FRACTAL_DIR, QDir::Files);
    while(it.hasNext())
    {
        FractalModel* newModel = new FractalModel(it.next());
        m_modelsMap.insert(newModel->name, newModel);
    }
    m_currentModel = m_modelsMap.first();

    // Create the FractalView widget and set model for view and generator
    m_fractalView = new FractalView(m_currentModel, this);
    m_fractalGenerator->setModel(m_currentModel);

    // connect signals and slots between view and generator
    connect(m_fractalView, &FractalView::glBufferCreated,
            m_fractalGenerator, &FractalGenerator::registerGLBuffer);
    connect(m_fractalGenerator, &FractalGenerator::fractalUpdated,
            [this](){m_fractalView->update();});

    // set up model synchronization between MainWidget/view/generator
    connect(this, &MainWidget::modelUpdated,
            m_fractalView, &FractalView::updateModel);
    connect(this, &MainWidget::modelUpdated,
            m_fractalGenerator, &FractalGenerator::updateModel);
    
    // seperating out UI initialization into seperate function for clarity
    // see 'mainwidget_ui.cpp'
    setUpUI();
}

MainWidget::~MainWidget()
{
    // Delete all FractalModels - 'foreach' is a Qt keyword
    foreach (FractalModel* model, m_modelsMap)
        delete model;

    // No need to delete fractalView - parented to this widget
    delete m_fractalGenerator;
}

/*
 * Updates the model and synchronizes UI elements
 * Emits the modelUpdated signal to regenerate and render fractal
 */
void MainWidget::updateModel()
{
    emit modelUpdated(m_currentModel);
}

/*
 * Overriding closeEvent function to allow for unregistering of CUDA resource
 * in the FractalGenerator before the OpenGL context is lost
 */
void MainWidget::closeEvent(QCloseEvent* event)
{
    m_fractalGenerator->cleanup();
    event->accept();
}
