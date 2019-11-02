/**
 * Matthew Smith
 * github.com/mattonaise
 * Created on Oct 01, 2019
 *
 * mainwidget.cpp
 **/

#include <QtWidgets>
#include <QPushButton>
#include <QSlider>

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
    
    m_currentModel = new FractalModel("./fractals/barnsleyfern.frac");

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
    setUpUI();
}

/*
 * Sets up the UI components for the MainWidget
 * Called in the constructor
 */
void MainWidget::setUpUI()
{
    // Set up horizontal layout
    QHBoxLayout* hLayout = new QHBoxLayout(this);
    hLayout->addWidget(m_fractalView);
    hLayout->setStretch(0, 3);
    setLayout(hLayout);

    // Set up vertical layout
    QVBoxLayout* vLayout = new QVBoxLayout();
    hLayout->addLayout(vLayout, 1);
    
    // TODO remove temporary widget
    QPushButton* button = new QPushButton("CUDA", this);
    button->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
    button->setFlat(true);
    connect(button, &QPushButton::clicked, m_fractalGenerator,
            &FractalGenerator::generateFractal);
    vLayout->addWidget(button);

    QSlider* slider = new QSlider(Qt::Horizontal, this);
    slider->setMinimum(MIN_POINTS);
    slider->setMaximum(MAX_POINTS);
    slider->setTickPosition(QSlider::TicksBothSides);
    slider->setTickInterval(MAX_POINTS / 4);
    slider->setValue(DEFAULT_POINTS);
    slider->setSingleStep(1000);
    connect(slider, &QSlider::valueChanged,
            [this](int value)
            {
                m_currentModel->m_numPoints = value;
                m_fractalGenerator->generateFractal();
            });
    vLayout->addWidget(slider);
}

MainWidget::~MainWidget()
{
    delete m_currentModel;

    // No need to delete fractalView - parented to this widget
    delete m_fractalGenerator;
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
