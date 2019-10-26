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
    m_currentModel = new FractalModel();

    m_fractalView = new FractalView(m_currentModel, this);
    m_fractalGenerator = new FractalGenerator(m_currentModel);

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
    vLayout->setStretch(0, 1);

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
    vLayout->setStretch(1, 1);

    QSlider* slider2 = new QSlider(Qt::Horizontal, this);
    slider2->setMinimum(-100);
    slider2->setMaximum(100);
    slider2->setValue(0);
    connect(slider2, &QSlider::valueChanged,
            [this](int value)
            {
                m_currentModel->m_mappingsPtr[1].x = value / 100.0f * 1.6f;
                m_fractalGenerator->generateFractal();
            });
    vLayout->addWidget(slider2);
    vLayout->setStretch(2, 1);
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
