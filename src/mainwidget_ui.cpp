/**
 * Matthew Smith
 * github.com/mattonaise
 * Created on Nov 09, 2019
 *
 * mainwidget_ui.cpp
 **/

#include <QtWidgets>
#include <QListWidget>
#include <QPushButton>
#include <QSlider>
#include <QLabel>
#include <QFrame>

#include "mainwidget.hpp"
#include "fractalview.hpp"
#include "fractalgenerator.hpp"
#include "defaultvalues.hpp"

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

    // Create ListWidget and populate with keys of the model map (the names of
    // the fractals)
    QListWidget* modelList = new QListWidget(this);
    QMapIterator<QString, FractalModel*> i(m_modelsMap);
    while(i.hasNext())
    {
        i.next();
        new QListWidgetItem(i.key(), modelList);
    }
    modelList->setCurrentRow(0);
    vLayout->addWidget(modelList);

    // Change fractal model when a new selection is made
    connect(modelList, &QListWidget::currentItemChanged,
        [this](QListWidgetItem* current)
        {
            m_currentModel = m_modelsMap.value(current->text());
            updateModel();
        });

    // QFrame line seperator
    QFrame* line1 = new QFrame(this);
    line1->setFrameShape(QFrame::HLine);
    vLayout->addWidget(line1);
    
    // QPushButton to regenerate fractal
    QPushButton* button = new QPushButton("Regenerate Fractal", this);
    button->setSizePolicy(QSizePolicy::Minimum, QSizePolicy::Minimum);
    connect(button, &QPushButton::clicked, m_fractalGenerator,
        &FractalGenerator::generateFractal);
    vLayout->addWidget(button);

    // QLabel to display number of threads running
    m_numThreadsLabel = new QLabel(this);
    m_numThreadsLabel->setAlignment(Qt::AlignCenter);
    vLayout->addWidget(m_numThreadsLabel);

    // QSlider for the number of threads to be ran
    m_numThreadsSlider = new QSlider(Qt::Horizontal, this);
    m_numThreadsSlider->setMinimum(MIN_THREADS);
    m_numThreadsSlider->setMaximum(MAX_THREADS);
    m_numThreadsSlider->setValue(m_currentModel->m_numThreads);
    m_numThreadsSlider->setTickPosition(QSlider::TicksBothSides);
    m_numThreadsSlider->setTickInterval(MAX_THREADS / 4 - MIN_THREADS);
    connect(m_numThreadsSlider, &QSlider::valueChanged,
        [this](int value)
        {
            m_currentModel->m_numThreads = value;
            updateModel();
        });
    vLayout->addWidget(m_numThreadsSlider);

    // QLabel for the number of points to be generated
    m_numPointsLabel = new QLabel(this);
    m_numPointsLabel->setAlignment(Qt::AlignCenter);
    vLayout->addWidget(m_numPointsLabel);

    // QSlider for the number of points to be generated
    m_numPointsSlider = new QSlider(Qt::Horizontal, this);
    m_numPointsSlider->setMinimum(MIN_POINTS);
    m_numPointsSlider->setMaximum(MAX_POINTS);
    m_numPointsSlider->setValue(m_currentModel->m_numPoints);
    m_numPointsSlider->setTickPosition(QSlider::TicksBothSides);
    m_numPointsSlider->setTickInterval(MAX_POINTS / 5 - MIN_POINTS);
    m_numPointsSlider->setSingleStep(1000);
    connect(m_numPointsSlider, &QSlider::valueChanged,
        [this](int value)
        {
            m_currentModel->m_numPoints = value;
            updateModel();
        });
    vLayout->addWidget(m_numPointsSlider);

    // QFrame line seperator
    QFrame* line2 = new QFrame();
    line2->setFrameShape(QFrame::HLine);
    vLayout->addWidget(line2);

    // QLabel for kernel runtime on each kernel execution
    m_kernelRuntimeLabel = new QLabel(this);
    m_kernelRuntimeLabel->setAlignment(Qt::AlignCenter);
    vLayout->addWidget(m_kernelRuntimeLabel);
}
