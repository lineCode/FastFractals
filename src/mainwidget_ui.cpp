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

#include "mainwidget.hpp"
#include "fractalview.hpp"
#include "fractalgenerator.hpp"

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
                updateModel();
            });
    vLayout->addWidget(slider);
}
