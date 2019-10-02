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

MainWidget::MainWidget(QWidget* parent) : QWidget(parent)
{
    fractalView = new FractalView(this);

    /* Set up horizontal layout */
    QHBoxLayout* hLayout = new QHBoxLayout;
    hLayout->addWidget(fractalView);
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
}
