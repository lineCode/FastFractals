/**
 * Matthew Smith
 * github.com/mattonaise
 * Created on Oct 01, 2019
 *
 * mainwidget.cpp
 **/

#include <QtWidgets>

#include "mainwidget.hpp"

MainWidget::MainWidget(QWidget* parent) : QWidget(parent)
{
    fractalView = new FractalView(this);
}

MainWidget::~MainWidget()
{
}
