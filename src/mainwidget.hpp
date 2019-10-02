/**
 * Matthew Smith
 * github.com/mattonaise
 * Created on Oct 01, 2019
 *
 * mainwidget.hpp
 **/

#ifndef MAINWIDGET_HPP
#define MAINWIDGET_HPP

#include <QWidget>

#include "fractalview.hpp"

class MainWidget : public QWidget
{
    Q_OBJECT

    public:
        explicit MainWidget(QWidget* parent = 0);
        ~MainWidget();

    private:
        FractalView* m_fractalView;

};

#endif
