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

class FractalView;
class FractalGenerator;

class MainWidget : public QWidget
{
    Q_OBJECT

    public:
        explicit MainWidget(QWidget* parent = nullptr);
        ~MainWidget();

    protected:
        void closeEvent(QCloseEvent* event) override;

    private:
        FractalView* m_fractalView;
        FractalGenerator* m_fractalGenerator;

};

#endif
