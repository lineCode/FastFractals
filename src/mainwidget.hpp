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
class FractalModel;

class MainWidget : public QWidget
{
    Q_OBJECT

    public:
        explicit MainWidget(QWidget* parent = nullptr);
        ~MainWidget();

    signals:
        void modelUpdated(FractalModel* newModel);

    protected:
        void closeEvent(QCloseEvent* event) override;

    private:
        void setUpUI();

        FractalView* m_fractalView;
        FractalGenerator* m_fractalGenerator;
        FractalModel* m_currentModel;

};

#endif
