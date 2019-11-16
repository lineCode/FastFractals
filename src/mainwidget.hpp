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
#include <QMap>

class FractalView;
class FractalGenerator;
class FractalModel;
class QSlider;
class QLabel;

class MainWidget : public QWidget
{
    Q_OBJECT

    public:
        explicit MainWidget(QWidget* parent = nullptr);
        ~MainWidget();

    signals:
        void modelUpdated(FractalModel* newModel);

    private slots:
        void fractalUpdated();

    protected:
        void showEvent(QShowEvent* event) override;
        void closeEvent(QCloseEvent* event) override;

    private:
        void setUpUI();
        void updateModel();

        FractalView* m_fractalView;
        FractalGenerator* m_fractalGenerator;
        FractalModel* m_currentModel;

        QMap<QString, FractalModel*> m_modelsMap;

        QLabel* m_numThreadsLabel;
        QSlider* m_numThreadsSlider;
        QLabel* m_numPointsLabel;
        QSlider* m_numPointsSlider;
        QLabel* m_kernelRuntimeLabel;
};

#endif
