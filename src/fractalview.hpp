#ifndef FRACTALVIEW_HPP
#define FRACTALVIEW_HPP

#include <QOpenGLWidget>
#include <QOpenGLFunctions>

class FractalView : public QOpenGLWidget, protected QOpenGLFunctions
{
    Q_OBJECT
    
    public:
        FractalView(QWidget* parent = 0);
        ~FractalView();

    protected:
        void initializeGL() override;
        void resizeGL(int w, int h) override;
        void paintGL() override;

    private:
        void cleanupGL();
        void printContextInfo();
};

#endif
