/**
 * Matthew Smith
 * github.com/mattonaise
 * Created on Oct 01, 2019
 *
 * main.cpp
 **/

#include <QtWidgets>

#include "mainwidget.hpp"

int main(int argc, char* argv[])
{
    QApplication app(argc, argv);

    MainWidget mainWidget;

    // Set size and position of main window
    mainWidget.resize(QDesktopWidget().availableGeometry(&mainWidget).size()
            * 0.5f);
    mainWidget.setGeometry(
        QStyle::alignedRect(
            Qt::LeftToRight,
            Qt::AlignCenter,
            mainWidget.size(),
            QDesktopWidget().availableGeometry(&mainWidget)
        )
    );
    mainWidget.setWindowTitle("Fast Fractals");
    mainWidget.show();

    return app.exec();
}
