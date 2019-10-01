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
    mainWidget.show();
    return app.exec();
}
