#
# Matthew Smith
# github.com/mattonaise
# Created on Sep 29, 2019
#
# FastFractals.pro
#

# Project Settings
TEMPLATE = app
TARGET = FastFractals

# QT modules
QT += widgets

# qmake directories
MOC_DIR = moc
OBJECTS_DIR = bin

# includes and libraries
INCLUDEPATH += .

# C/C++ compiler flags
QMAKE_CXXFLAGS += -g

# C/C++ source files
VPATH += src/
HEADERS += mainwidget.hpp \
    FractalView.hpp
SOURCES += main.cpp \
    mainwidget.cpp \
    FractalView.cpp
