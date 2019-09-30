#
# Matthew Smith
# github.com/mattonaise
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
HEADERS +=
SOURCES += main.cpp


