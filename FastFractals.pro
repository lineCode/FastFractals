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

# Includes and libraries
INCLUDEPATH += .

# C/C++ compiler flags
QMAKE_CXXFLAGS += -g

# C/C++ source files
VPATH += src/
HEADERS += mainwidget.hpp \
    fractalview.hpp \
    shaders.hpp \
    fractalgenerator.hpp \
    cuda.hpp \
    fractalmodel.hpp \
    defaultvalues.hpp \
    mapping.hpp
SOURCES += main.cpp \
    mainwidget.cpp \
    fractalview.cpp \
    fractalgenerator.cpp \
    fractalmodel.cpp

### CUDA ###

# CUDA source files
CUDA_SOURCES = cuda.cu

# CUDA install directory
CUDA_DIR = /opt/cuda

# CUDA includes and libraries
INCLUDE_PATH += $$CUDA_DIR/include
LIBS += -L $$CUDA_DIR/lib64 -lcudart -lcuda

# CUDA compiler settings
cuda.input = CUDA_SOURCES
cuda.output = ${OBJECTS_DIR}${QMAKE_FILE_BASE}.o
cuda.commands = $$CUDA_DIR/bin/nvcc -c -g -o ${QMAKE_FILE_OUT} \
    ${QMAKE_FILE_NAME}
cuda.dependency_type = TYPE_C
cuda.depend_command = $$CUDA_DIR/bin/nvcc -M ${QMAKE_FILE_NAME} \
    | sed \"s/^.*: //\"     # this line fixes empty target error in Makefile

# Add CUDA compiler
QMAKE_EXTRA_COMPILERS += cuda
