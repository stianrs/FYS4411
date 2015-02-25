TEMPLATE = app
CONFIG += console
CONFIG -= qt

LIBS += -llapack -larmadillo

SOURCES += main.cpp \
    vmcsolver.cpp \
    lib.cpp \
    investigate.cpp

HEADERS += \
    vmcsolver.h \
    lib.h \
    investigate.h
