TEMPLATE = app
CONFIG += console
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += main.cpp \
    investigate.cpp \
    lib.cpp \
    vmcsolver.cpp

OTHER_FILES +=

HEADERS += \
    investigate.h \
    lib.h \
    vmcsolver.h

LIBS += -larmadillo -lblas -llapack

