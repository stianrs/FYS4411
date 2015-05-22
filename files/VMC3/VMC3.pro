TEMPLATE = app
CONFIG += console
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += main.cpp \
    investigate.cpp \
    lib.cpp \
    vmcsolver.cpp \
    hydrogenic.cpp \
    #gaussian.cpp \
    #molecules.cpp

OTHER_FILES +=

HEADERS += \
    investigate.h \
    lib.h \
    vmcsolver.h \
    hydrogenic.h \
    #gaussian.h \
    #molecules.h

LIBS += -larmadillo -lblas -llapack


# MPI Settings
QMAKE_CXX = mpicxx
QMAKE_CXX_RELEASE = $$QMAKE_CXX
QMAKE_CXX_DEBUG = $$QMAKE_CXX
QMAKE_LINK = $$QMAKE_CXX
QMAKE_CC = mpicc

QMAKE_CFLAGS += $$system(mpicc --showme:compile)
QMAKE_LFLAGS += $$system(mpicxx --showme:link)
QMAKE_CXXFLAGS += $$system(mpicxx --showme:compile) -DMPICH_IGNORE_CXX_SEEK
QMAKE_CXXFLAGS_RELEASE += $$system(mpicxx --showme:compile) -DMPICH_IGNORE_CXX_SEEK
