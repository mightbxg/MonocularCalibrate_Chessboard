QT += widgets

CONFIG += c++11 console
CONFIG -= app_bundle

# The following define makes your compiler emit warnings if you use
# any feature of Qt which as been marked deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if you use deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES += main.cpp \
    myfunctions.cpp

RC_ICONS = bug.ico

# opencv3.4
PATH_OPENCV = D:/Library/opencv3.4_Qt
INCLUDEPATH += $$PATH_OPENCV/include\
               $$PATH_OPENCV/include/opencv\
               $$PATH_OPENCV/include/opencv2
Debug:LIBS += $$PATH_OPENCV/x64/vc15/lib/opencv_*340d.lib
Release:LIBS += $$PATH_OPENCV/x64/vc15/lib/opencv_*340.lib

HEADERS += \
    myfunctions.h

