#ifndef MYFUNCTIONS_H
#define MYFUNCTIONS_H

#include <cv.hpp>

#ifndef CS
#include <QDebug>
#define CS qDebug()
#endif

class Calibrator {
public:
    // source
    QString fpSrc;
    std::vector<cv::Mat> images;
    std::vector<int> idxChessboard;
    QStringList namesImage;
    // findChessboardCorners
    float cStep;
    cv::Size szPattern;
    int flagChessboard;
    std::vector<std::vector<cv::Point2f>> vptsImage;
    // cornerSubPix
    cv::Size szWindow;
    cv::Size szZeroZone;
    cv::TermCriteria critCornerSubPix;
    // calibrateCamera
    cv::Size szImage;
    int flagCalibrate;
    cv::TermCriteria critCalibrate;
    cv::Mat camMtx, disCef;
    // output
    QString fpDst;
    bool showCorners;

public:
    Calibrator();
    bool loadSettings(QString fileName);
    void saveSettings(QString fileName);
    int readImages(QStringList fileNamesFull);
    void detectChessboards();
    void detectChessboards(const cv::Mat& src, std::vector<std::vector<cv::Point2f>>& vpts2d);
    std::vector<cv::Point3f> defineWorldPts();
    double calibrate(QString fnInitialParams = "");
    void writeCalibResult(QString fileName);
};

QString getBaseName(QString fileName);

#endif // MYFUNCTIONS_H
