#include "myfunctions.h"
#include <QDir>
#include <QFileDialog>

using namespace cv;
using namespace std;

Calibrator::Calibrator()
{
    fpSrc = "./";
    // findChessboardCorners
    cStep = 30;
    szPattern = Size(5, 4);
    flagChessboard = CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE;
    // cornerSubPix
    szWindow = Size(5, 5);
    szZeroZone = Size(-1, -1);
    critCornerSubPix = TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 60, 0.001);
    // calibrateCamera
    szImage;
    flagCalibrate = 0;
    critCalibrate = TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 30, DBL_EPSILON);
    // output
    fpDst = "./";
    showCorners = true;
}

#define MY_AssertSetting(condition, elsedo)               \
    {                                                     \
        if (!(condition)) {                               \
            CS << "Setting not fulfill:" << (#condition); \
            ret = false;                                  \
        } else {                                          \
            elsedo;                                       \
        }                                                 \
    }

bool Calibrator::loadSettings(QString fileName)
{
    bool ret = true;
    FileStorage fs;
    if (!fs.open(fileName.toStdString(), FileStorage::READ))
        return false;

    std::string _fpSrc;
    float _cStep;
    cv::Size _szPattern;
    cv::Size _szWindow;
    cv::TermCriteria _critCornerSubPix;
    cv::TermCriteria _critCalibrate;
    std::string _fpDst;
    fs["fpSrc"] >> _fpSrc;
    MY_AssertSetting(_fpSrc.size() != 0, fpSrc = QString::fromStdString(_fpSrc));
    fs["cStep"] >> _cStep;
    MY_AssertSetting(_cStep > 0, cStep = _cStep);
    fs["szPattern"] >> _szPattern;
    MY_AssertSetting(_szPattern.area() > 0, szPattern = _szPattern);
    fs["flagChessboard"] >> flagChessboard;
    fs["szWindow"] >> _szWindow;
    MY_AssertSetting(_szWindow.area() > 0, szWindow = _szWindow);
    fs["critCornerSubPix_maxCount"] >> _critCornerSubPix.maxCount;
    MY_AssertSetting(_critCornerSubPix.maxCount > 20, critCornerSubPix.maxCount = _critCornerSubPix.maxCount);
    fs["critCornerSubPix_epsilon"] >> _critCornerSubPix.epsilon;
    MY_AssertSetting(_critCornerSubPix.epsilon > 0, critCornerSubPix.epsilon = _critCornerSubPix.epsilon);
    fs["flagCalibrate"] >> flagCalibrate;
    fs["critCalibrate_maxCount"] >> _critCalibrate.maxCount;
    MY_AssertSetting(_critCalibrate.maxCount > 20, critCalibrate.maxCount = _critCalibrate.maxCount);
    fs["critCalibrate_epsilon"] >> _critCalibrate.epsilon;
    MY_AssertSetting(_critCalibrate.epsilon > 0, critCalibrate.epsilon = _critCalibrate.epsilon);
    fs["fpDst"] >> _fpDst;
    MY_AssertSetting(_fpDst.size() > 0, fpDst = QString::fromStdString(_fpDst));
    fs["showCorners"] >> showCorners;
    return ret;
}

void Calibrator::saveSettings(QString fileName)
{
    FileStorage fs;
    if (!fs.open(fileName.toStdString(), FileStorage::WRITE))
        return;

    fs << "fpSrc" << fpSrc.toStdString();
    fs << "cStep" << cStep;
    fs << "szPattern" << szPattern;
    fs << "flagChessboard" << flagChessboard;
    fs << "szWindow" << szWindow;
    fs << "critCornerSubPix_maxCount" << critCornerSubPix.maxCount;
    fs << "critCornerSubPix_epsilon" << critCornerSubPix.epsilon;
    fs << "flagCalibrate" << flagCalibrate;
    fs << "critCalibrate_maxCount" << critCalibrate.maxCount;
    fs << "critCalibrate_epsilon" << critCalibrate.epsilon;
    fs << "fpDst" << fpDst.toStdString();
    fs << "showCorners" << showCorners;
}

int Calibrator::readImages(QStringList fileNamesFull)
{
    images.clear();
    idxChessboard.clear();
    namesImage.clear();
    foreach (QString fileName, fileNamesFull) {
        Mat src = imread(fileName.toStdString(), 0);
        if (src.empty())
            continue;
        szImage = src.size();
        images.push_back(src);
        namesImage.push_back(getBaseName(fileName));
    }
    return images.size();
}

void Calibrator::detectChessboards()
{
    idxChessboard.clear();
    vptsImage.clear();
    for (int i = 0; i < images.size(); i++) {
        idxChessboard.push_back(vptsImage.size());
        if (images[i].empty())
            continue;
        detectChessboards(images[i], vptsImage);
    }
    idxChessboard.push_back(vptsImage.size());

    if (showCorners) {
        QDir qd;
        QString fpCorners = fpDst + "/CornerImages";
        qd.mkpath(fpCorners);
        for (int i = 0; i < images.size(); i++) {
            if (images[i].empty())
                continue;
            Mat imgDst;
            cvtColor(images[i], imgDst, CV_GRAY2BGR);
            for (int j = idxChessboard[i]; j < idxChessboard[i + 1]; j++)
                drawChessboardCorners(imgDst, szPattern, vptsImage[j], true);
            QString fnDst = fpCorners + "/" + namesImage[i] + "_corners.png";
            imwrite(fnDst.toStdString(), imgDst);
        }
    }
}

void Calibrator::detectChessboards(const Mat& src, std::vector<std::vector<Point2f>>& vpts2d)
{
    std::vector<Point2f> corners;
    if (!findChessboardCorners(src, szPattern, corners, flagChessboard))
        return;
    cornerSubPix(src, corners, szWindow, szZeroZone, critCornerSubPix);
    vpts2d.push_back(corners);
    Mat _src = src.clone();
    Point2f ptsF[4];
    minAreaRect(corners).points(ptsF);
    Point ptsI[4];
    for (int i = 0; i < 4; i++)
        ptsI[i] = ptsF[i];
    fillConvexPoly(_src, ptsI, 4, Scalar::all(0));
    detectChessboards(_src, vpts2d);
}

std::vector<Point3f> Calibrator::defineWorldPts()
{
    vector<Point3f> pts;
    for (int i = 0; i < szPattern.height; i++)
        for (int j = 0; j < szPattern.width; j++)
            pts.push_back(Point3f(j * cStep, i * cStep, 0));
    return pts;
}

double Calibrator::calibrate(QString fnInitialParams)
{
    if (vptsImage.size() == 0 || szImage.area() == 0)
        return -1;
    vector<Point3f> pts3d = defineWorldPts();
    vector<vector<Point3f>> vptsWorld;
    vptsWorld.resize(vptsImage.size(), pts3d);
    vector<Mat> rvecs, tvecs;

    // load initial cam params
    if (!fnInitialParams.isEmpty()) {
        QString fnFull = fpSrc + "/" + fnInitialParams;
        FileStorage fs;
        if (fs.open(fnFull.toStdString(), FileStorage::READ)) {
            fs["camMtx"] >> camMtx;
            fs["disCef"] >> disCef;
            CS << "Initial intrinsic params read.";
        }
    }

    return calibrateCamera(vptsWorld, vptsImage, szImage, camMtx, disCef,
        rvecs, tvecs, flagCalibrate, critCalibrate);
}

void Calibrator::writeCalibResult(QString fileName)
{
    QDir qd;
    qd.mkpath(fpDst);
    QString fnFull = fpDst + "/" + fileName;
    FileStorage fs;
    if (!fs.open(fnFull.toStdString(), FileStorage::WRITE))
        return;
    fs << "camMtx" << camMtx;
    fs << "disCef" << disCef;
}

QString getBaseName(QString fileName)
{
    int id1 = 0, id2 = -1;
    id1 = fileName.lastIndexOf("/");
    id1 += 1;
    id2 = fileName.lastIndexOf(".");
    return fileName.mid(id1, id2 - id1);
}
