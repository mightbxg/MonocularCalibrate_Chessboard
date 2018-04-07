#include <QApplication>
#include <QFileDialog>
#include <iostream>
#include <myfunctions.h>

using namespace cv;
using namespace std;

#define FILTER_IMAGE "images (*.bmp *.png *.jpg *.jpeg)"

#define MY_AssertSetting(condition, elsedo)           \
    {                                                 \
        if (!(condition)) {                           \
            CS << "Invalid setting:" << (#condition); \
                                                      \
        } else {                                      \
            elsedo;                                   \
        }                                             \
    }

#define EXIT(code)       \
    {                    \
        CS << "";        \
        system("pause"); \
        return code;     \
    }

void greetings()
{
    CS << "***********************************************************";
    CS << "*            MonocularCalibrate with Chessboard           *";
    CS << "*                      by Victor Bian                     *";
    CS << "***********************************************************";
    CS << "";
}

int main(int argc, char* argv[])
{
    QApplication a(argc, argv);
    system("title MonocularCalibrate with Chessboard");
    greetings();

    // load settings
    QString fnSettings = "./settings.yml";
    Calibrator calib;
    calib.fpSrc = "D:/Experiments/calibTest180405";
    calib.fpDst = "D:/Experiments/calibTest180405";
    if (!calib.loadSettings(fnSettings)) {
        CS << "Settings invalid, task abort.";
        calib.saveSettings(fnSettings);
        EXIT(1);
    }

    // read images
    QStringList fnsImage = QFileDialog::getOpenFileNames(0, "select images which contain chessboards",
        calib.fpSrc, FILTER_IMAGE);
    system("cls");
    greetings();
    CS << "Settings loaded from settings.yml";
    if (fnsImage.isEmpty()) {
        CS << "No image read.";
        EXIT(2);
    }
    CS << "Image number:" << fnsImage.size();
    calib.readImages(fnsImage);
    calib.detectChessboards();
    CS << "Pattern found:" << calib.vptsImage.size();

    QString fnCalibResult = "CalibResult.yml";
    double rmse = calib.calibrate(fnCalibResult);
    CS << "Calibrate done. RMSE =" << rmse;
    calib.writeCalibResult(fnCalibResult);

    CS << "End.";
    EXIT(0);
}
