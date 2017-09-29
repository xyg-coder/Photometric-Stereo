#ifndef PCVASSIGNMENT1_PHOTOMETRICSTERO_H_
#define PCVASSIGNMENT1_PHOTOMETRICSTERO_H_

#include <string>
#include <vector>

#include <opencv2\core\core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>

#include "PFMAccess.h"

namespace phoSte {
struct circle {
    circle(double x = 0, double y = 0, double r = 0): mCenterX(x), mCenterY(y), mR(r) {
    }
    double mCenterX;
    double mCenterY;
    double mR;
};

struct light {
    light(double x, double y, double z): mx(x), my(y), mz(z) {
    }
    double mx;
    double my;
    double mz;
    double mIntensity;
};

class photometryStero {
  public:
    // fill the imageNames and maskNames
    // use image with euqal distance
    photometryStero(int n, int startI, int endI, std::string path,
                    std::string metal1Phere1Name, std::string metal2Phere1Name,
                    std::string lambertPhereName, std::string objectName, double discardRatio = 0.1);
    photometryStero(const photometryStero&) = delete;
    photometryStero& operator = (const photometryStero&) = delete;
    ~photometryStero();
    bool readImage(); // read the images and masks according to the ImageNames
    void getLightInformation(const int metalIndex, const int lambIndex);
    void getPixelNormAndAlbedo(const int objectIndex);
    cv::Mat outputNormalImage(int objectIndex);
    cv::Mat outputAlbedoImage(int objectIndex);
    cv::Mat outputNormalWithAlbedo(int objectIndex);
    cv::Mat getHeightMap(int mode, double parameter);
    // below are functions for test
    //
    void outputImage();
    // build one mask with size*size in the middle to test the N;
    void addSmallMaskForObject(int size, int midX, int midY);
  private:
    const int imageNum; // the num of images used to calculate;
    std::vector<std::string> mImageNames; // the name of images
    std::vector<std::string> mMaskNames; // the name of mask
    std::vector<cv::Mat> mp2Images;
    std::vector<cv::Mat> mp2Mask;
    // when the PFMACCess destructs, the data will be destroyed
    // so it's very dangerous, it should not be copied
    std::vector<CPFMAccess*> mImageStorage;
    // tht circle data for the metalSphere
    phoSte::circle m_metalSphere;
    phoSte::circle m_lambSpere;
    std::vector<phoSte::light> m_light;
    cv::Mat mN;
    cv::Mat mAlbedo;
    std::vector<int> mObjectX;
    std::vector<int> mObjectY;
    std::vector<int> mInvalidIndex;
    const double mDiscardRatio;
};
}

#endif
