#include "PhotometricStero.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <climits>
#include <cstdio>
#include<Eigen/SparseCholesky>

namespace phoSte {

namespace {
// use iniRSize points to get the initial r
const int iniRSize = 10;
// use 20 points to get the circle information
const int adjustSize = 20;
// iterate iterTimes to get the circle
const int iterTime = 10;
// threshold of stopping the iteration
const double normThresh = 0.01;

phoSte::circle getCircle(std::vector<cv::Point>& contour) {
    double circle_x = 0;
    double circle_y = 0;
    double r = 0;
    int pointSize = contour.size();
    if(pointSize < iniRSize) {
        std::cout << "the counter is error" << std::endl;
        exit(-1);
    }
    for(int i = 0; i < pointSize; i++) {
        circle_x += contour[i].x;
        circle_y += contour[i].y;
    }
    circle_x /= pointSize;
    circle_y /= pointSize;
    for(int i = 0; i < iniRSize; i++) {
        r += sqrt((contour[i].x - circle_x) * (contour[i].x - circle_x) +
                  (contour[i].y - circle_y) * (contour[i].y - circle_y));
    }
    r /= iniRSize;

    // use adjustment calculation to get the circle
    if(pointSize < adjustSize) {
        std::cout << "the counter is error" << std::endl;
        exit(-1);
    }
    std::vector<int> pointIndex;
    pointIndex.reserve(adjustSize);
    int distanceOfIndex = pointSize / adjustSize;
    pointIndex.push_back(0);
    for(int i = 1; i < adjustSize; i++) {
        pointIndex.push_back(pointIndex[i - 1] + distanceOfIndex);
    }

    cv::Mat B(adjustSize, 3, CV_64F);
    cv::Mat L(adjustSize, 1, CV_64F);
    double v_norm = INT_MAX;
    for(int i = 0; i < iterTime; i++) {
        double *pB = (double *)B.data;
        double *pL = (double *)L.data;
        for(int j = 0; j < adjustSize; j++) {
            *pB = contour[pointIndex[j] + i].x - circle_x;
            *(pB + 1) = contour[pointIndex[j] + i].y - circle_y;
            *(pB + 2) = r;
            pB += 3;
            *pL = ((contour[pointIndex[j] + i].x - circle_x) * (contour[pointIndex[j] + i].x - circle_x)
                   + (contour[pointIndex[j] + i].y - circle_y) * (contour[pointIndex[j] + i].y - circle_y)
                   - r * r) / 2;
            pL++;
        }
        cv::Mat pseudoInverse;
        cv::invert(B, pseudoInverse, cv::DECOMP_SVD);
        cv::Mat result = pseudoInverse * L;
        circle_x += result.at<double>(0, 0);
        circle_y += result.at<double>(1, 0);
        r += result.at<double>(2, 0);
    }
    return phoSte::circle(circle_x, circle_y, r);
}

// get the direction of light according to the metalCircle
phoSte::light getLightDirection(const phoSte::circle& metalCircle,
                                const cv::Point& maxPoint) {
    int maxX = maxPoint.x;
    int maxY = maxPoint.y;
    double nx = maxX - metalCircle.mCenterX;
    double ny = maxY - metalCircle.mCenterY;
    double nz = sqrt(metalCircle.mR * metalCircle.mR - nx * nx - ny * ny);
    double rootSquareSum = sqrt(nx * nx + ny * ny + nz * nz);
    nx /= rootSquareSum;
    ny /= rootSquareSum;
    nz /= rootSquareSum;
    double nDotR = nz;
    double lx = 2 * nDotR * nx;
    double ly = 2 * nDotR * ny;
    double lz = 2 * nDotR * nz - 1;

    return phoSte::light(lx, ly, lz);
}

void swapNum(double& a, double& b) {
    double tmp = a;
    a = b;
    b = tmp;

}

// find the num at the ratio of the nums
// always use the endI as the pivot
// use the random integer because there might be some order of the pixel value
double quickSelect(std::vector<double>& nums, int startI, int endI, int selectI) {
    if(startI == endI)
        return nums[startI];
    int pivotIndex = rand() % (endI - startI + 1) + startI;
    swapNum(nums[endI], nums[pivotIndex]);
    double pivot = nums[endI];
    int i = startI - 1;
    int j = startI;
    while(j != endI) {
        if(nums[j] < pivot) {
            i++;
            swapNum(nums[j], nums[i]);
        }
        j++;
    }
    i++;
    swapNum(nums[endI], nums[i]);
    if(i == selectI)
        return nums[i];
    else if(i < selectI) {
        return quickSelect(nums, i + 1, endI, selectI);
    } else
        return quickSelect(nums, startI, i - 1, selectI);
}

// do the transform to image to output
// maxValue to 255, minValue to 0
int linearTransform(double max, double min, double value) {
    return 255 / (max - min) * (value - min);
}
}

photometryStero::~photometryStero() {
    if(!mImageStorage.empty()) {
        for(int i = 0; i < mImageStorage.size(); i++) {
            delete mImageStorage[i];
        }
    }
}

photometryStero::photometryStero(int n, int startI, int endI, std::string path,
                                 std::string metal1Phere1Name, std::string metal2Phere1Name,
                                 std::string lambertPhereName, std::string objectName,
                                 double discardRatio): mDiscardRatio(discardRatio), imageNum(n) {
    int distance = (endI - startI + 1) / n;
    for (int i = 0; i < n; i++) {
        int index = i * distance + startI;
        char imageName[100];
        sprintf(imageName, "image%03d.pbm", index);
        std::string imageNameStr = imageName;
        imageNameStr = path + imageNameStr;
        mImageNames.push_back(imageNameStr);
    }
    std::string mask1Name = metal1Phere1Name;
    std::string mask2Name = metal2Phere1Name;
    std::string mask3Name = lambertPhereName;

    mMaskNames.push_back(path + mask1Name);
    mMaskNames.push_back(path + mask2Name);
    mMaskNames.push_back(path + mask3Name);
    mMaskNames.push_back(path + objectName);
}

// read image and mask image according to the name
// return true if read successfully
// return false if read fail
bool photometryStero::readImage() {
    if(mImageNames.empty() || mMaskNames.empty())
        return false;

    // read the image
    for(int i = 0; i < mImageNames.size(); i++) {
        CPFMAccess* PFMAccessI = new CPFMAccess();
        char *tpChar = new char[mImageNames[i].size() + 1];
        std::strcpy(tpChar, mImageNames[i].c_str());
        if(! (PFMAccessI -> LoadFromFile(tpChar)) )
            return false;
        delete[]tpChar;
        float *imageData = PFMAccessI -> GetData();
        mImageStorage.push_back(PFMAccessI);
        cv::Mat image((PFMAccessI->GetWidth()) * (PFMAccessI->GetHeight()), 3,
                      CV_32F, imageData);
        cv::Mat p2Image (0.2989 * image.col(0) + 0.5870 * image.col(1)
                         + 0.1140 * image.col(2));
        p2Image = p2Image.reshape(0, PFMAccessI->GetHeight());
        mp2Images.push_back(p2Image);
    }
    // read the maskImage
    for(int i = 0; i < mMaskNames.size(); i++) {
        cv::Mat p2Mask(cv::imread(mMaskNames[i], CV_LOAD_IMAGE_GRAYSCALE));
        if(! p2Mask.data )
            return false;
        mp2Mask.push_back(p2Mask);
    }
    return true;
}

void photometryStero::getLightInformation(const int metalIndex, const int lambIndex) {
    // get the metal circle
    cv::threshold(mp2Mask[metalIndex], mp2Mask[metalIndex], 255 / 2, 255, cv::THRESH_BINARY);
    std::vector<std::vector<cv::Point>> metalContour;
    cv::findContours(mp2Mask[metalIndex], metalContour, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
    // use the contour with the max size to calculate the circle
    int mostPoint = 0;
    int mostPointIdx = 0;
    for(int i = 0; i < metalContour.size(); i++) {
        if(metalContour.size() > mostPoint) {
            mostPoint = metalContour.size();
            mostPointIdx = i;
        }
    }
    m_metalSphere = getCircle(metalContour[mostPointIdx]);

    // get the lambertian circle
    cv::threshold(mp2Mask[lambIndex], mp2Mask[lambIndex], 255 / 2, 255, cv::THRESH_BINARY);
    std::vector<std::vector<cv::Point>> lambContour;
    cv::findContours(mp2Mask[lambIndex], lambContour, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
    // use the contour with the max size to calculate the circle
    mostPoint = 0;
    mostPointIdx = 0;
    for(int i = 0; i < lambContour.size(); i++) {
        if(lambContour.size() > mostPoint) {
            mostPoint = lambContour.size();
            mostPointIdx = i;
        }
    }
    m_lambSpere = getCircle(lambContour[mostPointIdx]);

    // get the direction and the intensity of every image
    for(int i = 0; i < mp2Images.size(); i++) {
        cv::Point metalMaxPoint;
        cv::minMaxLoc(mp2Images[i], NULL, NULL, NULL, &metalMaxPoint, mp2Mask[metalIndex]);
        m_light.push_back(getLightDirection(m_metalSphere, metalMaxPoint));
        double lightIntensity;
        cv::minMaxLoc(mp2Images[i], NULL, &lightIntensity, NULL, NULL, mp2Mask[lambIndex]);
        m_light[i].mIntensity = lightIntensity;
    }
}

// calculate the norm and albedo for every pixel
// here, threshold is for the dark discard
void photometryStero::getPixelNormAndAlbedo(const int objectIndex) {
    // calculate the pixel num
    std::vector<double> pixelThreshold;
    for(int i = 0; i < mp2Mask[objectIndex].rows; i++) {
        for(int j = 0; j < mp2Mask[objectIndex].cols; j++) {
            if(mp2Mask[objectIndex].at<unsigned char>(i, j) >= 255) {
                mObjectX.push_back(j);
                mObjectY.push_back(i);
            }
        }
    }
    std::vector<std::vector<double>> allPixelValue;
    allPixelValue.reserve(imageNum);
    int objectPixelNum = mObjectX.size();
    for(int i = 0; i < imageNum; i++) {
        std::vector<double> pixelValue;
        allPixelValue.push_back(pixelValue);
        allPixelValue[i].reserve(objectPixelNum);
        for(int j = 0; j < objectPixelNum; j++) {
            allPixelValue[i].push_back(mp2Images[i].at<float>(mObjectY[j], mObjectX[j]));
        }
    }
    // threshold for every image;
    std::vector<double> allThreshold;
    allThreshold.reserve(imageNum);
    int thresholdIndex = mDiscardRatio * objectPixelNum;
    for(int i = 0; i < imageNum; i++) {
        std::vector<double> tmpPixelValue = allPixelValue[i];
        allThreshold.push_back(
            quickSelect(tmpPixelValue, 0, objectPixelNum - 1, thresholdIndex));
    }

    cv::Mat I(imageNum, objectPixelNum, CV_64F);
    cv::Mat L(imageNum, 3, CV_64F);
    mN.create(3, objectPixelNum, CV_64F);
    mAlbedo.create(1, objectPixelNum, CV_64F);
    double *pL = (double *)L.data;
    for(int i = 0; i < imageNum; i++) {
        *pL = m_light[i].mx;
        *(pL + 1) = m_light[i].my;
        *(pL + 2) = m_light[i].mz;
        pL += 3;
    }
    cv::Mat LPseudoInvert;
    cv::invert(L, LPseudoInvert, cv::DECOMP_SVD);
    double *pI = (double *)I.data;
    double *pN = (double *)mN.data;

    for(int i = 0; i < objectPixelNum; i++) {
        int inValidNum = 0;
        cv::Mat specificL;
        for(int j = 0; j < imageNum; j++) {
            if(allPixelValue[j][i] < allThreshold[j]) {
                inValidNum ++;
                if(inValidNum == 1) {
                    L.copyTo(specificL);
                }
                double *p2LRow = specificL.ptr<double>(j);
                *(p2LRow) = 0;
                *(p2LRow + 1) = 0;
                *(p2LRow + 2) = 0;
                *(pI + j * objectPixelNum + i) = 0;
            } else {
                *(pI + j * objectPixelNum + i) = allPixelValue[j][i] / m_light[j].mIntensity;
            }
            if(imageNum - inValidNum < 3) {
                break;
            }
        }
        if(inValidNum == 0) {
            mN.col(i) = LPseudoInvert * I.col(i);
            double nx = mN.col(i).at<double>(0, 0);
            double ny = mN.col(i).at<double>(1, 0);
            double nz = mN.col(i).at<double>(2, 0);
            mAlbedo.at<double>(0, i) = sqrt(nx * nx + ny * ny + nz * nz);
        } else if(imageNum - inValidNum >= 3) {
            cv::Mat specificLPseudoInvert;
            cv::invert(specificL, specificLPseudoInvert, cv::DECOMP_SVD);
            mN.col(i) = specificLPseudoInvert * I.col(i);
            double nx = mN.col(i).at<double>(0, 0);
            double ny = mN.col(i).at<double>(1, 0);
            double nz = mN.col(i).at<double>(2, 0);
            mAlbedo.at<double>(0, i) = sqrt(nx * nx + ny * ny + nz * nz);
        } else {
            mN.at<double>(0, i) = 0;
            mN.at<double>(1, i) = 0;
            mN.at<double>(2, i) = 0;
            mAlbedo.at<double>(0, i) = 0;
            mInvalidIndex.push_back(i);
        }
    }
}

void photometryStero::outputImage() {
    double max = INT_MIN, min = INT_MAX;
    if (mp2Images.empty()) {
        std::cout << "no image to output" << std::endl;
    }
    cv::Mat out(mp2Images[0].rows, mp2Images[0].cols, CV_8U);
    for (int i = 0; i < mp2Images[0].rows; i++) {
        for (int j = 0; j < mp2Images[0].cols; j++) {
            double pixelValue = mp2Images[0].at<float>(i, j);
            max = pixelValue > max ? pixelValue : max;
            min = pixelValue < min ? pixelValue : min;
        }
    }
    for (int i = 0; i < mp2Images[0].rows; i++) {
        for (int j = 0; j < mp2Images[0].cols; j++) {
            double pixelValue = mp2Images[0].at<float>(i, j);
            out.at<unsigned char>(i, j) = linearTransform(max, min, pixelValue);
        }
    }
    cv::imshow("outputImage", out);
    cv::waitKey(0);
}

cv::Mat photometryStero::outputNormalImage(int objectIndex) {
    cv::Mat result = cv::Mat::zeros(mp2Images[0].rows, mp2Images[0].cols, CV_32FC3);
    int nextInvalid = mInvalidIndex.empty() ? INT_MAX : mInvalidIndex[0];
    int nextInvalidIndex = 0;
    int invalidPixelNum = 0;
    for (int i = 0; i < mObjectX.size(); i++) {
        double nx = mN.at<double>(0, i);
        double ny = mN.at<double>(1, i);
        double nz = mN.at<double>(2, i);
        if (i == nextInvalid) {
            invalidPixelNum++;
            result.at<cv::Vec3f>(mObjectY[i], mObjectX[i])[0] = 0;
            result.at<cv::Vec3f>(mObjectY[i], mObjectX[i])[1] = 0;
            result.at<cv::Vec3f>(mObjectY[i], mObjectX[i])[2] = 0;
            if (nextInvalidIndex + 1 == mInvalidIndex.size()) {
                nextInvalid = INT_MAX;
            } else {
                nextInvalidIndex++;
                nextInvalid = mInvalidIndex[nextInvalidIndex];
            }
        } else {
            double rootsquareSum = sqrt(nx * nx + ny * ny + nz * nz);
            float nxf = (1 + nx / rootsquareSum) / 2;
            float nyf = (1 + ny / rootsquareSum) / 2;
            float nzf = (1 + nz / rootsquareSum) / 2;
            result.at<cv::Vec3f>(mObjectY[i], mObjectX[i])[0] = nxf;
            result.at<cv::Vec3f>(mObjectY[i], mObjectX[i])[1] = nyf;
            result.at<cv::Vec3f>(mObjectY[i], mObjectX[i])[2] = nzf;
        }
    }
    std::cout << invalidPixelNum << " pixels are not drawn" << std::endl;
    return result;
}

// this function is only for debug
void photometryStero::addSmallMaskForObject(int size, int midX, int midY) {
    cv::Mat smallMask = cv::Mat::zeros(mp2Images[0].rows, mp2Images[0].cols, CV_8U);
    int num = 0;
    for (int i = midY - size / 2; i <= midY + size / 2; i++) {
        for (int j = midX - size / 2; j <= midX + size / 2; j++) {
            smallMask.at<unsigned char>(i, j) = 255;
            num++;
        }
    }
    mp2Mask.push_back(smallMask);
}

cv::Mat photometryStero::outputAlbedoImage(int objectIndex) {
    cv::Mat result = cv::Mat::zeros(mp2Images[0].rows, mp2Images[0].cols, CV_32F);
    int nextInvalid = mInvalidIndex.empty() ? INT_MAX : mInvalidIndex[0];
    int nextInvalidIndex = 0;
    int invalidPixelNum = 0;
    for (int i = 0; i < mObjectX.size(); i++) {
        float albedo = mAlbedo.at<double>(0, i);
        if (i == nextInvalid) {
            invalidPixelNum++;
            result.at<float>(mObjectY[i], mObjectX[i]) = 0;
            if (nextInvalidIndex + 1 == mInvalidIndex.size()) {
                nextInvalid = INT_MAX;
            } else {
                nextInvalidIndex++;
                nextInvalid = mInvalidIndex[nextInvalidIndex];
            }
        } else {
            result.at<float>(mObjectY[i], mObjectX[i]) = albedo;
        }
    }
    std::cout << invalidPixelNum << " pixels are not drawn" << std::endl;
    return result;
}

// use the normal direction dot the (0, 0, 1) and then times the albedo
cv::Mat photometryStero::outputNormalWithAlbedo(int objectIndex) {
    cv::Mat result = cv::Mat::zeros(mp2Images[0].rows, mp2Images[0].cols, CV_32F);
    int nextInvalid = mInvalidIndex.empty() ? INT_MAX : mInvalidIndex[0];
    int nextInvalidIndex = 0;
    int invalidPixelNum = 0;
    for (int i = 0; i < mObjectX.size(); i++) {
        float albedo = mAlbedo.at<double>(0, i);
        double nx = mN.at<double>(0, i);
        double ny = mN.at<double>(1, i);
        double nz = mN.at<double>(2, i);
        float intensity = nz / sqrt(nx * nx + ny * ny + nz * nz) * albedo;
        if (i == nextInvalid) {
            invalidPixelNum++;
            result.at<float>(mObjectY[i], mObjectX[i]) = 0;
            if (nextInvalidIndex + 1 == mInvalidIndex.size()) {
                nextInvalid = INT_MAX;
            } else {
                nextInvalidIndex++;
                nextInvalid = mInvalidIndex[nextInvalidIndex];
            }
        } else {
            result.at<float>(mObjectY[i], mObjectX[i]) = intensity;
        }
    }
    std::cout << invalidPixelNum << " pixels are not drawn" << std::endl;
    return result;
}

// use the normal map to get the height map
// mode 1: use the discard ratio to discard some low lengths
// mode 2: use sigmoid function to map to the depth
// in order to avoid the mistake due to the small nz, I plus 0.05 when nz is too small
cv::Mat photometryStero::getHeightMap(int mode, double parameter) {
    // make the map from object pixel to object index
    int nextInvalid = mInvalidIndex.empty() ? INT_MAX : mInvalidIndex[0];
    int nextInvalidIndex = 0;
    cv::Mat pixel2ObjectIndex = cv::Mat::zeros(mp2Images[0].rows,
                                mp2Images[0].cols, CV_32S);
    for (int i = 0; i < mObjectX.size(); i++) {
        if (i == nextInvalid) {
            pixel2ObjectIndex.at<int>(mObjectY[i], mObjectX[i]) = -1;
            if (nextInvalidIndex + 1 == mInvalidIndex.size()) {
                nextInvalid = INT_MAX;
            } else {
                nextInvalidIndex++;
                nextInvalid = mInvalidIndex[nextInvalidIndex];
            }
        } else {
            // use i + 1 to prevent the collision with 0
            pixel2ObjectIndex.at<int>(mObjectY[i], mObjectX[i]) = i + 1;

        }
    }
    // first of all, iterate until all object points have normal value
    // I use the 3*3 window to get average normal
    cv::Mat NForHeight;
    mN.copyTo(NForHeight);
    cv::Mat heightMap = cv::Mat::zeros(mp2Images[0].rows, mp2Images[0].rows, CV_32F);
    int nInvalid = mInvalidIndex.size();
    int iterateTime = 0;
    while (nInvalid > 0) {
        if (iterateTime > 10) {
            std::cout << "too many iterations during getting values to the invalid pixel"
                      << std::endl;
            return heightMap;
        }
        iterateTime++;
        for (int i = 0; i < mInvalidIndex.size(); i++) {
            int index = mInvalidIndex[i];
            if (pixel2ObjectIndex.at<int>(mObjectY[index], mObjectX[index]) == -1) {
                int x = mObjectX[index];
                int y = mObjectY[index];
                double nxSum = 0;
                double nySum = 0;
                double nzSum = 0;
                int nValid = 0;
                for (int dx = -1; dx <= 1; dx++) {
                    for (int dy = -1; dy <= 1; dy++) {
                        if (pixel2ObjectIndex.at<int>(y + dy, x + dx) > 0) {
                            nValid++;
                            nxSum += NForHeight.at<double>(0, pixel2ObjectIndex.at<int>(y + dy, x + dx));
                            nySum += NForHeight.at<double>(1, pixel2ObjectIndex.at<int>(y + dy, x + dx));
                            nzSum += NForHeight.at<double>(2, pixel2ObjectIndex.at<int>(y + dy, x + dx));
                        }
                    }
                }
                if (nValid > 0) {
                    pixel2ObjectIndex.at<int>(y, x) = index + 1;
                    nInvalid--;
                    nxSum /= nValid;
                    nySum /= nValid;
                    nzSum /= nValid;
                    double squareSum = sqrt(nxSum * nxSum + nySum * nySum + nzSum * nzSum);
                    NForHeight.at<double>(0, index) = nxSum / squareSum;
                    NForHeight.at<double>(1, index) = nySum / squareSum;
                    NForHeight.at<double>(2, index) = nzSum / squareSum;
                }
            }
        }
    }
    std::cout << "the iteration time is " << iterateTime;

    // form the sparse matrix and solve the linear equation
    // first of all, scan all the points to find the number of equations
    int nEquation = 0;
    for (int i = 0; i < mObjectX.size(); i++) {
        int x = mObjectX[i];
        int y = mObjectY[i];
        if (pixel2ObjectIndex.at<int>(y + 1, x) > 0) {
            nEquation++;
        }
        if (pixel2ObjectIndex.at<int>(y, x + 1) > 0) {
            nEquation++;
        }
    }
    // then use the sparse matrix to solve the linear equation
    // I set z(allIndex / 2)=0 as the coordinate origin
    int midIndex = mObjectX.size() / 2;
    int midX = mObjectX[midIndex];
    int midY = mObjectY[midIndex];
    double midNx_midNz = -(NForHeight.at<double>(0, midIndex)) / (NForHeight.at<double>(2, midIndex));
    double midNy_midNz = -(NForHeight.at<double>(1, midIndex)) / (NForHeight.at<double>(2, midIndex));
    Eigen::VectorXd b(nEquation);
    typedef Eigen::SparseMatrix<double> SpMat;
    // since the sparse matrix of eigen is col main, so use the transpose to store the data
    SpMat A(mObjectX.size() - 1, nEquation);
    typedef Eigen::Triplet<double> T;
    std::vector<T> tripletList;
    tripletList.reserve(nEquation * 3);
    int currentEquation = 0;
    for (int i = 0; i < mObjectX.size(); i++) {
        int x = mObjectX[i];
        int y = mObjectY[i];
        if (i != midIndex) {
            int thisZIndex = i < midIndex ? i : i - 1;
            if (pixel2ObjectIndex.at<int>(y + 1, x) > 0) {
                if (y + 1 != midY || x != midX) {
                    int index = pixel2ObjectIndex.at<int>(y + 1, x) - 1;
                    int upZIndex = index < midIndex ? index : index - 1;
                    tripletList.push_back(T(upZIndex, currentEquation, 1));
                    tripletList.push_back(T(thisZIndex, currentEquation, -1));
                    if (NForHeight.at<double>(2, index) < 0.001) {
                        b(currentEquation) = -(NForHeight.at<double>(1, index) + 0.05) / (NForHeight.at<double>(2, index) + 0.05);
                    } else {
                        b(currentEquation) = -(NForHeight.at<double>(1, index)) / (NForHeight.at<double>(2, index));
                    }
                    currentEquation++;
                } else {
                    int index = pixel2ObjectIndex.at<int>(y + 1, x) - 1;
                    tripletList.push_back(T(thisZIndex, currentEquation, -1));
                    if (NForHeight.at<double>(2, index) < 0.001) {
                        b(currentEquation) = -(NForHeight.at<double>(1, index) + 0.05) / (NForHeight.at<double>(2, index) + 0.05);
                    } else {
                        b(currentEquation) = -(NForHeight.at<double>(1, index)) / (NForHeight.at<double>(2, index));
                    }
                    currentEquation++;
                }
            }
            if (pixel2ObjectIndex.at<int>(y, x + 1) > 0) {
                if (y != midY || x + 1 != midX) {
                    int index = pixel2ObjectIndex.at<int>(y, x + 1) - 1;
                    int rightZIndex = index < midIndex ? index : index - 1;
                    tripletList.push_back(T(rightZIndex, currentEquation, 1));
                    tripletList.push_back(T(thisZIndex, currentEquation, -1));
                    if (NForHeight.at<double>(2, index) < 0.001) {
                        b(currentEquation) = -(NForHeight.at<double>(0, index) + 0.05) / (NForHeight.at<double>(2, index) + 0.05);
                    } else {
                        b(currentEquation) = -(NForHeight.at<double>(0, index)) / (NForHeight.at<double>(2, index));
                    }
                    currentEquation++;
                } else {
                    int index = pixel2ObjectIndex.at<int>(y, x + 1) - 1;
                    tripletList.push_back(T(thisZIndex, currentEquation, -1));
                    if (NForHeight.at<double>(2, index) < 0.001) {
                        b(currentEquation) = -(NForHeight.at<double>(0, index) + 0.05) / (NForHeight.at<double>(2, index) + 0.05);
                    } else {
                        b(currentEquation) = -(NForHeight.at<double>(0, index)) / (NForHeight.at<double>(2, index));
                    }
                    currentEquation++;
                }
            }
        } else {
            int thisIndex = midIndex;
            if (pixel2ObjectIndex.at<int>(y + 1, x) > 0) {
                int index = pixel2ObjectIndex.at<int>(y + 1, x) - 1;
                int upZIndex = index < midIndex ? index : index - 1;
                tripletList.push_back(T(upZIndex, currentEquation, 1));
                b(currentEquation) = -midNy_midNz;
                currentEquation++;
            }
            if (pixel2ObjectIndex.at<int>(y, x + 1) > 0) {
                int index = pixel2ObjectIndex.at<int>(y, x + 1) - 1;
                int rightZIndex = index < midIndex ? index : index - 1;
                tripletList.push_back(T(rightZIndex, currentEquation, 1));
                b(currentEquation) = -midNx_midNz;
                currentEquation++;
            }
        }
    }
    A.setFromTriplets(tripletList.begin(), tripletList.end());
    SpMat Atranspose = A.transpose();
    A = A * Atranspose;
    A.makeCompressed();
    Eigen::SimplicialLLT <Eigen::SparseMatrix<double>> solver;
    solver.compute(A);
    if (solver.info() != Eigen::Success) {
        std::cout << "decomposition fails" << std::endl;
        return heightMap;
    }
    b = Atranspose.transpose() * b;
    Eigen::VectorXd z = solver.solve(b);
    if (solver.info() != Eigen::Success) {
        std::cout << "solving failed" << std::endl;
        return heightMap;
    }

    // try to discard ratio to show;
    switch (mode) {
    case 1: {
        std::vector<double> zVec;
        zVec.reserve(mObjectX.size());
        for (int i = 0; i < mObjectX.size() - 1; i++) {
            zVec.push_back(z(i));
        }
        zVec.push_back(0);
        std::vector<double> zVec2(zVec.begin(), zVec.end());
        double minThreshold = quickSelect(zVec, 0, zVec.size() - 1, zVec.size() * parameter);
        double maxZ = z.maxCoeff();
        for (int i = 0; i < mObjectX.size(); i++) {
            int x = mObjectX[i];
            int y = mObjectY[i];
            double originz;
            if (i < midIndex) {
                originz = z(i);
            } else if (i > midIndex) {
                originz = z(i - 1);
            } else {
                originz = 0;
            }

            if (originz < minThreshold) {
                heightMap.at<float>(y, x) = 0;
            } else {
                float newz = (originz - minThreshold) / (maxZ - minThreshold);
                heightMap.at<float>(y, x) = newz;
            }
        }
        break;
    }
    case 2: {
        std::vector<double> zVec;
        zVec.reserve(mObjectX.size());
        for (int i = 0; i < mObjectX.size() - 1; i++) {
            zVec.push_back(z(i));
        }
        zVec.push_back(0);
        std::vector<double> zVec2(zVec.begin(), zVec.end());
        for (int i = 0; i < mObjectX.size(); i++) {
            int x = mObjectX[i];
            int y = mObjectY[i];
            double originz;
            if (i < midIndex) {
                originz = z(i);
            } else if (i > midIndex) {
                originz = z(i - 1);
            } else {
                originz = 0;
            }
            heightMap.at<float>(y, x) = 1 / (1 + exp(parameter * originz));
        }
        break;
    }
    default:
        break;
    }
    // output the height map to matrix;
    return heightMap;
}

}
