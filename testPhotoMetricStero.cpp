#include "PhotometricStero.h"

void drawAppleNormal();
void drawAppleAlbedo();
void drawElephantNormal();
void drawElephantAlbedo();
void drawPearNormal();
void drawPearAlbedo();
void drawAppleNormalWithAlbedo();
void drawElephantNormalwithAlbedo();
void drawPearNormalWithAlbedo();
void getAppleHeight();
void getElephantHeight();
void getPearHeight();

int main() {
    //drawAppleNormal();
    //drawElephantNormal();
    //drawPearNormal();
    //drawAppleAlbedo();
    //drawElephantAlbedo();
    //drawPearAlbedo();
    //drawAppleNormalWithAlbedo();
    //drawElephantNormalwithAlbedo();
    //drawPearNormalWithAlbedo();
    //getAppleHeight();
    getElephantHeight();
    //getPearHeight();
}

void drawAppleNormal() {
    phoSte::photometryStero A(21, 2, 22,
                              "C:\\xinyuan Gui\\pcv_Assignment_1\\Assignment_1\\Apple\\",
                              "mask_dir_1.png", "mask_dir_2.png", "mask_I.png", "applemask.png");
    A.readImage();
    A.getLightInformation(1, 2);
    A.getPixelNormAndAlbedo(3);
    cv::Mat result = A.outputNormalImage(3);
    cv::imshow("apple normal image", result);
    cv::imwrite("C:\\xinyuan Gui\\pcv_Assignment_1\\Assignment_1\\Apple\\normal.jpg", result);
    cv::waitKey(0);
}

void drawAppleAlbedo() {
    phoSte::photometryStero A(21, 2, 22,
                              "C:\\xinyuan Gui\\pcv_Assignment_1\\Assignment_1\\Apple\\",
                              "mask_dir_1.png", "mask_dir_2.png", "mask_I.png", "applemask.png");
    A.readImage();
    A.getLightInformation(1, 2);
    A.getPixelNormAndAlbedo(3);
    cv::Mat result = A.outputAlbedoImage(3);
    cv::imshow("apple albedo image", result);
    cv::waitKey(0);
}

void drawAppleNormalWithAlbedo() {
    phoSte::photometryStero A(21, 2, 22,
                              "C:\\xinyuan Gui\\pcv_Assignment_1\\Assignment_1\\Apple\\",
                              "mask_dir_1.png", "mask_dir_2.png", "mask_I.png", "applemask.png");
    A.readImage();
    A.getLightInformation(1, 2);
    A.getPixelNormAndAlbedo(3);
    cv::Mat result = A.outputNormalWithAlbedo(3);
    cv::imshow("apple albedo image", result);
    cv::waitKey(0);
}

void drawElephantNormal() {
    phoSte::photometryStero A(21, 1, 21,
                              "C:\\xinyuan Gui\\pcv_Assignment_1\\Assignment_1\\Elephant\\",
                              "mask_dir_1.png", "mask_dir_2.png", "mask_I.png", "mask.png");
    A.readImage();
    A.getLightInformation(1, 2);
    A.getPixelNormAndAlbedo(3);
    cv::Mat result = A.outputNormalImage(3);
    cv::imshow("elephant normal image", result);
    cv::imwrite("C:\\xinyuan Gui\\pcv_Assignment_1\\Assignment_1\\Elephant\\elephantNormal.jpg", result);
    cv::waitKey(0);
}

void drawElephantNormalwithAlbedo() {
    phoSte::photometryStero A(21, 1, 21,
                              "C:\\xinyuan Gui\\pcv_Assignment_1\\Assignment_1\\Elephant\\",
                              "mask_dir_1.png", "mask_dir_2.png", "mask_I.png", "mask.png");
    A.readImage();
    A.getLightInformation(1, 2);
    A.getPixelNormAndAlbedo(3);
    cv::Mat result = A.outputNormalWithAlbedo(3);
    cv::imshow("elephant normal image", result);
    cv::waitKey(0);
}

void drawElephantAlbedo() {
    phoSte::photometryStero A(21, 1, 21,
                              "C:\\xinyuan Gui\\pcv_Assignment_1\\Assignment_1\\Elephant\\",
                              "mask_dir_1.png", "mask_dir_2.png", "mask_I.png", "mask.png");
    A.readImage();
    A.getLightInformation(1, 2);
    A.getPixelNormAndAlbedo(3);
    cv::Mat result = A.outputAlbedoImage(3);
    cv::imshow("elephant albedo image", result);
    cv::waitKey(0);
}

void drawPearNormal() {
    phoSte::photometryStero A(21, 1, 21,
                              "C:\\xinyuan Gui\\pcv_Assignment_1\\Assignment_1\\Pear\\",
                              "mask_dir_1.png", "mask_dir_2.png", "mask_I.png", "pearmask.png");
    A.readImage();
    A.getLightInformation(1, 2);
    A.getPixelNormAndAlbedo(3);
    cv::Mat result = A.outputNormalImage(3);
    cv::imshow("pear normal image", result);
    cv::imwrite("C:\\xinyuan Gui\\pcv_Assignment_1\\Assignment_1\\Pear\\pearNormal.jpg", result);
    cv::waitKey(0);
}

void drawPearAlbedo() {
    phoSte::photometryStero A(21, 1, 21,
                              "C:\\xinyuan Gui\\pcv_Assignment_1\\Assignment_1\\Pear\\",
                              "mask_dir_1.png", "mask_dir_2.png", "mask_I.png", "pearmask.png");
    A.readImage();
    A.getLightInformation(1, 2);
    A.getPixelNormAndAlbedo(3);
    cv::Mat result = A.outputAlbedoImage(3);
    cv::imshow("pear albedo image", result);
    cv::waitKey(0);
}

void drawPearNormalWithAlbedo() {
    phoSte::photometryStero A(21, 1, 21,
                              "C:\\xinyuan Gui\\pcv_Assignment_1\\Assignment_1\\Pear\\",
                              "mask_dir_1.png", "mask_dir_2.png", "mask_I.png", "pearmask.png");
    A.readImage();
    A.getLightInformation(1, 2);
    A.getPixelNormAndAlbedo(3);
    cv::Mat result = A.outputNormalWithAlbedo(3);
    cv::imshow("pear albedo image", result);
    cv::waitKey(0);
}

void getAppleHeight() {
    std::string path = "C:\\xinyuan Gui\\pcv_Assignment_1\\Assignment_1\\Apple\\";
    phoSte::photometryStero A(21, 2, 22,
                              path, "mask_dir_1.png", "mask_dir_2.png", "mask_I.png", "applemask.png");
    A.readImage();
    A.getLightInformation(1, 2);
    A.getPixelNormAndAlbedo(3);
    cv::Mat heightMap = A.getHeightMap(2, -0.1);
    std::string fileName = "appleDepth.png";
    cv::imwrite(path + fileName, heightMap);
    cv::imshow(fileName, heightMap);
    cv::waitKey(0);
}

void getElephantHeight() {
    std::string path = "C:\\xinyuan Gui\\pcv_Assignment_1\\Assignment_1\\Elephant\\";
    phoSte::photometryStero A(21, 1, 21,
                              path, "mask_dir_1.png", "mask_dir_2.png", "mask_I.png", "mask.png");
    A.readImage();
    A.getLightInformation(1, 2);
    A.getPixelNormAndAlbedo(3);
    cv::Mat heightMap = A.getHeightMap(2, -0.1);
    std::string fileName = "elephantHeight.png";
    cv::imwrite(path + fileName, heightMap);
    cv::imshow(fileName, heightMap);
    cv::waitKey(0);
}

void getPearHeight() {
    std::string path = "C:\\xinyuan Gui\\pcv_Assignment_1\\Assignment_1\\Pear\\";
    phoSte::photometryStero A(21, 1, 21,
                              path, "mask_dir_1.png", "mask_dir_2.png", "mask_I.png", "pearmask.png");
    A.readImage();
    A.getLightInformation(1, 2);
    A.getPixelNormAndAlbedo(3);
    cv::Mat heightMap = A.getHeightMap(2, -0.1);
    std::string fileName = "pearHeight.png";
    cv::imwrite(path + fileName, heightMap);
    cv::imshow(fileName, heightMap);
    cv::waitKey(0);
}
