#pragma once
#include <opencv2/opencv.hpp>
void warp_triangle(cv::Mat& img1, cv::Mat& img2, std::vector<cv::Point2f> tri1, std::vector<cv::Point2f> tri2);