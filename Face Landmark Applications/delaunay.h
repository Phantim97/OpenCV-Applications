#pragma once
#include <opencv2/opencv.hpp>
void calculate_delaunay_triangles(cv::Rect rect, std::vector<cv::Point2f>& points, std::vector<std::vector<int>>& delaunay_tri);