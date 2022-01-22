#pragma once
#include <opencv2/opencv.hpp>

void calculate_delaunay_triangles(cv::Rect rect, std::vector<cv::Point2f>& points, std::vector<std::vector<int>>& delaunay_tri);
void landmarks_to_points(dlib::full_object_detection& landmarks, std::vector<cv::Point2f>& points);
std::vector<cv::Point2f> get_landmarks(dlib::frontal_face_detector& face_detector, const dlib::shape_predictor& landmark_detector, const cv::Mat& img, const float FACE_DOWNSAMPLE_RATIO = 1);
bool landmark_file_found(const std::string& file);
std::string remove_extension(const std::string& file);
std::vector<cv::Point2f> get_saved_points(const std::string& points_file_name);
void warp_triangle(cv::Mat& img1, cv::Mat& img2, std::vector<cv::Point2f> tri1, std::vector<cv::Point2f> tri2);
std::vector<cv::Point2f> get_landmark_point_vector(const cv::Mat& img, const std::string& dir, const std::string& filename, dlib::frontal_face_detector fd, const dlib::shape_predictor& pd);