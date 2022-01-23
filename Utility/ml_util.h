#pragma once
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/full_object_detection.h>
#include <dlib/image_processing/shape_predictor.h>
#include <opencv2/opencv.hpp>

void calculate_delaunay_triangles(cv::Rect rect, std::vector<cv::Point2f>& points, std::vector<std::vector<int>>& delaunay_tri);
void landmarks_to_points(dlib::full_object_detection& landmarks, std::vector<cv::Point2f>& points);
std::vector<cv::Point2f> get_landmarks(dlib::frontal_face_detector& face_detector, const dlib::shape_predictor& landmark_detector, const cv::Mat& img, float face_downsample_ratio);
bool landmark_file_found(const std::string& file);

std::string remove_extension(const std::string& file);
std::vector<cv::Point2f> get_saved_points(const std::string& points_file_name);
void warp_triangle(cv::Mat& img1, cv::Mat& img2, std::vector<cv::Point2f> tri1, std::vector<cv::Point2f> tri2);
std::vector<cv::Point2f> get_landmark_point_vector(const cv::Mat& img, const std::string& dir, const std::string& filename, dlib::frontal_face_detector fd, const dlib::shape_predictor& pd);

void constrain_point(cv::Point2f& p, const cv::Size& sz);
void warp_image(cv::Mat& img_in, cv::Mat& img_out, const std::vector<cv::Point2f>& points_in, const std::vector<cv::Point2f>& points_out, const std::vector<std::vector<int>>& delaunay_tri);
void get_eight_boundary_points(const cv::Size& size, std::vector<cv::Point2f>& boundary_pts);
cv::Mat correct_colors(const cv::Mat& im1, cv::Mat im2, const std::vector<cv::Point2f>& points2);// lower number --> output is closer to webcam and vice-versa

enum class MlsMode
{
	FAST,
	SINGLE
};

void mls_warp_image(cv::Mat& src, std::vector<cv::Point2f>& spts, cv::Mat& dst, std::vector<cv::Point2f>& dpts, MlsMode mode);

//Assumes using people directory
void face_landmark_writer(const cv::Mat& src, const std::string& res_file);