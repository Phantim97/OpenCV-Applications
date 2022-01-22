#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>
#include <dlib/image_io.h>
#include <dlib/gui_widgets.h>
#include <opencv2/imgproc.hpp> // If you are using OpenCV 3
#include <iostream>
#include <fstream>
#include <string>
#include <dlib/opencv.h>
#include <stdlib.h>

#include "env_util.h"
#include "ml_util.h"

#define FACE_DOWNSAMPLE_RATIO 1

static cv::Mat correct_colors(const cv::Mat& im1, cv::Mat im2, std::vector<cv::Point2f> points2)// lower number --> output is closer to webcam and vice-versa
{
	const cv::Point2f dist_between_eyes = points2[38] - points2[43];
	const float distance = cv::norm(dist_between_eyes);

	//using heuristics to calculate the amount of blur
	int blur_amount = static_cast<int>(0.5 * distance);

	if (blur_amount % 2 == 0)
	{
		blur_amount += 1;
	}

	cv::Mat im1_blur = im1.clone();
	cv::Mat im2_blur = im2.clone();

	cv::blur(im1_blur, im1_blur, cv::Size(blur_amount, blur_amount));
	cv::blur(im2_blur, im2_blur, cv::Size(blur_amount, blur_amount));
	// Avoid divide-by-zero errors.

	im2_blur += 2 * (im2_blur <= 1) / 255;
	im1_blur.convertTo(im1_blur, CV_32F);
	im2_blur.convertTo(im2_blur, CV_32F);
	im2.convertTo(im2, CV_32F);

	cv::Mat ret = im2.clone();
	ret = im2.mul(im1_blur).mul(1 / im2_blur);
	cv::threshold(ret, ret, 255, 255, cv::THRESH_TRUNC);
	ret.convertTo(ret, CV_8UC3);

	return ret;
}

// Constrains points to be inside boundary
static void constrain_point(cv::Point2f& p, const cv::Size& sz)
{
	p.x = std::min(std::max(static_cast<double>(p.x), 0.0), static_cast<double>(sz.width - 1));
	p.y = std::min(std::max(static_cast<double>(p.y), 0.0), static_cast<double>(sz.height - 1));
}

void beardify()
{
	int selected_points[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 31, 32, 33, 34, 35, 55, 56, 57, 58, 59 };
	std::vector<int> selected_index(selected_points, selected_points + sizeof(selected_points) / sizeof(int));

	// Load face detection and pose estimation models.
	std::string model_path = util::get_model_path() + "dlib_models/shape_predictor_68_face_landmarks.dat";
	dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
	dlib::shape_predictor predictor;
	dlib::deserialize(model_path) >> predictor;

	std::string overlay_file = util::get_data_path() + "images/filters/beardify/beard1.png";
	std::string image_file = util::get_data_path() + "images/people/jilly.jpg";

	// Read the beard image along with its alpha mask

	cv::Mat beard;
	cv::Mat target_image;
	cv::Mat beard_alpha_mask;
	cv::Mat img_with_mask = cv::imread(overlay_file, cv::IMREAD_UNCHANGED);
	std::vector<cv::Mat> rgba_channels(4);

	// Split into channels
	cv::split(img_with_mask, rgba_channels);

	// Extract the beard image
	std::vector<cv::Mat> bgr_channels;
	bgr_channels.push_back(rgba_channels[0]);
	bgr_channels.push_back(rgba_channels[1]);
	bgr_channels.push_back(rgba_channels[2]);

	cv::merge(bgr_channels, beard);
	beard.convertTo(beard, CV_32F, 1.0 / 255.0);

	// Extract the beard mask
	std::vector<cv::Mat> mask_channels;
	mask_channels.push_back(rgba_channels[3]);
	mask_channels.push_back(rgba_channels[3]);
	mask_channels.push_back(rgba_channels[3]);

	cv::merge(mask_channels, beard_alpha_mask);
	beard_alpha_mask.convertTo(beard_alpha_mask, CV_32FC3);

	//Read points for beard from file
	std::vector<cv::Point2f> feature_points1 = get_saved_points(overlay_file);

	// Calculate Delaunay triangles
	cv::Rect rect = cv::boundingRect(feature_points1);

	std::vector< std::vector<int> > dt;
	calculate_delaunay_triangles(rect, feature_points1, dt);

	float time_detector = static_cast<double>(cv::getTickCount());

	// Get the face image for putting the beard
	target_image = cv::imread(image_file);

	// int height = targetImage.rows;
	// float IMAGE_RESIZE = (float)height/RESIZE_HEIGHT;
	// cv::resize(targetImage, targetImage, cv::Size(), 1.0/IMAGE_RESIZE, 1.0/IMAGE_RESIZE);

	std::vector<cv::Point2f> points2 = get_landmark_point_vector(target_image, "images/people", "tian2.jpg", detector, predictor);
	// std::vector<Point2f> points2 = getLandmarks(detector, predictor, targetImage, (float)FACE_DOWNSAMPLE_RATIO);

	std::vector<cv::Point2f> feature_points2;
	for (int i = 0; i < selected_index.size(); i++)
	{
		feature_points2.push_back(points2[selected_index[i]]);
		constrain_point(feature_points2[i], target_image.size());
	}

	//convert Mat to float data type
	target_image.convertTo(target_image, CV_32F, 1.0 / 255.0);

	//empty warp image
	cv::Mat beard_warped = cv::Mat::zeros(target_image.size(), beard.type());
	cv::Mat beard_alpha_mask_warped = cv::Mat::zeros(target_image.size(), beard_alpha_mask.type());

	// Apply affine transformation to Delaunay triangles
	for (size_t i = 0; i < dt.size(); i++)
	{
		std::vector<cv::Point2f> t1, t2;
		// Get points for img1, targetImage corresponding to the triangles
		for (size_t j = 0; j < 3; j++)
		{
			t1.push_back(feature_points1[dt[i][j]]);
			t2.push_back(feature_points2[dt[i][j]]);
		}

		warp_triangle(beard, beard_warped, t1, t2);
		warp_triangle(beard_alpha_mask, beard_alpha_mask_warped, t1, t2);
	}

	cv::Mat mask1;
	beard_alpha_mask_warped.convertTo(mask1, CV_32FC3, 1.0 / 255.0);
	cv::Mat mask2 = cv::Scalar(1.0, 1.0, 1.0) - mask1;

	cv::Mat temp1 = target_image.mul(mask2);
	cv::Mat temp2 = beard_warped.mul(mask1);
	cv::Mat result = temp1 + temp2;

	cv::imshow("Output", result);
	cv::waitKey(5000);
}