#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include "env_util.h"

void similarityTransform(const std::vector<cv::Point2f>& vector, const std::vector<cv::Point2f>& points, const cv::Mat& mat)
{
	//Internal course function
	return;
}

void normalizeImagesAndLandmarks(const cv::Size out_size, const cv::Mat& img_in, cv::Mat& img_out, const std::vector<cv::Point2f>& points_in, std::vector<cv::Point2f>& points_out)
{
	const int h = out_size.height;
	const int w = out_size.width;

	std::vector<cv::Point2f> eyecorner_src;
	if (points_in.size() == 68)
	{
		// Get the locations of the left corner of left eye
		eyecorner_src.push_back(points_in[36]);
		// Get the locations of the right corner of right eye
		eyecorner_src.push_back(points_in[45]);
	}
	else if (points_in.size() == 5)
	{
		// Get the locations of the left corner of left eye
		eyecorner_src.push_back(points_in[2]);
		// Get the locations of the right corner of right eye
		eyecorner_src.push_back(points_in[0]);
	}

	std::vector<cv::Point2f> eyecorner_dst;
	// Location of the left corner of left eye in normalized image.
	eyecorner_dst.emplace_back(0.3 * w, h / 3);
	// Location of the right corner of right eye in normalized image.
	eyecorner_dst.emplace_back(0.7 * w, h / 3);

	// Calculate similarity transform
	cv::Mat tform;
	similarityTransform(eyecorner_src, eyecorner_dst, tform);

	// Apply similarity transform to input image
	img_out = cv::Mat::zeros(h, w, CV_32FC3);
	cv::warpAffine(img_in, img_out, tform, img_out.size());

	// Apply similarity transform to landmarks
	cv::transform(points_in, points_out, tform);
}

std::vector<cv::Point2f> getLandmarks(const dlib::frontal_face_detector& face_detector, const dlib::shape_predictor& shape_predictor, const cv::Mat& mat)
{
	std::cout << "Internal course function just here for compiling purpose\n";
	return std::vector<cv::Point2f>();
}

void face_alignment_main()
{
	// Get the face detector
	dlib::frontal_face_detector faceDetector = dlib::get_frontal_face_detector();
	// The landmark detector is implemented in the shape_predictor class
	dlib::shape_predictor landmarkDetector;
	std::string PREDICTOR_PATH = util::get_model_path() + "models/shape_predictor_5_face_landmarks.dat";
	// Load the landmark model
	dlib::deserialize(PREDICTOR_PATH) >> landmarkDetector;
	//Read image
	cv::Mat img = cv::imread(util::get_data_path() + "images/face2.png");
	// Detect landmarks
	std::vector<cv::Point2f> points = getLandmarks(faceDetector, landmarkDetector, img);

	// Convert image to floating point in the range 0 to 1
	img.convertTo(img, CV_32FC3, 1 / 255.0);
	// Dimensions of output image
	cv::Size size(600, 600);
	// Variables for storing normalized image
	cv::Mat imNorm;
	// Normalize image to output coordinates.
	normalizeImagesAndLandmarks(size, img, imNorm, points, points);
	imNorm.convertTo(imNorm, CV_8UC3, 255);
}