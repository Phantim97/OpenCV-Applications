#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>
#include <dlib/image_io.h>
#include <opencv2/imgproc.hpp> // If you are using OpenCV 3
#include <iostream>
#include <string>
#include <dlib/opencv.h>

#include "env_util.h"
#include "ml_util.h"

void lipstick()
{
	int selected_points[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 31, 32, 33, 34, 35, 55, 56, 57, 58, 59 };
	std::vector<int> selected_index(selected_points, selected_points + sizeof(selected_points) / sizeof(int));

	// Load face detection and pose estimation models.
	std::string model_path = util::get_model_path() + "dlib_models/shape_predictor_68_face_landmarks.dat";
	dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
	dlib::shape_predictor predictor;
	dlib::deserialize(model_path) >> predictor;

	std::string overlay_file = util::get_data_path() + "images/filters/makeup/lipstick.png";
	std::string image_file = util::get_data_path() + "images/people/girl-no-makeup.jpg";

	cv::Mat result_eyes;
	cv::Mat result_lips;

	cv::imshow("Eye change", result_eyes);
	cv::waitKey(5000);
}

void eye_change()
{
	int selected_points[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 31, 32, 33, 34, 35, 55, 56, 57, 58, 59 };
	std::vector<int> selected_index(selected_points, selected_points + sizeof(selected_points) / sizeof(int));

	// Load face detection and pose estimation models.
	std::string model_path = util::get_model_path() + "dlib_models/shape_predictor_68_face_landmarks.dat";
	dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
	dlib::shape_predictor predictor;
	dlib::deserialize(model_path) >> predictor;

	const std::string face_file_name = "girl-no-makeup.jpg";

	std::string overlay_file = util::get_data_path() + "images/filters/makeup/eyes.png";
	std::string image_file = util::get_data_path() + "images/people/" + face_file_name;

	cv::Mat filter;
	cv::Mat face = cv::imread(image_file, cv::IMREAD_UNCHANGED);

	face_landmark_writer(face, "landmark_check.jpg");

	std::vector<cv::Point2f> landmark_points = get_landmark_point_vector(face, "images/people", face_file_name, detector, predictor);

	cv::Mat result_eyes = face.clone();
	cv::Mat result_lips;

	cv::imshow("Eye change", result_eyes);
	cv::waitKey(5000);
}

void virtual_makeup()
{
	eye_change();

}