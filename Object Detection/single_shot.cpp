#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <stdlib.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>

#include "env_util.h"

constexpr size_t in_width = 300;
constexpr size_t in_height = 300;
constexpr double in_scale_factor = 1.0 / 127.5;
constexpr float confidence_threshold = 0.7;
const cv::Scalar mean_val(127.5, 127.5, 127.5);
std::vector<std::string> classes;

cv::Mat detect_objects(cv::dnn::Net net, const cv::Mat& frame)
{
	const cv::Mat input_blob = cv::dnn::blobFromImage(frame, in_scale_factor, cv::Size(in_width, in_height),
		mean_val, true, false);

	net.setInput(input_blob);

	cv::Mat detection = net.forward("detection_out");
	cv::Mat detection_mat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());

	return detection_mat;
}

void display_text(cv::Mat& img, const std::string& text, const int x, const int y)
{
	// Get text size
	int base_line;
	const cv::Size text_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.7, 1, &base_line);
	// Use text size to create a black rectangle
	cv::rectangle(img, cv::Point(x, y - text_size.height - base_line), cv::Point(x + text_size.width, y + base_line),
		cv::Scalar(0, 0, 0), -1);
	// Display text inside the rectangle
	cv::putText(img, text, cv::Point(x, y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 1, cv::LINE_AA);
}

void display_objects(cv::Mat& img, cv::Mat objects, const float threshold = 0.25)
{
	// For every detected object
	for (int i = 0; i < objects.rows; i++) 
	{
		const int class_id = objects.at<float>(i, 1);
		const float score = objects.at<float>(i, 2);

		// Recover original cordinates from normalized coordinates
		const int x = static_cast<int>(objects.at<float>(i, 3) * img.cols);
		const int y = static_cast<int>(objects.at<float>(i, 4) * img.rows);
		const int w = static_cast<int>(objects.at<float>(i, 5) * img.cols - x);
		const int h = static_cast<int>(objects.at<float>(i, 6) * img.rows - y);

		// Check if the detection is of good quality
		if (score > threshold)
		{
			display_text(img, classes[class_id], x, y);
			rectangle(img, cv::Point(x, y), cv::Point(x + w, y + h), cv::Scalar(255, 255, 255), 2);
		}
	}
}

void single_shot_main()
{
	const std::string config_file = util::get_model_path() + "ssd_mobilenet_v2_coco_2018_03_29.pbtxt";
	const std::string model_file = util::get_model_path() + "ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb";
	const std::string class_file = util::get_model_path() + "coco_class_labels.txt";

	cv::dnn::Net net = cv::dnn::readNetFromTensorflow(model_file, config_file);

	std::ifstream ifs(class_file.c_str());
	std::string line;

	while (std::getline(ifs, line))
	{
		classes.push_back(line);
	}

	cv::Mat img;
	cv::Mat objects;
	img = cv::imread(util::get_data_path() + "images/street.jpg");
	objects = detect_objects(net, img);
	display_objects(img, objects);
	cv::imshow("Objects (Single Shot)", img);
	cv::waitKey(5000);
}