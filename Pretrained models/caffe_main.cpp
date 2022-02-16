#include <fstream>
#include <sstream>
#include <iostream>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include "matplotlibcpp.h"
#include "displayImages.h"

#include "env_util.h"

void caffe_main_inference()
{
	std::string protoFile = MODEL_PATH + "deploy.prototxt";
	std::string weightFile = MODEL_PATH + "face_orientation_iter_200.caffemodel";
	std::string filename = DATA_PATH + "images/left.jpg";
	cv::Mat frame = imread(filename);
	std::vector<std::string> classes;
	std::string classFile = "class2label.txt";
	std::ifstream ifs(classFile.c_str());
	std::string line;

	while (std::getline(ifs, line))
	{
		classes.push_back(line);
	}

	float scale = 1.0;
	int inHeight = 224;
	int inWidth = 224;
	bool swapRB = false;
	cv::Scalar mean = Scalar(104, 117, 123);

	//! [Read and initialize network]
	cv::dnn::Net net = cv::readNetFromCaffe(protoFile, weightFile);

	// Process frames.
	cv::Mat blob;
	//! [Create a 4D blob from a frame]
	cv::blobFromImage(frame, blob, scale, cv::Size(inWidth, inHeight), mean, swapRB, false);

	//! [Set input blob]
	net.setInput(blob);

	//! [Make forward pass]
	cv::Mat prob = net.forward();

	//! [Get a class with a highest score]
	cv::Point classIdPoint;
	double confidence;
	cv::minMaxLoc(prob.reshape(1, 1), 0, &confidence, 0, &classIdPoint);
	int classId = classIdPoint.x;

	// Print predicted class.
	std::string label = format("Predicted Class : %s, confidence : %.3f", (classes[classId].c_str()), confidence);
	std::cout << label << '\n';
}