#include <fstream>
#include <sstream>
#include <iostream>

#include <string>
#include <vector>

#include "env_util.h"

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

// Initialize the parameters
//float objectnessThreshold = 0.5; // Objectness threshold

std::vector<std::string> classes;

// Get the names of the output layers
std::vector<std::string> get_outputs_names(const cv::dnn::Net& net)
{
	static std::vector<std::string> names;
	if (names.empty())
	{
		//Get the indices of the output layers, i.e. the layers with unconnected outputs
		const std::vector<int> out_layers = net.getUnconnectedOutLayers();

		//get the names of all the layers in the network
		const std::vector<std::string> layers_names = net.getLayerNames();

		// Get the names of the output layers in names
		names.resize(out_layers.size());
		for (size_t i = 0; i < out_layers.size(); ++i)
		{
			names[i] = layers_names[out_layers[i] - 1];
		}
	}

	return names;
}

// Draw the predicted bounding box
void draw_pred(const int class_id, const float conf, const int left, const int top, const int right, const int bottom, cv::Mat& frame)
{
	//Draw a rectangle displaying the bounding box
	cv::rectangle(frame, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(255, 178, 50), 3);

	//Get the label for the class name and its confidence
	std::string label = cv::format("%.2f", conf);

	if (!classes.empty())
	{
		CV_Assert(class_id < static_cast<int>(classes.size()));
		label = classes[class_id] + ":" + label;
	}

	//Display the label at the top of the bounding box
	int base_line;
	const cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &base_line);
	const int updated_top = cv::max(top, label_size.height);
	cv::rectangle(frame, cv::Point(left, updated_top - round(1.5 * label_size.height)), cv::Point(left + round(1.5 * label_size.width), updated_top + base_line), cv::Scalar(255, 255, 255), cv::FILLED);
	cv::putText(frame, label, cv::Point(left, updated_top), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 0, 0), 1);
}

// Remove the bounding boxes with low confidence using non-maxima suppression
void postprocess(cv::Mat& frame, const std::vector<cv::Mat>& outs)
{
	constexpr float conf_threshold = 0.5; // Confidence threshold
	constexpr float nms_threshold = 0.4;  // Non-maximum suppression threshold
	std::vector<int> class_ids;
	std::vector<float> confidences;
	std::vector<cv::Rect> boxes;

	for (size_t i = 0; i < outs.size(); ++i)
	{
		// Scan through all the bounding boxes output from the network and keep only the
		// ones with high confidence scores. Assign the box's class label as the class
		// with the highest score for the box.
		float* data = reinterpret_cast<float*>(outs[i].data);
		for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
		{
			cv::Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
			cv::Point class_id_point;
			double confidence;
			// Get the value and location of the maximum score
			cv::minMaxLoc(scores, nullptr, &confidence, nullptr, &class_id_point);
			if (confidence > conf_threshold)
			{
				const int center_x = static_cast<int>(data[0] * frame.cols);
				const int center_y = static_cast<int>(data[1] * frame.rows);
				const int width = static_cast<int>(data[2] * frame.cols);
				const int height = static_cast<int>(data[3] * frame.rows);
				const int left = center_x - width / 2;
				const int top = center_y - height / 2;

				class_ids.push_back(class_id_point.x);
				confidences.push_back(static_cast<float>(confidence));
				boxes.emplace_back(left, top, width, height);
			}
		}
	}

	// Perform non maximum suppression to eliminate redundant overlapping boxes with
	// lower confidences
	std::vector<int> indices;
	cv::dnn::NMSBoxes(boxes, confidences, conf_threshold, nms_threshold, indices);
	for (size_t i = 0; i < indices.size(); ++i)
	{
		const int idx = indices[i];
		const cv::Rect box = boxes[idx];
		draw_pred(class_ids[idx], confidences[idx], box.x, box.y,
			box.x + box.width, box.y + box.height, frame);
	}
}

void yolo_main()
{
	constexpr int inp_width = 416;  // Width of network's input image
	constexpr int inp_height = 416; // Height of network's input image
	// Load names of classes
	std::string classes_file = util::get_model_path() + "coco.names";
	std::ifstream ifs(classes_file.c_str());
	std::string line;
	while (std::getline(ifs, line))
	{
		classes.push_back(line);
	}

	// Give the configuration and weight files for the model
	std::string model_configuration = util::get_model_path() + "yolov3.cfg";
	std::string model_weights = util::get_model_path() + "yolov3.weights";

	// Load the network
	cv::dnn::Net net = cv::dnn::readNetFromDarknet(model_configuration, model_weights);

	std::string image_path = util::get_data_path() + "images/bird.jpg";
	cv::Mat frame = cv::imread(image_path);
	// Create a 4D blob from a frame.
	cv::Mat blob;
	cv::dnn::blobFromImage(frame, blob, 1 / 255.0, cv::Size(inp_width, inp_height), cv::Scalar(0, 0, 0), true, false);
	//Sets the input to the network
	net.setInput(blob);
	// Runs the forward pass to get output of the output layers
	std::vector<cv::Mat> outs;
	net.forward(outs, get_outputs_names(net));

	// Remove the bounding boxes with low confidence
	postprocess(frame, outs);
	// Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
	std::vector<double> layers_times;
	double freq = cv::getTickFrequency() / 1000;
	double t = net.getPerfProfile(layers_times) / freq;
	std::string label = cv::format("Inference time for a frame : %.2f ms", t);
	putText(frame, label, cv::Point(0, 15), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255));

	cv::imshow("Frame", frame);
	std::cout << "Label: " << label << '\n';
	cv::waitKey(5000);
}