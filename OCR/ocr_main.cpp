#include <opencv2/text.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

#include "env_util.h"



void decode(const cv::Mat& scores, const cv::Mat& geometry, const float score_thresh,
    std::vector<cv::RotatedRect>& detections, std::vector<float>& confidences)
{
    detections.clear();
    CV_Assert(scores.dims == 4); CV_Assert(geometry.dims == 4); CV_Assert(scores.size[0] == 1);
    CV_Assert(geometry.size[0] == 1); CV_Assert(scores.size[1] == 1); CV_Assert(geometry.size[1] == 5);
    CV_Assert(scores.size[2] == geometry.size[2]); CV_Assert(scores.size[3] == geometry.size[3]);

    const int height = scores.size[2];
    const int width = scores.size[3];
    for (int y = 0; y < height; ++y)
    {
        const float* scores_data = scores.ptr<float>(0, 0, y);
        const float* x0_data = geometry.ptr<float>(0, 0, y);
        const float* x1_data = geometry.ptr<float>(0, 1, y);
        const float* x2_data = geometry.ptr<float>(0, 2, y);
        const float* x3_data = geometry.ptr<float>(0, 3, y);
        const float* angles_data = geometry.ptr<float>(0, 4, y);

        for (int x = 0; x < width; ++x)
        {
            float score = scores_data[x];
            if (score < score_thresh)
            {
                continue;
            }

            const float offset_x = x * 4.0f;
            // Decode a prediction.
            // Multiple by 4 because feature maps are 4 time less than input image.
            const float offset_y = y * 4.0f;
            const float angle = angles_data[x];
            const float cos_a = std::cos(angle);
            const float sin_a = std::sin(angle);
            const float h = x0_data[x] + x2_data[x];
            const float w = x1_data[x] + x3_data[x];

            cv::Point2f offset(offset_x + cos_a * x1_data[x] + sin_a * x2_data[x],
            offset_y - sin_a * x1_data[x] + cos_a * x2_data[x]);
            cv::Point2f p1 = cv::Point2f(-sin_a * h, -cos_a * h) + offset;
            cv::Point2f p3 = cv::Point2f(-cos_a * w, sin_a * w) + offset;
            cv::RotatedRect r(0.5f * (p1 + p3), cv::Size2f(w, h), -angle * 180.0f / (float)CV_PI);
            detections.push_back(r);
            confidences.push_back(score);
        }
    }
}

const std::string model = util::get_model_path() + "models/frozen_east_text_detection.pb";
cv::dnn::Net net = cv::dnn::readNet(model);
int inp_width = 640;
int inp_height = 640;
float conf_threshold = 0.7;
float nms_threshold = 0.4;

cv::Mat east_text_detection(const std::string& image_name)
{
    cv::Mat image = cv::imread(image_name);
    //imageOut = image.copy()
    // Get Height and width of the image.
    int height_ = image.rows;
    int width_ = image.cols;

    // Get ratio by which the image is resized for using in the network
    float rW = (float)width_ / inp_width;
    float rH = (float)height_ / inp_height;

    // Create a blob and assign the image to the blob
    cv::Mat blob;
    cv::dnn::blobFromImage(image, blob, 1.0, cv::Size(inp_width, inp_height), cv::Scalar(123.68, 116.78, 103.94), true, false);
    net.setInput(blob);

    // Get the output using by passing the image through the network
    std::vector<cv::Mat> output;
    std::vector<std::string> outputLayers(2);
    outputLayers[0] = "feature_fusion/Conv_7/Sigmoid";
    outputLayers[1] = "feature_fusion/concat_3";
    net.forward(output, outputLayers);
    cv::Mat scores = output[0];
    cv::Mat geometry = output[1];

    // Get rotated rectangles using the decode function described above
    std::vector<cv::RotatedRect> boxes;
    std::vector<float> confidences;
    decode(scores, geometry, conf_threshold, boxes, confidences);

    // Apply non-maximum suppression procedure
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, conf_threshold, nms_threshold, indices);

    // Render detections
    for (size_t i = 0; i < indices.size(); ++i)
    {
	    cv::RotatedRect& box = boxes[indices[i]];

	    cv::Point2f vertices[4];
        box.points(vertices);

        for (int j = 0; j < 4; ++j)
        {
            vertices[j].x *= rW;
            vertices[j].y *= rH;
        }
        for (int j = 0; j < 4; ++j)
        {
			cv::line(image, vertices[j], vertices[(j + 1) % 4], cv::Scalar(0, 255, 0), 4, cv::LINE_AA);
        }
    }
    return image;
}

void text_detect_main()
{
    std::string model = util::get_model_path() +  "models/frozen_east_text_detection.pb";
    // Load network
    cv::dnn::Net net = cv::dnn::readNet(model);
    std::vector<std::string> outputLayers(2);
    outputLayers[0] = "feature_fusion/Conv_7/Sigmoid";
    outputLayers[1] = "feature_fusion/concat_3";
  
}

void ocr_driver()
{
	const std::string tessdata = util::get_tessdata_path();
	const cv::Mat img = cv::imread(util::get_data_path() + "/images/ocr/recipt_big.png");

	const cv::Ptr<cv::text::OCRTesseract> ocr = cv::text::OCRTesseract::create(tessdata.c_str(), "eng", "0123456789abcdefghiklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ-\n");
	const std::string data = ocr->run(img, -1);

	std::cout << "OCR Results: " << '\n';
	std::cout << data << '\n';
	std::cout << "Data size: " << data.size() << '\n';
}