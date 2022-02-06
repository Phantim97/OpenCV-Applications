#include <opencv2/text.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

#include "env_util.h"

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