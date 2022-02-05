#include <opencv2/text.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

#include "env_util.h"

void ocr_driver()
{
	const std::string tessdata = util::get_tessdata_path();
	cv::Mat img = cv::imread(util::get_data_path() + "/images/ocr/recipt.png");
	cv::Ptr<cv::text::OCRTesseract> ocr = cv::text::OCRTesseract::create(tessdata.c_str());

	std::string data = ocr->run(img, 0);

	std::cout << "OCR Results: " << '\n';
	std::cout << data << '\n';
}