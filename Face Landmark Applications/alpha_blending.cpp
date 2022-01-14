#include <opencv2/opencv.hpp>
#include "env_util.h"

// Alpha blending using multiply and add functions
void blend(const cv::Mat& alpha, const cv::Mat& foreground, const cv::Mat& background, cv::Mat& out_image)
{
	cv::Mat fore;
	cv::Mat back;
	cv::multiply(alpha, foreground, fore);
    cv::multiply(cv::Scalar::all(1.0) - alpha, background, back);
    cv::add(fore, back, out_image);
}

// Alpha Blending using direct pointer access (more optimal)
void alpha_blend_direct_access(const cv::Mat& alpha, const cv::Mat& foreground, const cv::Mat& background, cv::Mat& out_image)
{
	const int number_of_pixels = foreground.rows * foreground.cols * foreground.channels();

    float* fptr = reinterpret_cast<float*>(foreground.data);
    float* bptr = reinterpret_cast<float*>(background.data);
    float* aptr = reinterpret_cast<float*>(alpha.data);
    float* out_image_ptr = reinterpret_cast<float*>(out_image.data);

   
    for (int j = 0; j < number_of_pixels; ++j, out_image_ptr++, fptr++, aptr++, bptr++)
    {
        *out_image_ptr = (*fptr) * (*aptr) + (*bptr) * (1 - *aptr);
    }
}

void alpha_blending()
{
    // Read in the png foreground asset file that contains both rgb and alpha information
    cv::Mat foreground_image = cv::imread(util::get_data_path() + "images/foreGroundAssetLarge.png", -1);
    cv::Mat bgra[4];

    //split png foreground
    split(foreground_image, bgra);

    // Save the foregroung RGB content into a single Mat
    std::vector<cv::Mat> foreground_channels;
    foreground_channels.push_back(bgra[0]);
    foreground_channels.push_back(bgra[1]);
    foreground_channels.push_back(bgra[2]);

    cv::Mat foreground = cv::Mat::zeros(foreground_image.size(), CV_8UC3);
    cv::merge(foreground_channels, foreground);

    // Save the alpha information into a single Mat
    std::vector<cv::Mat> alpha_channels;
    alpha_channels.push_back(bgra[3]);
    alpha_channels.push_back(bgra[3]);
    alpha_channels.push_back(bgra[3]);

    cv::Mat alpha = cv::Mat::zeros(foreground_image.size(), CV_8UC3);
    cv::merge(alpha_channels, alpha);
    cv::Mat copy_with_mask = cv::Mat::zeros(foreground_image.size(), CV_8UC3);
    foreground.copyTo(copy_with_mask, bgra[3]);

    // Read background image
    cv::Mat background = cv::imread(util::get_data_path() + "images/backGroundLarge.jpg");

    // Convert Mat to float data type
    foreground.convertTo(foreground, CV_32FC3);
    background.convertTo(background, CV_32FC3);
    alpha.convertTo(alpha, CV_32FC3, 1.0 / 255); // keeps the alpha values betwen 0 and 1

    // Number of iterations to average the performane over
    int num_of_iterations = 1; //1000;

    // Alpha blending using functions multiply and add
    cv::Mat out_image = cv::Mat::zeros(foreground.size(), foreground.type());
    double t = static_cast<double>(cv::getTickCount());

    for (int i = 0; i < num_of_iterations; i++)
    {
        blend(alpha, foreground, background, out_image);
    }

    t = (static_cast<double>(cv::getTickCount()) - t) / cv::getTickFrequency();
    std::cout << "Time for alpha blending using multiply & add function : " << t * 1000 / num_of_iterations << " milliseconds\n";

	// Alpha blending using direct Mat access with for loop
	out_image = cv::Mat::zeros(foreground.size(), foreground.type());
    t = static_cast<double>(cv::getTickCount());

    for (int i = 0; i < num_of_iterations; i++) 
    {
        alpha_blend_direct_access(alpha, foreground, background, out_image);
    }

    t = (static_cast<double>(cv::getTickCount()) - t) / cv::getTickFrequency();
    std::cout << "Time for alpha blending using alphaBlendDirectAccess : " << t * 1000 / num_of_iterations << " milliseconds\n";

    cv::imshow("Image", out_image/255);
    cv::waitKey(250);
}