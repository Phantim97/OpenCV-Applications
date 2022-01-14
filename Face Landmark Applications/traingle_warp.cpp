#include <opencv2/opencv.hpp>
#include <iostream>

#include "env_util.h"

// Warps and alpha blends triangular regions from img1 and img2 to img
void warp_triangle(cv::Mat& img1, cv::Mat& img2, std::vector<cv::Point2f> tri1, std::vector<cv::Point2f> tri2)
{
    // Find bounding rectangle for each triangle
    const cv::Rect r1 = cv::boundingRect(tri1);
    cv::Rect r2 = cv::boundingRect(tri2);

    // Crop the input image to the bounding box of input triangle
    cv::Mat img1_cropped;
    img1(r1).copyTo(img1_cropped);

    // Once the bounding boxes are cropped, 
    // the triangle coordinates need to be 
    // adjusted by an offset to reflect the 
    // fact that they are now in a cropped image. 
    // Offset points by left top corner of the respective rectangles
	std::vector<cv::Point2f> tri1_cropped, tri2_cropped;
    std::vector<cv::Point> tri2_cropped_int;

	for (int i = 0; i < 3; i++)
    {
        tri1_cropped.emplace_back(tri1[i].x - r1.x, tri1[i].y - r1.y);
        tri2_cropped.emplace_back(tri2[i].x - r2.x, tri2[i].y - r2.y);

        // fillConvexPoly needs a vector of Point and not Point2f
        tri2_cropped_int.emplace_back(tri2[i].x - r2.x, tri2[i].y - r2.y);
    }

    // Given a pair of triangles, find the affine transform.
    const cv::Mat warp_mat = getAffineTransform(tri1_cropped, tri2_cropped);

    // Apply the Affine Transform just found to the src image
    cv::Mat img2_cropped = cv::Mat::zeros(r2.height, r2.width, img1_cropped.type());
    cv::warpAffine(img1_cropped, img2_cropped, warp_mat, img2_cropped.size(), cv::INTER_LINEAR, cv::BORDER_REFLECT_101);

    // We are interested in the pixels inside 
    // the triangle and not the entire bounding box. 

    // So we create a triangular mask using fillConvexPoly.
    // This mask has values 1 ( in all three channels ) 
    // inside the triangle and 0 outside.   
    // Get mask by filling triangle
    cv::Mat mask = cv::Mat::zeros(r2.height, r2.width, CV_32FC3);
    cv::fillConvexPoly(mask, tri2_cropped_int, cv::Scalar(1.0, 1.0, 1.0), 16, 0);

    // Copy triangular region of the rectangular patch to the output image
    cv::multiply(img2_cropped, mask, img2_cropped);
    cv::multiply(img2(r2), cv::Scalar(1.0, 1.0, 1.0) - mask, img2(r2));
    img2(r2) = img2(r2) + img2_cropped;
}

void warp_main()
{
    // Read input image and convert to float
    cv::Mat img_in = cv::imread(util::get_data_path() + "images/kingfisher.jpg");
    // Convert to floating point image in the range 0 to 1.
    img_in.convertTo(img_in, CV_32FC3, 1 / 255.0);
    // Create white output image the same size and type of input image
    cv::Mat img_out = cv::Mat::ones(img_in.size(), img_in.type());
    img_out = cv::Scalar(1.0, 1.0, 1.0);

    // Input triangle
    std::vector <cv::Point2f> tri_in;
    tri_in.emplace_back(360, 50);
    tri_in.emplace_back(60, 100);
    tri_in.emplace_back(300, 400);

    // Output triangle
    std::vector <cv::Point2f> tri_out;
    tri_out.emplace_back(400, 200);
    tri_out.emplace_back(160, 270);
    tri_out.emplace_back(400, 400);

    // Warp all pixels inside input triangle to output triangle
    warp_triangle(img_in, img_out, tri_in, tri_out);

    // Draw triangle on the input and output image.
	// Convert back to uint because OpenCV antialiasing
	// does not work on image of type CV_32FC3

    img_in.convertTo(img_in, CV_8UC3, 255.0);
    img_out.convertTo(img_out, CV_8UC3, 255.0);

    // Draw triangle using this color
    const cv::Scalar color = cv::Scalar(255, 150, 0);

    // cv::polylines needs vector of type Point and not Point2f
    std::vector<cv::Point> tri_in_int;
    std::vector<cv::Point> tri_out_int;

    for (int i = 0; i < 3; i++)
    {
        tri_in_int.emplace_back(tri_in[i].x, tri_in[i].y);
        tri_out_int.emplace_back(tri_out[i].x, tri_out[i].y);
    }

	// Draw triangles in input and output images
    constexpr int line_width = 2;
    cv::polylines(img_in, tri_in_int, true, color, line_width, cv::LINE_AA);
    cv::polylines(img_out, tri_out_int, true, color, line_width, cv::LINE_AA);

    cv::imshow("Input", img_in);
    cv::waitKey(1000);

    cv::imshow("Output", img_out);
    cv::waitKey(1000);
}