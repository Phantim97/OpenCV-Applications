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

#define FACE_DOWNSAMPLE_RATIO 1

cv::Mat correct_colors(const cv::Mat& im1, cv::Mat im2, std::vector<cv::Point2f> points2)// lower number --> output is closer to webcam and vice-versa
{
	const cv::Point2f dist_between_eyes = points2[38] - points2[43];
	const float distance = cv::norm(dist_between_eyes);

	//using heuristics to calculate the amount of blur
	int blur_amount = static_cast<int>(0.5 * distance);

	if (blur_amount % 2 == 0)
	{
		blur_amount += 1;
	}

	cv::Mat im1_blur = im1.clone();
	cv::Mat im2_blur = im2.clone();

	cv::blur(im1_blur, im1_blur, cv::Size(blur_amount, blur_amount));
	cv::blur(im2_blur, im2_blur, cv::Size(blur_amount, blur_amount));
	// Avoid divide-by-zero errors.

	im2_blur += 2 * (im2_blur <= 1) / 255;
	im1_blur.convertTo(im1_blur, CV_32F);
	im2_blur.convertTo(im2_blur, CV_32F);
	im2.convertTo(im2, CV_32F);

	cv::Mat ret = im2.clone();
	ret = im2.mul(im1_blur).mul(1 / im2_blur);
	cv::threshold(ret, ret, 255, 255, cv::THRESH_TRUNC);
	ret.convertTo(ret, CV_8UC3);

	return ret;
}

void beardify()
{
	int selected_points[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 31, 32, 33, 34, 35, 55, 56, 57, 58, 59 };
	std::vector<int> selected_index(selected_points, selected_points + sizeof(selected_points) / sizeof(int));

	// Load face detection and pose estimation models.
	std::string model_path = util::get_model_path() + "dlib_models/shape_predictor_68_face_landmarks.dat";
	dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
	dlib::shape_predictor predictor;
	dlib::deserialize(model_path) >> predictor;

	std::string overlay_file = util::get_data_path() + "images/filters/beardify/beard1.png";
	std::string image_file = util::get_data_path() + "images/people/jilly.jpg";

	// Read the beard image along with its alpha mask

	cv::Mat beard;
	cv::Mat target_image;
	cv::Mat beard_alpha_mask;
	cv::Mat img_with_mask = cv::imread(overlay_file, cv::IMREAD_UNCHANGED);
	std::vector<cv::Mat> rgba_channels(4);

	// Split into channels
	cv::split(img_with_mask, rgba_channels);

	// Extract the beard image
	std::vector<cv::Mat> bgr_channels;
	bgr_channels.push_back(rgba_channels[0]);
	bgr_channels.push_back(rgba_channels[1]);
	bgr_channels.push_back(rgba_channels[2]);

	cv::merge(bgr_channels, beard);
	beard.convertTo(beard, CV_32F, 1.0 / 255.0);

	// Extract the beard mask
	std::vector<cv::Mat> mask_channels;
	mask_channels.push_back(rgba_channels[3]);
	mask_channels.push_back(rgba_channels[3]);
	mask_channels.push_back(rgba_channels[3]);

	cv::merge(mask_channels, beard_alpha_mask);
	beard_alpha_mask.convertTo(beard_alpha_mask, CV_32FC3);

	//Read points for beard from file
	std::vector<cv::Point2f> feature_points1 = get_saved_points(overlay_file);

	// Calculate Delaunay triangles
	cv::Rect rect = cv::boundingRect(feature_points1);

	std::vector< std::vector<int> > dt;
	calculate_delaunay_triangles(rect, feature_points1, dt);

	//float time_detector = static_cast<double>(cv::getTickCount());

	// Get the face image for putting the beard
	target_image = cv::imread(image_file);

	std::vector<cv::Point2f> points2 = get_landmark_point_vector(target_image, "images/people", "tian2.jpg", detector, predictor);

	std::vector<cv::Point2f> feature_points2;
	for (int i = 0; i < selected_index.size(); i++)
	{
		feature_points2.push_back(points2[selected_index[i]]);
		constrain_point(feature_points2[i], target_image.size());
	}

	//convert Mat to float data type
	target_image.convertTo(target_image, CV_32F, 1.0 / 255.0);

	//empty warp image
	cv::Mat beard_warped = cv::Mat::zeros(target_image.size(), beard.type());
	cv::Mat beard_alpha_mask_warped = cv::Mat::zeros(target_image.size(), beard_alpha_mask.type());

	// Apply affine transformation to Delaunay triangles
	for (size_t i = 0; i < dt.size(); i++)
	{
		std::vector<cv::Point2f> t1, t2;
		// Get points for img1, targetImage corresponding to the triangles
		for (size_t j = 0; j < 3; j++)
		{
			t1.push_back(feature_points1[dt[i][j]]);
			t2.push_back(feature_points2[dt[i][j]]);
		}

		warp_triangle(beard, beard_warped, t1, t2);
		warp_triangle(beard_alpha_mask, beard_alpha_mask_warped, t1, t2);
	}

	cv::Mat mask1;
	beard_alpha_mask_warped.convertTo(mask1, CV_32FC3, 1.0 / 255.0);
	cv::Mat mask2 = cv::Scalar(1.0, 1.0, 1.0) - mask1;

	cv::Mat temp1 = target_image.mul(mask2);
	cv::Mat temp2 = beard_warped.mul(mask1);
	cv::Mat result = temp1 + temp2;

	cv::imshow("Output", result);
	cv::waitKey(5000);
}

//Age filter portion

// Alpha blending using multiply and add functions
cv::Mat& alpha_blend(const cv::Mat& alpha, const cv::Mat& foreground, const cv::Mat& background, cv::Mat& out_image)
{
	cv::Mat fore;
	cv::Mat back;

	cv::multiply(alpha, foreground, fore, 1 / 255.0);
	cv::multiply(cv::Scalar::all(255) - alpha, background, back, 1 / 255.0);

	cv::add(fore, back, out_image);

	return out_image;
}

// Desaturate image
void desaturate_image(cv::Mat& im, const double scale_by)
{
	// Convert input image to HSV
	cv::Mat img_hsv;
	cv::cvtColor(im, img_hsv, cv::COLOR_BGR2HSV);

	// Split HSV image into three channels.
	std::vector<cv::Mat> channels(3);
	cv::split(img_hsv, channels);

	// Multiple saturation by the scale.
	channels[1] = scale_by * channels[1];

	// Merge back the three channels
	cv::merge(channels, img_hsv);

	// Convert HSV to RGB
	cv::cvtColor(img_hsv, im, cv::COLOR_HSV2BGR);
}

void remove_polygon_from_mask(cv::Mat& mask, const std::vector<cv::Point2f>& points, const std::vector<int>& points_index)
{
	std::vector<cv::Point> hull_points;

	for (int i = 0; i < points_index.size(); i++)
	{
		cv::Point pt(points[points_index[i]].x, points[points_index[i]].y);
		hull_points.push_back(pt);
	}

	cv::fillConvexPoly(mask, &hull_points[0], hull_points.size(), cv::Scalar(0, 0, 0));
}

void append_forehead_points(std::vector<cv::Point2f>& points)
{
	constexpr double offset_scalp = 3.0;

	static int brows[] = { 25, 23, 20, 18 };
	const std::vector<int> brows_index(brows, brows + sizeof(brows) / sizeof(brows[0]));

	static int brows_reference[] = { 45, 47, 40, 36 };
	const std::vector<int> brows_reference_index(brows_reference, brows_reference + sizeof(brows_reference) / sizeof(brows_reference[0]));

	for (unsigned long k = 0; k < brows_index.size(); ++k)
	{
		cv::Point2f forehead_point = offset_scalp * (points[brows_index[k]] - points[brows_reference_index[k]]) + points[brows_reference_index[k]];
		points.push_back(forehead_point);
	}
}

cv::Mat get_face_mask(const cv::Size& size, const std::vector<cv::Point2f>& points)
{
	// Left eye polygon
	static int left_eye[] = { 36, 37, 38, 39, 40, 41 };
	const std::vector<int> left_eye_index(left_eye, left_eye + sizeof(left_eye) / sizeof(left_eye[0]));

	// Right eye polygon
	static int right_eye[] = { 42, 43, 44, 45, 46, 47 };
	const std::vector<int> right_eye_index(right_eye, right_eye + sizeof(right_eye) / sizeof(right_eye[0]));

	// Mouth polygon
	static int mouth[] = { 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59 };
	const std::vector<int> mouth_index(mouth, mouth + sizeof(mouth) / sizeof(mouth[0]));

	// Nose polygon
	static int nose[] = { 28, 31, 33, 35 };
	const std::vector<int> nose_index(nose, nose + sizeof(nose) / sizeof(nose[0]));

	// Find Convex hull of all points
	std::vector<cv::Point2f> hull;
	cv::convexHull(points, hull, false, true);

	// Convert to vector of Point2f to vector of Point
	std::vector<cv::Point> hull_int;
	for (int i = 0; i < hull.size(); i++)
	{
		cv::Point pt(hull[i].x, hull[i].y);
		hull_int.push_back(pt);
	}

	// Create mask such that convex hull is white.
	cv::Mat mask = cv::Mat::zeros(size.height, size.width, CV_8UC3);
	cv::fillConvexPoly(mask, &hull_int[0], hull_int.size(), cv::Scalar(255, 255, 255));

	// Remove eyes, mouth and nose from the mask.
	remove_polygon_from_mask(mask, points, left_eye_index);
	remove_polygon_from_mask(mask, points, right_eye_index);
	remove_polygon_from_mask(mask, points, nose_index);
	remove_polygon_from_mask(mask, points, mouth_index);

	return mask;
}

void age_filter()
{
	std::string model_path = util::get_model_path() + "dlib_models/shape_predictor_68_face_landmarks.dat";

	// Load face detector
	dlib::frontal_face_detector face_detector = dlib::get_frontal_face_detector();

	// Load landmark detector.
	dlib::shape_predictor landmark_detector;
	dlib::deserialize(model_path) >> landmark_detector;

	// File to copy wrinkles from
	std::string filename1 = util::get_data_path() + "images/filters/age/wrinkle2.jpg";

	// File to apply aging
	std::string filename2 = util::get_data_path() + "images/people/tian2.jpg";

	// Read images
	cv::Mat img1 = cv::imread(filename1);
	cv::Mat img2 = cv::imread(filename2);

	// Find landmarks.
	std::vector<cv::Point2f> points1;
	std::vector<cv::Point2f> points2;

	points1 = get_saved_points(filename1 + ".txt");
	points2 = get_landmark_point_vector(img2, "images/people/", "tian2.jpg", face_detector, landmark_detector);

	// Find forehead points.
	append_forehead_points(points1);
	append_forehead_points(points2);

	// Find Delaunay Triangulation
	std::vector<std::vector<int>> dt;
	cv::Rect rect(0, 0, img1.cols, img1.rows);
	calculate_delaunay_triangles(rect, points1, dt);

	// Convert image for warping.
	img1.convertTo(img1, CV_32F);
	img2.convertTo(img2, CV_32F);

	// Warp wrinkle image to face image.
	cv::Mat img1_warped = img2.clone();
	warp_image(img1, img1_warped, points1, points2, dt);
	img1_warped.convertTo(img1_warped, CV_8UC3);
	img2.convertTo(img2, CV_8UC3);

	// Calculate face mask for seamless cloning.
	cv::Mat mask = get_face_mask(img2.size(), points2);

	// Seamlessly clone the wrinkle image onto original face
	cv::Rect r1 = cv::boundingRect(points2);
	cv::Point center1 = (r1.tl() + r1.br()) / 2;
	cv::Mat cloned_output;
	cv::seamlessClone(img1_warped, img2, mask, center1, cloned_output, cv::MIXED_CLONE);

	// Blurring face mask to alpha blend to hide seams
	cv::Size size = mask.size();
	cv::Mat mask_small;

	cv::resize(mask, mask_small, cv::Size(256, static_cast<int>(size.height * 256.0 / static_cast<double>(size.width))));
	cv::erode(mask_small, mask_small, cv::Mat(), cv::Point(-1, -1), 5);
	cv::GaussianBlur(mask_small, mask_small, cv::Size(15, 15), 0, 0);
	cv::resize(mask_small, mask, size);

	cv::Mat aged_image = cloned_output.clone();
	alpha_blend(mask, cloned_output, img2, aged_image);

	//Desaturate output
	desaturate_image(aged_image, 0.8);

	cv::imshow("Output", aged_image);
	cv::waitKey(5000);
}

//Happify Filter

void happify()
{
	// Get the face detector
	dlib::frontal_face_detector faceDetector = dlib::get_frontal_face_detector();

	// The landmark detector is implemented in the shape_predictor class
	dlib::shape_predictor landmarkDetector;

	// Load the landmark model
	dlib::deserialize(util::get_model_path() + "dlib_models/shape_predictor_68_face_landmarks.dat") >> landmarkDetector;

	// Amount of deformation
	float offset1 = 1.3;
	float offset2 = 1.3;

	// Points that should not move
	int anchorPoints[] = { 8, 30 };
	std::vector<int> anchorPointsArray(anchorPoints, anchorPoints + sizeof(anchorPoints) / sizeof(int));

	// Points that will be deformed for lips
	int deformedPoints1[] = { 48, 57, 54 };
	std::vector<int> deformedPoints1Array(deformedPoints1, deformedPoints1 + sizeof(deformedPoints1) / sizeof(int));

	// Points that will be deformed for lips
	int deformedPoints2[] = { 21, 22, 36, 45 };
	std::vector<int> deformedPoints2Array(deformedPoints2, deformedPoints2 + sizeof(deformedPoints2) / sizeof(int));

	double t = (double)cv::getTickCount();

	// load a nice picture
	std::string filename = util::get_data_path() + "images/people/tian2.jpg";
	cv::Mat src = cv::imread(filename);

	std::vector<cv::Point2f> landmarks;
	landmarks = get_landmark_point_vector(src, "images/people/", "tian2.jpg", faceDetector, landmarkDetector);

	if (landmarks.empty())
	{
		std::cout << "No face found\n";
		return;
	}

	// Set the center to tip of chin
	cv::Point2f center1(landmarks[8]);
	// Set the center to point on nose
	cv::Point2f center2(landmarks[28]);

	// Variables for storing the original and deformed points
	std::vector<cv::Point2f> srcPoints;
	std::vector<cv::Point2f> dstPoints;

	// Adding the original and deformed points using the landmark points
	for (int i = 0; i < anchorPointsArray.size(); i++)
	{
		srcPoints.push_back(landmarks[anchorPointsArray[i]]);
		dstPoints.push_back(landmarks[anchorPointsArray[i]]);
	}

	for (int i = 0; i < deformedPoints1Array.size(); i++)
	{
		srcPoints.push_back(landmarks[deformedPoints1Array[i]]);
		cv::Point2f pt = offset1 * (landmarks[deformedPoints1Array[i]] - center1) + center1;
		dstPoints.push_back(pt);
	}
	for (int i = 0; i < deformedPoints2Array.size(); i++)
	{
		srcPoints.push_back(landmarks[deformedPoints2Array[i]]);
		cv::Point2f pt = offset2 * (landmarks[deformedPoints2Array[i]] - center2) + center2;
		dstPoints.push_back(pt);
	}

	// Adding the boundary points to keep the image stable globally
	get_eight_boundary_points(src.size(), srcPoints);
	get_eight_boundary_points(src.size(), dstPoints);
	// Performing moving least squares deformation on the image using the points gathered above
	cv::Mat dst = src.clone();
	mls_warp_image(src, srcPoints, dst, dstPoints, MlsMode::FAST);

	std::cout << "time taken " << (static_cast<double>(cv::getTickCount()) - t) / cv::getTickFrequency() << '\n';
	cv::Mat combined;
	cv::hconcat(src, dst, combined);

	cv::imshow("Output", combined);
	cv::waitKey(5000);
}

//Fatify Filter

void fatify()
{
	std::string model_path = util::get_model_path() + "dlib_models/shape_predictor_68_face_landmarks.dat";

	// Get the face detector
	dlib::frontal_face_detector face_detector = dlib::get_frontal_face_detector();

	// The landmark detector is implemented in the shape_predictor class
	dlib::shape_predictor landmark_detector;

	// Load the landmark model
	dlib::deserialize(model_path) >> landmark_detector;

	// Amount of bulge to be given for fatify

	// Points that should not move
	int anchor_points[] = { 1, 15, 30 };
	std::vector<int> anchor_points_array(anchor_points, anchor_points + sizeof(anchor_points) / sizeof(int));

	// Points that will be deformed
	int deformed_points[] = { 5, 6, 8, 10, 11 };
	std::vector<int> deformed_points_array(deformed_points, deformed_points + sizeof(deformed_points) / sizeof(int));

	double t = static_cast<double>(cv::getTickCount());

	std::string file;
	std::cout << "Select an image: ";
	std::cin >> file;

	// load image
	std::string filename = util::get_data_path() + "images/people/" + file;
	cv::Mat src = cv::imread(filename);

	cv::resize(src, src, cv::Size(480, 854));

	std::vector<cv::Point2f> landmarks;
	landmarks = get_landmark_point_vector(src, "images/people/", file, face_detector, landmark_detector);

	// Set the center of face to be the nose tip
	cv::Point2f center(landmarks[30]);

	// Variables for storing the original and deformed points
	std::vector<cv::Point2f> src_points;
	std::vector<cv::Point2f> dst_points;

	// Adding the original and deformed points using the landmark points
	for (int i = 0; i < anchor_points_array.size(); i++)
	{
		src_points.push_back(landmarks[anchor_points_array[i]]);
		dst_points.push_back(landmarks[anchor_points_array[i]]);
	}

	for (int i = 0; i < deformed_points_array.size(); i++)
	{
		constexpr float offset = 1.2f;
		src_points.push_back(landmarks[deformed_points_array[i]]);
		cv::Point2f pt(offset * (landmarks[deformed_points_array[i]].x - center.x) + center.x, offset * (landmarks[deformed_points_array[i]].y - center.y) + center.y);
		dst_points.push_back(pt);
	}

	// Adding the boundary points to keep the image stable globally
	get_eight_boundary_points(src.size(), src_points);
	get_eight_boundary_points(src.size(), dst_points);

	// Performing moving least squares deformation on the image using the points gathered above
	cv::Mat dst = cv::Mat::zeros(src.rows, src.cols, CV_8UC3);
	mls_warp_image(src, src_points, dst, dst_points, MlsMode::FAST);

	std::cout << "Time taken: " << (static_cast<double>(cv::getTickCount()) - t) / cv::getTickFrequency() << '\n';

	cv::Mat combined;
	cv::hconcat(src, dst, combined);

	cv::namedWindow("Output", cv::WINDOW_FREERATIO);
	cv::imshow("Output", combined);
	cv::waitKey(10000);
}