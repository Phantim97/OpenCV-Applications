#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>
#include <dlib/image_io.h>
#include <opencv2/imgproc.hpp> // If you are using OpenCV 3
#include <iostream>
#include <string>
#include <dlib/opencv.h>

#include <functional>

#include "env_util.h"
#include "ml_util.h"

static void remove_polygon_from_mask(cv::Mat& mask, const std::vector<cv::Point2f>& points, const std::vector<int>& points_index)
{
	std::vector<cv::Point> hull_points;

	for (int i = 0; i < points_index.size(); i++)
	{
		cv::Point pt(points[points_index[i]].x, points[points_index[i]].y);
		hull_points.push_back(pt);
	}

	cv::fillConvexPoly(mask, &hull_points[0], hull_points.size(), cv::Scalar(0, 0, 0));
}

static cv::Mat get_face_mask(const cv::Size& size, const std::vector<cv::Point2f>& points)
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

static int frame_count = 0;
static int gif_idx_current = 0;
static std::vector<cv::Mat> gif_frames;
bool anim_first_pass = true;

static dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
static dlib::shape_predictor predictor;

// Left eye polygon
static const int left_eye[] = { 36, 37, 38, 39, 40, 41 };
static const std::vector<int> left_eye_index(left_eye, left_eye + sizeof(left_eye) / sizeof(left_eye[0]));

// Right eye polygon
static const int right_eye[] = { 42, 43, 44, 45, 46, 47 };
static const std::vector<int> right_eye_index(right_eye, right_eye + sizeof(right_eye) / sizeof(right_eye[0]));

// Mouth polygon
static int mouth[] = { 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59 };
static const std::vector<int> mouth_index(mouth, mouth + sizeof(mouth) / sizeof(mouth[0]));

static cv::Mat& alpha_blend(const cv::Mat& alpha, const cv::Mat& foreground, const cv::Mat& background, cv::Mat& out_image)
{
	cv::Mat fore;
	cv::Mat back;
	cv::multiply(alpha, foreground, fore, 1 / 255.0);
	cv::multiply(cv::Scalar::all(255) - alpha, background, back, 1 / 255.0);
	cv::add(fore, back, out_image);

	return out_image;
}

static void add_polygon_to_mask(cv::Mat& mask, const std::vector<cv::Point2f>& points, const std::vector<int>& points_index)
{
	std::vector<cv::Point> hull_points;

	for (int i = 0; i < points_index.size(); i++)
	{
		cv::Point pt(points[points_index[i]].x, points[points_index[i]].y);
		hull_points.push_back(pt);
	}

	cv::fillConvexPoly(mask, &hull_points[0], hull_points.size(), cv::Scalar(255, 255, 255));
}

void inline pass_filter(cv::Mat& src, const std::vector<cv::Point2f> landmarks)
{
}

//TODO: Lipstick applier
/*
 * mask lip region
 * apply lipstick to lip region
 */

//The mouth can be accessed through points 49 - 67.
void lip_filter(cv::Mat& src, const std::vector<cv::Point2f> landmarks)
{
	//Get our dimensions
	const cv::Size size = src.size();
	//const std::string lip_filter = util::get_data_path() + "images/filters/lips/lips.jpg";
	//cv::Mat lip_face = cv::imread(lip_filter, cv::IMREAD_UNCHANGED);
	//const cv::Size lip_filter_sz = lip_face.size();
	//const std::vector<cv::Point2f> lip_landmarks = get_landmark_point_vector(lip_face, "images/filters/lips/", "lips.jpg", detector, predictor);

	////Get mouth mask of our image
	cv::Mat mask = cv::Mat::zeros(size.height, size.width, CV_8UC3);
	add_polygon_to_mask(mask, landmarks, mouth_index);

	//Get center point of lips
	const cv::Point lip_center((landmarks[54].x + landmarks[48].x) / 2, (landmarks[54].y + landmarks[48].y) / 2);

	//TODO: Get the lip colored region
	//TODO: Select the option for lip coloring

	//Get the lip region of the chosen face for the filter
	/*cv::Mat lip_mask = cv::Mat::zeros(lip_filter_sz.height, lip_filter_sz.width, CV_8UC3);
	add_polygon_to_mask(lip_mask, lip_landmarks, mouth_index);*/

	//Take lips off the lip mask
	/*cv::Mat lips;
	lip_face.copyTo(lips, lip_mask);*/

	//cv::seamlessClone(lips, src, mask, lip_center, src, cv::MIXED_CLONE);
}

//TODO: Eye Change (Heart Eye Tik Tok effect)
/*
 * Mask out eye region for both eyes (Done)
 * Get Iris specifically
 * Seamless clone eye change
 * Add color option
 */
cv::Mat eye_filter(cv::Mat& src, const std::vector<cv::Point2f> landmarks)
{
	const cv::Size size = src.size();
	cv::Mat gif_frame;

	if (anim_first_pass)
	{
		const std::string eye_filter = util::get_data_path() + "images/filters/eyes/heart_pulse.gif";
		
		cv::VideoCapture gif_capture(eye_filter.c_str());

		while (gif_capture.read(gif_frame))
		{
			gif_frames.push_back(gif_frame.clone()); //Note necessity for clone here
		}

		anim_first_pass = false;

		gif_capture.release();
	}

	//If anim complete we loop it back to beginning
	if (gif_idx_current == gif_frames.size())
	{
		gif_idx_current = 0;
	}

	//Get current frame of anim
	gif_frame = gif_frames[gif_idx_current++];

	//Handle both eyes
	cv::Mat left_eye_mask= cv::Mat::zeros(size.height, size.width, CV_8UC3);
	cv::Mat right_eye_mask = left_eye_mask.clone();

	add_polygon_to_mask(left_eye_mask, landmarks, left_eye_index);
	add_polygon_to_mask(right_eye_mask, landmarks, right_eye_index);

	//TODO: Somewhere around here we need to get the iris

	const cv::Point left_eye_center((landmarks[39].x + landmarks[40].x) / 2, (landmarks[39].y + landmarks[36].y) / 2);
	const cv::Point right_eye_center((landmarks[43].x + landmarks[44].x) / 2, (landmarks[45].y + landmarks[42].y) / 2);

	/*cv::Mat left_eye_frame;
	cv::Mat right_eye_frame;

	const cv::Size l_eye_size(landmarks[39].x - landmarks[36].x, landmarks[40].y - landmarks[38].y);
	const cv::Size r_eye_size(landmarks[45].x - landmarks[42].x, landmarks[46].y - landmarks[44].y);

	std::cout << "Eye sizes: \n";
	std::cout << l_eye_size << '\n';
	std::cout << r_eye_size << '\n';

	cv::resize(gif_frame, left_eye_frame, l_eye_size);
	cv::resize(gif_frame, right_eye_frame, r_eye_size);

	std::cout << "Sizes: " << left_eye_frame.size() << " vs. " << left_eye_mask.size() << '\n';*/

	//Next blend filter into eyes
	cv::resize(gif_frame, gif_frame, left_eye_mask.size());
	cv::seamlessClone(gif_frame, src, left_eye_mask, left_eye_center, src, cv::MIXED_CLONE);
	cv::resize(gif_frame, gif_frame, right_eye_mask.size());
	cv::seamlessClone(gif_frame, src, right_eye_mask, right_eye_center, src, cv::MIXED_CLONE);

	return src;
}

#define RESIZE_HEIGHT 480
#define FACE_DOWNSAMPLE_RATIO 1.5
#define SKIP_FRAMES 2

enum class Colors
{
	NONE,
	RED,
	ORANGE,
	YELLOW,
	GREEN,
	BLUE,
	INDIGO,
	VIOLET,
	SPECIAL,
	COUNT
};

enum class Filter
{
    NONE,
	LIPS,
    EYES,
    COUNT
};

static int filter_value = static_cast<int>(Filter::LIPS);
static int filter_color = static_cast<int>(Colors::NONE);

void virtual_makeup()
{
	const std::function<void(cv::Mat&, std::vector<cv::Point2f>)> filters[static_cast<int>(Filter::COUNT)] = { pass_filter, lip_filter, eye_filter };

	//eye_change();
    const std::string win_name = "Live Feed";

    cv::namedWindow(win_name);
    cv::createTrackbar("Filter", win_name, &filter_value, static_cast<int>(Filter::COUNT) - 1);
    cv::createTrackbar("Filter", win_name, &filter_color, static_cast<int>(Colors::COUNT) - 1);

    // Load face detection and pose estimation models. Will resolve for all detection purposes
	const std::string model_path = util::get_model_path() + "dlib_models/shape_predictor_68_face_landmarks.dat";
    dlib::deserialize(model_path) >> predictor;

    //process input from webcam or video file
    cv::VideoCapture cap;
   
	cap.open(0);
	if (!cap.isOpened())
    {
		std::cout << "Camera not open breaking\n";
		return;
    }

    cv::Mat frame;

    // Read a frame initially to assign memory for the frame and calculate new height
    cap >> frame;
	const int height = frame.rows;
	const float image_resize = static_cast<float>(height) / RESIZE_HEIGHT;

    // Declare the variable for landmark points
    std::vector<cv::Point2f> landmarks;

	cv::Mat output;

    //Video Loop
	while (true)
	{
		cap >> frame;

		cv::resize(frame, frame, cv::Size(), 1.0 / image_resize, 1.0 / image_resize);

		if (filter_value != static_cast<int>(Filter::NONE)) //Only if we are using face detection filter
		{
			//Find landmarks after skipping SKIP_Frames number of frames
			if (frame_count % SKIP_FRAMES == 0)
			{
				landmarks = get_landmarks(detector, predictor, frame, (float)FACE_DOWNSAMPLE_RATIO);
			}

			//If face is partially detected
			if (landmarks.size() != 68)
			{
				//To not freeze stream if face not detected
				cv::imshow(win_name, frame);
				cv::waitKey(1);

				continue;
			}
		}

		//Now we check for which filter we want to use
		filters[filter_value](frame, landmarks);

        cv::imshow(win_name, frame);

		const int k = cv::waitKey(1);

        if (k == 27) //ESC
        {
            break;
        }

		frame_count++;
    }

    cap.release();
}