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

// Left iris polygon
static const int left_iris[] = { 37, 38, 40, 41 };
static const std::vector<int> left_iris_index(left_iris, left_iris + sizeof(left_iris) / sizeof(left_iris[0]));

// Right iris polygon
static const int right_iris[] = { 43, 44, 46, 47 };
static const std::vector<int> right_iris_index(right_iris, right_iris + sizeof(right_iris) / sizeof(right_iris[0]));

// Mouth polygon
static const int mouth[] = { 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59 };
static const std::vector<int> mouth_index(mouth, mouth + sizeof(mouth) / sizeof(mouth[0]));

// Innner Mouth polygon
static int mouth_no_lip[] = { 60, 61, 62, 63, 64, 65, 66, 67 };
static const std::vector<int> mouth_no_lip_index(mouth_no_lip, mouth_no_lip + sizeof(mouth_no_lip) / sizeof(mouth_no_lip[0]));

enum class Colors
{
	NONE,
	RED,
	ORANGE,
	YELLOW,
	GREEN,
	BLUE,
	PURPLE,
	PINK,
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

static int filter_value = static_cast<int>(Filter::EYES);
static int filter_color = static_cast<int>(Colors::NONE);

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

void change_color(cv::Mat& obj)
{
	cv::Mat channels[3];

	cv::split(obj, channels); //BGR

	switch (filter_color)
	{
	case static_cast<int>(Colors::RED): //pure r modification
		channels[0] *= .75;
		channels[1] *= .75;
		channels[2] *= 1.5;
		break;
	case static_cast<int>(Colors::ORANGE):
		channels[0] *= .75;
		channels[1] *= 1.25;
		channels[2] *= 1.5;
		break;
	case static_cast<int>(Colors::YELLOW):
		channels[0] *= .75;
		channels[1] *= 1.5;
		channels[2] *= 1.5;
		break;
	case static_cast<int>(Colors::GREEN):
		channels[0] *= .75;
		channels[1] *= 1.5;
		channels[2] *= .75;
		break;
	case static_cast<int>(Colors::BLUE):
		channels[0] *= 1.5;
		channels[1] *= .5;
		channels[2] *= .5;
		break;
	case static_cast<int>(Colors::PURPLE):
		channels[0] *= 1.5;
		channels[1] *= .75;
		channels[2] *= .75;
		break;
	case static_cast<int>(Colors::PINK):
		channels[0] *= 1.5;
		channels[1] *= .75;
		channels[2] *= 1.5;
		break;
	case static_cast<int>(Colors::SPECIAL):
		channels[0] *= .5;
		channels[1] *= .5;
		channels[2] *= .5;
		break;
	default: 
		break;
	}

	cv::merge(channels, 3, obj);
}

void lip_filter(cv::Mat& src, const std::vector<cv::Point2f> landmarks)
{
	//Get our dimensions
	const cv::Size size = src.size();

	//Get mouth mask of our image
	cv::Mat mask = cv::Mat::zeros(size.height, size.width, CV_8UC3);

	add_polygon_to_mask(mask, landmarks, mouth_index);
	remove_polygon_from_mask(mask, landmarks, mouth_no_lip_index);

	//Get the lip region
	cv::Mat lips;
	src.copyTo(lips, mask);

	change_color(lips);

	lips.copyTo(src, mask);
}

cv::Mat get_iris(cv::Mat eye)
{
	cv::Mat iris;

	return iris;
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
		const std::string eye_filter = util::get_data_path() + "images/filters/eyes/rainbow_spiral.gif";
		
		cv::VideoCapture gif_capture(eye_filter.c_str());

		while (gif_capture.read(gif_frame))
		{
			gif_frames.push_back(gif_frame.clone()); //Note necessity for clone here
		}

		anim_first_pass = false;

		gif_capture.release();
	}

	//Mask the iris region
	cv::Mat left_iris_mask= cv::Mat::zeros(size.height, size.width, CV_8UC3);
	cv::Mat right_iris_mask = left_iris_mask.clone();

	cv::Mat left_iris;
	cv::Mat right_iris;

	add_polygon_to_mask(left_iris_mask, landmarks, left_iris_index);
	add_polygon_to_mask(right_iris_mask, landmarks, right_iris_index);

	src.copyTo(left_iris, left_iris_mask);
	src.copyTo(right_iris, right_iris_mask);

	//Next blend filter into eyes
	change_color(left_iris);
	change_color(right_iris);

	//Get current frame of anim
	if (filter_color == static_cast<int>(Colors::SPECIAL))
	{
		//If anim complete we loop it back to beginning
		if (gif_idx_current == gif_frames.size())
		{
			gif_idx_current = 0;
		}

		gif_frame = gif_frames[gif_idx_current].clone();
		gif_idx_current++;

		cv::resize(gif_frame, gif_frame, left_iris_mask.size());
		gif_frame.copyTo(left_iris, left_iris_mask);

		cv::resize(gif_frame, gif_frame, right_iris_mask.size());
		gif_frame.copyTo(right_iris, right_iris_mask);
	}

	left_iris.copyTo(src, left_iris_mask);
	right_iris.copyTo(src, right_iris_mask);

	return src;
}

void virtual_makeup()
{
	constexpr int resize_height = 480;

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
	const float image_resize = static_cast<float>(height) / resize_height;

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
			constexpr int skip_frames = 2;
			//Find landmarks after skipping SKIP_Frames number of frames
			if (frame_count % skip_frames == 0)
			{
				constexpr double face_downsample_ratio = 1.5;
				landmarks = get_landmarks(detector, predictor, frame, (float)face_downsample_ratio);
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