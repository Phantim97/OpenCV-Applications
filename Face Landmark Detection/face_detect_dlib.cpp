#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <fstream>
#include <iostream>

#include "env_util.h"

void write_landmarks_to_file(dlib::full_object_detection& landmarks, const std::string& filename)
{
    // Open file
    std::ofstream ofs;
    ofs.open(filename);

    // Loop over all landmark points
    for (int i = 0; i < landmarks.num_parts(); i++)
    {
        // Print x and y coordinates to file
        ofs << landmarks.part(i).x() << " " << landmarks.part(i).y() << '\n';

    }
    // Close file
    ofs.close();
}

void draw_polyline
(
    cv::Mat& img,
    const dlib::full_object_detection& landmarks,
    const int start,
    const int end,
    const bool is_closed = false
)
{
    std::vector <cv::Point> points;
    for (int i = start; i <= end; ++i)
    {
        points.emplace_back(landmarks.part(i).x(), landmarks.part(i).y());
    }
    cv::polylines(img, points, is_closed, cv::Scalar(255, 200, 0), 2, 16);

}

// Draw face for the 68-point model.
void render_face(cv::Mat& img, const dlib::full_object_detection& landmarks)
{
    draw_polyline(img, landmarks, 0, 16);           // Jaw line
    draw_polyline(img, landmarks, 17, 21);          // Left eyebrow
    draw_polyline(img, landmarks, 22, 26);          // Right eyebrow
    draw_polyline(img, landmarks, 27, 30);          // Nose bridge
    draw_polyline(img, landmarks, 30, 35, true);    // Lower nose
    draw_polyline(img, landmarks, 36, 41, true);    // Left eye
    draw_polyline(img, landmarks, 42, 47, true);    // Right Eye
    draw_polyline(img, landmarks, 48, 59, true);    // Outer lip
    draw_polyline(img, landmarks, 60, 67, true);    // Inner lip
}

void render_face(
    cv::Mat& img, // Image to draw the points on
    const std::vector<cv::Point2f>& points, // Vector of points
    const cv::Scalar& color, // color points
    const int radius = 3) // Radius of points.
{
    for (int i = 0; i < points.size(); i++)
    {
        cv::circle(img, points[i], radius, color, -1);
    }
}

void dlib_detect_main()
{
    std::string model_path = util::get_model_path();
    std::string data_path = util::get_data_path();

    // Get the face detector
    dlib::frontal_face_detector face_detector = dlib::get_frontal_face_detector();

    // The landmark detector is implemented in the shape_predictor class
    dlib::shape_predictor landmark_detector;
    std::string predictor_path(model_path + "dlib_models/shape_predictor_68_face_landmarks.dat");

    // Load the landmark model
    dlib::deserialize(predictor_path) >> landmark_detector;
    
    // Read Image
    std::string image_filename(data_path + "images/family.jpg");
    cv::Mat img = cv::imread(image_filename);

    // landmarks will be stored in results/family_0.txt
    std::string landmarks_basename("results/family");

    // Convert OpenCV image format to Dlib's image format
    dlib::cv_image<dlib::bgr_pixel> dlib_img(img);

    // Detect faces in the image
    std::vector<dlib::rectangle> face_rects = face_detector(dlib_img);
    std::cout << "Number of faces detected: " << face_rects.size() << '\n';

    // Vector to store landmarks of all detected faces
    std::vector<dlib::full_object_detection> landmarks_all;

    // Loop over all detected face rectangles
    for (int i = 0; i < face_rects.size(); i++)
    {
        // For every face rectangle, run landmarkDetector
        dlib::full_object_detection landmarks = landmark_detector(dlib_img, face_rects[i]);

        // Print number of landmarks
        if (i == 0)
        {
            std::cout << "Number of landmarks : " << landmarks.num_parts() << '\n';
        }

        // Store landmarks for current face
        landmarks_all.push_back(landmarks);

        // Next, we render the outline of the face using detected landmarks.
        // Draw landmarks on face
        render_face(img, landmarks); //Internal function

        // The code below saves the landmarks to results/family_0.txt … results/family_4.txt
    	// Write landmarks to disk
        std::stringstream landmarks_filename;
        landmarks_filename << landmarks_basename << "_" << i << ".txt";
        std::cout << "Saving landmarks to " << landmarks_filename.str() << '\n';
        write_landmarks_to_file(landmarks, landmarks_filename.str());
    }

    // Save image
    std::string output_filename("results/familyLandmarks.jpg");
    std::cout << "Saving output image to " << output_filename << '\n';
    cv::imwrite(output_filename, img);
    cv::imshow("Image", img);
    cv::waitKey(5000);
}

#define RESIZE_HEIGHT 480
#define SKIP_FRAMES 2

void dlib_fast_face_detect()
{
	std::string PREDICTOR_PATH = util::get_model_path() + "dlib_models/shape_predictor_68_face_landmarks.dat";

    // Create an imshow window
	std::string win_name("Fast Facial Landmark Detector");
    cv::namedWindow(win_name, cv::WINDOW_NORMAL);

    // Create a VideoCapture object
    cv::VideoCapture cap(0);
    // Check if OpenCV is able to read feed from camera
    if (!cap.isOpened())
    {
	    std::cerr << "Unable to connect to camera\n";
        return;
    }

    // Just a place holder. Actual value calculated after 100 frames.
    double fps = 120.0;

    // Get first frame and allocate memory.
    cv::Mat im;
    cap >> im;

    // We will use a fixed height image as input to face detector
    cv::Mat im_small, imDisplay;
    float height = im.rows;
    // calculate resize scale
    float resize_scale = height / RESIZE_HEIGHT;
    cv::Size size = im.size();

    // Load face detection and pose estimation models
	dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
	dlib::shape_predictor predictor;
	dlib::deserialize(PREDICTOR_PATH) >> predictor;

    // initiate the tickCounter
    double t = static_cast<double>(cv::getTickCount());
    int count = 0;

    std::vector<dlib::rectangle> faces;
    // Grab and process frames until the main window is closed by the user.
    while (true)
    {
        if (count == 0)
        {
            t = cv::getTickCount();
        }

        // Grab a frame
        cap >> im;
        // create imSmall by resizing image by resize scale
        cv::resize(im, im_small, cv::Size(), 1.0 / resize_scale, 1.0 / resize_scale);
        // Change to dlib's image format. No memory is copied
        dlib::cv_image<dlib::bgr_pixel> cimgSmall(im_small);
        dlib::cv_image<dlib::bgr_pixel> cimg(im);

        // Process frames at an interval of SKIP_FRAMES.
        // This value should be set depending on your system hardware
        // and camera fps.
        // To reduce computations, this value should be increased
        if (count % SKIP_FRAMES == 0)
        {
            // Detect faces
            faces = detector(cimgSmall);
        }

        // Find facial landmarks for each face.
        std::vector<dlib::full_object_detection> shapes;
        // Iterate over faces
        for (unsigned long i = 0; i < faces.size(); ++i)
        {
            // Since we ran face detection on a resized image,
            // we will scale up coordinates of face rectangle
            dlib::rectangle r(faces[i].left() * resize_scale,faces[i].top() * resize_scale,faces[i].right() * resize_scale,faces[i].bottom() * resize_scale);

            // Find face landmarks by providing reactangle for each face
            dlib::full_object_detection shape = predictor(cimg, r);
            shapes.push_back(shape);
            // Draw facial landmarks
            render_face(im, shape);
        }

        // Put fps at which we are processing camera feed on frame
        cv::putText(im, cv::format("fps %.2f", fps), cv::Point(50, size.height - 50), cv::FONT_HERSHEY_COMPLEX, 1.5, cv::Scalar(0, 0, 255), 3);

        // Display it all on the screen
        cv::imshow(win_name, im);

        // Wait for keypress
        char key = cv::waitKey(1);
        if (key == 27) // ESC
        {
            // If ESC is pressed, break out of loop.
            break;
        }

        // increment frame counter
        count++;
        // calculate fps after each 100 frames are processed
        if (count == 100)
        {
            t = (static_cast<double>(cv::getTickCount()) - t) / cv::getTickFrequency();
            fps = 100.0 / t;
            count = 0;
        }
    }

    cap.release();
    cv::destroyAllWindows();
}

// Constrains points to be inside boundary
void constrainPoint(cv::Point2f& p, const cv::Size& sz)
{
    p.x = cv::min(cv::max(static_cast<double>(p.x), 0.0), static_cast<double>(sz.width - 1));
    p.y = cv::min(cv::max(static_cast<double>(p.y), 0.0), static_cast<double>(sz.height - 1));

}
double interEyeDistance(dlib::full_object_detection& shape)
{
	const cv::Point2f leftEyeLeftCorner(shape.part(36).x(), shape.part(36).y());
	const cv::Point2f rightEyeRightCorner(shape.part(45).x(), shape.part(45).y());
	const double distance = norm(rightEyeRightCorner - leftEyeLeftCorner);
    return distance;
}

void landmark_stabilization()
{
    constexpr int resize_height = 360;
	constexpr int num_frames_for_fps = 100;
	constexpr int skip_frames = 1;

    std::string win_name("Stabilized Facial Landmark Detector");
    cv::namedWindow(win_name, cv::WINDOW_NORMAL);

    cv::VideoCapture cap(0);
    if (!cap.isOpened())
    {
	    std::cerr << "Unable to connect to camera\n";
        return;
    }

    // Set up optical flow params
    cv::TermCriteria termcrit(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 20, 0.03);
    cv::Size win_size(101, 101);
    double eye_distance;
    double dot_radius;
    double sigma;
    bool eye_distance_not_calculated = true;
    int max_level = 5;
    std::vector<uchar> status;
    std::vector<float> err;

    // Just a place holder for frame rate.
    // Actual value calculated after 100 frames.
    double fps = 30.0;

    cv::Mat im;
    cv::Mat im_prev;
    cv::Mat im_gray;
    cv::Mat im_gray_prev;

    std::vector<cv::Mat> im_gray_pyr;
    std::vector<cv::Mat> im_gray_prev_pyr;

    // Get first frame and allocate memory.
    cap >> im_prev;

    // Convert to grayscale for optical flow calculation
    cv::cvtColor(im_prev, im_gray_prev, cv::COLOR_BGR2GRAY);

    // Build image pyramid for fast optical flow calculation
    cv::buildOpticalFlowPyramid(im_gray_prev, im_gray_prev_pyr, win_size, max_level);

    // Get image size
    cv::Size size = im_prev.size();

    // imSmall will be used for storing a resized image.
    cv::Mat im_small;
    // Load Dlib's face detection
    dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();

    // Load Facial Landmark Detector
    dlib::shape_predictor landmark_detector;
    dlib::deserialize(util::get_model_path() + "dlib_models/shape_predictor_68_face_landmarks.dat") >> landmark_detector;
    // Vector to store face rectangles
    std::vector<dlib::rectangle> faces;

    std::vector<cv::Point2f> points;
    std::vector<cv::Point2f> points_prev;
    std::vector<cv::Point2f> points_detected_cur;
    std::vector<cv::Point2f> points_detected_prev;

    // Initialize point arrays with (0,0)
    for (unsigned long k = 0; k < landmark_detector.num_parts(); ++k)
    {
        points_prev.emplace_back(0, 0);
        points.emplace_back(0, 0);
        points_detected_cur.emplace_back(0, 0);
        points_detected_prev.emplace_back(0, 0);
    }

    // First frame is handled differently.
    bool is_first_frame = true;

    // Show stabilized video flag
    bool show_stabilized = false;

    // Variables used for Frame rate calculation
    int count = 0;
    double t;

    while (true)
    {
        if (count == 0) t = (double)cv::getTickCount();

        // Grab a frame
        cap >> im;

        cv::cvtColor(im, im_gray, cv::COLOR_BGR2GRAY);
        float height = im.rows;
        float image_resize = height / resize_height;
        // Resize image for faster face detection
        cv::resize(im, im_small, cv::Size(), 1.0 / image_resize, 1.0 / image_resize);

        // Change to dlib's image format. No memory is copied.
        dlib::cv_image<dlib::bgr_pixel> cimg_small(im_small);
        dlib::cv_image<dlib::bgr_pixel> cimg(im);


        // Detect faces. Some frames are skipped for speed.
        if (count % skip_frames == 0)
        {
            faces = detector(cimg_small);
        }

        if (faces.empty())
        {
            continue;
        }

        // Space for landmarks on multiple faces.
        std::vector<dlib::full_object_detection> shapes;

        // Loop over all faces
        for (unsigned long i = 0; i < faces.size(); ++i)
        {

            // Face detector was found over a smaller image.
            // So, we scale face rectangle to correct size.
            dlib::rectangle r(
                faces[i].left() * image_resize,
                faces[i].top() * image_resize,
                faces[i].right() * image_resize,
                faces[i].bottom() * image_resize
            );

            // Run landmark detector on current frame
            dlib::full_object_detection shape = landmark_detector(cimg, r);

            // Save current face in a vector
            shapes.push_back(shape);

            // Loop over every point
            for (unsigned long k = 0; k < shape.num_parts(); ++k)
            {

                if (is_first_frame)
                {
                    // If it is the first frame copy the current frame points
                    points_prev[k].x = points_detected_prev[k].x = shape.part(k).x();
                    points_prev[k].y = points_detected_prev[k].y = shape.part(k).y();
                }
                else
                {
                    // If not the first frame, copy points from previous frame.
                    points_prev[k] = points[k];
                    points_detected_prev[k] = points_detected_cur[k];
                }

                // pointsDetectedCur stores results returned by the facial landmark detector
                // points stores the stabilized landmark points
                points[k].x = points_detected_cur[k].x = shape.part(k).x();
                points[k].y = points_detected_cur[k].y = shape.part(k).y();
            }

            if (eye_distance_not_calculated)
            {
                eye_distance = interEyeDistance(shape);
                win_size = cv::Size(2 * static_cast<int>(eye_distance / 4) + 1, 2 * static_cast<int>(eye_distance / 4) + 1);
                eye_distance_not_calculated = false;
                dot_radius = eye_distance > 100 ? 3 : 2;
                sigma = eye_distance * eye_distance / 400;
            }

            // Build an image pyramid to speed up optical flow
            cv::buildOpticalFlowPyramid(im_gray, im_gray_pyr, win_size, max_level);

            // Predict landmarks based on optical flow. points stores the new location of points.
            cv::calcOpticalFlowPyrLK(im_gray_prev_pyr, im_gray_pyr, points_prev, points, status, err, win_size, max_level, termcrit, 0, 0.0001);

            // Final landmark points are a weighted average of
            // detected landmarks and tracked landmarks

            for (unsigned long k = 0; k < shape.num_parts(); ++k)
            {
                double n = norm(points_detected_prev[k] - points_detected_cur[k]);
                double alpha = exp(-n * n / sigma);
                points[k] = (1 - alpha) * points_detected_cur[k] + alpha * points[k];
            }

            if (show_stabilized)
            {
                // Show optical flow stabilized points
                render_face(im, points, cv::Scalar(255, 0, 0), dot_radius);
            }
            else
            {
                // Show landmark points (unstabilized)
                render_face(im, points_detected_cur, cv::Scalar(0, 0, 255), dot_radius);
            }
        }

        // Display on screen
        cv::imshow(win_name, im);

        // Wait for keypress
        char key = cv::waitKey(1);

        if (key == 32)
        {
            // If space is pressed toggle showStabilized
            show_stabilized = !show_stabilized;
        }
        else if (key == 27) // ESC
        {
            // If ESC is pressed, exit.
            break;
        }

        // Get ready for next frame.
        im_prev = im.clone();
        im_gray_prev = im_gray.clone();
        im_gray_prev_pyr = im_gray_pyr;
        im_gray_pyr = std::vector<cv::Mat>();

        is_first_frame = false;

        // Calculate framerate
        count++;
        if (count == num_frames_for_fps)
        {
            t = (static_cast<double>(cv::getTickCount()) - t) / cv::getTickFrequency();
            fps = num_frames_for_fps / t;
            count = 0;
        }
        cv::putText(im, cv::format("fps %.2f", fps), cv::Point(50, size.height - 50), cv::FONT_HERSHEY_COMPLEX, 1.5, cv::Scalar(0, 0, 255), 3);
    }

    cap.release();
    cv::destroyAllWindows();
}

// Fill the vector with random colors
void get_random_colors(std::vector<cv::Scalar>& colors, int numColors)
{
	cv::RNG rng(0);
    for (int i = 0; i < numColors; i++)
    {
        colors.emplace_back(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
    }
}

void lucas_kanade_tracker()
{
	std::string videoFileName = util::get_data_path() + "videos/cycle.mp4";
    cv::VideoCapture cap(videoFileName);
	const int width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
	const int height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
	cv::VideoWriter out("sparse-output.mp4", cv::VideoWriter::fourcc('M', 'P', '4', 'V'), 20, cv::Size(width, height));

	cv::TermCriteria termcrit(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 10, 0.03);
    // Take first frame and find corners in it
	cv::Mat old_frame;
    cap >> old_frame;

    cv::imshow("Old frame", old_frame);
    cv::waitKey(1000);

	cv::Mat old_gray;
    cv::cvtColor(old_frame, old_gray, cv::COLOR_BGR2GRAY);

	std::vector<cv::Point2f> old_points;
	std::vector<uchar> status;
	std::vector<float> err;
	std::vector<cv::Point2f> new_points;
	std::vector<cv::Point2f> good_new;
	std::vector<cv::Point2f> good_old;
	std::vector<cv::Scalar> colors;
	cv::Point2f pt1;
	cv::Point2f pt2;

	cv::goodFeaturesToTrack(old_gray,old_points,100, 0.3, 7, cv::Mat(), 7);

	cv::Mat display_frame;
    // Create a mask image for drawing the tracks
	cv::Mat mask = cv::Mat::zeros(old_frame.size().height, old_frame.size().width, CV_8UC3);

	int count = 0;

	cv::Mat frame;
	cv::Mat frame_gray;

	while (true) 
    {
        cap >> frame;

        if (frame.empty())
        {
            std::cout << "over\n";
        }

        cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);
        count += 1;

        // calculate optical flow
        cv::calcOpticalFlowPyrLK(old_gray, frame_gray, old_points, new_points, status, err, cv::Size(15, 15), 2, termcrit);

        for (int i = 0; i < new_points.size(); i++)
        {
            if (status[i] == 1) 
            {
                good_new.push_back(new_points[i]);
                good_old.push_back(old_points[i]);
            }
        }

        get_random_colors(colors, new_points.size());

        // draw the tracks
        for (int j = 0; j < new_points.size(); j++)
        {
            pt1 = new_points[j];
            pt2 = old_points[j];
            cv::line(mask, pt1, pt2, colors[j], 2, cv::LINE_AA);
            cv::circle(frame, pt1, 3, colors[j], -1);
        }

        add(frame, mask, display_frame);
        out.write(display_frame);

        if (count % 5 == 0) 
        {
            cv::imshow("Frame", display_frame);
            cv::waitKey(250);
        }
        if (count > 50)
        {
            break;
        }

        // Now update the previous frame and previous_points
        old_gray = frame_gray.clone();
        std::copy(new_points.begin(), new_points.end(), old_points.begin());
    }
}