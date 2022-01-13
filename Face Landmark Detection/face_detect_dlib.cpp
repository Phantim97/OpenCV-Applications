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
#define OPENCV_FACE_RENDER

void render_face
(
    cv::Mat& img, // Image to draw the points on
    const std::vector<cv::Point2f>& points, // Vector of points
    cv::Scalar color, // color points
    int radius = 3) // Radius of points.
{

    for (int i = 0; i < points.size(); i++)
    {
        cv::circle(img, points[i], radius, color, -1);
    }
}

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
    while (1)
    {
        if (count == 0) t = cv::getTickCount();

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