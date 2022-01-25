#include "face_swapping.h"

#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>
#include <dlib/image_io.h>
#include <dlib/gui_widgets.h>

#include "env_util.h"
#include "ml_util.h"

void face_swap_main()
{
    // Load face detection and pose estimation models.
    dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
    dlib::shape_predictor predictor;
    const std::string model_path = util::get_model_path() + "dlib_models/shape_predictor_68_face_landmarks.dat";
    dlib::deserialize(model_path) >> predictor;

    double t = static_cast<double>(cv::getTickCount());

    std::string s1;
    std::string s2;

    std::cout << "Enter Face Image: ";
    std::cin >> s1;
    std::cout << "Enter Body Image: ";
    std::cin >> s2;

    //Read two images
    cv::Mat img1 = cv::imread(util::get_data_path() + "images/people/" + s1);
    cv::Mat img2 = cv::imread(util::get_data_path() + "images/people/" + s2);
    cv::Mat img1_warped = img2.clone();

    //Read points
    std::vector<cv::Point2f> points1 = get_landmark_point_vector(img1, "images/people", s1, detector, predictor);
    std::vector<cv::Point2f> points2 = get_landmark_point_vector(img2, "images/people", s2, detector, predictor);

    //convert Mat to float data type
    img1.convertTo(img1, CV_32F);
    img1_warped.convertTo(img1_warped, CV_32F);

    // Find convex hull
    std::vector<cv::Point2f> hull1;
    std::vector<cv::Point2f> hull2;
    std::vector<int> hull_index;

    cv::convexHull(points2, hull_index, false, false);;

    for (int i = 0; i < hull_index.size(); i++)
    {
        hull1.push_back(points1[hull_index[i]]);
        hull2.push_back(points2[hull_index[i]]);
    }

    // Find delaunay triangulation for points on the convex hull
    std::vector<std::vector<int>> dt;
    cv::Rect rect(0, 0, img1_warped.cols, img1_warped.rows);
    
    calculate_delaunay_triangles(rect, hull2, dt);

    // Apply affine transformation to Delaunay triangles
    for (size_t i = 0; i < dt.size(); i++)
    {
        std::vector<cv::Point2f> t1, t2;
        // Get points for img1, img2 corresponding to the triangles
        for (size_t j = 0; j < 3; j++)
        {
            t1.push_back(hull1[dt[i][j]]);
            t2.push_back(hull2[dt[i][j]]);
        }
        warp_triangle(img1, img1_warped, t1, t2);
    }

    double t_clone = (double)cv::getTickCount();
    std::cout << "Benchmark = " << (t_clone - t) / cv::getTickFrequency() << '\n';

    // Calculate mask for seamless cloning
    std::vector<cv::Point> hull8U;
    for (int i = 0; i < hull2.size(); i++)
    {
	    cv::Point pt(hull2[i].x, hull2[i].y);
        hull8U.push_back(pt);
    }

    cv::Mat mask = cv::Mat::zeros(img2.rows, img2.cols, img2.depth());
    cv::fillConvexPoly(mask, &hull8U[0], hull8U.size(), cv::Scalar(255, 255, 255));

    cv::namedWindow("Mask", cv::WINDOW_FREERATIO);
    cv::imshow("Mask", mask);
    cv::waitKey(5000);

    // find center of the mask to be cloned with the destination image
    cv::Rect r = cv::boundingRect(hull2);
    cv::Point center = (r.tl() + r.br()) / 2;

    cv::Mat output;
    img1_warped.convertTo(img1_warped, CV_8UC3);
    cv::seamlessClone(img1_warped, img2, mask, center, output, cv::NORMAL_CLONE);

    cv::namedWindow("Before Seamless Clone", cv::WINDOW_FREERATIO);
    cv::imshow("Before Seamless Clone", img1_warped);
    cv::waitKey(5000);
    cv::namedWindow("Output", cv::WINDOW_FREERATIO);
    cv::imshow("Output", output);
    cv::waitKey(10000);
}

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <string>

#define RESIZE_HEIGHT 480
#define FACE_DOWNSAMPLE_RATIO 1.5
#define SKIP_FRAMES 2

static bool get_output_option()
{
    std::string opt;
    while (opt != "y" && opt != "n")
    {
        std::cin >> opt;

        if (opt != "y" && opt != "n")
        {
            std::cout << "invalid option";
        }
    }

    return opt == "y";
}

void face_swap_video(VidcapMode vm)
{
    // Load face detection and pose estimation models.
    dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
    dlib::shape_predictor predictor;
    std::string model_path = util::get_model_path() + "dlib_models/shape_predictor_68_face_landmarks.dat";

    // Processing input file
    std::string face_file;
    std::cout << "Enter Face Image: ";
    std::cin >> face_file;

    dlib::deserialize(model_path) >> predictor;

    //Read input image and resize it
    std::vector<cv::Point2f> points1;
    cv::Mat img1 = cv::imread(util::get_data_path() + "images/people/" + face_file);

    int height = img1.rows;
    float image_resize = static_cast<float>(height) / RESIZE_HEIGHT;
    cv::resize(img1, img1, cv::Size(), 1.0 / image_resize, 1.0 / image_resize);

    //Find landmark points
    points1 = get_landmark_point_vector(img1, "images/people", face_file, detector, predictor);
    img1.convertTo(img1, CV_32F);

    if (points1.empty())
    {
        std::cout << "Face not detected in image\n";
        return;
    }

    // Find convex hull for delaunay triangulation using the landmark points
    std::vector<int> hull_index;
    cv::convexHull(points1, hull_index, false, false);

    // Add the points on the mouth to the convex hull to create delaunay triangles
    for (int i = 48; i < 59; i++)
    {
        hull_index.push_back(i);
    }

    // Find Delaunay triangulation for convex hull points
    std::vector<std::vector<int>> dt;
    cv::Rect rect(0, 0, img1.cols, img1.rows);
    std::vector<cv::Point2f> hull1;

    hull1.reserve(hull_index.size());

    for (int i = 0; i < hull_index.size(); i++)
    {
        hull1.push_back(points1[hull_index[i]]);
    }

	calculate_delaunay_triangles(rect, hull1, dt);

	std::cout << "Processed input image\n";

    std::cout << "Enable Benchmarking? (y/n): ";
    const bool benchmarking = get_output_option();

    std::cout << "Enable Debug Messages? (y/n): ";
    const bool debug_msg = get_output_option();

    //process input from webcam or video file
    cv::VideoCapture cap;
    if (vm == VidcapMode::VIDEO)
    {
        std::string vid_name;
        std::cout << "Enter video name: ";
        std::cin >> vid_name;

       cap.open(util::get_data_path() + "videos/" + vid_name);
    }
    else
    {
        cap.open(0);
        if (!cap.isOpened())
        {
            std::cout << "Camera not open breaking\n";
            return;
        }
    }

    cv::Mat frame;

    // Read a frame initially to assign memory for the frame and calculate new height
    cap >> frame;
    height = frame.rows;
    image_resize = static_cast<float>(height) / RESIZE_HEIGHT;

    // Declare the variable for landmark points
    std::vector<cv::Point2f> points2;

    // Some variables for tracking time
    int count = 0;
    double t = static_cast<double>(cv::getTickCount());
    double fps = 30.0;

    // Variables for Optical flow  calculation
    cv::TermCriteria termcrit(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 20, 0.03);
    cv::Size sub_pix_win_size(10, 10), winSize(101, 101);
    double eye_distance, sigma;
    bool eye_distance_not_calculated = true;

    std::vector<cv::Point2f> hull2_prev;
    std::vector<cv::Point2f> hull2_next;

    cv::Mat img2_gray;
    cv::Mat img2_gray_prev;

    cv::Mat result;
    cv::Mat output;
    cv::Mat img1_warped;

    std::string win_name;

    if (vm == VidcapMode::LIVE)
    {
        win_name = "Live Feed";
    }

    cv::namedWindow(win_name);

    // Main Loop
    while (true)
    {
        if (vm == VidcapMode::LIVE)
        {
            cap >> frame;
        }
        else if (vm == VidcapMode::VIDEO && !cap.read(frame))
        {
            break;
        }

        if (count == 0)
        {
            t = static_cast<double>(cv::getTickCount());
        }

        double time_detector = static_cast<double>(cv::getTickCount());

        cv::resize(frame, frame, cv::Size(), 1.0 / image_resize, 1.0 / image_resize);

        if (debug_msg)
        {
            std::cout << "Resize complete\n";
        }

        // find landmarks after skipping SKIP_Frames number of frames
        if (count % SKIP_FRAMES == 0)
        {
            points2 = get_landmarks(detector, predictor, frame, (float)FACE_DOWNSAMPLE_RATIO);

        	if (debug_msg)
            {
                std::cout << "Face Detector\n";
            }
        }

        // if face is partially detected
        if (points2.size() != 68)
        {
            if (debug_msg)
            {
                std::cout << "Points not detected\n";
            }

            //To not freeze stream if face not detected
            cv::imshow(win_name, frame);
            cv::waitKey(10);

            continue;
        }

        //convert Mat to float data type
        img1_warped = frame.clone();
        img1_warped.convertTo(img1_warped, CV_32F);

        // Find convex hull
        std::vector<cv::Point2f> hull2;

        hull2.reserve(hull_index.size());

        for (int i = 0; i < hull_index.size(); i++)
        {
            hull2.push_back(points2[hull_index[i]]);
        }

        ////////// Calculation of Optical flow and Stabilization of Landmark points ////////////
        if (hull2_prev.empty())
        {
            hull2_prev = hull2;
        }

        double t1 = static_cast<double>(cv::getTickCount());

        if (eye_distance_not_calculated)
        {
            eye_distance = norm(points2[36] - points2[45]);
            winSize = cv::Size(2 * static_cast<int>(eye_distance / 4) + 1, 2 * static_cast<int>(eye_distance / 4) + 1);
            eye_distance_not_calculated = false;
            sigma = eye_distance * eye_distance / 400;
        }

        cv::cvtColor(frame, img2_gray, cv::COLOR_BGR2GRAY);

        if (img2_gray_prev.empty())
        {
            img2_gray_prev = img2_gray.clone();
        }

        std::vector<uchar> status;
        std::vector<float> err;

        // Calculate Optical Flow based estimate of the point in this frame
        cv::calcOpticalFlowPyrLK(img2_gray_prev, img2_gray, hull2_prev, hull2_next, status, err, winSize,
            5, termcrit, 0, 0.001);

        // Final landmark points are a weighted average of detected landmarks and tracked landmarks
        for (unsigned long k = 0; k < hull2.size(); ++k)
        {
            double n = norm(hull2_next[k] - hull2[k]);
            double alpha = exp(-n * n / sigma);
            hull2[k] = (1 - alpha) * hull2[k] + alpha * hull2_next[k];
            constrain_point(hull2[k], frame.size());
        }

        // Update varibales for next pass
        hull2_prev = hull2;
        img2_gray_prev = img2_gray.clone();

        /////////// Finished Stabilization code   //////////////////////////////////

        // Apply affine transformation to Delaunay triangles
        for (size_t i = 0; i < dt.size(); i++)
        {
	        std::vector<cv::Point2f> t1;
	        std::vector<cv::Point2f> t2;

	        // Get points for img1, img2 corresponding to the triangles
            for (size_t j = 0; j < 3; j++)
            {
                t1.push_back(hull1[dt[i][j]]);
                t2.push_back(hull2[dt[i][j]]);
            }

            warp_triangle(img1, img1_warped, t1, t2);
        }

        if (benchmarking)
        {
            std::cout << "Stabilize and Warp time" << (static_cast<double>(cv::getTickCount()) - t1) / cv::getTickFrequency() << '\n';
        }

        /////////////////////////   Blending   /////////////////////////////////////////////////////////////

        img1_warped.convertTo(img1_warped, CV_8UC3);

        // Color Correction of the warped image so that the source color matches that of the destination
        output = correct_colors(frame, img1_warped, points2);

        // imshow("Before Blending", output);

        // Create a Mask around the face
        cv::Rect re = cv::boundingRect(hull2);
        cv::Point center = (re.tl() + re.br()) / 2;
        std::vector<cv::Point> hull3;

        for (int i = 0; i < hull2.size() - 12; i++)
        {
            //Take the points just inside of the convex hull
            cv::Point pt1(0.95 * (hull2[i].x - center.x) + center.x, 0.95 * (hull2[i].y - center.y) + center.y);
            hull3.push_back(pt1);
        }

        cv::Mat mask1 = cv::Mat::zeros(frame.rows, frame.cols, frame.type());

        cv::fillConvexPoly(mask1, &hull3[0], hull3.size(), cv::Scalar(255, 255, 255));

        // Blur the mask before blending
        cv::GaussianBlur(mask1, mask1, cv::Size(21, 21), 10);

        cv::Mat mask2 = cv::Scalar(255, 255, 255) - mask1;
        // imshow("mask1",mask1);
        // imshow("mask2",mask2);

        // Perform alpha blending of the two images
        cv::Mat temp1 = output.mul(mask1, 1.0 / 255);
        cv::Mat temp2 = frame.mul(mask2, 1.0 / 255);
        result = temp1 + temp2;
        // imshow("temp1",temp1);
        // imshow("temp2",temp2);

    //////////////////////////////////////////////////////////////////////////////////////////////////////
        if (benchmarking)
        {
            std::cout << "Total time" << (static_cast<double>(cv::getTickCount()) - time_detector) / cv::getTickFrequency() << '\n';
        }

        cv::imshow(win_name, result);

        int k = cv::waitKey(1);
        // Quit if  ESC is pressed
        if (k == 27)
        {
            break;
        }

        count++;

        if (count == 10)
        {
            fps = 10.0 * cv::getTickFrequency() / (static_cast<double>(cv::getTickCount()) - t);
            count = 0;
        }

        if (benchmarking)
        {
            std::cout << "FPS " << fps << '\n';
        }
    }

    cap.release();
}