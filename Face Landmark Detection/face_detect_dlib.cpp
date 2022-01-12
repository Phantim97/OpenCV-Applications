#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
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